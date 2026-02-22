"""ENS v3 scheduler scaffold.

This module provides queue + tick primitives behind v3 flags while preserving
the existing ENSRuntime ingest path until full rollout.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional
import logging
import time
import uuid

from sqlalchemy import case
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chorus_engine.ens.models import ENSOutcome, Signal
from chorus_engine.models.ens import ENSSchedulerTick, ENSSignalQueue
from chorus_engine.ens.time_utils import next_created_at_us


_PRIORITY_RANK = {"user": 0, "system": 1, "loop": 2}
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

logger = logging.getLogger(__name__)

def _priority_tier_for_signal(signal: Signal) -> str:
    signal_type = (signal.type or "").lower()
    if signal_type in ("user.message", "chat.message", "voice.transcript.final"):
        return "user"
    if "loop" in signal_type:
        return "loop"
    return "system"


def _idempotency_key_for_signal(signal: Signal) -> Optional[str]:
    key = str(getattr(signal, "idempotency_key", "") or "").strip()
    if key:
        return key
    payload = dict(getattr(signal, "payload", {}) or {})
    payload_key = str(payload.get("idempotency_key") or "").strip()
    return payload_key or None


class ENSScheduler:
    """Persistent queue/tick selector for v3 rollout."""

    def enqueue(self, db: Session, signal: Signal) -> ENSSignalQueue:
        existing = (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.signal_id == signal.signal_id)
            .first()
        )
        if existing:
            return existing

        idempotency_key = _idempotency_key_for_signal(signal)
        if idempotency_key:
            existing_by_key = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.idempotency_key == idempotency_key)
                .order_by(ENSSignalQueue.created_at.asc())
                .first()
            )
            if existing_by_key:
                return existing_by_key

        payload = dict(getattr(signal, "__dict__", {}) or {})
        row = ENSSignalQueue(
            queue_id=str(uuid.uuid4()),
            signal_id=str(signal.signal_id),
            signal_type=str(signal.type),
            relationship_id=payload.get("relationship_hint"),
            conversation_id=payload.get("payload", {}).get("conversation_id"),
            surface_id=payload.get("surface_id"),
            priority_tier=_priority_tier_for_signal(signal),
            created_at_us=int(getattr(signal, "created_at_us", 0) or next_created_at_us()),
            idempotency_key=idempotency_key,
            signal_json=payload,
            status=STATUS_PENDING,
        )
        db.add(row)
        try:
            db.commit()
            db.refresh(row)
            return row
        except IntegrityError:
            db.rollback()
            if idempotency_key:
                existing_by_key = (
                    db.query(ENSSignalQueue)
                    .filter(ENSSignalQueue.idempotency_key == idempotency_key)
                    .order_by(ENSSignalQueue.created_at.asc())
                    .first()
                )
                if existing_by_key:
                    return existing_by_key
            existing_by_signal = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_id == signal.signal_id)
                .first()
            )
            if existing_by_signal:
                return existing_by_signal
            raise

    def _select_next(self, db: Session) -> Optional[ENSSignalQueue]:
        return (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.status == STATUS_PENDING)
            .order_by(
                case(
                    (ENSSignalQueue.priority_tier == "user", 0),
                    (ENSSignalQueue.priority_tier == "system", 1),
                    (ENSSignalQueue.priority_tier == "loop", 2),
                    else_=1,
                ).asc(),
                ENSSignalQueue.created_at_us.asc(),
                ENSSignalQueue.signal_id.asc(),
            )
            .first()
        )

    def _claim_next(self, db: Session) -> Optional[ENSSignalQueue]:
        """Atomically claim one pending signal for execution."""
        claim_us = next_created_at_us()
        for _ in range(8):
            candidate = self._select_next(db)
            if not candidate:
                return None
            claimed = (
                db.query(ENSSignalQueue)
                .filter(
                    ENSSignalQueue.queue_id == candidate.queue_id,
                    ENSSignalQueue.status == STATUS_PENDING,
                )
                .update(
                    {
                        ENSSignalQueue.status: STATUS_RUNNING,
                        ENSSignalQueue.selected_at: datetime.utcnow(),
                        ENSSignalQueue.claimed_at_us: claim_us,
                    },
                    synchronize_session=False,
                )
            )
            db.commit()
            if claimed == 1:
                return (
                    db.query(ENSSignalQueue)
                    .filter(ENSSignalQueue.queue_id == candidate.queue_id)
                    .first()
                )
        return None

    def _running_age_us(self, row: ENSSignalQueue, now_us: int) -> int:
        claimed_at_us = int(getattr(row, "claimed_at_us", 0) or 0)
        if claimed_at_us > 0:
            return max(0, now_us - claimed_at_us)
        selected_at = getattr(row, "selected_at", None)
        if selected_at:
            return max(0, now_us - int(selected_at.timestamp() * 1_000_000))
        return 0

    def recover_stuck_running(
        self,
        db: Session,
        *,
        running_ttl_us: int,
    ) -> int:
        """Recover stale running signals back to pending status."""
        now_us = time.time_ns() // 1000
        recovered = 0
        rows = db.query(ENSSignalQueue).filter(ENSSignalQueue.status == STATUS_RUNNING).all()
        for row in rows:
            age_us = self._running_age_us(row, now_us)
            if age_us <= running_ttl_us:
                continue
            row.status = STATUS_PENDING
            row.error_message = "stuck_running_recovered"
            row.selected_at = None
            row.claimed_at_us = None
            row.completed_at = None
            recovered += 1
            logger.warning(
                "signal_recovered_from_stuck_running signal_id=%s age_us=%s queue_id=%s",
                row.signal_id,
                age_us,
                row.queue_id,
            )
        if recovered > 0:
            db.commit()
        return recovered

    async def tick(
        self,
        db: Session,
        *,
        execute_signal: Callable[[Signal], Awaitable[ENSOutcome]],
    ) -> Optional[ENSOutcome]:
        candidate_count = (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.status == STATUS_PENDING)
            .count()
        )
        row = self._claim_next(db)
        if not row:
            return None

        tick = ENSSchedulerTick(
            tick_id=str(uuid.uuid4()),
            queue_id=row.queue_id,
            selected_signal_id=row.signal_id,
            reason_trace_json={
                "selection": "priority_then_created_at_then_signal_id",
                "candidate_count": candidate_count,
                "selected_signal_id": row.signal_id,
                "selected_priority_tier": row.priority_tier,
                "priority_tier": row.priority_tier,
                "status": row.status,
            },
            tie_break_json={
                "applied": candidate_count > 1,
                "created_at_us": row.created_at_us,
                "signal_id": row.signal_id,
            },
            created_at_us=next_created_at_us(),
        )
        db.add(tick)
        db.commit()

        signal_doc: Dict[str, Any] = dict(row.signal_json or {})
        signal = Signal(**signal_doc)

        try:
            outcome = await execute_signal(signal)
            row.status = STATUS_DONE
            row.completed_at = datetime.utcnow()
            row.claimed_at_us = None
            row.error_message = None
            db.commit()
            return outcome
        except Exception as exc:
            row.status = STATUS_FAILED
            row.completed_at = datetime.utcnow()
            row.claimed_at_us = None
            row.error_message = str(exc)
            db.commit()
            raise

