"""ENS v3 scheduler scaffold.

This module provides queue + tick primitives behind v3 flags while preserving
the existing ENSRuntime ingest path until full rollout.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional
import uuid

from sqlalchemy import case
from sqlalchemy.orm import Session

from chorus_engine.ens.models import ENSOutcome, SignalEnvelope
from chorus_engine.models.ens import ENSSchedulerTick, ENSSignalQueue
from chorus_engine.ens.time_utils import next_created_at_us


_PRIORITY_RANK = {"user": 0, "system": 1, "loop": 2}
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _priority_tier_for_signal(signal: SignalEnvelope) -> str:
    signal_type = (signal.type or "").lower()
    if signal_type in ("user.message", "chat.message", "voice.transcript.final"):
        return "user"
    if "loop" in signal_type:
        return "loop"
    return "system"


class ENSScheduler:
    """Persistent queue/tick selector for v3 rollout."""

    def enqueue(self, db: Session, signal: SignalEnvelope) -> ENSSignalQueue:
        existing = (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.signal_id == signal.signal_id)
            .first()
        )
        if existing:
            return existing

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
            idempotency_key=payload.get("idempotency_key"),
            signal_json=payload,
            status=STATUS_PENDING,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

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

    async def tick(
        self,
        db: Session,
        *,
        execute_signal: Callable[[SignalEnvelope], Awaitable[ENSOutcome]],
    ) -> Optional[ENSOutcome]:
        row = self._claim_next(db)
        if not row:
            return None

        tick = ENSSchedulerTick(
            tick_id=str(uuid.uuid4()),
            queue_id=row.queue_id,
            selected_signal_id=row.signal_id,
            reason_trace_json={
                "selection": "priority_then_created_at_then_signal_id",
                "priority_tier": row.priority_tier,
                "status": row.status,
            },
            tie_break_json={
                "created_at_us": row.created_at_us,
                "signal_id": row.signal_id,
            },
            created_at_us=next_created_at_us(),
        )
        db.add(tick)
        db.commit()

        signal_doc: Dict[str, Any] = dict(row.signal_json or {})
        signal = SignalEnvelope(**signal_doc)

        try:
            outcome = await execute_signal(signal)
            row.status = STATUS_DONE
            row.completed_at = datetime.utcnow()
            row.error_message = None
            db.commit()
            return outcome
        except Exception as exc:
            row.status = STATUS_FAILED
            row.completed_at = datetime.utcnow()
            row.error_message = str(exc)
            db.commit()
            raise
