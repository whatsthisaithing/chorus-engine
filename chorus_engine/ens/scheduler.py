"""ENS v3 scheduler scaffold.

This module provides queue + tick primitives behind v3 flags while preserving
the existing ENSRuntime ingest path until full rollout.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
import logging
import time
import uuid

from sqlalchemy import case
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chorus_engine.ens.models import ENSOutcome, Signal
from chorus_engine.ens.arbitration import ArbitrationEngine, ArbitrationSelection
from chorus_engine.models.ens import ENSFloorControlState, ENSLoopSession, ENSSchedulerTick, ENSSignalQueue
from chorus_engine.ens.time_utils import next_created_at_us


_PRIORITY_RANK = {"user": 0, "system": 1, "loop": 2}
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

logger = logging.getLogger(__name__)
FALLBACK_RELATIONSHIP_ID = "system/unknown"

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


def _relationship_id_for_signal(signal: Signal) -> Tuple[str, bool]:
    payload = dict(getattr(signal, "payload", {}) or {})
    raw = (
        getattr(signal, "relationship_hint", None)
        or payload.get("relationship_hint")
        or payload.get("relationship_id")
    )
    value = str(raw or "").strip()
    if value:
        return value, False
    return FALLBACK_RELATIONSHIP_ID, True


class ENSScheduler:
    """Persistent queue/tick selector for v3 rollout."""

    def __init__(self) -> None:
        self.arbitration = ArbitrationEngine()
        self.unresolved_relationship_fallback_count = 0

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
        signal_type = str(signal.type or "")
        signal_payload = dict(payload.get("payload", {}) or {})
        loop_id = str(signal_payload.get("loop_id") or "").strip() or None
        if signal_type == "loop_progression" and not loop_id:
            raise ValueError("loop_progression requires payload.loop_id")
        if signal_type == "loop_progression" and loop_id:
            existing_loop_pending = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.status == STATUS_PENDING)
                .filter(ENSSignalQueue.loop_id == loop_id)
                .order_by(ENSSignalQueue.created_at.asc())
                .first()
            )
            if existing_loop_pending:
                return existing_loop_pending

        relationship_id, used_fallback_relationship = _relationship_id_for_signal(signal)
        if used_fallback_relationship:
            self.unresolved_relationship_fallback_count += 1
            logger.info(
                "signal_relationship_fallback_applied signal_id=%s fallback_relationship_id=%s count=%s",
                signal.signal_id,
                relationship_id,
                self.unresolved_relationship_fallback_count,
            )
        row = ENSSignalQueue(
            queue_id=str(uuid.uuid4()),
            signal_id=str(signal.signal_id),
            signal_type=signal_type,
            relationship_id=relationship_id,
            loop_id=loop_id,
            conversation_id=signal_payload.get("conversation_id"),
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
            if signal_type == "loop_progression" and loop_id:
                existing_loop_pending = (
                    db.query(ENSSignalQueue)
                    .filter(ENSSignalQueue.signal_type == "loop_progression")
                    .filter(ENSSignalQueue.status == STATUS_PENDING)
                    .filter(ENSSignalQueue.loop_id == loop_id)
                    .order_by(ENSSignalQueue.created_at.asc())
                    .first()
                )
                if existing_loop_pending:
                    return existing_loop_pending
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

    def update_floor_state_from_signal(
        self,
        db: Session,
        signal: Signal,
        *,
        attention_lock_seconds: int,
    ) -> None:
        """Upsert floor/attention-lock state from user ingress signals.

        De-confliction only: this influences arbitration weighting and must not
        be used to hard-gate runnable signals.
        """
        if _priority_tier_for_signal(signal) != "user":
            return
        relationship_id, used_fallback = _relationship_id_for_signal(signal)
        surface_id = str(getattr(signal, "surface_id", "") or "")
        if used_fallback or not relationship_id or not surface_id:
            return
        ttl_us = max(0, int(attention_lock_seconds)) * 1_000_000
        if ttl_us <= 0:
            return
        now_us = time.time_ns() // 1000
        lock_until = now_us + ttl_us
        lock_source_signal_id = str(signal.signal_id)

        def _apply_state(state: ENSFloorControlState) -> None:
            state.active_surface_id = surface_id
            state.attention_lock_until_us = max(int(state.attention_lock_until_us or 0), lock_until)
            state.lock_source_signal_id = lock_source_signal_id
            meta = dict(state.metadata_json or {})
            meta["mode"] = "deconfliction_weight_only"
            state.metadata_json = meta

        state = (
            db.query(ENSFloorControlState)
            .filter(ENSFloorControlState.relationship_id == relationship_id)
            .first()
        )
        if not state:
            db.add(
                ENSFloorControlState(
                    id=str(uuid.uuid4()),
                    relationship_id=relationship_id,
                    active_surface_id=surface_id,
                    attention_lock_until_us=lock_until,
                    lock_source_signal_id=lock_source_signal_id,
                    metadata_json={"mode": "deconfliction_weight_only"},
                )
            )
            try:
                db.commit()
                return
            except IntegrityError:
                # Concurrent ingress may win the insert race; recover with an update path.
                db.rollback()
                logger.info(
                    "floor_state_insert_race_recovered relationship_id=%s signal_id=%s",
                    relationship_id,
                    lock_source_signal_id,
                )

        state = (
            db.query(ENSFloorControlState)
            .filter(ENSFloorControlState.relationship_id == relationship_id)
            .first()
        )
        if not state:
            logger.warning(
                "floor_state_upsert_missing_after_race relationship_id=%s signal_id=%s",
                relationship_id,
                lock_source_signal_id,
            )
            return
        _apply_state(state)
        db.commit()

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

    def _last_non_user_selection(self, db: Session) -> Tuple[Optional[str], Optional[str]]:
        row = (
            db.query(ENSSignalQueue)
            .join(
                ENSSchedulerTick,
                ENSSchedulerTick.selected_signal_id == ENSSignalQueue.signal_id,
            )
            .filter(ENSSignalQueue.priority_tier.in_(("system", "loop")))
            .order_by(ENSSchedulerTick.created_at_us.desc())
            .first()
        )
        if not row:
            return None, None
        return str(row.surface_id or ""), str(row.relationship_id or "")

    def _pending_candidates(self, db: Session) -> list[ENSSignalQueue]:
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
            .all()
        )

    def _active_floor_locks(
        self,
        db: Session,
        *,
        now_us: int,
        relationship_ids: list[str],
    ) -> Dict[str, Dict[str, Any]]:
        ids = sorted(set([rid for rid in relationship_ids if rid and rid != FALLBACK_RELATIONSHIP_ID]))
        if not ids:
            return {}
        rows = (
            db.query(ENSFloorControlState)
            .filter(ENSFloorControlState.relationship_id.in_(ids))
            .filter(ENSFloorControlState.attention_lock_until_us.isnot(None))
            .filter(ENSFloorControlState.attention_lock_until_us > now_us)
            .all()
        )
        return {
            str(row.relationship_id): {
                "active_surface_id": str(row.active_surface_id or ""),
                "attention_lock_until_us": int(row.attention_lock_until_us or 0),
                "lock_source_signal_id": str(row.lock_source_signal_id or ""),
            }
            for row in rows
        }

    def _select_with_arbitration(self, db: Session) -> ArbitrationSelection:
        candidates = self._pending_candidates(db)
        last_surface, last_relationship = self._last_non_user_selection(db)
        now_us = time.time_ns() // 1000
        locks = self._active_floor_locks(
            db,
            now_us=now_us,
            relationship_ids=[str(c.relationship_id or "") for c in candidates],
        )
        return self.arbitration.select(
            candidates,
            last_non_user_surface_id=last_surface,
            last_non_user_relationship_id=last_relationship,
            floor_locks_by_relationship=locks,
        )

    def _assistant_id_for_row(self, row: ENSSignalQueue) -> str:
        signal_doc = dict(getattr(row, "signal_json", {}) or {})
        payload = dict(signal_doc.get("payload") or {})
        return str(
            signal_doc.get("assistant_id")
            or payload.get("assistant_id")
            or ""
        ).strip()

    def _evaluate_limits(
        self,
        db: Session,
        *,
        candidates: list[ENSSignalQueue],
        now_us: int,
        surface_rate_cap_per_minute: int,
        assistant_rate_cap_per_minute: int,
        surface_cooldown_ms: int,
    ) -> Tuple[list[ENSSignalQueue], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        ordered = sorted(
            list(candidates or []),
            key=lambda r: (int(r.created_at_us or 0), str(r.signal_id)),
        )
        if not ordered:
            return [], {}, {
                "budgets_check": "pass",
                "cooldowns_check": "pass",
                "blocked_candidate_count": 0,
            }

        surface_cap = max(0, int(surface_rate_cap_per_minute or 0))
        assistant_cap = max(0, int(assistant_rate_cap_per_minute or 0))
        cooldown_us = max(0, int(surface_cooldown_ms or 0)) * 1000
        limits_enabled = bool(surface_cap > 0 or assistant_cap > 0 or cooldown_us > 0)
        if not limits_enabled:
            return ordered, {}, {
                "budgets_check": "pass",
                "cooldowns_check": "pass",
                "blocked_candidate_count": 0,
            }

        now_dt = datetime.utcnow()
        window_start_dt = now_dt - timedelta(seconds=60)
        recent_rows = (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.status.in_(("done", "completed")))
            .filter(ENSSignalQueue.completed_at.isnot(None))
            .filter(ENSSignalQueue.completed_at >= window_start_dt)
            .filter(ENSSignalQueue.completed_at <= now_dt)
            .all()
        )
        running_rows = (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.status == STATUS_RUNNING)
            .filter(ENSSignalQueue.selected_at.isnot(None))
            .filter(ENSSignalQueue.selected_at >= window_start_dt)
            .filter(ENSSignalQueue.selected_at <= now_dt)
            .all()
        )
        surface_counts: Dict[str, int] = {}
        assistant_counts: Dict[str, int] = {}
        surface_last_completed_at: Dict[str, datetime] = {}
        for row in recent_rows:
            completion_at = getattr(row, "completed_at", None)
            if not completion_at:
                continue
            surface = str(getattr(row, "surface_id", "") or "")
            if surface:
                surface_counts[surface] = int(surface_counts.get(surface, 0)) + 1
                prior = surface_last_completed_at.get(surface)
                if prior is None or completion_at > prior:
                    surface_last_completed_at[surface] = completion_at
            assistant_id = self._assistant_id_for_row(row)
            if assistant_id:
                assistant_counts[assistant_id] = int(assistant_counts.get(assistant_id, 0)) + 1

        # Reserve budget for in-flight claims so concurrent ticks cannot overrun caps.
        for row in running_rows:
            surface = str(getattr(row, "surface_id", "") or "")
            if surface:
                surface_counts[surface] = int(surface_counts.get(surface, 0)) + 1
            assistant_id = self._assistant_id_for_row(row)
            if assistant_id:
                assistant_counts[assistant_id] = int(assistant_counts.get(assistant_id, 0)) + 1

        eligible: list[ENSSignalQueue] = []
        blocked: Dict[str, Dict[str, Any]] = {}
        for row in ordered:
            surface_id = str(getattr(row, "surface_id", "") or "")
            assistant_id = self._assistant_id_for_row(row)
            stop_reason = ""
            reason_details: Dict[str, Any] = {}
            if surface_cap > 0 and surface_id:
                used = int(surface_counts.get(surface_id, 0))
                if used >= surface_cap:
                    stop_reason = "surface_rate_cap_exhausted"
                    reason_details = {"surface_id": surface_id, "used": used, "limit": surface_cap}
            if not stop_reason and assistant_cap > 0 and assistant_id:
                used = int(assistant_counts.get(assistant_id, 0))
                if used >= assistant_cap:
                    stop_reason = "assistant_rate_cap_exhausted"
                    reason_details = {"assistant_id": assistant_id, "used": used, "limit": assistant_cap}
            if not stop_reason and cooldown_us > 0 and surface_id:
                last_completed_at = surface_last_completed_at.get(surface_id)
                if last_completed_at and now_dt < (last_completed_at + timedelta(microseconds=cooldown_us)):
                    last_completed_us = int(last_completed_at.timestamp() * 1_000_000)
                    stop_reason = "surface_cooldown_active"
                    reason_details = {
                        "surface_id": surface_id,
                        "last_completed_us": last_completed_us,
                        "cooldown_ms": int(surface_cooldown_ms or 0),
                    }
            if stop_reason:
                blocked[str(row.signal_id)] = {
                    "stop_reason": stop_reason,
                    "details": reason_details,
                }
                continue
            eligible.append(row)

        return eligible, blocked, {
            "budgets_check": "enforced",
            "cooldowns_check": "enforced" if cooldown_us > 0 else "pass",
            "blocked_candidate_count": len(blocked),
            "surface_rate_cap_per_minute": surface_cap,
            "assistant_rate_cap_per_minute": assistant_cap,
            "surface_cooldown_ms": int(surface_cooldown_ms or 0),
        }

    def _stop_blocked_signal(
        self,
        db: Session,
        *,
        row: ENSSignalQueue,
        stop_reason: str,
        stop_details: Dict[str, Any],
        limits_trace: Dict[str, Any],
    ) -> None:
        row.status = STATUS_FAILED
        row.completed_at = datetime.utcnow()
        row.claimed_at_us = None
        row.error_message = str(stop_reason)
        if str(row.signal_type or "") == "loop_progression" and str(row.loop_id or ""):
            loop = (
                db.query(ENSLoopSession)
                .filter(ENSLoopSession.loop_id == row.loop_id)
                .first()
            )
            if loop is not None:
                loop.state = "stopped"
                loop.stop_reason = str(stop_reason)
        reason_trace = {
            "selection": "arbitration_v3",
            "phase": "budget_cooldown_stop",
            "selected_signal_id": row.signal_id,
            "selected_priority_tier": row.priority_tier,
            "stop_reason": stop_reason,
            "stop_details": dict(stop_details or {}),
            "status": STATUS_FAILED,
            **dict(limits_trace or {}),
        }
        tick = ENSSchedulerTick(
            tick_id=str(uuid.uuid4()),
            queue_id=row.queue_id,
            selected_signal_id=row.signal_id,
            reason_trace_json=reason_trace,
            tie_break_json={"applied": False, "signal_id": row.signal_id},
            created_at_us=next_created_at_us(),
        )
        db.add(tick)
        db.commit()
        logger.info(
            "signal_stopped_by_scheduler_policy signal_id=%s stop_reason=%s queue_id=%s",
            row.signal_id,
            stop_reason,
            row.queue_id,
        )

    def _claim_next(
        self,
        db: Session,
        *,
        arbitration_enabled: bool,
        surface_rate_cap_per_minute: int = 0,
        assistant_rate_cap_per_minute: int = 0,
        surface_cooldown_ms: int = 0,
    ) -> Tuple[Optional[ENSSignalQueue], Dict[str, Any], Dict[str, Any]]:
        """Atomically claim one pending signal for execution."""
        claim_us = next_created_at_us()
        for _ in range(8):
            if arbitration_enabled:
                candidates = self._pending_candidates(db)
                now_us = time.time_ns() // 1000
                eligible, blocked, limits_trace = self._evaluate_limits(
                    db,
                    candidates=candidates,
                    now_us=now_us,
                    surface_rate_cap_per_minute=surface_rate_cap_per_minute,
                    assistant_rate_cap_per_minute=assistant_rate_cap_per_minute,
                    surface_cooldown_ms=surface_cooldown_ms,
                )
                if not eligible:
                    if candidates and blocked:
                        stop_row = sorted(
                            candidates,
                            key=lambda r: (int(r.created_at_us or 0), str(r.signal_id)),
                        )[0]
                        blocked_item = dict(blocked.get(str(stop_row.signal_id)) or {})
                        stop_reason = str(blocked_item.get("stop_reason") or "scheduler_policy_blocked")
                        stop_details = dict(blocked_item.get("details") or {})
                        self._stop_blocked_signal(
                            db,
                            row=stop_row,
                            stop_reason=stop_reason,
                            stop_details=stop_details,
                            limits_trace=limits_trace,
                        )
                    return None, {}, {}
                last_surface, last_relationship = self._last_non_user_selection(db)
                locks = self._active_floor_locks(
                    db,
                    now_us=now_us,
                    relationship_ids=[str(c.relationship_id or "") for c in eligible],
                )
                arbitration = self.arbitration.select(
                    eligible,
                    last_non_user_surface_id=last_surface,
                    last_non_user_relationship_id=last_relationship,
                    floor_locks_by_relationship=locks,
                )
                candidate = arbitration.selected
                reason_trace = dict(arbitration.reason_trace or {})
                reason_trace.update(dict(limits_trace or {}))
                tie_break = dict(arbitration.tie_break or {})
            else:
                candidate = self._select_next(db)
                reason_trace = {}
                tie_break = {}
            if not candidate:
                return None, reason_trace, tie_break
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
                row = (
                    db.query(ENSSignalQueue)
                    .filter(ENSSignalQueue.queue_id == candidate.queue_id)
                    .first()
                )
                return row, reason_trace, tie_break
        return None, {}, {}

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
        arbitration_enabled: bool = False,
        surface_rate_cap_per_minute: int = 0,
        assistant_rate_cap_per_minute: int = 0,
        surface_cooldown_ms: int = 0,
    ) -> Optional[ENSOutcome]:
        candidate_count_before_claim = (
            db.query(ENSSignalQueue)
            .filter(ENSSignalQueue.status == STATUS_PENDING)
            .count()
        )
        row, arbitration_reason_trace, arbitration_tie_break = self._claim_next(
            db,
            arbitration_enabled=arbitration_enabled,
            surface_rate_cap_per_minute=surface_rate_cap_per_minute,
            assistant_rate_cap_per_minute=assistant_rate_cap_per_minute,
            surface_cooldown_ms=surface_cooldown_ms,
        )
        if not row:
            return None

        if arbitration_enabled:
            reason_trace = dict(arbitration_reason_trace or {})
            tie_break = dict(arbitration_tie_break or {})
            reason_trace.setdefault("status", row.status)
            reason_trace.setdefault("priority_tier", row.priority_tier)
        else:
            reason_trace = {
                "selection": "priority_then_created_at_then_signal_id",
                "candidate_count": candidate_count_before_claim,
                "selected_signal_id": row.signal_id,
                "selected_priority_tier": row.priority_tier,
                "priority_tier": row.priority_tier,
                "status": row.status,
            }
            tie_break = {
                "applied": candidate_count_before_claim > 1,
                "created_at_us": row.created_at_us,
                "signal_id": row.signal_id,
            }

        tick = ENSSchedulerTick(
            tick_id=str(uuid.uuid4()),
            queue_id=row.queue_id,
            selected_signal_id=row.signal_id,
            reason_trace_json=reason_trace,
            tie_break_json=tie_break,
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

