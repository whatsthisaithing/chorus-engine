"""ENS v3 development test harness CLI.

Dev-only utility for queue/scheduler/loop testing without UI.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func

from chorus_engine.config import ConfigLoader
from chorus_engine.db.migrations import ensure_database_ready
from chorus_engine.db.database import DATABASE_URL
from chorus_engine.db.database import SessionLocal
from chorus_engine.ens.models import Signal
from chorus_engine.ens.runtime import ENSContext, ENSRuntime
from chorus_engine.models.ens import ENSDecision, ENSLoopSession, ENSSchedulerTick, ENSSignalQueue


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    excerpts: List[str]


class HarnessError(RuntimeError):
    """Harness fatal error."""


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.finish_reason = "stop"
        self.usage = {"total_tokens": 64}


class _HarnessLLMClient:
    base_url = "harness://llm"

    async def health_check(self) -> bool:
        return True

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> _DummyResponse:
        _ = (prompt, system_prompt, model, kwargs)
        return _DummyResponse(
            "Harness loop response.\n"
            "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
            '{"version":1,"control":{"action":"YIELD"},"tool_calls":[]}\n'
            "---CHORUS_TOOL_PAYLOAD_END---"
        )

    async def generate_with_history(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> _DummyResponse:
        _ = (messages, temperature, max_tokens, model)
        return _DummyResponse("Harness chat response.")


class ENSV3Harness:
    """Dev harness wrapper around ENS runtime + DB models."""

    def __init__(self) -> None:
        ensure_database_ready(DATABASE_URL)
        loader = ConfigLoader()
        system_config = loader.load_system_config()
        characters = loader.load_all_characters()
        app_state: Dict[str, Any] = {
            "system_config": system_config,
            "characters": characters,
            "llm_client": _HarnessLLMClient(),
            "llm_invocation_service": None,
            "ens_tool_executor": None,
            "ens_scene_preview_executor": None,
        }
        # Dev harness forces scheduler + loop sessions on in-memory to exercise v3.
        ens_cfg = system_config.ens
        ens_cfg.enabled = True
        ens_cfg.v3_scheduler_enabled = True
        ens_cfg.v3_loop_sessions_enabled = True
        app_state["ens_runtime"] = ENSRuntime(app_state)
        self.app_state = app_state
        self.runtime: ENSRuntime = app_state["ens_runtime"]

    def _queue_row_for_signal(self, signal_id: str) -> Optional[ENSSignalQueue]:
        db = SessionLocal()
        try:
            return (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_id == signal_id)
                .first()
            )
        finally:
            db.close()

    def _wait_for_signal_completion(self, signal_id: str, max_ticks: int = 20) -> Optional[ENSSignalQueue]:
        row = self._queue_row_for_signal(signal_id)
        if row is not None and str(row.status) in ("done", "failed", "completed"):
            return row
        for _ in range(max(1, int(max_ticks))):
            _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
            row = self._queue_row_for_signal(signal_id)
            if row is not None and str(row.status) in ("done", "failed", "completed"):
                return row
        return row

    @staticmethod
    def _print_header(title: str) -> None:
        print(f"\n=== {title} ===")

    @staticmethod
    def _fmt_ts(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _signal_excerpt(row: ENSSignalQueue) -> str:
        return (
            f"signal_id={row.signal_id} type={row.signal_type} tier={row.priority_tier} status={row.status} "
            f"rel={row.relationship_id} surface={row.surface_id} loop_id={row.loop_id} "
            f"created_at_us={row.created_at_us} idem={row.idempotency_key}"
        )

    @staticmethod
    def _loop_excerpt(row: ENSLoopSession) -> str:
        return (
            f"loop_id={row.loop_id} kind={row.loop_kind} state={row.state} "
            f"step_index={row.step_index} step_count={row.step_count} "
            f"token_budget_used={row.token_budget_used} tool_budget_used={row.tool_budget_used} "
            f"updated_at={ENSV3Harness._fmt_ts(row.updated_at)}"
        )

    @staticmethod
    def _tick_excerpt(row: ENSSchedulerTick) -> str:
        reason = dict(row.reason_trace_json or {})
        return (
            f"tick_id={row.tick_id} selected_signal_id={row.selected_signal_id} "
            f"selection={reason.get('selection')} tier={reason.get('selected_priority_tier') or reason.get('priority_tier')} "
            f"phase={reason.get('phase')} stop_reason={reason.get('stop_reason')}"
        )

    def cmd_status(self, top: int) -> int:
        db = SessionLocal()
        try:
            self._print_header("Scheduler / Queue Status")
            by_status = (
                db.query(ENSSignalQueue.status, func.count(ENSSignalQueue.queue_id))
                .group_by(ENSSignalQueue.status)
                .all()
            )
            print("Signal counts by status:")
            for status, count in sorted([(str(s or ""), int(c or 0)) for s, c in by_status], key=lambda x: x[0]):
                print(f"  - {status}: {count}")

            by_type = (
                db.query(ENSSignalQueue.signal_type, func.count(ENSSignalQueue.queue_id))
                .group_by(ENSSignalQueue.signal_type)
                .all()
            )
            print("Signal counts by type:")
            for signal_type, count in sorted(
                [(str(t or ""), int(c or 0)) for t, c in by_type], key=lambda x: (-x[1], x[0])
            ):
                print(f"  - {signal_type}: {count}")

            self._print_header(f"Top {top} Pending Signals")
            rows = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.status == "pending")
                .order_by(ENSSignalQueue.created_at_us.asc(), ENSSignalQueue.signal_id.asc())
                .limit(max(1, int(top)))
                .all()
            )
            if not rows:
                print("  (none)")
            for row in rows:
                print(f"  - {self._signal_excerpt(row)}")

            self._print_header(f"Top {top} Active Loops")
            loop_rows = (
                db.query(ENSLoopSession)
                .filter(ENSLoopSession.state.in_(("running", "paused", "waiting_for_user")))
                .order_by(ENSLoopSession.updated_at.desc())
                .limit(max(1, int(top)))
                .all()
            )
            if not loop_rows:
                print("  (none)")
            for row in loop_rows:
                print(f"  - {self._loop_excerpt(row)}")
            return 0
        finally:
            db.close()

    @staticmethod
    def _canonical_type_for_tier(tier: str, requested: str) -> str:
        value = str(requested or "").strip()
        if value:
            return value
        t = str(tier or "").upper()
        if t == "USER":
            return "user.message"
        if t == "LOOP":
            return "loop.synthetic"
        return "system.noop"

    async def _enqueue_signal(
        self,
        *,
        signal_type: str,
        surface_id: Optional[str],
        relationship_id: Optional[str],
        conversation_id: Optional[str],
        idempotency_key: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload_doc = dict(payload or {})
        if conversation_id:
            payload_doc.setdefault("conversation_id", conversation_id)
        if signal_type == "user.message":
            payload_doc.setdefault("thread_id", payload_doc.get("thread_id") or f"harness-thread-{conversation_id or 'na'}")
            payload_doc.setdefault("content", payload_doc.get("content") or "harness test message")
            payload_doc.setdefault("client_message_id", payload_doc.get("client_message_id") or str(uuid.uuid4()))
        signal = Signal(
            type=signal_type,
            scope="SESSION" if signal_type in ("user.message", "loop_progression") else "SYSTEM",
            source="ens_v3_harness",
            payload=payload_doc,
            surface_id=surface_id,
            relationship_hint=relationship_id,
            idempotency_key=idempotency_key,
        )
        return await self.runtime.enqueue_signal(signal)

    def cmd_enqueue(
        self,
        *,
        tier: str,
        signal_type: str,
        count: int,
        surface: Optional[str],
        relationship: Optional[str],
        conversation: Optional[str],
        repeatable: bool,
    ) -> int:
        signal_type = self._canonical_type_for_tier(tier, signal_type)
        if signal_type == "loop_progression":
            raise HarnessError("Use 'loop-enqueue' for loop_progression signals (requires loop_id).")
        created = []
        for idx in range(max(1, int(count))):
            idem = None
            if repeatable:
                idem = f"harness:enqueue:{tier}:{signal_type}:{conversation or 'na'}:{surface or 'na'}:{relationship or 'na'}:{idx}"
            queued = asyncio.run(
                self._enqueue_signal(
                    signal_type=signal_type,
                    surface_id=surface,
                    relationship_id=relationship,
                    conversation_id=conversation,
                    idempotency_key=idem,
                    payload={
                        "harness": True,
                        "requested_tier": tier,
                        "requested_type": signal_type,
                    },
                )
            )
            created.append(queued)
        self._print_header("Enqueue Result")
        for row in created:
            print(
                "  - "
                f"queue_id={row.get('queue_id')} signal_id={row.get('signal_id')} "
                f"status={row.get('status')} tier={row.get('priority_tier')} created_at_us={row.get('created_at_us')}"
            )
        return 0

    def cmd_loop_create(
        self,
        *,
        kind: str,
        relationship: str,
        surface: Optional[str],
        conversation: Optional[str],
    ) -> int:
        loop_id = str(uuid.uuid4())
        signal = Signal(
            type="loop.session.create_requested",
            scope="SESSION",
            source="ens_v3_harness",
            payload={
                "loop_id": loop_id,
                "loop_kind": kind,
                "relationship_id": relationship,
                "conversation_id": conversation,
                "surface_id": surface,
                "step_prompt": "Harness loop progression step.",
            },
            relationship_hint=relationship,
            surface_id=surface,
        )
        outcome = asyncio.run(self.runtime.ingest(signal, ENSContext(app_state=self.app_state, surface="system", source="harness")))
        payload = dict(outcome.response_payload or {})
        if "progression_enqueued" not in payload:
            _ = self._wait_for_signal_completion(signal.signal_id, max_ticks=25)
            replay = self.runtime._replay_outcome_for_signal(signal.signal_id, signal.trace_id)  # noqa: SLF001
            if replay is not None:
                payload = dict(replay.response_payload or {})
        self._print_header("Loop Create")
        print(f"  - loop_id: {payload.get('loop_id') or loop_id}")
        print(f"  - progression_enqueued: {payload.get('progression_enqueued')}")
        progression = payload.get("progression") or {}
        print(f"  - progression_signal_id: {progression.get('signal_id')}")
        return 0

    def cmd_loop_enqueue(self, *, loop_id: str, count: int) -> int:
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            if not loop:
                raise HarnessError(f"Loop not found: {loop_id}")
            loop_kind = str(loop.loop_kind)
            relationship_id = str(loop.relationship_id)
            conversation_id = loop.conversation_id
            surface_id = loop.surface_id
        finally:
            db.close()

        queue_ids: List[str] = []
        signal_ids: List[str] = []
        for _ in range(max(1, int(count))):
            queued = asyncio.run(
                self.runtime.enqueue_loop_progression(
                    loop_id=loop_id,
                    loop_kind=loop_kind,
                    relationship_id=relationship_id,
                    conversation_id=conversation_id,
                    surface_id=surface_id,
                    step_prompt="Harness loop progression step.",
                )
            )
            queue_ids.append(str(queued.get("queue_id")))
            signal_ids.append(str(queued.get("signal_id")))

        db = SessionLocal()
        try:
            pending_count = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
        finally:
            db.close()

        unique_signal_count = len(set(signal_ids))
        noops = max(0, int(count) - unique_signal_count)
        self._print_header("Loop Enqueue")
        print(f"  - loop_id: {loop_id}")
        print(f"  - attempts: {count}")
        print(f"  - no_op_due_to_coalescing: {noops}")
        print(f"  - pending_loop_progression_count: {pending_count}")
        return 0

    def _tick_last_row_for_signal(self, signal_id: str) -> Optional[ENSSchedulerTick]:
        db = SessionLocal()
        try:
            return (
                db.query(ENSSchedulerTick)
                .filter(ENSSchedulerTick.selected_signal_id == signal_id)
                .order_by(ENSSchedulerTick.created_at_us.desc())
                .first()
            )
        finally:
            db.close()

    def cmd_tick(self, *, count: int, max_ms: Optional[int]) -> int:
        start = time.perf_counter()
        ran = 0
        for _ in range(max(1, int(count))):
            if max_ms is not None:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if elapsed_ms >= max(1, int(max_ms)):
                    break
            ran += 1
            outcome = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
            if outcome is None:
                print(f"tick={ran} selected=none runnable outcome=none")
                continue
            tick_row = self._tick_last_row_for_signal(outcome.signal_id)
            reason = dict((tick_row.reason_trace_json if tick_row else {}) or {})
            tier = reason.get("selected_priority_tier") or reason.get("priority_tier") or "na"
            selection = reason.get("selection") or "na"
            phase = reason.get("phase") or "execute"
            outbox_count = len([r for r in (outcome.action_results or []) if r.get("kind") == "surface.egress.persist_intent"])
            status = "done"
            if reason.get("status") == "failed":
                status = "failed"
            print(
                f"tick={ran} tick_id={(tick_row.tick_id if tick_row else 'na')} selected={outcome.signal_id} "
                f"selection={selection} tier={tier} phase={phase} outcome={status} outbox={outbox_count}"
            )
        return 0

    def _reset_queue_rows(
        self,
        *,
        relationship: Optional[str] = None,
        loop_id: Optional[str] = None,
        signal_type: Optional[str] = None,
    ) -> Dict[str, int]:
        db = SessionLocal()
        try:
            query = db.query(ENSSignalQueue).filter(ENSSignalQueue.status.in_(("pending", "running")))
            if signal_type:
                query = query.filter(ENSSignalQueue.signal_type == signal_type)
            if relationship:
                query = query.filter(ENSSignalQueue.relationship_id == relationship)
            if loop_id:
                query = query.filter(ENSSignalQueue.loop_id == loop_id)
            rows = query.all()
            now = datetime.utcnow()
            updated = 0
            for row in rows:
                row.status = "failed"
                row.completed_at = now
                row.claimed_at_us = None
                row.error_message = "harness_reset"
                updated += 1
            db.commit()
            return {"queue_rows_marked_failed": updated}
        finally:
            db.close()

    def _reset_loops(self, *, loop_id: Optional[str] = None) -> Dict[str, int]:
        db = SessionLocal()
        try:
            loop_query = db.query(ENSLoopSession)
            if loop_id:
                loop_query = loop_query.filter(ENSLoopSession.loop_id == loop_id)
            loops = loop_query.all()
            loop_updates = 0
            for row in loops:
                if row.state != "paused" or str(row.stop_reason or "") != "harness_loop_reset":
                    row.state = "paused"
                    row.stop_reason = "harness_loop_reset"
                    loop_updates += 1
            db.commit()
            return {"loops_paused": loop_updates, "loops_matched": len(loops)}
        finally:
            db.close()

    def cmd_reset(
        self,
        *,
        scope: str,
        queue_flag: bool,
        loop_flag: bool,
        all_flag: bool,
    ) -> int:
        selected_scope = (scope or "").strip().lower()
        if queue_flag:
            selected_scope = "queue"
        if loop_flag:
            selected_scope = "loop"
        if all_flag:
            selected_scope = "all"
        if selected_scope not in ("queue", "loop", "all"):
            raise HarnessError(f"Invalid reset scope: {scope}")

        self._print_header(f"Reset ({selected_scope})")
        summary: Dict[str, int] = {}
        if selected_scope in ("queue", "all"):
            summary.update(self._reset_queue_rows())
        if selected_scope in ("loop", "all"):
            summary.update(self._reset_loops())
            # Clear any pending/running loop progression work as part of loop reset scope.
            summary.update(self._reset_queue_rows(signal_type="loop_progression"))

        for key, value in summary.items():
            print(f"  - {key}: {value}")
        return 0

    def cmd_loop_reset(self, *, loop_id: str) -> int:
        loop_id = str(loop_id or "").strip()
        if not loop_id:
            raise HarnessError("loop-reset requires --loop")

        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            if not loop:
                raise HarnessError(f"Loop not found: {loop_id}")
            relationship = str(loop.relationship_id)
        finally:
            db.close()

        loop_summary = self._reset_loops(loop_id=loop_id)
        queue_summary = self._reset_queue_rows(relationship=relationship, loop_id=loop_id)

        db = SessionLocal()
        try:
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            if loop is None:
                raise HarnessError(f"Loop disappeared during reset: {loop_id}")
            loop_excerpt = self._loop_excerpt(loop)
        finally:
            db.close()

        self._print_header("Loop Reset")
        print(f"  - loop_id: {loop_id}")
        print(f"  - loops_paused: {loop_summary.get('loops_paused', 0)}")
        print(f"  - queue_rows_marked_failed: {queue_summary.get('queue_rows_marked_failed', 0)}")
        print(f"  - pending_progression_after_reset: {pending_progressions}")
        print(f"  - loop_state: {loop.state}")
        print(f"  - loop_excerpt: {loop_excerpt}")
        return 0

    @staticmethod
    def _check(name: str, passed: bool, details: str, excerpts: Optional[List[str]] = None) -> CheckResult:
        return CheckResult(name=name, passed=bool(passed), details=details, excerpts=list(excerpts or []))

    def _last_tick_excerpts(self, limit: int = 5) -> List[str]:
        db = SessionLocal()
        try:
            rows = (
                db.query(ENSSchedulerTick)
                .order_by(ENSSchedulerTick.created_at_us.desc())
                .limit(max(1, int(limit)))
                .all()
            )
            return [self._tick_excerpt(r) for r in rows]
        finally:
            db.close()

    def _verify_profile_3_5(self) -> List[CheckResult]:
        results: List[CheckResult] = []
        db = SessionLocal()
        try:
            pending_preexisting = db.query(ENSSignalQueue).filter(ENSSignalQueue.status == "pending").count()
            if pending_preexisting > 0:
                pending_rows = (
                    db.query(ENSSignalQueue)
                    .filter(ENSSignalQueue.status == "pending")
                    .order_by(ENSSignalQueue.created_at_us.asc())
                    .limit(10)
                    .all()
                )
                results.append(
                    self._check(
                        "isolation_guard",
                        False,
                        f"Expected clean queue for deterministic verify; found {pending_preexisting} pre-existing pending signals.",
                        excerpts=[self._signal_excerpt(r) for r in pending_rows] + self._last_tick_excerpts(5),
                    )
                )
                return results
        finally:
            db.close()

        run_id = f"harness35-{int(time.time())}"
        relationship = f"rel-{run_id}"
        conversation = f"conv-{run_id}"
        surface = "web"

        # 1) loop creation enqueues one progression, no immediate execution
        loop_id = str(uuid.uuid4())
        create_signal = Signal(
            type="loop.session.create_requested",
            scope="SESSION",
            source="ens_v3_harness",
            payload={
                "loop_id": loop_id,
                "loop_kind": "generic",
                "relationship_id": relationship,
                "conversation_id": conversation,
                "surface_id": surface,
                "step_prompt": "Harness verify 3.5 step",
                "character_id": "test_char",
            },
            relationship_hint=relationship,
            surface_id=surface,
        )
        outcome = asyncio.run(self.runtime.ingest(create_signal, ENSContext(app_state=self.app_state, surface="system", source="harness")))
        _ = outcome
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            passed = bool(loop is not None and int(loop.step_count or 0) == 0 and pending_progressions == 1)
            excerpts: List[str] = []
            if loop is not None:
                excerpts.append(self._loop_excerpt(loop))
            pending_rows = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.loop_id == loop_id)
                .order_by(ENSSignalQueue.created_at_us.asc())
                .all()
            )
            excerpts.extend([self._signal_excerpt(r) for r in pending_rows])
            results.append(
                self._check(
                    "loop_create_enqueues_once_no_immediate_exec",
                    passed,
                    f"loop_exists={loop is not None} step_count={(loop.step_count if loop else 'na')} pending_progressions={pending_progressions}",
                    excerpts,
                )
            )
        finally:
            db.close()

        # 2) coalescing
        for _ in range(5):
            asyncio.run(
                self.runtime.enqueue_loop_progression(
                    loop_id=loop_id,
                    loop_kind="generic",
                    relationship_id=relationship,
                    conversation_id=conversation,
                    surface_id=surface,
                    step_prompt="Harness verify 3.5 coalescing",
                    character_id="test_char",
                )
            )
        db = SessionLocal()
        try:
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            pending_rows = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .order_by(ENSSignalQueue.created_at_us.asc())
                .all()
            )
            results.append(
                self._check(
                    "coalescing_max_one_pending_progression",
                    pending_progressions <= 1,
                    f"pending_progressions={pending_progressions}",
                    [self._signal_excerpt(r) for r in pending_rows],
                )
            )
        finally:
            db.close()

        # 3) state gating paused loop has no follow-up
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            if loop is None:
                raise HarnessError(f"Loop missing during state-gating check: {loop_id}")
            loop.state = "paused"
            db.commit()
        finally:
            db.close()
        asyncio.run(
            self.runtime.enqueue_loop_progression(
                loop_id=loop_id,
                loop_kind="generic",
                relationship_id=relationship,
                conversation_id=conversation,
                surface_id=surface,
                step_prompt="paused-step",
                character_id="test_char",
            )
        )
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        db = SessionLocal()
        try:
            pending_after = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id)
                .order_by(ENSSignalQueue.created_at_us.asc())
                .all()
            )
            passed = bool(loop is not None and loop.state == "paused" and pending_after == 0)
            excerpts = ([self._loop_excerpt(loop)] if loop else []) + [self._signal_excerpt(r) for r in progressions]
            results.append(
                self._check(
                    "paused_state_gating_no_followup",
                    passed,
                    f"loop_state={(loop.state if loop else 'na')} pending_after={pending_after}",
                    excerpts,
                )
            )
        finally:
            db.close()

        # 4) no double execute with concurrent ticks
        signal = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="ens_v3_harness",
            payload={"harness": True, "check": "no_double_exec", "run_id": run_id},
            relationship_hint=relationship,
            surface_id=surface,
            idempotency_key=f"harness:{run_id}:no-double-exec",
        )
        queued = asyncio.run(self.runtime.enqueue_signal(signal))
        _ = queued

        async def _run_two_ticks() -> Tuple[Optional[Any], Optional[Any]]:
            a = asyncio.create_task(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
            b = asyncio.create_task(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
            o1, o2 = await asyncio.gather(a, b)
            return o1, o2

        o1, o2 = asyncio.run(_run_two_ticks())
        selected_ids = [o.signal_id for o in (o1, o2) if o is not None]
        db = SessionLocal()
        try:
            decision_count = db.query(ENSDecision).filter(ENSDecision.signal_id == signal.signal_id).count()
            row = db.query(ENSSignalQueue).filter(ENSSignalQueue.signal_id == signal.signal_id).first()
            passed = decision_count == 1 and selected_ids.count(signal.signal_id) == 1 and row is not None and row.status in ("done", "failed")
            excerpts = ([self._signal_excerpt(row)] if row else []) + self._last_tick_excerpts(5)
            results.append(
                self._check(
                    "no_double_execute_under_concurrent_ticks",
                    passed,
                    f"decision_count={decision_count} selected_occurrences={selected_ids.count(signal.signal_id)} final_status={(row.status if row else 'na')}",
                    excerpts,
                )
            )
        finally:
            db.close()

        # 5) no reopen gating proxy (drains without new ingress once tick runs later)
        proxy_signal = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="ens_v3_harness",
            payload={"harness": True, "check": "no_reopen_proxy", "run_id": run_id},
            relationship_hint=relationship,
            surface_id=surface,
            idempotency_key=f"harness:{run_id}:no-reopen-proxy",
        )
        asyncio.run(self.runtime.enqueue_signal(proxy_signal))
        time.sleep(0.2)
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        db = SessionLocal()
        try:
            row = db.query(ENSSignalQueue).filter(ENSSignalQueue.signal_id == proxy_signal.signal_id).first()
            passed = bool(row is not None and row.status in ("done", "failed"))
            excerpts = ([self._signal_excerpt(row)] if row else []) + self._last_tick_excerpts(5)
            results.append(
                self._check(
                    "no_reopen_gating_proxy_drains_without_new_ingress",
                    passed,
                    f"final_status={(row.status if row else 'missing')}",
                    excerpts,
                )
            )
        finally:
            db.close()

        return results

    def cmd_verify(self, *, profile: str) -> int:
        normalized = str(profile or "").strip().lower()
        if normalized != "3_5":
            raise HarnessError(f"Unsupported profile: {profile}")
        checks = self._verify_profile_3_5()
        self._print_header("VERIFY REPORT (profile=3_5)")
        failed = 0
        for idx, check in enumerate(checks, start=1):
            status = "PASS" if check.passed else "FAIL"
            print(f"{idx}. [{status}] {check.name}: {check.details}")
            if not check.passed:
                failed += 1
                for ex in check.excerpts:
                    print(f"   -> {ex}")
        print(f"\nSummary: {len(checks) - failed}/{len(checks)} checks passed.")
        return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ENS v3 dev harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show queue/scheduler/loop status snapshot")
    p_status.add_argument("--top", type=int, default=10)

    p_enqueue = sub.add_parser("enqueue", help="Enqueue no-op signals for arbitration/backlog testing")
    p_enqueue.add_argument("--tier", type=str, required=True, choices=["USER", "SYSTEM", "LOOP"])
    p_enqueue.add_argument("--type", type=str, required=True, dest="signal_type")
    p_enqueue.add_argument("--n", type=int, required=True)
    p_enqueue.add_argument("--surface", type=str, default=None)
    p_enqueue.add_argument("--relationship", type=str, default=None)
    p_enqueue.add_argument("--conversation", type=str, default=None)
    p_enqueue.add_argument("--repeatable", action="store_true")

    p_loop_create = sub.add_parser("loop-create", help="Create loop session and enqueue initial progression")
    p_loop_create.add_argument("--kind", type=str, required=True)
    p_loop_create.add_argument("--relationship", type=str, required=True)
    p_loop_create.add_argument("--surface", type=str, default=None)
    p_loop_create.add_argument("--conversation", type=str, default=None)

    p_loop_enqueue = sub.add_parser("loop-enqueue", help="Attempt repeated progression enqueue for a loop")
    p_loop_enqueue.add_argument("--loop", type=str, required=True, dest="loop_id")
    p_loop_enqueue.add_argument("--n", type=int, required=True)

    p_tick = sub.add_parser("tick", help="Run scheduler ticks directly")
    p_tick.add_argument("--n", type=int, required=True)
    p_tick.add_argument("--max-ms", type=int, default=None)

    p_reset = sub.add_parser("reset", help="Reset harness-managed scheduler/loop state (non-destructive)")
    p_reset.add_argument("--scope", type=str, choices=["queue", "loop", "all"], default="queue")
    p_reset.add_argument("--queue", action="store_true")
    p_reset.add_argument("--loop", action="store_true")
    p_reset.add_argument("--all", action="store_true")

    p_loop_reset = sub.add_parser("loop-reset", help="Pause one loop and clear its runnable progression work")
    p_loop_reset.add_argument("--loop", type=str, required=True, dest="loop_id")

    p_verify = sub.add_parser("verify", help="Run invariant verification profile")
    p_verify.add_argument("--profile", type=str, required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    harness = ENSV3Harness()
    try:
        if args.command == "status":
            return harness.cmd_status(top=args.top)
        if args.command == "enqueue":
            return harness.cmd_enqueue(
                tier=args.tier,
                signal_type=args.signal_type,
                count=args.n,
                surface=args.surface,
                relationship=args.relationship,
                conversation=args.conversation,
                repeatable=bool(args.repeatable),
            )
        if args.command == "loop-create":
            return harness.cmd_loop_create(
                kind=args.kind,
                relationship=args.relationship,
                surface=args.surface,
                conversation=args.conversation,
            )
        if args.command == "loop-enqueue":
            return harness.cmd_loop_enqueue(loop_id=args.loop_id, count=args.n)
        if args.command == "tick":
            return harness.cmd_tick(count=args.n, max_ms=args.max_ms)
        if args.command == "reset":
            return harness.cmd_reset(
                scope=args.scope,
                queue_flag=bool(args.queue),
                loop_flag=bool(args.loop),
                all_flag=bool(args.all),
            )
        if args.command == "loop-reset":
            return harness.cmd_loop_reset(loop_id=args.loop_id)
        if args.command == "verify":
            return harness.cmd_verify(profile=args.profile)
    except HarnessError as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
