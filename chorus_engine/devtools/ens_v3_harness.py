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
from chorus_engine.models.ens import (
    ENSDecision,
    ENSLoopCompressionArtifact,
    ENSLoopSession,
    ENSLoopStepEvent,
    ENSSchedulerTick,
    ENSSignalQueue,
)


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
    _VALID_MODES = {
        "yield_control",
        "freeform_yield_no_control",
        "sentinel_tool_call",
        "continue_control",
        "complete_control",
    }

    def __init__(self, mode: str = "yield_control") -> None:
        self.mode = mode if mode in self._VALID_MODES else "yield_control"

    def set_mode(self, mode: str) -> None:
        if mode not in self._VALID_MODES:
            raise HarnessError(f"Unsupported mock assistant result mode: {mode}")
        self.mode = mode

    async def health_check(self) -> bool:
        return True

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> _DummyResponse:
        _ = (prompt, system_prompt, model, kwargs)
        if self.mode == "freeform_yield_no_control":
            return _DummyResponse("I can YIELD or COMPLETE if needed, but this is plain freeform text only.")
        if self.mode == "sentinel_tool_call":
            return _DummyResponse(
                "Creating tool request from sentinel fallback.\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"tool_calls":[{"id":"h_tool_1","tool":"image.generate","requires_approval":true,"args":{"prompt":"harness image"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            )
        if self.mode == "continue_control":
            return _DummyResponse(
                "Continue step.\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"control":{"action":"CONTINUE","args":{}},"tool_calls":[]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            )
        if self.mode == "complete_control":
            return _DummyResponse(
                "Complete step.\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"control":{"action":"COMPLETE","args":{}},"tool_calls":[]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            )
        return _DummyResponse(
            "I may COMPLETE eventually, but follow structured control.\n"
            "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
            '{"version":1,"control":{"action":"YIELD","args":{}},"tool_calls":[]}\n'
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

    def __init__(self, *, mock_assistant_result: str = "yield_control") -> None:
        ensure_database_ready(DATABASE_URL)
        loader = ConfigLoader()
        system_config = loader.load_system_config()
        characters = loader.load_all_characters()
        app_state: Dict[str, Any] = {
            "system_config": system_config,
            "characters": characters,
            "llm_client": _HarnessLLMClient(mode=mock_assistant_result),
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

    def _set_mock_mode(self, mode: str) -> None:
        llm = self.app_state.get("llm_client")
        if hasattr(llm, "set_mode"):
            llm.set_mode(mode)

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

    def _create_loop_and_progression(
        self,
        *,
        loop_kind: str,
        relationship: str,
        conversation: str,
        surface: str,
        step_prompt: str,
        character_id: str = "test_char",
    ) -> Tuple[str, Optional[str], str]:
        loop_id = str(uuid.uuid4())
        progression_idempotency_key = f"harness:loop_progression:{loop_id}:initial"
        signal = Signal(
            type="loop.session.create_requested",
            scope="SESSION",
            source="ens_v3_harness",
            payload={
                "loop_id": loop_id,
                "loop_kind": loop_kind,
                "relationship_id": relationship,
                "conversation_id": conversation,
                "surface_id": surface,
                "step_prompt": step_prompt,
                "character_id": character_id,
                "progression_idempotency_key": progression_idempotency_key,
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
        progression = payload.get("progression") or {}
        return loop_id, progression.get("signal_id"), signal.signal_id

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
        progression_idempotency_key = f"harness:loop_progression:{loop_id}:initial"
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
                "progression_idempotency_key": progression_idempotency_key,
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
            attempt = len(queue_ids)
            queued = asyncio.run(
                self.runtime.enqueue_loop_progression(
                    loop_id=loop_id,
                    loop_kind=loop_kind,
                    relationship_id=relationship_id,
                    conversation_id=conversation_id,
                    surface_id=surface_id,
                    step_prompt="Harness loop progression step.",
                    idempotency_key=f"harness:loop_progression:{loop_id}:enqueue:{attempt}",
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

    def _collect_loop_compression_snapshot(self, loop_id: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            artifacts = (
                db.query(ENSLoopCompressionArtifact)
                .filter(ENSLoopCompressionArtifact.loop_id == loop_id)
                .order_by(ENSLoopCompressionArtifact.to_step_index.asc(), ENSLoopCompressionArtifact.created_at_us.asc())
                .all()
            )
            events = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id)
                .order_by(ENSLoopStepEvent.step_index_after.asc(), ENSLoopStepEvent.created_at_us.asc())
                .all()
            )
            session = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id).first()
            triggered_steps = [int(e.step_index_after or 0) for e in events if str(e.compression_artifact_id or "").strip()]
            return {
                "loop": session,
                "events": events,
                "artifacts": artifacts,
                "triggered_steps": triggered_steps,
            }
        finally:
            db.close()

    def _run_loop_steps_for_compression(
        self,
        *,
        loop_kind: str,
        relationship: str,
        conversation: str,
        surface: str,
        target_steps: int,
    ) -> Dict[str, Any]:
        loop_id, progression_signal_id, _ = self._create_loop_and_progression(
            loop_kind=loop_kind,
            relationship=relationship,
            conversation=conversation,
            surface=surface,
            step_prompt="compression profile deterministic step",
        )
        _ = progression_signal_id
        max_ticks = max(10, int(target_steps) * 4)
        for _ in range(max_ticks):
            _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
            snap = self._collect_loop_compression_snapshot(loop_id)
            loop = snap.get("loop")
            if loop is not None and int(loop.step_index or 0) >= int(target_steps):
                return {"loop_id": loop_id, **snap}
        raise HarnessError(f"Compression profile failed to reach target_steps={target_steps} for loop_id={loop_id}")

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

    def _verify_profile_3_6(self) -> List[CheckResult]:
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

        run_id = f"harness36-{int(time.time())}"
        relationship = f"rel-{run_id}"
        conversation = f"conv-{run_id}"
        surface = "web"

        # Check A - StepEvent created exactly once
        self._set_mock_mode("freeform_yield_no_control")
        loop_id_a, progression_signal_id_a, _create_signal_id_a = self._create_loop_and_progression(
            loop_kind="generic",
            relationship=relationship,
            conversation=conversation,
            surface=surface,
            step_prompt="check A step event exact-once",
        )
        db = SessionLocal()
        try:
            pre_tick_events = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_a)
                .count()
            )
        finally:
            db.close()
        first_outcome = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        _ = first_outcome
        db = SessionLocal()
        try:
            events = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_a)
                .order_by(ENSLoopStepEvent.created_at_us.asc())
                .all()
            )
            event_count_after_first = len(events)
            selected_tick = (
                db.query(ENSSchedulerTick)
                .filter(ENSSchedulerTick.selected_signal_id == progression_signal_id_a)
                .order_by(ENSSchedulerTick.created_at_us.desc())
                .first()
            )
            loop_row = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id_a).first()
            excerpts = [self._loop_excerpt(loop_row)] if loop_row else []
            excerpts.extend([f"step_event event_id={e.event_id} signal_id={e.signal_id} tick_id={e.tick_id}" for e in events])
            if selected_tick:
                excerpts.append(self._tick_excerpt(selected_tick))
        finally:
            db.close()
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        replay = self.runtime._replay_outcome_for_signal(str(progression_signal_id_a or ""), str(uuid.uuid4()))
        _ = replay
        db = SessionLocal()
        try:
            events_after = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_a)
                .order_by(ENSLoopStepEvent.created_at_us.asc())
                .all()
            )
            event = events_after[0] if events_after else None
            passed_a = bool(
                pre_tick_events == 0
                and event_count_after_first == 1
                and len(events_after) == 1
                and event is not None
                and str(event.signal_id or "") == str(progression_signal_id_a or "")
                and bool(str(event.tick_id or "").strip())
            )
            results.append(
                self._check(
                    "step_event_exactly_once_and_linked",
                    passed_a,
                    (
                        f"pre_tick_events={pre_tick_events} "
                        f"event_count_after_first={event_count_after_first} "
                        f"event_count_after_replay={len(events_after)} "
                        f"event_signal_id={(event.signal_id if event else 'na')} "
                        f"progression_signal_id={progression_signal_id_a}"
                    ),
                    excerpts + [f"replay_called=True", *(f"step_event_post event_id={e.event_id}" for e in events_after)],
                )
            )
        finally:
            db.close()

        # Check B - Sentinel Option A structured control works
        self._set_mock_mode("yield_control")
        loop_id_b, _progression_signal_id_b, _ = self._create_loop_and_progression(
            loop_kind="generic",
            relationship=f"{relationship}-b",
            conversation=f"{conversation}-b",
            surface=surface,
            step_prompt="check B sentinel option A control",
        )
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id_b).first()
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id_b)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_b)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            out = dict((event.output_json if event else {}) or {})
            passed_b = bool(
                loop is not None
                and loop.state == "running"
                and str(loop.stop_reason or "") == "yielded"
                and pending_progressions == 0
                and event is not None
                and str(event.control_action or "") == "YIELD"
                and out.get("control_channel") == "structured_control_present"
                and str(out.get("assistant_result_tier") or "") == "sentinel_fallback"
            )
            results.append(
                self._check(
                    "sentinel_option_a_control_yield_applied",
                    passed_b,
                    (
                        f"loop_state={(loop.state if loop else 'na')} loop_stop_reason={(loop.stop_reason if loop else 'na')} "
                        f"pending_progressions={pending_progressions} "
                        f"control_action={(event.control_action if event else 'na')} "
                        f"control_channel={out.get('control_channel')} "
                        f"assistant_result_tier={out.get('assistant_result_tier')}"
                    ),
                    ([self._loop_excerpt(loop)] if loop else [])
                    + ([f"step_event={event.event_id} output={out}"] if event else []),
                )
            )
        finally:
            db.close()

        # Check C - freeform YIELD/COMPLETE without structured control does nothing
        self._set_mock_mode("freeform_yield_no_control")
        loop_id_c, _progression_signal_id_c, _ = self._create_loop_and_progression(
            loop_kind="generic",
            relationship=f"{relationship}-c",
            conversation=f"{conversation}-c",
            surface=surface,
            step_prompt="check C freeform yield does nothing",
        )
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id_c).first()
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_c)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            out = dict((event.output_json if event else {}) or {})
            passed_c = bool(
                loop is not None
                and loop.state == "running"
                and event is not None
                and event.control_action is None
                and out.get("control_channel") == "no_control_present"
                and out.get("parsed_from_text") is False
            )
            results.append(
                self._check(
                    "freeform_yield_word_no_control_directive",
                    passed_c,
                    (
                        f"loop_state={(loop.state if loop else 'na')} control_action={(event.control_action if event else 'na')} "
                        f"control_channel={out.get('control_channel')} parsed_from_text={out.get('parsed_from_text')}"
                    ),
                    ([self._loop_excerpt(loop)] if loop else [])
                    + ([f"step_event={event.event_id} output={out}"] if event else []),
                )
            )
        finally:
            db.close()

        # Check D - Sentinel fallback tool calls route via AssistantResult.tool_requests
        self._set_mock_mode("sentinel_tool_call")
        loop_id_d, _progression_signal_id_d, _ = self._create_loop_and_progression(
            loop_kind="generic",
            relationship=f"{relationship}-d",
            conversation=f"{conversation}-d",
            surface=surface,
            step_prompt="check D sentinel tool fallback",
        )
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        db = SessionLocal()
        try:
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_d)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            out = dict((event.output_json if event else {}) or {})
            passed_d = bool(
                event is not None
                and int(event.tool_requests_count or 0) >= 1
                and str(out.get("assistant_result_tier") or "") == "sentinel_fallback"
            )
            results.append(
                self._check(
                    "sentinel_tool_calls_route_via_assistant_result_tool_requests",
                    passed_d,
                    (
                        f"tool_requests_count={(event.tool_requests_count if event else 'na')} "
                        f"assistant_result_tier={out.get('assistant_result_tier')}"
                    ),
                    ([f"step_event={event.event_id} output={out}"] if event else []),
                )
            )
        finally:
            db.close()

        return results

    def _verify_profile_3_7(self) -> List[CheckResult]:
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

        run_id = f"harness37-{int(time.time())}"
        relationship = f"rel-{run_id}"
        conversation = f"conv-{run_id}"
        surface = "web"

        # Check 1: Visible one step emits once and does not auto-enqueue on YIELD/default.
        self._set_mock_mode("yield_control")
        loop_id_1, progression_signal_id_1, _ = self._create_loop_and_progression(
            loop_kind="generic",
            relationship=f"{relationship}-v1",
            conversation=f"{conversation}-v1",
            surface=surface,
            step_prompt="3.7 check 1 visible one-step emit",
        )
        _ = self._wait_for_signal_completion(str(progression_signal_id_1 or ""), max_ticks=25)
        db = SessionLocal()
        try:
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_1)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id_1)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            out = dict((event.output_json if event else {}) or {})
            outbox_count = int(out.get("outbox_count") or 0)
            passed = bool(
                event is not None
                and outbox_count == 1
                and pending_progressions == 0
            )
            results.append(
                self._check(
                    "visible_one_step_one_emit",
                    passed,
                    f"outbox_count={outbox_count} pending_progressions={pending_progressions}",
                    ([f"step_event={event.event_id} output={out}"] if event else []),
                )
            )
        finally:
            db.close()

        # Check 2: Visible CONTINUE enqueues exactly one follow-up.
        self._set_mock_mode("continue_control")
        loop_id_2, progression_signal_id_2, _ = self._create_loop_and_progression(
            loop_kind="generic",
            relationship=f"{relationship}-v2",
            conversation=f"{conversation}-v2",
            surface=surface,
            step_prompt="3.7 check 2 visible continue",
        )
        _ = self._wait_for_signal_completion(str(progression_signal_id_2 or ""), max_ticks=25)
        db = SessionLocal()
        try:
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_2)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id_2)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            out = dict((event.output_json if event else {}) or {})
            outbox_count = int(out.get("outbox_count") or 0)
            passed = bool(
                event is not None
                and str(event.control_action or "") == "CONTINUE"
                and outbox_count == 1
                and pending_progressions == 1
            )
            results.append(
                self._check(
                    "visible_continue_enqueues_exactly_one_followup",
                    passed,
                    (
                        f"control_action={(event.control_action if event else 'na')} "
                        f"outbox_count={outbox_count} pending_progressions={pending_progressions}"
                    ),
                    ([f"step_event={event.event_id} output={out}"] if event else []),
                )
            )
        finally:
            db.close()

        # Check 3: Hidden auto-continues with no per-step emission.
        self._set_mock_mode("freeform_yield_no_control")
        loop_id_3, progression_signal_id_3, _ = self._create_loop_and_progression(
            loop_kind="generic.hidden",
            relationship=f"{relationship}-h3",
            conversation=f"{conversation}-h3",
            surface=surface,
            step_prompt="3.7 check 3 hidden auto-continue",
        )
        _ = self._wait_for_signal_completion(str(progression_signal_id_3 or ""), max_ticks=25)
        db = SessionLocal()
        try:
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_3)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id_3)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            out = dict((event.output_json if event else {}) or {})
            outbox_count = int(out.get("outbox_count") or 0)
            passed = bool(
                event is not None
                and outbox_count == 0
                and pending_progressions == 1
            )
            results.append(
                self._check(
                    "hidden_auto_continue_without_per_step_emit",
                    passed,
                    f"outbox_count={outbox_count} pending_progressions={pending_progressions}",
                    ([f"step_event={event.event_id} output={out}"] if event else []),
                )
            )
        finally:
            db.close()

        # Check 4: Hidden COMPLETE emits once and stops with no pending follow-up.
        self._set_mock_mode("complete_control")
        loop_id_4, progression_signal_id_4, _ = self._create_loop_and_progression(
            loop_kind="generic.hidden",
            relationship=f"{relationship}-h4",
            conversation=f"{conversation}-h4",
            surface=surface,
            step_prompt="3.7 check 4 hidden complete",
        )
        _ = self._wait_for_signal_completion(str(progression_signal_id_4 or ""), max_ticks=25)
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id_4).first()
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_4)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id_4)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            out = dict((event.output_json if event else {}) or {})
            outbox_count = int(out.get("outbox_count") or 0)
            passed = bool(
                loop is not None
                and loop.state == "stopped"
                and event is not None
                and str(event.control_action or "") == "COMPLETE"
                and outbox_count == 1
                and pending_progressions == 0
            )
            results.append(
                self._check(
                    "hidden_complete_emits_once_and_stops",
                    passed,
                    (
                        f"loop_state={(loop.state if loop else 'na')} "
                        f"control_action={(event.control_action if event else 'na')} "
                        f"outbox_count={outbox_count} pending_progressions={pending_progressions}"
                    ),
                    (([self._loop_excerpt(loop)] if loop else []) + ([f"step_event={event.event_id} output={out}"] if event else [])),
                )
            )
        finally:
            db.close()

        # Optional Check 5: Hidden preempt emits nothing.
        self._set_mock_mode("freeform_yield_no_control")
        loop_id_5, progression_signal_id_5, _ = self._create_loop_and_progression(
            loop_kind="generic.hidden",
            relationship=f"{relationship}-h5",
            conversation=f"{conversation}-h5",
            surface=surface,
            step_prompt="3.7 optional hidden preempt",
        )
        queued_user = asyncio.run(
            self._enqueue_signal(
                signal_type="user.message",
                surface_id=surface,
                relationship_id=f"{relationship}-h5",
                conversation_id=f"{conversation}-h5",
                idempotency_key=f"harness:{run_id}:hidden-preempt-user",
                payload={"content": "preempt hidden loop now"},
            )
        )
        # First tick processes newer user work.
        _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        # Second tick attempts loop progression; force pre-step preemption check for deterministic verification.
        dispatcher = self.runtime.dispatcher
        original_preempt_check = getattr(dispatcher, "_has_newer_pending_user_signal")
        setattr(dispatcher, "_has_newer_pending_user_signal", lambda *args, **kwargs: True)
        try:
            _ = asyncio.run(self.runtime.scheduler_tick(ENSContext(app_state=self.app_state, surface="system", source="harness")))
        finally:
            setattr(dispatcher, "_has_newer_pending_user_signal", original_preempt_check)
        db = SessionLocal()
        try:
            loop = db.query(ENSLoopSession).filter(ENSLoopSession.loop_id == loop_id_5).first()
            event = (
                db.query(ENSLoopStepEvent)
                .filter(ENSLoopStepEvent.loop_id == loop_id_5)
                .order_by(ENSLoopStepEvent.created_at_us.desc())
                .first()
            )
            pending_progressions = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_type == "loop_progression")
                .filter(ENSSignalQueue.loop_id == loop_id_5)
                .filter(ENSSignalQueue.status == "pending")
                .count()
            )
            row = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_id == progression_signal_id_5)
                .first()
            )
            user_row = (
                db.query(ENSSignalQueue)
                .filter(ENSSignalQueue.signal_id == str(queued_user.get("signal_id") or ""))
                .first()
            )
            out = dict((event.output_json if event else {}) or {})
            outbox_count = int(out.get("outbox_count") or 0)
            passed = bool(
                user_row is not None
                and user_row.status in ("done", "failed")
                and row is not None
                and row.status in ("done", "failed")
                and loop is not None
                and loop.state == "stopped"
                and str(loop.stop_reason or "") == "USER_PREEMPT"
                and event is not None
                and outbox_count == 0
                and pending_progressions == 0
            )
            results.append(
                self._check(
                    "hidden_preempt_emits_nothing_optional",
                    passed,
                    (
                        f"user_status={(user_row.status if user_row else 'na')} "
                        f"loop_state={(loop.state if loop else 'na')} loop_stop_reason={(loop.stop_reason if loop else 'na')} "
                        f"progression_status={(row.status if row else 'na')} outbox_count={outbox_count} "
                        f"pending_progressions={pending_progressions}"
                    ),
                    (([self._loop_excerpt(loop)] if loop else []) + ([self._signal_excerpt(user_row)] if user_row else []) + ([self._signal_excerpt(row)] if row else []) + ([f"step_event={event.event_id} output={out}"] if event else [])),
                )
            )
        finally:
            db.close()

        return results

    def _verify_profile_3_8(self) -> List[CheckResult]:
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

        ens_cfg = self.app_state["system_config"].ens
        ens_cfg.v3_context_compression_enabled = True
        ens_cfg.loop_memory_compress_every_n_steps = 4
        ens_cfg.loop_memory_keep_last_k_steps = 2

        run_id = f"harness38-{int(time.time())}"
        surface = "web"
        n = int(ens_cfg.loop_memory_compress_every_n_steps)
        k = int(ens_cfg.loop_memory_keep_last_k_steps)
        target_steps = 12

        self._set_mock_mode("continue_control")
        run_a = self._run_loop_steps_for_compression(
            loop_kind="generic.hidden",
            relationship=f"rel-{run_id}-a",
            conversation=f"conv-{run_id}-a",
            surface=surface,
            target_steps=target_steps,
        )
        run_b = self._run_loop_steps_for_compression(
            loop_kind="generic.hidden",
            relationship=f"rel-{run_id}-b",
            conversation=f"conv-{run_id}-b",
            surface=surface,
            target_steps=target_steps,
        )

        artifacts_a: List[ENSLoopCompressionArtifact] = list(run_a.get("artifacts") or [])
        artifacts_b: List[ENSLoopCompressionArtifact] = list(run_b.get("artifacts") or [])
        events_a: List[ENSLoopStepEvent] = list(run_a.get("events") or [])
        loop_a: Optional[ENSLoopSession] = run_a.get("loop")
        triggered_steps_a = [int(x) for x in (run_a.get("triggered_steps") or [])]
        triggered_steps_b = [int(x) for x in (run_b.get("triggered_steps") or [])]
        expected_trigger_steps = [step for step in range(1, target_steps + 1) if step > k and (step % n) == 0]

        # 1) deterministic trigger point
        results.append(
            self._check(
                "compression_trigger_point_deterministic",
                triggered_steps_a == expected_trigger_steps,
                f"compression_triggered_at_steps={triggered_steps_a}",
                excerpts=[
                    f"expected_trigger_steps={expected_trigger_steps}",
                    *(f"artifact_id={a.artifact_id} from={a.from_step_index} to={a.to_step_index}" for a in artifacts_a),
                ],
            )
        )

        # 2) incremental window correctness (first + second compression)
        window_ok = False
        window_details = "insufficient_artifacts"
        excerpts: List[str] = []
        if len(artifacts_a) >= 2:
            first = artifacts_a[0]
            second = artifacts_a[1]
            first_to_expected = expected_trigger_steps[0] - k
            second_to_expected = expected_trigger_steps[1] - k
            window_ok = bool(
                int(first.from_step_index) == 0
                and int(first.to_step_index) == int(first_to_expected)
                and int(second.from_step_index) == int(first.to_step_index) + 1
                and int(second.to_step_index) == int(second_to_expected)
            )
            window_details = (
                f"first=({first.from_step_index},{first.to_step_index}) expected=(0,{first_to_expected}) "
                f"second=({second.from_step_index},{second.to_step_index}) expected=({int(first.to_step_index)+1},{second_to_expected})"
            )
            excerpts = [f"artifact_id={a.artifact_id} from={a.from_step_index} to={a.to_step_index}" for a in artifacts_a[:3]]
        results.append(self._check("incremental_window_correctness", window_ok, window_details, excerpts))

        # 3) deterministic artifact hashing across equivalent runs
        hashes_a = [(a.input_hash, a.output_hash, a.config_hash) for a in artifacts_a]
        hashes_b = [(a.input_hash, a.output_hash, a.config_hash) for a in artifacts_b]
        results.append(
            self._check(
                "deterministic_artifact_hashes_across_runs",
                hashes_a == hashes_b and len(hashes_a) > 0,
                f"hash_triplets_a={len(hashes_a)} hash_triplets_b={len(hashes_b)} equal={hashes_a == hashes_b}",
                excerpts=[
                    *(f"a[{idx}] input={h[0]} output={h[1]} config={h[2]}" for idx, h in enumerate(hashes_a)),
                    *(f"b[{idx}] input={h[0]} output={h[1]} config={h[2]}" for idx, h in enumerate(hashes_b)),
                ][:10],
            )
        )

        # 4) no compression of recent K steps
        recent_ok = False
        details_recent = "missing_loop_or_artifact"
        if loop_a is not None and artifacts_a:
            final_artifact = artifacts_a[-1]
            loop_step_index = int(loop_a.step_index or 0)
            expected_to = loop_step_index - k
            recent_step_threshold = int(final_artifact.to_step_index) + 1
            recent_events = [e for e in events_a if int(e.step_index_after or 0) >= recent_step_threshold]
            recent_ok = bool(
                int(final_artifact.to_step_index) == expected_to
                and len(recent_events) >= k
                and all(int(e.step_index_after or 0) > int(final_artifact.to_step_index) for e in recent_events)
                and all(isinstance(e.memory_payload_json, dict) for e in recent_events)
            )
            details_recent = (
                f"loop_step_index={loop_step_index} final_artifact_to={final_artifact.to_step_index} expected_to={expected_to} "
                f"recent_events_count={len(recent_events)} keep_k={k}"
            )
        results.append(self._check("no_compression_of_recent_k_steps", recent_ok, details_recent))

        # 5) replay stability proxy (second deterministic run)
        replay_ok = bool(
            triggered_steps_a == triggered_steps_b
            and hashes_a == hashes_b
            and len(artifacts_a) == len(artifacts_b)
        )
        results.append(
            self._check(
                "replay_stability_same_triggers_hashes_counts",
                replay_ok,
                (
                    f"trigger_steps_a={triggered_steps_a} trigger_steps_b={triggered_steps_b} "
                    f"artifact_count_a={len(artifacts_a)} artifact_count_b={len(artifacts_b)}"
                ),
                excerpts=[
                    f"run_a_loop_id={run_a.get('loop_id')}",
                    f"run_b_loop_id={run_b.get('loop_id')}",
                ],
            )
        )

        return results

    def cmd_verify(self, *, profile: str) -> int:
        normalized = str(profile or "").strip().lower()
        if normalized not in ("3_5", "3_6", "3_7", "3_8"):
            raise HarnessError(f"Unsupported profile: {profile}")
        if normalized == "3_5":
            checks = self._verify_profile_3_5()
        elif normalized == "3_6":
            checks = self._verify_profile_3_6()
        elif normalized == "3_7":
            checks = self._verify_profile_3_7()
        else:
            checks = self._verify_profile_3_8()
        self._print_header(f"VERIFY REPORT (profile={normalized})")
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
    parser.add_argument(
        "--mock-assistant-result",
        type=str,
        default="yield_control",
        choices=["yield_control", "freeform_yield_no_control", "sentinel_tool_call", "continue_control", "complete_control"],
        help="Harness-only mock LLM response mode used by loop progression checks",
    )
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
    harness = ENSV3Harness(mock_assistant_result=args.mock_assistant_result)
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
