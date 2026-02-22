"""ENS runtime orchestrator for Slice 0/1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import uuid
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from chorus_engine.db.database import SessionLocal
from chorus_engine.ens.models import ENSAction, ENSOutcome, SignalEnvelope
from chorus_engine.ens.dispatcher import ENSDispatcher
from chorus_engine.ens.decision_store import ENSDecisionStore
from chorus_engine.ens.scheduler import ENSScheduler
from chorus_engine.ens.session_registry import ENSSessionRegistry
from chorus_engine.ens.surface_identity import canonicalize_surface_id
from chorus_engine.ens.surface_router import SurfaceRouter
from chorus_engine.repositories import ThreadRepository
from chorus_engine.models.conversation import Conversation, ConversationSummary, Memory, Message, MessageRole, Thread

logger = logging.getLogger(__name__)


@dataclass
class ENSContext:
    """Optional request context for ingest; contains no open DB sessions."""

    app_state: Dict[str, Any]
    surface: str = "web"
    source: str = "web"
    data: Dict[str, Any] = field(default_factory=dict)


class ENSRuntime:
    """Stateless ENS runtime that owns action transaction boundaries."""

    def __init__(self, app_state: Dict[str, Any]) -> None:
        self.app_state = app_state
        self.dispatcher = ENSDispatcher(app_state)
        self.scheduler = ENSScheduler()
        self.session_registry = ENSSessionRegistry()
        self.decision_store = ENSDecisionStore()

    def _ens_cfg(self) -> Any:
        cfg = self.app_state.get("system_config")
        return getattr(cfg, "ens", None) if cfg else None

    def _v3_scheduler_enabled(self) -> bool:
        ens_cfg = self._ens_cfg()
        return bool(
            ens_cfg
            and getattr(ens_cfg, "enabled", False)
            and getattr(ens_cfg, "v3_scheduler_enabled", False)
        )

    async def enqueue_signal(self, signal: SignalEnvelope) -> Dict[str, Any]:
        """Queue a signal for v3 scheduler processing."""
        db = SessionLocal()
        try:
            row = self.scheduler.enqueue(db, signal)
            return {
                "queue_id": row.queue_id,
                "signal_id": row.signal_id,
                "status": row.status,
                "priority_tier": row.priority_tier,
                "created_at_us": row.created_at_us,
            }
        finally:
            db.close()

    async def scheduler_tick(self, ctx: Optional[ENSContext] = None) -> Optional[ENSOutcome]:
        """Execute one scheduler tick using current runtime ingest path."""
        if ctx is None:
            ctx = ENSContext(app_state=self.app_state)
        db = SessionLocal()
        try:
            return await self.scheduler.tick(
                db,
                execute_signal=lambda signal: self.ingest(signal, ctx, _force_legacy_execute=True),
            )
        finally:
            db.close()

    async def ingest(
        self,
        signal: SignalEnvelope,
        ctx: Optional[ENSContext] = None,
        *,
        _force_legacy_execute: bool = False,
    ) -> ENSOutcome:
        if ctx is None:
            ctx = ENSContext(app_state=self.app_state)

        if not _force_legacy_execute and self._v3_scheduler_enabled():
            queued = await self.enqueue_signal(signal)
            ens_cfg = self._ens_cfg()
            max_ticks = int(getattr(ens_cfg, "scheduler_sync_ticks_per_ingress", 1) or 0)
            for _ in range(max_ticks):
                outcome = await self.scheduler_tick(ctx)
                if outcome is None:
                    break
                if outcome.signal_id == signal.signal_id:
                    return outcome
            return ENSOutcome(
                decision_id=f"queued:{queued.get('queue_id')}",
                trace_id=signal.trace_id,
                signal_id=signal.signal_id,
                actions=[],
                action_results=[],
                response_payload={
                    "queued": True,
                    "queue_id": queued.get("queue_id"),
                    "status": queued.get("status"),
                },
            )

        db = SessionLocal()
        try:
            decision_id = str(uuid.uuid4())
            resolved_signal = self._resolve_signal_session(db, signal, ctx)
            actions = self._propose_actions(db, resolved_signal, ctx)
            action_results: List[Dict[str, Any]] = []
            for action in actions:
                if action.kind in (
                    "segment.ensure_for_turn",
                    "attachments.link_to_message",
                    "attachments.process_vision",
                    "media.gating.evaluate",
                    "llm.invoke.chat",
                    "tool_payload.adjudicate",
                    "message.write_assistant",
                ):
                    user_write = next(
                        (
                            r
                            for r in action_results
                            if r.get("kind") in ("message.write_user", "message.write_history")
                            and r.get("status") in ("success", "skipped")
                        ),
                        None,
                    )
                    user_message_id = (user_write or {}).get("output", {}).get("message_id")
                    attachment_ids = action.params.get("image_attachment_ids") or []
                    attachment_digest = hashlib.sha256(
                        "|".join(sorted([str(aid) for aid in attachment_ids])).encode("utf-8")
                    ).hexdigest()[:20] if attachment_ids else "none"
                    if action.kind == "attachments.link_to_message" and user_message_id:
                        action.idempotency_key = f"attach:link:{resolved_signal.session_id}:{user_message_id}:{attachment_digest}"
                        action.params["message_id"] = user_message_id
                    if action.kind == "attachments.process_vision" and user_message_id:
                        action.idempotency_key = f"attach:vision:{resolved_signal.session_id}:{user_message_id}:{attachment_digest}"
                        action.params["message_id"] = user_message_id
                    if action.kind == "media.gating.evaluate" and user_message_id:
                        action.idempotency_key = f"gate:media:{resolved_signal.session_id}:{user_message_id}"
                    if action.kind == "segment.ensure_for_turn" and user_message_id:
                        action.idempotency_key = (
                            f"segment:ensure:{resolved_signal.payload.get('conversation_id')}:"
                            f"{resolved_signal.payload.get('thread_id')}:{user_message_id}:v1"
                        )
                        action.params["user_message_id"] = user_message_id
                        action.params["surface_id"] = resolved_signal.payload.get("surface_id")
                        action.params["surface_instance_id"] = resolved_signal.payload.get("surface_instance_id")
                    if action.kind == "llm.invoke.chat" and user_message_id:
                        action.idempotency_key = f"llm:chat:{resolved_signal.session_id}:{user_message_id}"
                        action.params["user_message_id"] = user_message_id
                        segment_result = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "segment.ensure_for_turn" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        action.params["segment_context"] = (segment_result or {}).get("output", {})
                        media_gate = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "media.gating.evaluate" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        action.params["media_gate_snapshot"] = (media_gate or {}).get("output", {}).get("media_gate_snapshot", {})
                    if action.kind == "tool_payload.adjudicate" and user_message_id:
                        action.idempotency_key = f"gate:adjudicate:{resolved_signal.session_id}:{user_message_id}"
                        llm_result = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "llm.invoke.chat" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        if not llm_result:
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="llm_missing",
                                )
                            )
                            continue
                        media_gate = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "media.gating.evaluate" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        action.params["llm_output"] = (llm_result or {}).get("output", {})
                        action.params["media_gate_snapshot"] = (media_gate or {}).get("output", {}).get("media_gate_snapshot", {})
                        action.params["thread_id"] = resolved_signal.payload.get("thread_id")
                        action.params["character_id"] = resolved_signal.assistant_id
                        action.params["user_id"] = resolved_signal.user_id
                    if action.kind == "message.write_assistant" and user_message_id:
                        action.idempotency_key = f"msg:assistant:{resolved_signal.session_id}:{user_message_id}"
                        llm_result = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "llm.invoke.chat" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        llm_content = (llm_result or {}).get("output", {}).get("content", "")
                        llm_metadata = (llm_result or {}).get("output", {}).get("assistant_metadata") or {}
                        action.params["content"] = llm_content
                        if llm_metadata:
                            existing_metadata = dict(action.params.get("metadata") or {})
                            existing_metadata.update(llm_metadata)
                            action.params["metadata"] = existing_metadata
                if action.kind == "tool_call.persist_pending":
                    if resolved_signal.type == "scene_capture.preview_requested":
                        preview_result = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "scene_capture.prompt_generate" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        if not preview_result:
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="preview_not_available",
                                )
                            )
                            continue
                        preview_output = (preview_result or {}).get("output") or {}
                        tool_call_id = preview_output.get("tool_call_id")
                        if not tool_call_id:
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="preview_missing_tool_call_id",
                                )
                            )
                            continue
                        action.params["session_id"] = resolved_signal.session_id or "na"
                        action.params["assistant_message_id"] = None
                        action.params["tool_call"] = {
                            "tool_call_id": tool_call_id,
                            "tool_name": "scene_capture.generate",
                            "args_json": {
                                "conversation_id": resolved_signal.payload.get("conversation_id"),
                                "thread_id": resolved_signal.payload.get("thread_id"),
                                "media_type": resolved_signal.payload.get("media_type"),
                                "client_capture_id": preview_output.get("client_capture_id"),
                                "preview": {
                                    "prompt": preview_output.get("prompt"),
                                    "negative_prompt": preview_output.get("negative_prompt"),
                                    "reasoning": preview_output.get("reasoning"),
                                    "type": preview_output.get("type"),
                                    "needs_trigger": preview_output.get("needs_trigger"),
                                },
                                "prompt": preview_output.get("prompt"),
                                "negative_prompt": preview_output.get("negative_prompt"),
                                "workflow_id": resolved_signal.payload.get("workflow_id"),
                            },
                            "status": "pending",
                            "idempotency_key": f"scene:preview:{resolved_signal.payload.get('conversation_id')}:{resolved_signal.payload.get('thread_id')}:{resolved_signal.payload.get('media_type')}:{preview_output.get('client_capture_id')}",
                            "client_payload": {
                                "id": tool_call_id,
                                "tool": "scene_capture.generate",
                                "args": {
                                    "media_type": resolved_signal.payload.get("media_type"),
                                    "prompt": preview_output.get("prompt"),
                                    "negative_prompt": preview_output.get("negative_prompt"),
                                },
                                "requires_approval": True,
                                "classification": "explicit_request",
                                "needs_confirmation": True,
                            },
                        }
                    else:
                        assistant_write = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "message.write_assistant" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        if not llm_result or not assistant_write:
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="llm_or_assistant_missing",
                                )
                            )
                            continue
                        adjudication = next(
                            (
                                r
                                for r in action_results
                                if r.get("kind") == "tool_payload.adjudicate" and r.get("status") in ("success", "skipped")
                            ),
                            None,
                        )
                        if adjudication:
                            action.params["pending_tool_calls"] = (
                                (adjudication or {}).get("output", {}).get("pending_tool_calls", [])
                            )
                        else:
                            llm_result = next(
                                (
                                    r
                                    for r in action_results
                                    if r.get("kind") == "llm.invoke.chat" and r.get("status") in ("success", "skipped")
                                ),
                                None,
                            )
                            action.params["pending_tool_calls"] = (llm_result or {}).get("output", {}).get("pending_tool_calls", [])
                        action.params["assistant_message_id"] = (assistant_write or {}).get("output", {}).get("message_id")
                        action.params["session_id"] = resolved_signal.session_id or "na"
                        if not action.params["pending_tool_calls"] or not action.params["assistant_message_id"]:
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="no_tool_calls",
                                    output={"reason": "no_tool_calls", "pending_tool_calls": []},
                                )
                            )
                            continue

                if action.kind == "conversation.title.maybe_update":
                    assistant_write = next(
                        (
                            r
                            for r in action_results
                            if r.get("kind") == "message.write_assistant" and r.get("status") in ("success", "skipped")
                        ),
                        None,
                    )
                    assistant_message_id = (assistant_write or {}).get("output", {}).get("message_id")
                    if not assistant_message_id:
                        action_results.append(
                            self._skipped_action_result(
                                action=action,
                                decision_id=decision_id,
                                reason="assistant_missing",
                            )
                        )
                        continue
                    action.idempotency_key = f"title:maybe:{resolved_signal.session_id}:{assistant_message_id}"
                    action.params.setdefault("thread_id", resolved_signal.payload.get("thread_id"))
                    action.params.setdefault("character_id", resolved_signal.assistant_id)

                if action.kind == "continuity.bootstrap":
                    character_id = action.params.get("character_id") or resolved_signal.assistant_id or "unknown"
                    force = bool(action.params.get("force", False))
                    watermark = self._continuity_inputs_watermark(db, character_id)
                    action.idempotency_key = f"continuity:bootstrap:{character_id}:{force}:{watermark}"

                if action.kind in (
                    "config.system.apply",
                    "config.system.post_apply",
                    "config.character.apply_yaml",
                    "config.character.apply_profile_asset",
                    "config.character.reload_runtime",
                    "config.conversation.apply",
                    "message.mutation.apply",
                    "memory.moderation.apply",
                    "config.admin.apply",
                    "config.workflow.apply_file",
                    "config.workflow.apply_db",
                    "config.workflow.rollback_file_or_db",
                    "config.core_memory.apply_db",
                    "config.core_memory.apply_vectors",
                ):
                    validators = {
                        "config.system.apply": "config.system.validate",
                        "config.system.post_apply": "config.system.validate",
                        "config.character.apply_yaml": "config.character.validate",
                        "config.character.apply_profile_asset": "config.character.validate",
                        "config.character.reload_runtime": "config.character.validate",
                        "config.conversation.apply": "config.conversation.validate",
                        "message.mutation.apply": "message.mutation.validate",
                        "memory.moderation.apply": "memory.moderation.validate",
                        "config.admin.apply": "config.admin.validate",
                        "config.workflow.apply_file": "config.workflow.validate",
                        "config.workflow.apply_db": "config.workflow.validate",
                        "config.workflow.rollback_file_or_db": "config.workflow.validate",
                        "config.core_memory.apply_db": "config.core_memory.diff",
                        "config.core_memory.apply_vectors": "config.core_memory.diff",
                    }
                    gate_kind = validators.get(action.kind)
                    gate_result = next((r for r in action_results if r.get("kind") == gate_kind), None)
                    if gate_result and gate_result.get("status") in ("success", "skipped"):
                        gate_output = (gate_result or {}).get("output") or {}
                        if gate_output.get("ok") is False:
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="validation_failed",
                                    output={"reason": "validation_failed", "errors": gate_output.get("errors", [])},
                                )
                            )
                            continue
                        if gate_output.get("no_change") and action.kind != "config.workflow.rollback_file_or_db":
                            action_results.append(
                                self._skipped_action_result(
                                    action=action,
                                    decision_id=decision_id,
                                    reason="no_change",
                                    output={"reason": "no_change"},
                                )
                            )
                            continue
                    if action.kind == "config.workflow.apply_db":
                        file_step = next((r for r in action_results if r.get("kind") == "config.workflow.apply_file"), None)
                        action.params["file_result"] = dict((file_step or {}).get("output") or {})
                    if action.kind == "config.workflow.rollback_file_or_db":
                        db_step = next((r for r in action_results if r.get("kind") == "config.workflow.apply_db"), None)
                        rollback_ctx = ((db_step or {}).get("output") or {}).get("rollback")
                        if not rollback_ctx:
                            file_step = next((r for r in action_results if r.get("kind") == "config.workflow.apply_file"), None)
                            rollback_ctx = ((file_step or {}).get("output") or {}).get("rollback_context")
                        action.params["rollback_context"] = rollback_ctx or {}
                    if action.kind in ("config.core_memory.apply_db", "config.core_memory.apply_vectors"):
                        diff_step = next((r for r in action_results if r.get("kind") == "config.core_memory.diff"), None)
                        action.params["diff_result"] = dict((diff_step or {}).get("output") or {})

                result = await self.dispatcher.execute(db, action, decision_id=decision_id)
                action_results.append(result)
                if result["status"] == "failure":
                    break

            outcome = self._build_outcome(
                decision_id=decision_id,
                signal=resolved_signal,
                actions=actions,
                action_results=action_results,
            )
            self._persist(db, decision_id, resolved_signal, actions, action_results)
            return outcome
        finally:
            db.close()

    def _skipped_action_result(
        self,
        *,
        action: ENSAction,
        decision_id: str,
        reason: str,
        output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "action_result_id": str(uuid.uuid4()),
            "decision_id": decision_id,
            "action_id": action.action_id,
            "idempotency_key": action.idempotency_key,
            "kind": action.kind,
            "execution_class": action.execution_class,
            "status": "skipped",
            "error_code": None,
            "error_message": None,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {"skipped": True},
            "output": output if output is not None else {"reason": reason},
        }

    def _resolve_signal_session(self, db, signal: SignalEnvelope, ctx: ENSContext) -> SignalEnvelope:
        if signal.scope == "SESSION" and not signal.session_id:
            ens_cfg = getattr(self.app_state.get("system_config"), "ens", None)
            slice6_enabled = bool(
                ens_cfg
                and getattr(ens_cfg, "enabled", False)
                and getattr(ens_cfg, "slice6_surface_routing_ownership", False)
            )
            thread_id = signal.payload.get("thread_id")
            if signal.type == "message.mutation_requested" and not thread_id:
                nested_payload = signal.payload.get("payload") or {}
                nested_thread_id = nested_payload.get("thread_id")
                if nested_thread_id:
                    thread_id = nested_thread_id
                    signal.payload["thread_id"] = thread_id
                if not signal.payload.get("conversation_id"):
                    nested_conversation_id = nested_payload.get("conversation_id")
                    if nested_conversation_id:
                        signal.payload["conversation_id"] = nested_conversation_id
            if not thread_id:
                conversation_id = signal.payload.get("conversation_id")
                if conversation_id:
                    thread_repo = ThreadRepository(db)
                    threads = thread_repo.list_by_conversation(conversation_id)
                    if threads:
                        thread_id = threads[0].id
                        signal.payload["thread_id"] = thread_id
            assistant_id = signal.assistant_id or signal.payload.get("assistant_id")
            if slice6_enabled and assistant_id and signal.type != "message.mutation_requested":
                surface_id = signal.surface_id or signal.payload.get("surface_id") or ctx.surface or signal.source
                source = canonicalize_surface_id(surface_id)
                external_thread_id = signal.external_thread_id or signal.payload.get("external_thread_id") or thread_id
                resolver = SurfaceRouter(db)
                resolved = resolver.resolve(
                    assistant_id=assistant_id,
                    surface_id=source,
                    surface_instance_id=signal.surface_instance_id or signal.payload.get("surface_instance_id"),
                    external_thread_id=external_thread_id,
                    relationship_hint=signal.relationship_hint or signal.payload.get("relationship_hint"),
                    target_hint=signal.target_hint or signal.payload.get("target_hint"),
                    conversation_id_hint=signal.payload.get("conversation_id"),
                    thread_id_hint=thread_id,
                )
                signal.payload["conversation_id"] = resolved.conversation_id
                signal.payload["thread_id"] = resolved.thread_id
                signal.payload["surface_id"] = source
                signal.payload["external_thread_id"] = external_thread_id
                if resolved.ignored_target_hint:
                    signal.payload["ignored_target_hint"] = resolved.ignored_target_hint
                signal.surface_id = source
                signal.external_thread_id = external_thread_id
                thread_id = resolved.thread_id
            if thread_id and assistant_id:
                session = self.session_registry.resolve_thread_session(
                    db,
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                    conversation_id=signal.payload.get("conversation_id"),
                    surface=canonicalize_surface_id(signal.surface_id or ctx.surface),
                    source=canonicalize_surface_id(signal.surface_id or signal.source or ctx.source),
                    latency_sensitive=bool(signal.payload.get("latency_sensitive", False)),
                )
                signal.session_id = session.session_id
                signal.user_id = session.user_id
        return signal

    def _propose_actions(self, db, signal: SignalEnvelope, ctx: ENSContext) -> List[ENSAction]:
        if signal.type == "user.message":
            ens_cfg = getattr(self.app_state.get("system_config"), "ens", None)
            slice2_tool_parsing_ownership = bool(
                ens_cfg and getattr(ens_cfg, "enabled", False) and getattr(ens_cfg, "slice2_tool_parsing_ownership", False)
            )
            slice25_media_gating_ownership = bool(
                ens_cfg and getattr(ens_cfg, "enabled", False) and getattr(ens_cfg, "slice25_media_gating_ownership", False)
            )
            thread_id = signal.payload["thread_id"]
            content = signal.payload["content"]
            metadata = signal.payload.get("metadata")
            is_private = bool(signal.payload.get("is_private", False))
            client_message_id = signal.payload.get("client_message_id")
            message_external_id = signal.payload.get("message_external_id")
            id_key = message_external_id or client_message_id
            user_key = self.dispatcher.user_message_key(signal.session_id or "na", content, id_key)

            actions = [
                ENSAction(
                    kind="message.write_user",
                    idempotency_key=user_key,
                    params={
                        "thread_id": thread_id,
                        "content": content,
                        "metadata": metadata,
                        "is_private": is_private,
                    },
                ),
                ENSAction(
                    kind="segment.ensure_for_turn",
                    params={
                        "conversation_id": signal.payload.get("conversation_id"),
                        "thread_id": thread_id,
                        "character_id": signal.assistant_id,
                    },
                ),
                ENSAction(
                    kind="attachments.link_to_message",
                    params={
                        "thread_id": thread_id,
                        "conversation_id": signal.payload.get("conversation_id"),
                        "character_id": signal.assistant_id,
                        "image_attachment_ids": signal.payload.get("image_attachment_ids") or [],
                    },
                ),
                ENSAction(
                    kind="attachments.process_vision",
                    params={
                        "thread_id": thread_id,
                        "conversation_id": signal.payload.get("conversation_id"),
                        "character_id": signal.assistant_id,
                        "content": content,
                        "image_attachment_ids": signal.payload.get("image_attachment_ids") or [],
                    },
                ),
            ]
            if slice2_tool_parsing_ownership and slice25_media_gating_ownership:
                actions.append(
                    ENSAction(
                        kind="media.gating.evaluate",
                        params={
                            "thread_id": thread_id,
                            "character_id": signal.assistant_id,
                            "user_id": signal.user_id,
                            "user_content": content,
                            "conversation_source": signal.payload.get("conversation_source"),
                        },
                    )
                )
            actions.extend(
                [
                ENSAction(
                    kind="llm.invoke.chat",
                    params={
                        "thread_id": thread_id,
                        "character_id": signal.assistant_id,
                        "user_id": signal.user_id,
                        "user_content": content,
                        "conversation_source": signal.payload.get("conversation_source"),
                        "surface_id": signal.payload.get("surface_id"),
                        "surface_instance_id": signal.payload.get("surface_instance_id"),
                        "target_hint": signal.payload.get("target_hint"),
                        "slice2_tool_parsing_ownership": slice2_tool_parsing_ownership,
                    },
                ),
                ENSAction(
                    kind="message.write_assistant",
                    params={
                        "thread_id": thread_id,
                        "content": "",
                        "metadata": {
                            "ens_trace_id": signal.trace_id,
                        },
                        "is_private": is_private,
                    },
                ),
                ]
            )
            if slice2_tool_parsing_ownership and slice25_media_gating_ownership:
                actions.append(
                    ENSAction(
                        kind="tool_payload.adjudicate",
                        params={},
                    )
                )
            if slice2_tool_parsing_ownership:
                actions.append(
                    ENSAction(
                        kind="tool_call.persist_pending",
                        params={},
                    )
                )
            actions.append(
                ENSAction(
                    kind="conversation.title.maybe_update",
                    execution_class="background",
                    params={},
                )
            )
            return actions

        if signal.type == "external.history.message":
            role = signal.payload.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            return [
                ENSAction(
                    kind="message.write_history",
                    idempotency_key=self.dispatcher.user_message_key(
                        signal.session_id or "na",
                        signal.payload["content"],
                        signal.payload.get("message_external_id") or signal.payload.get("client_message_id"),
                    ),
                    params={
                        "thread_id": signal.payload["thread_id"],
                        "role": role,
                        "content": signal.payload["content"],
                        "metadata": signal.payload.get("metadata"),
                        "is_private": bool(signal.payload.get("is_private", False)),
                    },
                ),
                ENSAction(
                    kind="attachments.link_to_message",
                    params={
                        "thread_id": signal.payload["thread_id"],
                        "conversation_id": signal.payload.get("conversation_id"),
                        "character_id": signal.assistant_id,
                        "image_attachment_ids": signal.payload.get("image_attachment_ids") or [],
                    },
                ),
                ENSAction(
                    kind="attachments.process_vision",
                    params={
                        "thread_id": signal.payload["thread_id"],
                        "conversation_id": signal.payload.get("conversation_id"),
                        "character_id": signal.assistant_id,
                        "content": signal.payload.get("content", ""),
                        "role": role,
                        "image_attachment_ids": signal.payload.get("image_attachment_ids") or [],
                    },
                ),
            ]

        if signal.type == "chat.simple":
            return [
                ENSAction(
                    kind="llm.invoke.simple",
                    params={
                        "character_id": signal.assistant_id,
                        "content": signal.payload["content"],
                    },
                )
            ]

        if signal.type == "user.message.stream_intake":
            return []

        if signal.type == "user.message.nonstream_intake":
            return []

        if signal.type == "tool.execute_requested":
            return [
                ENSAction(
                    kind="tool.execute_media",
                    idempotency_key=f"tool:dispatch:{signal.payload.get('tool_call_id')}",
                    params=dict(signal.payload),
                )
            ]
        if signal.type == "scene_capture.preview_requested":
            media_type = signal.payload.get("media_type")
            client_capture_id = signal.payload.get("client_capture_id")
            preview_idempotency = (
                f"scene:preview:{signal.payload.get('conversation_id')}:"
                f"{signal.payload.get('thread_id')}:{media_type}:{client_capture_id}"
            )
            return [
                ENSAction(
                    kind="scene_capture.prompt_generate",
                    idempotency_key=preview_idempotency,
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="tool_call.persist_pending",
                    idempotency_key=preview_idempotency,
                    params={},
                ),
            ]

        if signal.type in ("analysis.manual_requested", "analysis.heartbeat_requested"):
            conversation_id = signal.payload["conversation_id"]
            analysis_kind = signal.payload.get("analysis_kind", "both")
            character_id = signal.assistant_id or signal.payload.get("character_id")
            payload = dict(signal.payload)
            if not payload.get("range_start_message_id") or not payload.get("range_end_message_id"):
                derived_start, derived_end = self._resolve_analysis_idempotency_range(
                    db,
                    conversation_id=conversation_id,
                    analysis_kind=analysis_kind,
                )
                if derived_start != "na" and not payload.get("range_start_message_id"):
                    payload["range_start_message_id"] = derived_start
                if derived_end != "na" and not payload.get("range_end_message_id"):
                    payload["range_end_message_id"] = derived_end
            return [
                ENSAction(
                    kind="analysis.execute",
                    idempotency_key=(
                        f"analysis:{conversation_id}:{analysis_kind}:{payload.get('range_start_message_id') or 'na'}:"
                        f"{payload.get('range_end_message_id') or 'na'}:ens.slice3.v1"
                    ),
                    params={
                        "conversation_id": conversation_id,
                        "character_id": character_id,
                        "analysis_kind": analysis_kind,
                        "range_start_message_id": payload.get("range_start_message_id"),
                        "range_end_message_id": payload.get("range_end_message_id"),
                        "manual": signal.type == "analysis.manual_requested",
                    },
                )
            ]

        if signal.type == "memory.explicit_user_create_requested":
            conversation_id = signal.payload["conversation_id"]
            client_memory_id = signal.payload.get("client_memory_id")
            key = (
                f"mem:explicit:user:{conversation_id}:{client_memory_id}"
                if client_memory_id
                else None
            )
            return [
                ENSAction(
                    kind="memory.write_explicit_user",
                    idempotency_key=key,
                    params=dict(signal.payload),
                )
            ]

        if signal.type == "memory.explicit_vision_create_requested":
            conversation_id = signal.payload["conversation_id"]
            thread_id = signal.payload.get("thread_id")
            message_id = signal.payload.get("message_id")
            vision_model = signal.payload.get("vision_model") or "unknown"
            observation_text = signal.payload.get("observation_text") or signal.payload.get("content") or ""
            normalized_observation = " ".join(observation_text.lower().strip().split())
            source_fingerprint = hashlib.sha256(
                f"{normalized_observation}|{vision_model}|{message_id}".encode("utf-8")
            ).hexdigest()
            return [
                ENSAction(
                    kind="memory.write_explicit_vision",
                    idempotency_key=f"mem:explicit:vision:{conversation_id}:{thread_id}:{message_id}:{source_fingerprint}",
                    params={**dict(signal.payload), "source_fingerprint": source_fingerprint},
                )
            ]

        if signal.type == "pin.create_requested":
            selected_message_ids = signal.payload.get("selected_message_ids") or []
            selection_fingerprint = hashlib.sha256("|".join(selected_message_ids).encode("utf-8")).hexdigest()
            return [
                ENSAction(
                    kind="pin.create",
                    idempotency_key=f"pin:create:{signal.payload.get('conversation_id')}:{selection_fingerprint}:moment_pin.v1",
                    params={**dict(signal.payload), "selection_fingerprint": selection_fingerprint},
                )
            ]

        if signal.type == "conversation.branch_requested":
            selected_message_ids = [
                str(mid).strip()
                for mid in (signal.payload.get("selected_message_ids") or [])
                if str(mid).strip()
            ]
            canonical_ids = sorted(set(selected_message_ids))
            selection_fingerprint = hashlib.sha256("|".join(canonical_ids).encode("utf-8")).hexdigest()
            return [
                ENSAction(
                    kind="conversation.branch_from_general_chat",
                    idempotency_key=(
                        f"branch:{signal.payload.get('source_conversation_id')}:{selection_fingerprint}:v2"
                    ),
                    params={**dict(signal.payload), "selected_message_ids": canonical_ids},
                )
            ]

        if signal.type == "pin.update_requested":
            update_digest = hashlib.sha256(json.dumps(signal.payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
            return [
                ENSAction(
                    kind="pin.update",
                    idempotency_key=f"pin:update:{signal.payload.get('pin_id')}:{update_digest}",
                    params=dict(signal.payload),
                )
            ]

        if signal.type == "pin.delete_requested":
            return [
                ENSAction(
                    kind="pin.delete",
                    idempotency_key=f"pin:delete:{signal.payload.get('pin_id')}",
                    params=dict(signal.payload),
                )
            ]

        if signal.type == "continuity.bootstrap_requested":
            return [
                ENSAction(
                    kind="continuity.bootstrap",
                    idempotency_key=f"continuity:bootstrap:{signal.assistant_id}:{signal.payload.get('force', False)}",
                    params={
                        "character_id": signal.assistant_id,
                        "conversation_id": signal.payload.get("conversation_id"),
                        "force": bool(signal.payload.get("force", False)),
                    },
                )
            ]

        if signal.type == "config.system.change_requested":
            payload_hash = hashlib.sha256(
                json.dumps(signal.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            return [
                ENSAction(
                    kind="config.system.validate",
                    idempotency_key=f"config:system:global:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.system.apply",
                    idempotency_key=f"config:system:global:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.system.post_apply",
                    idempotency_key=f"config:system:global:{payload_hash}",
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "config.character.change_requested":
            payload_hash = hashlib.sha256(
                json.dumps(signal.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            character_id = signal.payload.get("character_id") or "global"
            operation = signal.payload.get("operation")
            actions = [
                ENSAction(
                    kind="config.character.validate",
                    idempotency_key=f"config:character:{character_id}:{payload_hash}",
                    params=dict(signal.payload),
                ),
            ]
            if operation == "reload_runtime_only":
                actions.append(
                    ENSAction(
                        kind="config.character.reload_runtime",
                        idempotency_key=f"config:character:{character_id}:{payload_hash}",
                        params=dict(signal.payload),
                    )
                )
                return actions
            if operation in ("set_profile_image", "upload_profile_image"):
                actions.append(
                    ENSAction(
                        kind="config.character.apply_profile_asset",
                        idempotency_key=f"config:character:{character_id}:{payload_hash}",
                        params=dict(signal.payload),
                    )
                )
            else:
                actions.append(
                    ENSAction(
                        kind="config.character.apply_yaml",
                        idempotency_key=f"config:character:{character_id}:{payload_hash}",
                        params=dict(signal.payload),
                    )
                )
            actions.append(
                ENSAction(
                    kind="config.character.reload_runtime",
                    idempotency_key=f"config:character:{character_id}:{payload_hash}",
                    params=dict(signal.payload),
                )
            )
            return actions

        if signal.type == "config.conversation.change_requested":
            payload_hash = hashlib.sha256(
                json.dumps(signal.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            conversation_id = signal.payload.get("conversation_id") or "global"
            operation = signal.payload.get("operation")
            if operation == "delete_conversation":
                payload = signal.payload.get("payload") or {}
                mode = f"mem{1 if payload.get('delete_memories') else 0}-pins{1 if payload.get('delete_moment_pins') else 0}"
                idempotency_key = f"conv:delete:{conversation_id}:{mode}"
            else:
                idempotency_key = f"conv:update:{conversation_id}:{payload_hash}"
            return [
                ENSAction(
                    kind="config.conversation.validate",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.conversation.apply",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "message.mutation_requested":
            operation = signal.payload.get("operation")
            payload = signal.payload.get("payload") or {}
            if operation == "soft_delete":
                message_ids = sorted([str(mid) for mid in (payload.get("message_ids") or [])])
                if len(message_ids) == 1:
                    idempotency_key = f"msg:soft_delete:{message_ids[0]}"
                else:
                    list_hash = hashlib.sha256("|".join(message_ids).encode("utf-8")).hexdigest()[:20]
                    idempotency_key = f"msg:soft_delete:{list_hash}"
            else:
                payload_hash = hashlib.sha256(
                    json.dumps(signal.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                message_id = payload.get("message_id") or "na"
                idempotency_key = f"msg:metadata:{message_id}:{payload_hash}"
            return [
                ENSAction(
                    kind="message.mutation.validate",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="message.mutation.apply",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "memory.moderation_requested":
            operation = signal.payload.get("operation")
            payload = signal.payload.get("payload") or {}
            if operation in ("approve", "reject"):
                memory_id = str(payload.get("memory_id") or "na")
                idempotency_key = f"memory:{operation}:{memory_id}"
            else:
                memory_ids = sorted([str(mid) for mid in (payload.get("memory_ids") or [])])
                list_hash = hashlib.sha256("|".join(memory_ids).encode("utf-8")).hexdigest()[:20]
                idempotency_key = f"memory:batch_approve:{list_hash}"
            return [
                ENSAction(
                    kind="memory.moderation.validate",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="memory.moderation.apply",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "config.admin.change_requested":
            operation = str(signal.payload.get("operation") or "unknown")
            payload = signal.payload.get("payload") or {}
            payload_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:20]
            idempotency_key = f"config:admin:{operation}:{payload_hash}"
            return [
                ENSAction(
                    kind="config.admin.validate",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.admin.apply",
                    idempotency_key=idempotency_key,
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "config.workflow.change_requested":
            payload_hash = hashlib.sha256(
                json.dumps(signal.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            workflow_scope_key = (
                f"{signal.payload.get('character_id')}:{signal.payload.get('operation')}:"
                f"{signal.payload.get('workflow_name') or signal.payload.get('workflow_id') or signal.payload.get('old_name') or 'na'}"
            )
            return [
                ENSAction(
                    kind="config.workflow.validate",
                    idempotency_key=f"config:workflow:{workflow_scope_key}:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.workflow.apply_file",
                    idempotency_key=f"config:workflow:{workflow_scope_key}:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.workflow.apply_db",
                    idempotency_key=f"config:workflow:{workflow_scope_key}:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.workflow.rollback_file_or_db",
                    idempotency_key=f"config:workflow:{workflow_scope_key}:{payload_hash}",
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "config.core_memory.sync_requested":
            payload_hash = hashlib.sha256(
                json.dumps(signal.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            character_id = signal.payload.get("character_id") or "global"
            return [
                ENSAction(
                    kind="config.core_memory.diff",
                    idempotency_key=f"config:core_memory_sync:{character_id}:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.core_memory.apply_db",
                    idempotency_key=f"config:core_memory_sync:{character_id}:{payload_hash}",
                    params=dict(signal.payload),
                ),
                ENSAction(
                    kind="config.core_memory.apply_vectors",
                    idempotency_key=f"config:core_memory_sync:{character_id}:{payload_hash}",
                    params=dict(signal.payload),
                ),
            ]

        if signal.type == "surface.send_message_requested":
            ens_cfg = getattr(self.app_state.get("system_config"), "ens", None)
            if not bool(ens_cfg and getattr(ens_cfg, "enabled", False) and getattr(ens_cfg, "slice65_egress_outbox_ownership", False)):
                return []
            return [
                ENSAction(
                    kind="surface.egress.persist_intent",
                    params=dict(signal.payload),
                )
            ]

        if signal.type == "llm.control.requested":
            idempotency_key = signal.payload.get("idempotency_key")
            return [
                ENSAction(
                    kind="llm.control.execute",
                    idempotency_key=str(idempotency_key) if idempotency_key else None,
                    params=dict(signal.payload),
                )
            ]

        return []

    @staticmethod
    def _message_is_after_cursor(
        *,
        created_at: Optional[datetime],
        message_id: Optional[str],
        cursor_created_at: Optional[datetime],
        cursor_message_id: Optional[str],
    ) -> bool:
        if not cursor_created_at:
            return True
        if created_at and created_at > cursor_created_at:
            return True
        if created_at and created_at < cursor_created_at:
            return False
        if not cursor_message_id:
            return True
        return str(message_id or "") > str(cursor_message_id)

    def _resolve_analysis_idempotency_range(
        self,
        db,
        *,
        conversation_id: str,
        analysis_kind: str,
    ) -> tuple[str, str]:
        """
        Derive stable message-range markers for analysis idempotency keys.

        - Standard conversations: full non-deleted transcript range.
        - General chat memories: strictly-new incremental range using tuple cursor semantics.
        """
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
        if not conversation:
            return "na", "na"

        rows = (
            db.query(Message.id, Message.created_at)
            .join(Thread, Message.thread_id == Thread.id)
            .filter(Thread.conversation_id == conversation_id)
            .filter(Message.deleted_at.is_(None))
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )
        if not rows:
            return "na", "na"

        if conversation.conversation_kind == "general_chat" and analysis_kind == "memories":
            cursor_created_at = getattr(conversation, "general_chat_memories_processed_through_created_at", None)
            cursor_message_id = getattr(conversation, "general_chat_memories_processed_through_message_id", None)
            new_rows = [
                row for row in rows
                if self._message_is_after_cursor(
                    created_at=row.created_at,
                    message_id=row.id,
                    cursor_created_at=cursor_created_at,
                    cursor_message_id=cursor_message_id,
                )
            ]
            if not new_rows:
                return "na", "na"
            return str(new_rows[0].id), str(new_rows[-1].id)

        return str(rows[0].id), str(rows[-1].id)

    def _build_outcome(
        self,
        *,
        decision_id: str,
        signal: SignalEnvelope,
        actions: List[ENSAction],
        action_results: List[Dict[str, Any]],
    ) -> ENSOutcome:
        response_payload: Dict[str, Any] = {}

        user_write = next((r for r in action_results if r.get("kind") in ("message.write_user", "message.write_history")), None)
        assistant_write = next((r for r in action_results if r.get("kind") == "message.write_assistant"), None)
        pending_write = next((r for r in action_results if r.get("kind") == "tool_call.persist_pending"), None)
        tool_exec = next((r for r in action_results if r.get("kind") == "tool.execute_media"), None)
        scene_preview = next((r for r in action_results if r.get("kind") == "scene_capture.prompt_generate"), None)

        if signal.type == "external.history.message":
            response_payload = {
                "history_message_id": (user_write or {}).get("output", {}).get("message_id")
            }
        elif signal.type == "chat.simple":
            llm_result = next((r for r in action_results if r.get("kind") == "llm.invoke.simple"), None)
            if llm_result and llm_result.get("output"):
                response_payload = {
                    "content": llm_result["output"].get("content", ""),
                    "character_name": llm_result["output"].get("character_name"),
                }
        elif user_write and assistant_write:
            title_update = next((r for r in action_results if r.get("kind") == "conversation.title.maybe_update"), None)
            updated_title = None
            if title_update and title_update.get("status") in ("success", "skipped"):
                title_output = (title_update or {}).get("output") or {}
                if title_output.get("updated"):
                    updated_title = title_output.get("title")
            response_payload = {
                "user_message_id": user_write["output"]["message_id"],
                "assistant_message_id": assistant_write["output"]["message_id"],
                "pending_tool_calls": (pending_write or {}).get("output", {}).get("pending_tool_calls", []),
                "conversation_title_updated": updated_title,
            }
        elif signal.type == "tool.execute_requested" and tool_exec:
            tool_status = (tool_exec or {}).get("status")
            if tool_status == "failure":
                response_payload = {
                    "success": False,
                    "error": (tool_exec or {}).get("error_message") or (tool_exec or {}).get("error_code") or "tool execution failed",
                }
            elif tool_status == "skipped":
                skipped_output = dict((tool_exec or {}).get("output") or {})
                if "success" in skipped_output:
                    response_payload = skipped_output
                else:
                    response_payload = {
                        "success": False,
                        "error": skipped_output.get("reason") or "tool execution skipped",
                    }
            else:
                response_payload = dict((tool_exec or {}).get("output") or {})
        elif signal.type == "scene_capture.preview_requested" and scene_preview:
            preview_output = dict((scene_preview or {}).get("output") or {})
            if pending_write:
                pending = (pending_write or {}).get("output", {}).get("pending_tool_calls", [])
                if pending:
                    preview_output["tool_call_id"] = pending[0].get("id")
            response_payload = preview_output
        elif signal.type in ("analysis.manual_requested", "analysis.heartbeat_requested"):
            analysis_exec = next((r for r in action_results if r.get("kind") == "analysis.execute"), None)
            response_payload = dict((analysis_exec or {}).get("output") or {})
        elif signal.type in ("memory.explicit_user_create_requested", "memory.explicit_vision_create_requested"):
            mem_write = next(
                (r for r in action_results if r.get("kind") in ("memory.write_explicit_user", "memory.write_explicit_vision")),
                None,
            )
            response_payload = dict((mem_write or {}).get("output") or {})
        elif signal.type == "pin.create_requested":
            pin_create = next((r for r in action_results if r.get("kind") == "pin.create"), None)
            response_payload = dict((pin_create or {}).get("output") or {})
        elif signal.type == "conversation.branch_requested":
            branch_result = next(
                (r for r in action_results if r.get("kind") == "conversation.branch_from_general_chat"),
                None,
            )
            response_payload = dict((branch_result or {}).get("output") or {})
        elif signal.type == "pin.update_requested":
            pin_update = next((r for r in action_results if r.get("kind") == "pin.update"), None)
            response_payload = dict((pin_update or {}).get("output") or {})
        elif signal.type == "pin.delete_requested":
            pin_delete = next((r for r in action_results if r.get("kind") == "pin.delete"), None)
            response_payload = dict((pin_delete or {}).get("output") or {})
        elif signal.type == "continuity.bootstrap_requested":
            cont = next((r for r in action_results if r.get("kind") == "continuity.bootstrap"), None)
            response_payload = dict((cont or {}).get("output") or {})
        elif signal.type == "config.system.change_requested":
            apply_result = next((r for r in action_results if r.get("kind") == "config.system.apply"), None)
            post_result = next((r for r in action_results if r.get("kind") == "config.system.post_apply"), None)
            response_payload = dict((apply_result or {}).get("output") or {})
            response_payload.update(dict((post_result or {}).get("output") or {}))
        elif signal.type == "config.character.change_requested":
            apply_yaml = next((r for r in action_results if r.get("kind") == "config.character.apply_yaml"), None)
            apply_asset = next((r for r in action_results if r.get("kind") == "config.character.apply_profile_asset"), None)
            reload_result = next((r for r in action_results if r.get("kind") == "config.character.reload_runtime"), None)
            response_payload = dict((apply_yaml or apply_asset or {}).get("output") or {})
            if reload_result:
                response_payload["runtime_reloaded"] = bool(((reload_result or {}).get("output") or {}).get("reloaded"))
        elif signal.type == "config.conversation.change_requested":
            apply_result = next((r for r in action_results if r.get("kind") == "config.conversation.apply"), None)
            response_payload = dict((apply_result or {}).get("output") or {})
        elif signal.type == "message.mutation_requested":
            apply_result = next((r for r in action_results if r.get("kind") == "message.mutation.apply"), None)
            response_payload = dict((apply_result or {}).get("output") or {})
        elif signal.type == "memory.moderation_requested":
            apply_result = next((r for r in action_results if r.get("kind") == "memory.moderation.apply"), None)
            response_payload = dict((apply_result or {}).get("output") or {})
        elif signal.type == "config.admin.change_requested":
            apply_result = next((r for r in action_results if r.get("kind") == "config.admin.apply"), None)
            response_payload = dict((apply_result or {}).get("output") or {})
        elif signal.type == "config.workflow.change_requested":
            apply_result = next((r for r in action_results if r.get("kind") == "config.workflow.apply_db"), None)
            rollback_result = next((r for r in action_results if r.get("kind") == "config.workflow.rollback_file_or_db"), None)
            response_payload = dict((apply_result or {}).get("output") or {})
            if rollback_result:
                response_payload["rollback"] = dict((rollback_result or {}).get("output") or {})
        elif signal.type == "config.core_memory.sync_requested":
            db_result = next((r for r in action_results if r.get("kind") == "config.core_memory.apply_db"), None)
            vector_result = next((r for r in action_results if r.get("kind") == "config.core_memory.apply_vectors"), None)
            response_payload = dict((db_result or {}).get("output") or {})
            if vector_result:
                response_payload["vectors"] = dict((vector_result or {}).get("output") or {})
        elif signal.type == "surface.send_message_requested":
            intent_result = next((r for r in action_results if r.get("kind") == "surface.egress.persist_intent"), None)
            response_payload = dict((intent_result or {}).get("output") or {})
        elif signal.type == "llm.control.requested":
            control_result = next((r for r in action_results if r.get("kind") == "llm.control.execute"), None)
            response_payload = dict((control_result or {}).get("output") or {})

        return ENSOutcome(
            decision_id=decision_id,
            trace_id=signal.trace_id,
            signal_id=signal.signal_id,
            actions=actions,
            action_results=action_results,
            response_payload=response_payload,
        )

    def _continuity_inputs_watermark(self, db, character_id: str) -> str:
        """Compute a stable watermark from latest continuity-relevant inputs."""
        latest_summary = (
            db.query(func.max(ConversationSummary.created_at))
            .join(Conversation, ConversationSummary.conversation_id == Conversation.id)
            .filter(Conversation.character_id == character_id)
            .scalar()
        )
        latest_memory = (
            db.query(func.max(Memory.created_at))
            .filter(Memory.character_id == character_id)
            .scalar()
        )
        summary_part = latest_summary.isoformat() if latest_summary else "none"
        memory_part = latest_memory.isoformat() if latest_memory else "none"
        return hashlib.sha256(f"{summary_part}|{memory_part}".encode("utf-8")).hexdigest()[:20]

    def _persist(
        self,
        db,
        decision_id: str,
        signal: SignalEnvelope,
        actions: List[ENSAction],
        action_results: List[Dict[str, Any]],
    ) -> None:
        decision_doc = {
            "decision_id": decision_id,
            "trace_id": signal.trace_id,
            "signal_id": signal.signal_id,
            "session_id": signal.session_id,
            "assistant_id": signal.assistant_id,
            "user_id": signal.user_id,
            "scope": signal.scope,
            "signal_type": signal.type,
            "signal_payload": signal.payload,
            "timestamp": datetime.utcnow().isoformat(),
            "appraisal": {
                "urgency": 0.5,
                "risk": 0.2,
                "confidence": 0.9,
            },
            "constraints": [],
            "intent_proposals": [{"name": f"intent:{signal.type}"}],
            "arbitration": {"selected": [a.kind for a in actions]},
            "actions": [
                {
                    "action_id": a.action_id,
                    "kind": a.kind,
                    "execution_class": a.execution_class,
                    "idempotency_key": a.idempotency_key,
                }
                for a in actions
            ],
            "explanation": f"Slice 0/1 route for signal {signal.type}",
        }
        media_gate = next((r for r in action_results if r.get("kind") == "media.gating.evaluate"), None)
        if media_gate and isinstance(media_gate.get("output"), dict):
            decision_doc["media_gate_snapshot"] = media_gate["output"].get("media_gate_snapshot")

        sql_action_results = self._sql_safe_action_results(action_results)
        self.decision_store.persist(
            db,
            decision_doc,
            action_results,
            sql_action_results=sql_action_results,
        )

    @staticmethod
    def _sql_safe_action_results(action_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sql_docs: List[Dict[str, Any]] = []
        for item in action_results:
            row = dict(item)
            output = row.get("output")
            if row.get("kind") == "llm.invoke.chat" and isinstance(output, dict):
                full_content = output.get("content") or ""
                sanitized_output = {k: v for k, v in output.items() if k != "content"}
                sanitized_output.setdefault("content_length", len(full_content))
                sanitized_output.setdefault("response_excerpt", full_content[:240])
                sanitized_output.setdefault("response_sha256", hashlib.sha256(full_content.encode("utf-8")).hexdigest())
                row["output"] = sanitized_output
            sql_docs.append(row)
        return sql_docs
