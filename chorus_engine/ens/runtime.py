"""ENS runtime orchestrator for Slice 0/1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from chorus_engine.db.database import SessionLocal
from chorus_engine.ens.models import ENSAction, ENSOutcome, SignalEnvelope
from chorus_engine.ens.dispatcher import ENSDispatcher
from chorus_engine.ens.decision_store import ENSDecisionStore
from chorus_engine.ens.session_registry import ENSSessionRegistry
from chorus_engine.models.conversation import MessageRole

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
        self.session_registry = ENSSessionRegistry()
        self.decision_store = ENSDecisionStore()

    async def ingest(self, signal: SignalEnvelope, ctx: Optional[ENSContext] = None) -> ENSOutcome:
        if ctx is None:
            ctx = ENSContext(app_state=self.app_state)

        db = SessionLocal()
        try:
            decision_id = str(uuid.uuid4())
            resolved_signal = self._resolve_signal_session(db, signal, ctx)
            actions = self._propose_actions(resolved_signal, ctx)
            action_results: List[Dict[str, Any]] = []
            for action in actions:
                if action.kind in ("media.gating.evaluate", "llm.invoke.chat", "tool_payload.adjudicate", "message.write_assistant"):
                    user_write = next(
                        (
                            r
                            for r in action_results
                            if r.get("kind") == "message.write_user" and r.get("status") in ("success", "skipped")
                        ),
                        None,
                    )
                    user_message_id = (user_write or {}).get("output", {}).get("message_id")
                    if action.kind == "media.gating.evaluate" and user_message_id:
                        action.idempotency_key = f"gate:media:{resolved_signal.session_id}:{user_message_id}"
                    if action.kind == "llm.invoke.chat" and user_message_id:
                        action.idempotency_key = f"llm:chat:{resolved_signal.session_id}:{user_message_id}"
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
            thread_id = signal.payload.get("thread_id")
            assistant_id = signal.assistant_id or signal.payload.get("assistant_id")
            if thread_id and assistant_id:
                session = self.session_registry.resolve_thread_session(
                    db,
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                    conversation_id=signal.payload.get("conversation_id"),
                    surface=ctx.surface,
                    source=ctx.source,
                    latency_sensitive=bool(signal.payload.get("latency_sensitive", False)),
                )
                signal.session_id = session.session_id
                signal.user_id = session.user_id
        return signal

    def _propose_actions(self, signal: SignalEnvelope, ctx: ENSContext) -> List[ENSAction]:
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
            user_key = self.dispatcher.user_message_key(signal.session_id or "na", content, client_message_id)

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
                        signal.payload.get("client_message_id"),
                    ),
                    params={
                        "thread_id": signal.payload["thread_id"],
                        "role": role,
                        "content": signal.payload["content"],
                        "metadata": signal.payload.get("metadata"),
                        "is_private": bool(signal.payload.get("is_private", False)),
                    },
                )
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

        return []

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
            response_payload = dict((tool_exec or {}).get("output") or {})
        elif signal.type == "scene_capture.preview_requested" and scene_preview:
            preview_output = dict((scene_preview or {}).get("output") or {})
            if pending_write:
                pending = (pending_write or {}).get("output", {}).get("pending_tool_calls", [])
                if pending:
                    preview_output["tool_call_id"] = pending[0].get("id")
            response_payload = preview_output

        return ENSOutcome(
            decision_id=decision_id,
            trace_id=signal.trace_id,
            signal_id=signal.signal_id,
            actions=actions,
            action_results=action_results,
            response_payload=response_payload,
        )

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
