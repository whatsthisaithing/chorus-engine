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
                if action.kind in ("llm.invoke.chat", "message.write_assistant"):
                    user_write = next(
                        (
                            r
                            for r in action_results
                            if r.get("kind") == "message.write_user" and r.get("status") in ("success", "skipped")
                        ),
                        None,
                    )
                    user_message_id = (user_write or {}).get("output", {}).get("message_id")
                    if action.kind == "llm.invoke.chat" and user_message_id:
                        action.idempotency_key = f"llm:chat:{resolved_signal.session_id}:{user_message_id}"
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
                        action.params["content"] = llm_content

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
            thread_id = signal.payload["thread_id"]
            content = signal.payload["content"]
            metadata = signal.payload.get("metadata")
            is_private = bool(signal.payload.get("is_private", False))
            client_message_id = signal.payload.get("client_message_id")
            user_key = self.dispatcher.user_message_key(signal.session_id or "na", content, client_message_id)

            return [
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
                    kind="llm.invoke.chat",
                    params={
                        "thread_id": thread_id,
                        "character_id": signal.assistant_id,
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
            response_payload = {
                "user_message_id": user_write["output"]["message_id"],
                "assistant_message_id": assistant_write["output"]["message_id"],
            }

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
