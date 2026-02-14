"""ENS action dispatcher."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import MessageRole
from chorus_engine.models.ens import ENSActionResult
from chorus_engine.repositories import ConversationRepository, MessageRepository, ThreadRepository
from chorus_engine.ens.models import ENSAction
from chorus_engine.services.tool_payload import extract_tool_payload, parse_tool_payload, validate_tool_payload

logger = logging.getLogger(__name__)


class ENSDispatcher:
    """Executes ENS actions with idempotency safeguards."""

    def __init__(self, app_state: Dict[str, Any]) -> None:
        self.app_state = app_state

    async def execute(
        self,
        db: Session,
        action: ENSAction,
        *,
        decision_id: str,
    ) -> Dict[str, Any]:
        if action.idempotency_key:
            existing = self._find_existing_by_key(db, action.idempotency_key)
            if existing and existing.output_json is not None:
                return {
                    "action_result_id": str(uuid.uuid4()),
                    "decision_id": decision_id,
                    "action_id": action.action_id,
                    "idempotency_key": action.idempotency_key,
                    "kind": action.kind,
                    "execution_class": action.execution_class,
                    "status": "skipped",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {"replayed": True},
                    "output": existing.output_json,
                }

        started = datetime.utcnow()
        try:
            if action.kind == "message.write_user":
                output = self._write_message(db, action.params, role=MessageRole.USER)
            elif action.kind == "message.write_assistant":
                output = self._write_message(db, action.params, role=MessageRole.ASSISTANT)
            elif action.kind == "llm.invoke.chat":
                output = await self._invoke_llm_chat(db, action.params)
            elif action.kind == "llm.invoke.simple":
                output = await self._invoke_llm_simple(action.params)
            elif action.kind == "message.write_history":
                output = self._write_message(
                    db,
                    action.params,
                    role=MessageRole(action.params["role"]),
                )
            else:
                raise ValueError(f"Unsupported action kind: {action.kind}")

            return {
                "action_result_id": str(uuid.uuid4()),
                "decision_id": decision_id,
                "action_id": action.action_id,
                "idempotency_key": action.idempotency_key,
                "kind": action.kind,
                "execution_class": action.execution_class,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {"latency_ms": int((datetime.utcnow() - started).total_seconds() * 1000)},
                "output": output,
            }
        except Exception as e:
            logger.error("ENS action failed kind=%s error=%s", action.kind, e)
            return {
                "action_result_id": str(uuid.uuid4()),
                "decision_id": decision_id,
                "action_id": action.action_id,
                "idempotency_key": action.idempotency_key,
                "kind": action.kind,
                "execution_class": action.execution_class,
                "status": "failure",
                "error_code": "action_error",
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {"latency_ms": int((datetime.utcnow() - started).total_seconds() * 1000)},
                "output": None,
            }

    def _find_existing_by_key(self, db: Session, key: str) -> Optional[ENSActionResult]:
        return (
            db.query(ENSActionResult)
            .filter(
                ENSActionResult.idempotency_key == key,
                ENSActionResult.status == "success",
            )
            .order_by(ENSActionResult.created_at.desc())
            .first()
        )

    def _write_message(self, db: Session, params: Dict[str, Any], *, role: MessageRole) -> Dict[str, Any]:
        msg_repo = MessageRepository(db)
        message = msg_repo.create(
            thread_id=params["thread_id"],
            role=role,
            content=params["content"],
            metadata=params.get("metadata"),
            is_private=params.get("is_private", False),
        )
        return {"message_id": message.id, "thread_id": message.thread_id}

    async def _invoke_llm_chat(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")

        thread_id = params["thread_id"]
        character_id = params["character_id"]
        msg_repo = MessageRepository(db)
        thread_repo = ThreadRepository(db)
        conv_repo = ConversationRepository(db)

        thread = thread_repo.get_by_id(thread_id)
        if not thread:
            raise RuntimeError("Thread not found")
        conversation = conv_repo.get_by_id(thread.conversation_id)
        if not conversation:
            raise RuntimeError("Conversation not found")

        character = self.app_state["characters"].get(character_id)
        if not character:
            raise RuntimeError("Character not found")

        history = msg_repo.get_thread_history(thread_id)
        messages = [{"role": "system", "content": character.system_prompt}] + history

        temperature = character.preferred_llm.temperature
        max_tokens = character.preferred_llm.max_tokens
        model = character.preferred_llm.model or self.app_state["system_config"].llm.model

        response = await llm_client.generate_with_history(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        content = response.content or ""
        payload_extraction = extract_tool_payload(content)
        payload_obj = parse_tool_payload(payload_extraction.payload_text)
        tool_calls = validate_tool_payload(payload_obj)
        return {
            "content": content,
            "model": model,
            "content_length": len(content),
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "response_excerpt": content[:240],
            "tool_payload_present": payload_extraction.payload_text is not None,
            "tool_payload_parseable": payload_obj is not None,
            "tool_call_count": len(tool_calls),
            "tool_names": sorted({call.tool for call in tool_calls}),
        }

    async def _invoke_llm_simple(self, params: Dict[str, Any]) -> Dict[str, Any]:
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")

        character = self.app_state["characters"].get(params["character_id"])
        if not character:
            raise RuntimeError("Character not found")

        model = character.preferred_llm.model or self.app_state["system_config"].llm.model
        response = await llm_client.generate(
            prompt=params["content"],
            system_prompt=character.system_prompt,
            model=model,
        )
        return {"content": response.content or "", "model": model, "character_name": character.name}

    @staticmethod
    def user_message_key(session_id: str, content: str, client_message_id: Optional[str]) -> str:
        if client_message_id:
            return f"msg:user:{session_id}:{client_message_id}"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]
        return f"msg:user:{session_id}:{digest}"
