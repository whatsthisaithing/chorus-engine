"""ENS action dispatcher."""

from __future__ import annotations

import hashlib
import logging
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import MessageRole
from chorus_engine.models.ens import ENSActionResult, ENSToolCallRequest
from chorus_engine.repositories import ConversationRepository, MessageRepository, ThreadRepository
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.repositories.moment_pin_repository import MomentPinRepository
from chorus_engine.repositories.continuity_repository import ContinuityRepository
from chorus_engine.ens.models import ENSAction
from chorus_engine.models.conversation import MemoryType
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.services.moment_pin_extraction_service import MomentPinExtractionService
from chorus_engine.services.prompt_assembly import PromptAssemblyService
from chorus_engine.services.media_turn_classifier import classify_media_turn
from chorus_engine.services.media_offer_policy import (
    compute_turn_media_permissions,
    is_offer_allowed,
    record_offer,
    resolve_effective_offer_policy,
)
from chorus_engine.services.tool_payload import (
    MOMENT_PIN_COLD_RECALL_TOOL,
    detect_malformed_tool_payload_block,
    extract_tool_payload,
    parse_tool_payload,
    strip_malformed_tool_payload_block,
    validate_cold_recall_payload,
    validate_tool_payload,
)
from chorus_engine.services.structured_response import (
    parse_structured_response,
    serialize_structured_response,
    template_rules,
)

logger = logging.getLogger(__name__)


def _get_effective_template(character) -> str:
    if getattr(character, "response_template", None):
        return character.response_template
    level = getattr(character, "immersion_level", "balanced")
    if level in ("full", "unbounded"):
        return "A"
    return "C"


def _looks_like_structured_content(text: str) -> bool:
    if not text:
        return False
    markers = (
        "<assistant_response",
        "<speech>",
        "<physicalaction>",
        "<innerthought>",
        "<narration>",
        "<action>",
    )
    return any(marker in text for marker in markers)

def _attempt_media_payload_repair_prompt(
    *,
    allowed_tools: List[str],
    requested_media_type: str,
    is_iteration_request: bool,
) -> str:
    tools_text = ", ".join(sorted(allowed_tools)) if allowed_tools else "none"
    request_type = "iteration request" if is_iteration_request else "explicit request"
    return (
        "Your last reply did not include a valid media tool payload.\n"
        f"This is a {request_type}. Requested media type: {requested_media_type}.\n"
        f"Allowed tools this turn: {tools_text}.\n"
        "Respond with assistant text followed by exactly one tool payload block:\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        "{\n"
        "  \"version\": 1,\n"
        "  \"tool_calls\": [\n"
        "    {\n"
        "      \"id\": \"unique_call_identifier\",\n"
        "      \"tool\": \"<allowed_tool>\",\n"
        "      \"requires_approval\": true,\n"
        "      \"confidence\": 1.0,\n"
        "      \"args\": {\"prompt\": \"Full generation prompt text\"}\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "---CHORUS_TOOL_PAYLOAD_END---\n"
        "Do not add any extra text after the END sentinel."
    )


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
            elif action.kind == "attachments.link_to_message":
                output = self._link_attachments_to_message(db, action.params)
            elif action.kind == "attachments.process_vision":
                output = await self._process_vision_attachments(db, action.params)
            elif action.kind == "media.gating.evaluate":
                output = self._evaluate_media_gating(db, action.params)
            elif action.kind == "llm.invoke.chat":
                output = await self._invoke_llm_chat(db, action.params)
            elif action.kind == "tool_payload.adjudicate":
                output = self._adjudicate_tool_payload(db, action.params)
            elif action.kind == "tool_call.persist_pending":
                output = self._persist_pending_tool_calls(db, action.params)
            elif action.kind == "conversation.title.maybe_update":
                output = await self._maybe_update_conversation_title(db, action.params)
            elif action.kind == "scene_capture.prompt_generate":
                output = await self._generate_scene_capture_preview(db, action.params)
            elif action.kind == "tool.execute_media":
                output = await self._execute_tool_via_app_executor(db, action.params)
            elif action.kind == "llm.invoke.simple":
                output = await self._invoke_llm_simple(action.params)
            elif action.kind == "message.write_history":
                output = self._write_message(
                    db,
                    action.params,
                    role=MessageRole(action.params["role"]),
                )
            elif action.kind == "analysis.execute":
                output = await self._execute_analysis(db, action.params)
            elif action.kind == "memory.write_explicit_user":
                output = self._write_explicit_user_memory(db, action.params)
            elif action.kind == "memory.write_explicit_vision":
                output = self._write_explicit_vision_memory(db, action.params)
            elif action.kind == "pin.create":
                output = await self._create_moment_pin(db, action.params)
            elif action.kind == "pin.update":
                output = self._update_moment_pin(db, action.params)
            elif action.kind == "pin.delete":
                output = self._delete_moment_pin(db, action.params)
            elif action.kind == "continuity.bootstrap":
                output = await self._run_continuity_bootstrap(db, action.params)
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

    def _append_conversation_ens_debug_log(self, conversation_id: Optional[str], event: Dict[str, Any]) -> None:
        if not conversation_id:
            return
        try:
            conv_dir = Path("data/debug_logs/conversations") / str(conversation_id)
            conv_dir.mkdir(parents=True, exist_ok=True)
            log_file = conv_dir / "ens_conversation.jsonl"
            doc = {"timestamp": datetime.utcnow().isoformat(), **event}
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed writing ENS conversation debug log: %s", e)

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

    def _link_attachments_to_message(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        from chorus_engine.models.conversation import ImageAttachment

        message_id = params.get("message_id")
        attachment_ids = params.get("image_attachment_ids") or []
        if not message_id or not attachment_ids:
            return {"linked_count": 0, "linked_attachment_ids": [], "missing_attachment_ids": []}

        linked_attachment_ids: List[str] = []
        missing_attachment_ids: List[str] = []
        conversation_id = params.get("conversation_id")
        character_id = params.get("character_id")
        for attachment_id in attachment_ids:
            attachment = db.query(ImageAttachment).filter(ImageAttachment.id == attachment_id).first()
            if not attachment:
                missing_attachment_ids.append(str(attachment_id))
                continue
            attachment.message_id = message_id
            if conversation_id:
                attachment.conversation_id = conversation_id
            if character_id:
                attachment.character_id = character_id
            linked_attachment_ids.append(str(attachment_id))
        db.commit()
        return {
            "message_id": message_id,
            "linked_count": len(linked_attachment_ids),
            "linked_attachment_ids": linked_attachment_ids,
            "missing_attachment_ids": missing_attachment_ids,
        }

    async def _process_vision_attachments(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        from chorus_engine.models.conversation import ImageAttachment

        message_id = params.get("message_id")
        attachment_ids = params.get("image_attachment_ids") or []
        role = params.get("role", "user")
        if not message_id or not attachment_ids:
            return {"processed_count": 0, "already_processed_count": 0, "memory_ids": [], "current_turn_visual_context_count": 0}
        if role != "user":
            return {"processed_count": 0, "already_processed_count": 0, "memory_ids": [], "current_turn_visual_context_count": 0, "reason": "non_user_role"}

        vision_service = self.app_state.get("vision_service")
        if not vision_service:
            return {"processed_count": 0, "already_processed_count": 0, "memory_ids": [], "current_turn_visual_context_count": 0, "reason": "vision_service_unavailable"}

        processed_count = 0
        already_processed_count = 0
        memory_ids: List[str] = []
        memory_config = vision_service.config.get("memory", {}) if hasattr(vision_service, "config") else {}
        min_confidence = float(memory_config.get("min_confidence", 0.6))
        default_priority = memory_config.get("default_priority", 70)
        context = params.get("content", "")
        character_id = params.get("character_id")
        conversation_id = params.get("conversation_id")

        for attachment_id in attachment_ids:
            attachment = db.query(ImageAttachment).filter(ImageAttachment.id == attachment_id).first()
            if not attachment:
                continue
            if attachment.vision_processed == "true":
                already_processed_count += 1
                continue
            try:
                result = await vision_service.analyze_image(
                    image_path=Path(attachment.original_path),
                    context=context,
                    character_id=character_id,
                )
                attachment.vision_processed = "true"
                attachment.vision_model = result.model
                attachment.vision_backend = result.backend
                attachment.vision_processed_at = datetime.utcnow()
                attachment.vision_processing_time_ms = result.processing_time_ms
                attachment.vision_observation = result.observation
                attachment.vision_confidence = result.confidence
                attachment.vision_tags = json.dumps(result.tags) if result.tags else None
                processed_count += 1

                if memory_config.get("auto_create", True) and result.confidence >= min_confidence:
                    vision_data = None
                    if result.observation:
                        try:
                            vision_data = json.loads(result.observation) if isinstance(result.observation, str) else result.observation
                        except Exception:
                            vision_data = {"description": result.observation}
                    if not vision_data:
                        vision_data = {"description": "Image analyzed but no details available"}
                    description_text = vision_data.get("description") if isinstance(vision_data, dict) else None
                    if not description_text and result.observation:
                        description_text = result.observation
                    if description_text:
                        content = f"User showed me an image: {str(description_text).strip()}"
                    else:
                        content = "User showed me an image."

                    memory_out = self._write_explicit_vision_memory(
                        db,
                        {
                            "conversation_id": conversation_id,
                            "thread_id": params.get("thread_id"),
                            "character_id": character_id,
                            "content": content,
                            "message_id": message_id,
                            "vision_model": result.model,
                            "observation_text": result.observation,
                            "confidence": result.confidence,
                            "category": "visual",
                            "priority": default_priority,
                            "status": "auto_approved",
                            "metadata": {
                                "source_messages": [message_id],
                                "image_attachment_id": attachment.id,
                                "vision_model": result.model,
                                "vision_backend": result.backend,
                            },
                            "source": "web",
                        },
                    )
                    memory_id = memory_out.get("memory_id")
                    if memory_id:
                        memory_ids.append(str(memory_id))
            except Exception as e:
                attachment.vision_skipped = "true"
                attachment.vision_skip_reason = f"analysis_failed: {str(e)[:80]}"

        db.commit()
        current_turn_visual_context_count = (
            db.query(ImageAttachment)
            .filter(
                ImageAttachment.message_id == message_id,
                ImageAttachment.vision_processed == "true",
            )
            .count()
        )
        return {
            "processed_count": processed_count,
            "already_processed_count": already_processed_count,
            "memory_ids": memory_ids,
            "current_turn_visual_context_count": current_turn_visual_context_count,
        }

    def _evaluate_media_gating(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params["thread_id"]
        character_id = params["character_id"]
        user_content = params.get("user_content") or ""
        user_id = params.get("user_id")

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

        semantic_intents = []
        try:
            from chorus_engine.services.semantic_intent_detection import get_intent_detector

            detector = get_intent_detector()
            semantic_intents = detector.detect(user_content, enable_multi_intent=True, debug=False)
        except Exception as semantic_error:
            logger.warning("ENS semantic intent detection failed: %s", semantic_error)

        media_cfg = getattr(self.app_state["system_config"], "media_tooling", None)
        explicit_image_threshold = media_cfg.explicit_min_confidence_image if media_cfg else 0.5
        explicit_video_threshold = media_cfg.explicit_min_confidence_video if media_cfg else 0.45
        turn_signals = classify_media_turn(
            message=user_content,
            semantic_intents=semantic_intents,
            explicit_image_threshold=explicit_image_threshold,
            explicit_video_threshold=explicit_video_threshold,
        )

        effective_policy = resolve_effective_offer_policy(self.app_state["system_config"], character)
        current_message_count = msg_repo.count_thread_messages(thread_id)
        source = (params.get("conversation_source") or conversation.source or "web")
        messages_for_media_type = msg_repo.get_thread_history_objects(thread_id)
        preferred_iteration_media_type = "none"
        for msg in reversed(messages_for_media_type):
            if msg.role != MessageRole.ASSISTANT:
                continue
            meta = msg.meta_data or {}
            if isinstance(meta, dict) and meta.get("video_id"):
                preferred_iteration_media_type = "video"
                break
            if isinstance(meta, dict) and meta.get("image_id"):
                preferred_iteration_media_type = "image"
                break

        media_permissions = compute_turn_media_permissions(
            turn_signals=turn_signals,
            policy=effective_policy,
            conversation=conversation,
            source=source,
            current_message_count=current_message_count,
            image_generation_enabled=bool(character.image_generation and character.image_generation.enabled),
            video_generation_enabled=bool(getattr(character, "video_generation", None) and character.video_generation.enabled),
            preferred_iteration_media_type=preferred_iteration_media_type,
        )

        turn_classification = "none"
        if media_permissions.explicit_allowed and media_permissions.is_iteration_request:
            turn_classification = "iterate_media"
        elif media_permissions.explicit_allowed:
            turn_classification = "explicit_request"
        elif media_permissions.offer_allowed:
            turn_classification = "proactive_offer"

        media_gate_snapshot = {
            "conversation_id": conversation.id,
            "thread_id": thread_id,
            "turn_classification": turn_classification,
            "requested_media_type": media_permissions.requested_media_type,
            "media_tool_calls_allowed": media_permissions.media_tool_calls_allowed,
            "explicit_allowed": media_permissions.explicit_allowed,
            "offer_allowed": media_permissions.offer_allowed,
            "cooldown_active": media_permissions.cooldown_active,
            "is_iteration_request": media_permissions.is_iteration_request,
            "allowed_tools_input": media_permissions.allowed_tools_input,
            "allowed_tools_final": media_permissions.allowed_tools_final,
            "media_offer_allowed_this_turn": media_permissions.media_offer_allowed_this_turn,
            "capability_state": {
                "image_enabled": bool(character.image_generation and character.image_generation.enabled),
                "video_enabled": bool(getattr(character, "video_generation", None) and character.video_generation.enabled),
            },
            "source_restrictions": {"source": source},
            "iteration_state": {"preferred_iteration_media_type": preferred_iteration_media_type},
            "cooldown_state": {
                "time_active": media_permissions.cooldown_time_active,
                "message_active": media_permissions.cooldown_message_active,
            },
            "offer_min_confidence": {
                "image": float(effective_policy.image_min_confidence),
                "video": float(effective_policy.video_min_confidence),
            },
            "current_message_count": int(current_message_count),
            "user_id": user_id,
            "image_confirmation_disabled": conversation.image_confirmation_disabled == "true",
            "video_confirmation_disabled": conversation.video_confirmation_disabled == "true",
        }
        return {
            "media_gate_snapshot": media_gate_snapshot,
            "semantic_intents_detected": [
                {"name": getattr(i, "name", None), "confidence": float(getattr(i, "confidence", 0.0) or 0.0)}
                for i in (semantic_intents or [])
            ],
        }

    async def _invoke_llm_chat(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        from chorus_engine.models.conversation import ImageAttachment

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

        user_content = params.get("user_content")
        user_message_id = params.get("user_message_id")
        if not isinstance(user_content, str) or not user_content.strip():
            history_probe = msg_repo.get_thread_history(thread_id)
            for item in reversed(history_probe):
                if item.get("role") == "user":
                    user_content = item.get("content", "")
                    break
            else:
                user_content = ""

        media_gate_snapshot = params.get("media_gate_snapshot") or {}
        source = (params.get("conversation_source") or conversation.source or "web")

        prompt_assembler = PromptAssemblyService(
            db=db,
            character_id=character.id,
            model_name=self.app_state["system_config"].llm.model,
            context_window=character.preferred_llm.context_window or self.app_state["system_config"].llm.context_window,
        )
        prompt_components = prompt_assembler.assemble_prompt(
            thread_id=thread_id,
            include_memories=True,
            primary_user=conversation.primary_user,
            conversation_source=source,
            conversation_id=conversation.id,
            user_id=params.get("user_id"),
            include_conversation_context=True,
            allowed_media_tools=set(media_gate_snapshot.get("allowed_tools_final") or []),
            allow_proactive_media_offers=bool(media_gate_snapshot.get("offer_allowed")),
            media_gate_context={
                "media_tool_calls_allowed": bool(media_gate_snapshot.get("media_tool_calls_allowed")),
                "allowed_tools": media_gate_snapshot.get("allowed_tools_final") or [],
                "requested_media_type": media_gate_snapshot.get("requested_media_type") or "none",
                "is_iteration_request": bool(media_gate_snapshot.get("is_iteration_request")),
            },
        )
        messages = prompt_assembler.format_for_api(prompt_components)

        temperature = character.preferred_llm.temperature
        max_tokens = character.preferred_llm.max_tokens
        model = character.preferred_llm.model or self.app_state["system_config"].llm.model

        response = await llm_client.generate_with_history(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        raw_content = response.content or ""
        payload_extraction = extract_tool_payload(raw_content)
        payload_obj = parse_tool_payload(payload_extraction.payload_text)
        display_text = payload_extraction.display_text
        malformed_tool_payload_non_sentinel = False
        malformed_payload_type: Optional[str] = None
        assistant_metadata: Dict[str, Any] = {}
        if payload_extraction.payload_text is None:
            stripped_text, stripped, payload_type = strip_malformed_tool_payload_block(display_text)
            if stripped:
                malformed_tool_payload_non_sentinel = True
                malformed_payload_type = payload_type
                display_text = stripped_text
                assistant_metadata["malformed_tool_payload_non_sentinel"] = True
                assistant_metadata["malformed_payload_type"] = payload_type
                logger.warning(
                    "[ENS_SANITIZER] Stripped malformed non-sentinel payload block type=%s thread_id=%s",
                    payload_type,
                    thread_id,
                )
        if _looks_like_structured_content(display_text):
            template = _get_effective_template(character)
            allowed_channels, required_channels = template_rules(template)
            parsed = parse_structured_response(
                display_text,
                allowed_channels=allowed_channels,
                required_channels=required_channels,
            )
            display_text = serialize_structured_response(parsed.segments)
            assistant_metadata["structured_response"] = {
                "is_fallback": parsed.is_fallback,
                "parse_error": parsed.parse_error,
                "had_untagged": parsed.had_untagged,
                "template": template,
                "raw_response": raw_content,
            }
        detected_raw_malformed, detected_raw_payload_type = detect_malformed_tool_payload_block(raw_content)
        if detected_raw_malformed and not malformed_tool_payload_non_sentinel:
            malformed_tool_payload_non_sentinel = True
            malformed_payload_type = detected_raw_payload_type
            assistant_metadata["malformed_tool_payload_non_sentinel"] = True
            assistant_metadata["malformed_payload_type"] = detected_raw_payload_type
        pending_tool_calls: List[Dict[str, Any]] = []
        tool_names: List[str] = []
        tool_call_count = 0
        if bool(params.get("slice2_tool_parsing_ownership")) and not media_gate_snapshot:
            legacy_calls = validate_tool_payload(payload_obj)
            if isinstance(payload_obj, dict):
                raw_calls = payload_obj.get("tool_calls") or []
                has_cold = any(
                    isinstance(item, dict) and item.get("tool") == MOMENT_PIN_COLD_RECALL_TOOL
                    for item in raw_calls
                )
                if has_cold and len(raw_calls) != 1:
                    legacy_calls = []
            pending_tool_calls = [
                {
                    "id": call.id,
                    "tool": call.tool,
                    "requires_approval": call.requires_approval,
                    "args": {"prompt": call.prompt},
                    "classification": "explicit_request",
                    "needs_confirmation": True,
                }
                for call in legacy_calls
            ]
            tool_names = sorted({call.tool for call in legacy_calls})
            tool_call_count = len(legacy_calls)

        result = {
            "content": display_text,
            "raw_content": raw_content,
            "model": model,
            "content_length": len(display_text),
            "response_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
            "response_excerpt": raw_content[:240],
            "tool_payload_present": payload_extraction.payload_text is not None,
            "tool_payload_parseable": payload_obj is not None,
            "tool_parse_status": (
                "malformed_non_sentinel"
                if malformed_tool_payload_non_sentinel and payload_obj is None
                else ("ok" if payload_obj is not None else "none_or_invalid")
            ),
            "tool_call_count": tool_call_count,
            "tool_names": tool_names,
            "pending_tool_calls": pending_tool_calls,
            "assistant_metadata": assistant_metadata,
            "malformed_tool_payload_non_sentinel": malformed_tool_payload_non_sentinel,
            "malformed_payload_type": malformed_payload_type,
            "current_turn_visual_context_count": (
                db.query(ImageAttachment)
                .filter(
                    ImageAttachment.message_id == user_message_id,
                    ImageAttachment.vision_processed == "true",
                )
                .count()
                if user_message_id
                else 0
            ),
        }
        self._append_conversation_ens_debug_log(
            conversation.id,
            {
                "type": "ens_llm_turn",
                "thread_id": thread_id,
                "character_id": character_id,
                "user_content_excerpt": (user_content or "")[:240],
                "media_gate_snapshot": media_gate_snapshot,
                "tool_parse_status": result["tool_parse_status"],
                "tool_payload_present": result["tool_payload_present"],
                "malformed_tool_payload_non_sentinel": malformed_tool_payload_non_sentinel,
                "malformed_payload_type": malformed_payload_type,
                "messages_tail": messages[-6:],
                "raw_content": raw_content,
                "display_content": display_text,
            },
        )
        return result

    def _adjudicate_tool_payload(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        llm_output = params.get("llm_output") or {}
        media_gate_snapshot = params.get("media_gate_snapshot") or {}
        raw_content = llm_output.get("raw_content") or llm_output.get("content") or ""
        payload_extraction = extract_tool_payload(raw_content)
        payload_obj = parse_tool_payload(payload_extraction.payload_text)
        media_tool_calls = validate_tool_payload(payload_obj)
        cold_recall_call = validate_cold_recall_payload(payload_obj)
        parse_status = "ok" if payload_obj is not None else "none_or_invalid"

        blocked_reasons: List[str] = []
        blocked_calls: List[Dict[str, Any]] = []
        accepted: List[Dict[str, Any]] = []
        media_tool_calls_allowed = bool(media_gate_snapshot.get("media_tool_calls_allowed"))
        allowed_tools_final = list(media_gate_snapshot.get("allowed_tools_final") or [])
        requested_media_type = media_gate_snapshot.get("requested_media_type") or "none"
        explicit_allowed = bool(media_gate_snapshot.get("explicit_allowed"))
        is_iteration_request = bool(media_gate_snapshot.get("is_iteration_request"))
        offer_allowed = bool(media_gate_snapshot.get("offer_allowed"))
        image_confirm_disabled = bool(media_gate_snapshot.get("image_confirmation_disabled"))
        video_confirm_disabled = bool(media_gate_snapshot.get("video_confirmation_disabled"))
        explicit_candidates: List[str] = []
        if requested_media_type == "image":
            explicit_candidates = ["image.generate"]
        elif requested_media_type == "video":
            explicit_candidates = ["video.generate"]
        elif requested_media_type == "either":
            explicit_candidates = ["image.generate", "video.generate"]

        if isinstance(payload_obj, dict):
            raw_calls = payload_obj.get("tool_calls") or []
            has_cold = any(isinstance(item, dict) and item.get("tool") == MOMENT_PIN_COLD_RECALL_TOOL for item in raw_calls)
            if has_cold and len(raw_calls) != 1:
                blocked_reasons.append("tool_chaining_not_allowed")
                media_tool_calls = []
                cold_recall_call = None

        if cold_recall_call:
            blocked_reasons.append("cold_recall_deferred")
            media_tool_calls = []

        malformed_non_sentinel = bool(llm_output.get("malformed_tool_payload_non_sentinel"))
        malformed_payload_type = llm_output.get("malformed_payload_type")
        if malformed_non_sentinel:
            blocked_reasons.append("malformed_non_sentinel_payload")
            parse_status = "malformed_non_sentinel"

        for call in media_tool_calls:
            if not media_tool_calls_allowed:
                blocked_reasons.append("media_tooling_disabled")
                blocked_calls.append({"id": call.id, "tool": call.tool, "reason": "media_tooling_disabled"})
                continue
            if call.tool not in allowed_tools_final:
                blocked_reasons.append("tool_not_allowed")
                blocked_calls.append({"id": call.id, "tool": call.tool, "reason": "tool_not_allowed"})
                continue

            is_explicit = bool(explicit_allowed and call.tool in explicit_candidates)
            if is_explicit:
                classification = "iterate_media" if is_iteration_request else "explicit_request"
            else:
                if not offer_allowed:
                    reason = "cooldown_active" if media_gate_snapshot.get("cooldown_active") else "offers_disabled"
                    blocked_reasons.append(reason)
                    blocked_calls.append({"id": call.id, "tool": call.tool, "reason": reason})
                    continue
                media_kind = "image" if call.tool == "image.generate" else "video"
                min_conf_map = media_gate_snapshot.get("offer_min_confidence") or {}
                min_conf = float(min_conf_map.get(media_kind, 0.0) or 0.0)
                if float(call.confidence) < min_conf:
                    blocked_reasons.append("confidence_too_low")
                    blocked_calls.append({"id": call.id, "tool": call.tool, "reason": "confidence_too_low"})
                    continue
                classification = "proactive_offer"

            accepted.append(
                {
                    "id": call.id,
                    "tool": call.tool,
                    "requires_approval": call.requires_approval,
                    "args": {"prompt": call.prompt},
                    "classification": classification,
                    "needs_confirmation": (
                        True
                        if classification == "proactive_offer"
                        else (not image_confirm_disabled if call.tool == "image.generate" else not video_confirm_disabled)
                    ),
                    "confidence": float(call.confidence),
                }
            )

        if accepted:
            conversation_id = media_gate_snapshot.get("conversation_id")
            current_message_count = int(media_gate_snapshot.get("current_message_count") or 0)
            if conversation_id:
                conv_repo = ConversationRepository(db)
                conversation = conv_repo.get_by_id(conversation_id)
                if conversation:
                    for item in accepted:
                        if item.get("classification") != "proactive_offer":
                            continue
                        media_kind = "image" if item.get("tool") == "image.generate" else "video"
                        record_offer(
                            conversation=conversation,
                            media_kind=media_kind,
                            current_message_count=current_message_count,
                        )
                    db.commit()

        result = {
            "tool_payload_present": payload_extraction.payload_text is not None,
            "tool_payload_parseable": payload_obj is not None,
            "tool_parse_status": parse_status,
            "accepted_tool_calls": accepted,
            "pending_tool_calls": [
                {
                    "id": item["id"],
                    "tool": item["tool"],
                    "requires_approval": item["requires_approval"],
                    "args": item["args"],
                    "classification": item["classification"],
                    "needs_confirmation": item["needs_confirmation"],
                }
                for item in accepted
            ],
            "tool_call_count": len(accepted),
            "tool_names": sorted({item["tool"] for item in accepted}),
            "blocked_reasons": sorted(set(blocked_reasons)),
            "blocked_calls": blocked_calls,
            "cold_recall_requested": cold_recall_call is not None,
            "cold_recall_executed": False,
            "malformed_tool_payload_non_sentinel": malformed_non_sentinel,
            "malformed_payload_type": malformed_payload_type,
        }
        self._append_conversation_ens_debug_log(
            media_gate_snapshot.get("conversation_id"),
            {
                "type": "ens_tool_adjudication",
                "thread_id": params.get("thread_id"),
                "tool_parse_status": result["tool_parse_status"],
                "tool_payload_present": result["tool_payload_present"],
                "tool_payload_parseable": result["tool_payload_parseable"],
                "accepted_tool_calls": result["accepted_tool_calls"],
                "blocked_reasons": result["blocked_reasons"],
                "malformed_tool_payload_non_sentinel": malformed_non_sentinel,
                "malformed_payload_type": malformed_payload_type,
            },
        )
        return result

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

    def _persist_pending_tool_calls(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        explicit_tool = params.get("tool_call")
        if explicit_tool:
            tool_call_id = explicit_tool["tool_call_id"]
            existing = db.query(ENSToolCallRequest).filter(ENSToolCallRequest.tool_call_id == tool_call_id).first()
            if not existing:
                row = ENSToolCallRequest(
                    tool_call_id=tool_call_id,
                    session_id=params["session_id"],
                    assistant_message_id=params.get("assistant_message_id"),
                    tool_name=explicit_tool["tool_name"],
                    args_json=explicit_tool.get("args_json") or {},
                    status=explicit_tool.get("status") or "pending",
                    idempotency_key=explicit_tool["idempotency_key"],
                    result_ref=explicit_tool.get("result_ref"),
                )
                db.add(row)
                db.commit()
            else:
                existing.args_json = explicit_tool.get("args_json") or {}
                existing.status = explicit_tool.get("status") or existing.status
                db.commit()
            return {"pending_tool_calls": [explicit_tool.get("client_payload") or {"id": tool_call_id}]}

        session_id = params["session_id"]
        assistant_message_id = params["assistant_message_id"]
        pending_tool_calls: List[Dict[str, Any]] = params.get("pending_tool_calls") or []
        persisted: List[Dict[str, Any]] = []

        for call in pending_tool_calls:
            raw_id = str(call.get("id") or uuid.uuid4())
            tool_call_id = f"tc:{assistant_message_id}:{raw_id}"
            existing = db.query(ENSToolCallRequest).filter(ENSToolCallRequest.tool_call_id == tool_call_id).first()
            if not existing:
                row = ENSToolCallRequest(
                    tool_call_id=tool_call_id,
                    session_id=session_id,
                    assistant_message_id=assistant_message_id,
                    tool_name=call.get("tool") or "unknown",
                    args_json=call.get("args") or {},
                    status="pending",
                    idempotency_key=f"tool:pending:{session_id}:{assistant_message_id}:{raw_id}",
                    result_ref=None,
                )
                db.add(row)
                db.commit()
            persisted.append(
                {
                    "id": tool_call_id,
                    "tool": call.get("tool"),
                    "requires_approval": bool(call.get("requires_approval", True)),
                    "args": call.get("args") or {},
                    "classification": call.get("classification") or "explicit_request",
                    "needs_confirmation": bool(call.get("needs_confirmation", True)),
                }
            )

        return {"pending_tool_calls": persisted}

    async def _generate_scene_capture_preview(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        executor = self.app_state.get("ens_scene_preview_executor")
        if not executor:
            raise RuntimeError("ENS scene preview executor not initialized")
        return await executor(db, params)

    async def _execute_tool_via_app_executor(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        executor = self.app_state.get("ens_tool_executor")
        if not executor:
            raise RuntimeError("ENS tool executor not initialized")
        return await executor(db, params)

    async def _execute_analysis(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        analysis_service: ConversationAnalysisService = self.app_state.get("analysis_service")
        if not analysis_service:
            raise RuntimeError("Analysis service not initialized")

        conversation_id = params["conversation_id"]
        character_id = params["character_id"]
        manual = bool(params.get("manual", False))
        analysis_kind = params.get("analysis_kind", "both")
        character = self.app_state["characters"].get(character_id)
        if not character:
            raise RuntimeError(f"Character not found: {character_id}")

        if analysis_kind == "summary":
            analysis = await analysis_service.analyze_summary_only(
                conversation_id=conversation_id,
                character=character,
                manual=manual,
            )
            saved = await analysis_service.save_summary_only(
                conversation_id=conversation_id,
                character_id=character_id,
                analysis=analysis,
                manual=manual,
            ) if analysis else False
        elif analysis_kind == "memories":
            analysis = await analysis_service.analyze_memories_only(
                conversation_id=conversation_id,
                character=character,
                manual=manual,
            )
            saved = await analysis_service.save_memories_only(
                conversation_id=conversation_id,
                character_id=character_id,
                analysis=analysis,
            ) if analysis else False
        else:
            analysis = await analysis_service.analyze_conversation(
                conversation_id=conversation_id,
                character=character,
                manual=manual,
            )
            saved = await analysis_service.save_analysis(
                conversation_id=conversation_id,
                character_id=character_id,
                analysis=analysis,
                manual=manual,
            ) if analysis else False

        if not analysis or not saved:
            return {
                "status": "no_result",
                "conversation_id": conversation_id,
                "character_id": character_id,
                "analysis_kind": analysis_kind,
                "manual": manual,
                "saved": bool(saved),
                "memories_extracted": 0,
                "summary_length": 0,
            }

        conv_repo = ConversationRepository(db)
        conversation = conv_repo.get_by_id(conversation_id)
        memory_counts: Dict[str, int] = {}
        memory_payload: List[Dict[str, Any]] = []
        for memory in analysis.memories or []:
            mem_type = memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type)
            memory_counts[mem_type] = memory_counts.get(mem_type, 0) + 1
            memory_payload.append(
                {
                    "type": mem_type,
                    "content": memory.content,
                    "confidence": memory.confidence,
                    "emotional_weight": memory.emotional_weight,
                    "reasoning": memory.reasoning,
                    "durability": getattr(memory, "durability", None),
                    "pattern_eligible": getattr(memory, "pattern_eligible", None),
                }
            )
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "character_id": character_id,
            "analysis_kind": analysis_kind,
            "manual": manual,
            "saved": True,
            "memories_extracted": len(analysis.memories or []),
            "memory_counts": memory_counts,
            "memories": memory_payload,
            "summary_length": len(analysis.summary or ""),
            "summary": analysis.summary,
            "key_topics": analysis.key_topics,
            "tone": analysis.tone,
            "emotional_arc": analysis.emotional_arc,
            "participants": analysis.participants,
            "open_questions": analysis.open_questions,
            "current_summary_id": getattr(conversation, "current_summary_id", None) if conversation else None,
        }

    def _write_explicit_user_memory(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = params["conversation_id"]
        character_id = params["character_id"]
        content = params["content"]
        thread_id = params.get("thread_id")
        tags = params.get("tags")
        priority = params.get("priority")
        client_memory_id = params.get("client_memory_id")

        memory_repo = MemoryRepository(db)
        # Avoid importing model at file top just for this check.
        from chorus_engine.models.conversation import Memory

        if client_memory_id:
            existing = (
                db.query(Memory)
                .filter(
                    Memory.conversation_id == conversation_id,
                    Memory.client_memory_id == client_memory_id,
                )
                .first()
            )
            if existing:
                return {"memory_id": existing.id, "replayed": True, "client_memory_id": client_memory_id}

        memory = memory_repo.create(
            content=content,
            character_id=character_id,
            memory_type=MemoryType.EXPLICIT,
            conversation_id=conversation_id,
            thread_id=thread_id,
            tags=tags,
            priority=priority,
            client_memory_id=client_memory_id,
            source_kind="explicit_user",
            source="web",
        )
        return {"memory_id": memory.id, "replayed": False, "client_memory_id": client_memory_id}

    @staticmethod
    def _normalize_observation(text: str) -> str:
        return " ".join((text or "").lower().strip().split())

    def _write_explicit_vision_memory(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        from chorus_engine.models.conversation import Memory

        conversation_id = params["conversation_id"]
        thread_id = params.get("thread_id")
        character_id = params["character_id"]
        content = params["content"]
        message_id = params["message_id"]
        vision_model = params.get("vision_model") or "unknown"
        observation_text = params.get("observation_text") or content

        normalized_observation = self._normalize_observation(observation_text)
        source_fingerprint = hashlib.sha256(
            f"{normalized_observation}|{vision_model}|{message_id}".encode("utf-8")
        ).hexdigest()
        source_kind = "explicit_vision"
        existing = (
            db.query(Memory)
            .filter(
                Memory.conversation_id == conversation_id,
                Memory.thread_id == thread_id,
                Memory.source_kind == source_kind,
                Memory.source_fingerprint == source_fingerprint,
            )
            .first()
        )
        if existing:
            return {"memory_id": existing.id, "replayed": True, "source_fingerprint": source_fingerprint}

        memory_repo = MemoryRepository(db)
        memory = memory_repo.create(
            content=content,
            character_id=character_id,
            memory_type=MemoryType.EXPLICIT,
            conversation_id=conversation_id,
            thread_id=thread_id,
            category=params.get("category", "visual"),
            priority=params.get("priority"),
            confidence=params.get("confidence"),
            status=params.get("status", "auto_approved"),
            source_messages=[message_id] if message_id else None,
            metadata=params.get("metadata") or {},
            source_kind=source_kind,
            source_fingerprint=source_fingerprint,
            source=params.get("source", "web"),
        )
        return {"memory_id": memory.id, "replayed": False, "source_fingerprint": source_fingerprint}

    async def _create_moment_pin(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = params["conversation_id"]
        selected_message_ids = params["selected_message_ids"]
        character_id = params["character_id"]
        model = params["model"]
        user_id = params.get("user_id") or "User"
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")

        selection_fingerprint = hashlib.sha256(
            "|".join(selected_message_ids).encode("utf-8")
        ).hexdigest()
        from chorus_engine.models.conversation import MomentPin

        existing = (
            db.query(MomentPin)
            .filter(
                MomentPin.conversation_id == conversation_id,
                MomentPin.selection_fingerprint == selection_fingerprint,
            )
            .order_by(MomentPin.created_at.desc())
            .first()
        )
        if existing:
            return {"pin_id": existing.id, "replayed": True}

        extraction = MomentPinExtractionService(db=db, llm_client=llm_client, model=model)
        snapshot_json, selected_with_margin = extraction.build_snapshot(
            conversation_id=conversation_id,
            selected_message_ids=selected_message_ids,
        )
        extraction_result = await extraction.extract_moment(snapshot_json)
        extracted = extraction_result.parsed if extraction_result else None
        if not extracted:
            raise RuntimeError("Failed to extract moment pin fields")
        what_happened = str(extracted.get("what_happened", "")).strip()
        why_model = str(extracted.get("why_it_mattered", "")).strip()
        if not what_happened or not why_model:
            raise RuntimeError("Moment extraction returned incomplete fields")
        quote_snippet = extracted.get("quote_snippet")
        if quote_snippet is not None:
            quote_snippet = str(quote_snippet).strip() or None
        tags = extracted.get("tags") if isinstance(extracted.get("tags"), list) else []
        tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        telemetry_flags = extracted.get("telemetry_flags") if isinstance(extracted.get("telemetry_flags"), dict) else {
            "contains_roleplay": False,
            "contains_directives": False,
            "contains_sensitive_content": False,
        }

        pin_repo = MomentPinRepository(db)
        pin = pin_repo.create(
            user_id=user_id,
            character_id=character_id,
            conversation_id=conversation_id,
            selected_message_ids=selected_with_margin,
            transcript_snapshot=snapshot_json,
            what_happened=what_happened,
            why_model=why_model,
            why_user=None,
            quote_snippet=quote_snippet,
            tags=tags,
            telemetry_flags=telemetry_flags,
        )
        pin.selection_fingerprint = selection_fingerprint
        pin.extractor_version = "moment_pin.v1"
        db.commit()
        db.refresh(pin)

        vector_store = self.app_state.get("moment_pin_vector_store")
        embedding_service = self.app_state.get("embedding_service")
        if vector_store and embedding_service:
            hot_text = "\n".join(
                [
                    pin.what_happened,
                    pin.why_user or pin.why_model,
                    pin.quote_snippet or "",
                    ", ".join(pin.tags or []),
                ]
            ).strip()
            embedding = embedding_service.embed(hot_text)
            if vector_store.upsert_pin(
                character_id=pin.character_id,
                pin_id=pin.id,
                hot_text=hot_text,
                embedding=embedding,
                metadata={"user_id": pin.user_id, "conversation_id": pin.conversation_id or ""},
            ):
                pin_repo.set_vector_id(pin.id, pin.id)
                pin = pin_repo.get_by_id(pin.id) or pin

        return {"pin_id": pin.id, "replayed": False}

    def _update_moment_pin(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        pin_repo = MomentPinRepository(db)
        pin = pin_repo.update_fields(
            pin_id=params["pin_id"],
            why_user=params.get("why_user"),
            tags=params.get("tags"),
            archived=params.get("archived"),
        )
        if not pin:
            raise RuntimeError("Moment pin not found")

        if params.get("why_user") is not None or params.get("tags") is not None:
            vector_store = self.app_state.get("moment_pin_vector_store")
            embedding_service = self.app_state.get("embedding_service")
            if vector_store and embedding_service:
                hot_text = "\n".join(
                    [
                        pin.what_happened,
                        pin.why_user or pin.why_model,
                        pin.quote_snippet or "",
                        ", ".join(pin.tags or []),
                    ]
                ).strip()
                embedding = embedding_service.embed(hot_text)
                vector_store.upsert_pin(
                    character_id=pin.character_id,
                    pin_id=pin.id,
                    hot_text=hot_text,
                    embedding=embedding,
                    metadata={"user_id": pin.user_id, "conversation_id": pin.conversation_id or ""},
                )

        return {"pin_id": pin.id}

    def _delete_moment_pin(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        pin_id = params["pin_id"]
        pin_repo = MomentPinRepository(db)
        pin = pin_repo.get_by_id(pin_id)
        if not pin:
            raise RuntimeError("Moment pin not found")
        vector_store = self.app_state.get("moment_pin_vector_store")
        if vector_store:
            vector_store.delete_pin(character_id=pin.character_id, pin_id=pin.id)
        pin_repo.delete(pin_id)
        return {"pin_id": pin_id, "deleted": True}

    async def _run_continuity_bootstrap(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        continuity_service = self.app_state.get("continuity_service")
        if not continuity_service:
            raise RuntimeError("Continuity service not initialized")
        character_id = params["character_id"]
        character = self.app_state["characters"].get(character_id)
        if not character:
            raise RuntimeError(f"Character not found: {character_id}")
        force = bool(params.get("force", False))
        result = await continuity_service.generate_and_save(
            character=character,
            conversation_id=params.get("conversation_id"),
            force=force,
        )
        return {
            "character_id": character_id,
            "skipped": bool((result or {}).get("skipped")),
            "has_cache": bool((result or {}).get("cache")),
        }

    async def _maybe_update_conversation_title(self, db: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENS-native auto title generation parity with legacy non-stream behavior.
        Trigger only at 4 total messages (2 turns) and only while title is auto-managed.
        """
        thread_id = params.get("thread_id")
        character_id = params.get("character_id")
        if not thread_id or not character_id:
            return {"updated": False, "reason": "missing_context"}

        title_service = self.app_state.get("title_service")
        if not title_service:
            return {"updated": False, "reason": "title_service_unavailable"}

        thread_repo = ThreadRepository(db)
        conv_repo = ConversationRepository(db)
        msg_repo = MessageRepository(db)

        thread = thread_repo.get_by_id(thread_id)
        if not thread:
            return {"updated": False, "reason": "thread_not_found"}
        conversation = conv_repo.get_by_id(thread.conversation_id)
        if not conversation:
            return {"updated": False, "reason": "conversation_not_found"}
        if not bool(getattr(conversation, "title_auto_generated", False)):
            return {"updated": False, "reason": "title_already_user_managed"}

        threads = thread_repo.list_by_conversation(conversation.id)
        total_messages = 0
        for t in threads:
            total_messages += msg_repo.count_thread_messages(t.id)
        if total_messages != 4:
            return {"updated": False, "reason": "turn_threshold_not_reached", "total_messages": total_messages}

        character = self.app_state["characters"].get(character_id)
        if not character:
            return {"updated": False, "reason": "character_not_found"}

        # Build full conversation context across threads.
        all_messages = []
        for t in threads:
            all_messages.extend(msg_repo.list_by_thread(t.id))
        all_messages.sort(key=lambda m: m.created_at)

        model = character.preferred_llm.model or self.app_state["system_config"].llm.model
        comfyui_lock = self.app_state.get("comfyui_lock")
        try:
            result = await title_service.generate_title(
                messages=all_messages,
                character_name=character.name,
                model=model,
                comfyui_lock=comfyui_lock,
            )
            if not result.success or not result.title:
                return {"updated": False, "reason": "title_generation_failed", "error": result.error}

            conv_repo.update(conversation.id, title=result.title)
            return {"updated": True, "title": result.title}
        except Exception as e:
            logger.warning("ENS title generation failed: %s", e)
            return {"updated": False, "reason": "title_generation_exception", "error": str(e)}
