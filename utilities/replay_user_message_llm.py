"""
Replay a conversation turn directly against the configured LLM (without ENS).

Primary goal:
- Recreate an assistant response as closely as possible.

Behavior:
1) Given a target message ID, detect role (assistant/user) unless overridden.
2) For assistant targets, prefer exact replay payload from conversation debug capture
   (`prompt_capture.messages_for_llm`) when available.
3) Fall back to prompt reconstruction via PromptAssemblyService when debug capture
   is unavailable.
4) Invoke LLM directly (optional) and write a unique JSON debug bundle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chorus_engine.config.loader import ConfigLoader
from chorus_engine.db.database import SessionLocal, init_db
from chorus_engine.llm.client import create_llm_client
from chorus_engine.models.conversation import MessageRole
from chorus_engine.repositories.conversation_repository import ConversationRepository
from chorus_engine.repositories.message_repository import MessageRepository
from chorus_engine.repositories.thread_repository import ThreadRepository
from chorus_engine.services.prompt_assembly import PromptAssemblyService
from chorus_engine.services.structured_response import ALLOWED_CHANNELS_ALL


def _coerce_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def _build_output_path(output_dir: Path, message_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = str(uuid.uuid4())[:8]
    return output_dir / f"llm_replay_{message_id}_{stamp}_{suffix}.json"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except Exception:
            continue
    return rows


def _safe_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _load_conversation_debug_events(conversation_id: str) -> List[Dict[str, Any]]:
    conv_dir = Path("data/debug_logs/conversations") / conversation_id
    if not conv_dir.exists():
        return []
    files = sorted(conv_dir.glob("ens_conversation_*.jsonl"))
    events: List[Dict[str, Any]] = []
    for f in files:
        events.extend(_read_jsonl(f))
    return events


def _load_loop_action_result_index() -> Dict[str, Dict[str, Any]]:
    path = Path("data/debug_logs/ens/action_results.jsonl")
    index: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("kind") != "loop.progression.step":
            continue
        output = row.get("output") or {}
        assistant_message_id = output.get("assistant_message_id")
        if assistant_message_id:
            index[str(assistant_message_id)] = row
    return index


def _find_exact_event_for_assistant(
    *,
    assistant_message_id: str,
    assistant_content: str,
    assistant_created_at: datetime,
    events: List[Dict[str, Any]],
    loop_action_index: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Return best-matching debug event containing prompt_capture.messages_for_llm.
    """
    debug: Dict[str, Any] = {"strategy": None}
    loop_action = loop_action_index.get(assistant_message_id)

    # Strategy 1: loop action result timestamp -> nearest ens_loop_step_turn
    if loop_action:
        debug["strategy"] = "loop_action_timestamp_match"
        ts = _safe_dt(loop_action.get("timestamp"))
        candidates = [e for e in events if e.get("type") == "ens_loop_step_turn"]
        if ts and candidates:
            candidates = sorted(
                candidates,
                key=lambda e: abs((_safe_dt(e.get("timestamp")) or ts) - ts),
            )
            best = candidates[0]
            debug["loop_action_timestamp"] = loop_action.get("timestamp")
            debug["matched_event_timestamp"] = best.get("timestamp")
            return best, debug

    # Strategy 2: direct content match for llm/loop event
    debug["strategy"] = "display_content_match"
    candidates = [
        e for e in events if e.get("type") in ("ens_llm_turn", "ens_loop_step_turn")
    ]
    for e in candidates:
        if (e.get("display_content") or "") == assistant_content:
            debug["matched_event_timestamp"] = e.get("timestamp")
            debug["matched_event_type"] = e.get("type")
            return e, debug

    # Strategy 3: nearest event by timestamp and type
    window = timedelta(seconds=20)
    nearest: Optional[Dict[str, Any]] = None
    nearest_delta: Optional[float] = None
    for e in candidates:
        evt_dt = _safe_dt(e.get("timestamp"))
        if not evt_dt:
            continue
        delta = abs((evt_dt - assistant_created_at).total_seconds())
        if delta <= window.total_seconds() and (nearest_delta is None or delta < nearest_delta):
            nearest = e
            nearest_delta = delta
    if nearest is not None:
        debug["strategy"] = "nearest_timestamp"
        debug["matched_event_timestamp"] = nearest.get("timestamp")
        debug["matched_event_type"] = nearest.get("type")
        debug["timestamp_delta_seconds"] = nearest_delta
        return nearest, debug

    debug["strategy"] = "no_match"
    return None, debug


def _slice_history_up_to_message(messages: List[Any], message_id: str) -> List[Any]:
    sliced: List[Any] = []
    for msg in messages:
        sliced.append(msg)
        if str(msg.id) == str(message_id):
            break
    return sliced


def _find_previous_user_message_id(messages: List[Any], assistant_message_id: str) -> Optional[str]:
    prev_user: Optional[str] = None
    for msg in messages:
        if str(msg.id) == str(assistant_message_id):
            return prev_user
        if str(msg.role) == str(MessageRole.USER):
            prev_user = str(msg.id)
    return prev_user


def _determine_cutoff_message_id(
    *,
    target_role: str,
    target_message_id: str,
    full_history: List[Any],
) -> Optional[str]:
    if target_role == "assistant":
        cutoff_message_id = _find_previous_user_message_id(full_history, target_message_id)
        if cutoff_message_id is None:
            # Loop-step style fallback: use last message before target.
            for msg in full_history:
                if str(msg.id) == str(target_message_id):
                    break
                cutoff_message_id = str(msg.id)
        return cutoff_message_id
    return target_message_id


def _analyze_payload(text: str) -> Dict[str, Any]:
    begin = "---CHORUS_TOOL_PAYLOAD_BEGIN---"
    end = "---CHORUS_TOOL_PAYLOAD_END---"
    result: Dict[str, Any] = {
        "has_sentinel_begin": begin in text,
        "has_sentinel_end": end in text,
        "has_sentinel_payload": False,
        "sentinel_payload_raw": None,
        "sentinel_payload_parsed": None,
        "sentinel_payload_parse_error": None,
        "likely_payload_outside_sentinel": False,
        "likely_payload_snippets": [],
        "multiple_payload_markers": False,
    }

    begin_count = text.count(begin)
    end_count = text.count(end)
    result["multiple_payload_markers"] = begin_count > 1 or end_count > 1

    if begin in text and end in text:
        start = text.find(begin) + len(begin)
        finish = text.find(end, start)
        if finish != -1:
            payload_text = text[start:finish].strip()
            result["has_sentinel_payload"] = True
            result["sentinel_payload_raw"] = payload_text
            try:
                parsed = json.loads(payload_text)
                result["sentinel_payload_parsed"] = parsed
            except Exception as exc:
                result["sentinel_payload_parse_error"] = str(exc)

    if not result["has_sentinel_payload"]:
        snippets: List[str] = []
        patterns = [
            r'(?is)\{\s*"version"\s*:\s*1.*?\}',
            r'(?is)\{\s*"control"\s*:\s*\{.*?\}\s*\}',
            r'(?is)---CHORUS.*?payload.*?---',
            r'(?is)\bcontrol\b\s*:\s*"(?:CONTINUE|YIELD|COMPLETE|WAIT_FOR_USER)"',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                snippet = m.group(0).strip().replace("\n", " ")
                if len(snippet) > 280:
                    snippet = snippet[:280] + "..."
                snippets.append(snippet)
        if snippets:
            result["likely_payload_outside_sentinel"] = True
            result["likely_payload_snippets"] = snippets

    return result


def _analyze_display_format(text: str) -> Dict[str, Any]:
    begin = "---CHORUS_TOOL_PAYLOAD_BEGIN---"
    end = "---CHORUS_TOOL_PAYLOAD_END---"

    def _remove_sentinel_blocks(source: str) -> str:
        cleaned = source
        while True:
            start = cleaned.find(begin)
            if start == -1:
                break
            finish = cleaned.find(end, start + len(begin))
            if finish == -1:
                break
            cleaned = cleaned[:start] + cleaned[finish + len(end):]
        return cleaned

    def _snippet(value: str, limit: int = 200) -> str:
        compact = re.sub(r"\s+", " ", value).strip()
        if len(compact) > limit:
            return compact[:limit] + "..."
        return compact

    def _visible_text(value: str) -> str:
        return re.sub(r"<[^>]+>", "", value).strip()

    result: Dict[str, Any] = {
        "assistant_response_block_count": 0,
        "has_single_assistant_response_block": False,
        "has_display_text_outside_assistant_response": False,
        "display_text_outside_assistant_response_snippets": [],
        "has_untagged_display_text_inside_assistant_response": False,
        "untagged_inside_assistant_response_snippets": [],
        "has_unknown_tags_inside_assistant_response": False,
        "unknown_tags_inside_assistant_response": [],
        "allowed_template_tags": sorted(ALLOWED_CHANNELS_ALL),
    }

    cleaned = _remove_sentinel_blocks(text or "")
    root_pattern = re.compile(r"<assistant_response>([\s\S]*?)</assistant_response>")
    roots = list(root_pattern.finditer(cleaned))
    result["assistant_response_block_count"] = len(roots)
    result["has_single_assistant_response_block"] = len(roots) == 1

    outside = root_pattern.sub("", cleaned)
    if _visible_text(outside):
        result["has_display_text_outside_assistant_response"] = True
        result["display_text_outside_assistant_response_snippets"].append(_snippet(_visible_text(outside)))

    tag_pattern = re.compile(r"<([a-z][a-z0-9_]*)>([\s\S]*?)</\1>")
    unknown_tags: set[str] = set()
    for root in roots:
        body = root.group(1)
        cursor = 0
        for tag_match in tag_pattern.finditer(body):
            start, end = tag_match.span()
            raw_prefix = body[cursor:start]
            if _visible_text(raw_prefix):
                result["has_untagged_display_text_inside_assistant_response"] = True
                result["untagged_inside_assistant_response_snippets"].append(_snippet(_visible_text(raw_prefix)))

            channel = tag_match.group(1)
            if channel not in ALLOWED_CHANNELS_ALL:
                unknown_tags.add(channel)
                inner_visible = _visible_text(tag_match.group(2))
                if inner_visible:
                    result["has_untagged_display_text_inside_assistant_response"] = True
                    result["untagged_inside_assistant_response_snippets"].append(_snippet(inner_visible))

            cursor = end

        tail = body[cursor:]
        if _visible_text(tail):
            result["has_untagged_display_text_inside_assistant_response"] = True
            result["untagged_inside_assistant_response_snippets"].append(_snippet(_visible_text(tail)))

    if unknown_tags:
        result["has_unknown_tags_inside_assistant_response"] = True
        result["unknown_tags_inside_assistant_response"] = sorted(unknown_tags)

    # Keep snippets unique and bounded for readability.
    if result["display_text_outside_assistant_response_snippets"]:
        result["display_text_outside_assistant_response_snippets"] = list(
            dict.fromkeys(result["display_text_outside_assistant_response_snippets"])
        )[:5]
    if result["untagged_inside_assistant_response_snippets"]:
        result["untagged_inside_assistant_response_snippets"] = list(
            dict.fromkeys(result["untagged_inside_assistant_response_snippets"])
        )[:10]

    return result


async def _run(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    output: Dict[str, Any] = {
        "script": "utilities/replay_user_message_llm.py",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "input": {
            "target_message_id": args.message_id,
            "target_role": args.target_role,
            "force_reconstruct": args.force_reconstruct,
            "rebuild_system_prompt": args.rebuild_system_prompt,
            "no_invoke": args.no_invoke,
            "no_memories": args.no_memories,
            "max_history_messages": args.max_history_messages,
            "model_override": args.model,
            "temperature_override": args.temperature,
            "max_tokens_override": args.max_tokens,
            "primary_user_override": args.primary_user,
            "conversation_source_override": args.conversation_source,
            "user_id_override": args.user_id,
        },
        "status": "started",
    }

    try:
        message_repo = MessageRepository(db)
        thread_repo = ThreadRepository(db)
        conversation_repo = ConversationRepository(db)
        target_message = message_repo.get_by_id(args.message_id)
        if target_message is None:
            output["status"] = "error"
            output["error"] = f"Message not found: {args.message_id}"
            return 1

        thread = thread_repo.get_by_id(str(target_message.thread_id))
        if thread is None:
            output["status"] = "error"
            output["error"] = f"Thread not found for message: {target_message.thread_id}"
            return 1

        conversation = conversation_repo.get_by_id(str(thread.conversation_id))
        if conversation is None:
            output["status"] = "error"
            output["error"] = f"Conversation not found for thread: {thread.conversation_id}"
            return 1

        config_loader = ConfigLoader()
        system_config = config_loader.load_system_config()
        character = config_loader.load_character(str(conversation.character_id))

        target_role = args.target_role
        if target_role == "auto":
            if str(target_message.role) == str(MessageRole.ASSISTANT):
                target_role = "assistant"
            elif str(target_message.role) == str(MessageRole.USER):
                target_role = "user"
            else:
                target_role = "assistant"

        metadata = _coerce_metadata(target_message.meta_data)
        primary_user = args.primary_user if args.primary_user is not None else (conversation.primary_user or None)
        conversation_source = (
            args.conversation_source
            if args.conversation_source is not None
            else (conversation.source or metadata.get("platform") or "web")
        )
        user_id = args.user_id if args.user_id is not None else metadata.get("user_id")

        full_history = message_repo.list_by_thread(
            thread_id=str(thread.id),
            skip=0,
            limit=args.max_history_messages,
        )

        messages_for_llm: Optional[List[Dict[str, str]]] = None
        replay_source: Dict[str, Any] = {"mode": None}
        loop_action_index = _load_loop_action_result_index()
        loop_action = loop_action_index.get(str(target_message.id))
        matched_event_for_output: Optional[Dict[str, Any]] = None

        # Preferred path: exact debug-captured payload for assistant response replay.
        if target_role == "assistant" and not args.force_reconstruct:
            events = _load_conversation_debug_events(str(conversation.id))
            matched_event, match_debug = _find_exact_event_for_assistant(
                assistant_message_id=str(target_message.id),
                assistant_content=str(target_message.content or ""),
                assistant_created_at=target_message.created_at,
                events=events,
                loop_action_index=loop_action_index,
            )
            replay_source["debug_match"] = match_debug
            if matched_event:
                matched_event_for_output = matched_event
                prompt_capture = matched_event.get("prompt_capture") or {}
                payload = prompt_capture.get("messages_for_llm")
                if isinstance(payload, list) and payload:
                    messages_for_llm = payload
                    replay_source["mode"] = "debug_capture_exact"
                    replay_source["event_type"] = matched_event.get("type")
                    replay_source["event_timestamp"] = matched_event.get("timestamp")
                    replay_source["event_control_action"] = matched_event.get("control_action")

        # Reconstruction fallback
        if messages_for_llm is None:
            replay_source["mode"] = "reconstructed"
            assembler = PromptAssemblyService(
                db=db,
                character_id=str(conversation.character_id),
                model_name=str(system_config.llm.model),
                context_window=int(character.preferred_llm.context_window or system_config.llm.context_window),
                startup_monotonic=None,
            )
            original_list_by_thread = assembler.message_repository.list_by_thread

            # Determine cut-off message for reconstruction
            cutoff_message_id = _determine_cutoff_message_id(
                target_role=target_role,
                target_message_id=str(target_message.id),
                full_history=full_history,
            )

            if not cutoff_message_id:
                output["status"] = "error"
                output["error"] = "Unable to determine cut-off message for reconstruction."
                return 1

            def _list_by_thread_cutoff(thread_id: str, skip: int = 0, limit: int = 1000):
                messages = original_list_by_thread(thread_id=thread_id, skip=skip, limit=limit)
                return _slice_history_up_to_message(messages, cutoff_message_id)

            assembler.message_repository.list_by_thread = _list_by_thread_cutoff  # type: ignore[assignment]

            loop_step = bool(loop_action)
            loop_kind = None
            if loop_action:
                loop_kind = (loop_action.get("output") or {}).get("loop_kind")
                replay_source["loop_step_from_action_results"] = {
                    "timestamp": loop_action.get("timestamp"),
                    "loop_id": (loop_action.get("output") or {}).get("loop_id"),
                    "loop_kind": loop_kind,
                    "assistant_message_id": (loop_action.get("output") or {}).get("assistant_message_id"),
                }

            include_conversation_context = True
            try:
                components = assembler.assemble_prompt(
                    thread_id=str(thread.id),
                    include_memories=not args.no_memories,
                    max_history_messages=args.max_history_messages,
                    primary_user=primary_user,
                    conversation_source=conversation_source,
                    conversation_kind=conversation.conversation_kind,
                    conversation_id=conversation.id,
                    user_id=user_id,
                    include_conversation_context=include_conversation_context,
                    loop_step=loop_step,
                    loop_kind=loop_kind,
                )
            except Exception as first_error:
                # Offline-safe fallback when retrieval backends are unavailable.
                components = assembler.assemble_prompt(
                    thread_id=str(thread.id),
                    include_memories=False,
                    max_history_messages=args.max_history_messages,
                    primary_user=primary_user,
                    conversation_source=conversation_source,
                    conversation_kind=conversation.conversation_kind,
                    conversation_id=conversation.id,
                    user_id=user_id,
                    include_conversation_context=False,
                    loop_step=loop_step,
                    loop_kind=loop_kind,
                )
                output.setdefault("warnings", []).append(
                    f"Initial assembly failed; retried without retrieval context: {first_error}"
                )
                replay_source["fallback_without_retrieval"] = True

            messages_for_llm = assembler.format_for_api(components)
            replay_source["reconstructed_cutoff_message_id"] = cutoff_message_id
            replay_source["token_breakdown"] = components.token_breakdown

            # Loop-step behavior: progression uses synthetic user "continue".
            if target_role == "assistant" and loop_step:
                messages_for_llm = list(messages_for_llm)
                messages_for_llm.append({"role": "user", "content": "continue"})
                replay_source["appended_synthetic_continue"] = True

        if not messages_for_llm:
            output["status"] = "error"
            output["error"] = "Unable to build replay payload."
            return 1

        # Optional mode: regenerate only the system prompt using current prompt builder,
        # while preserving the rest of the replay transcript messages.
        if args.rebuild_system_prompt:
            rebuild_assembler = PromptAssemblyService(
                db=db,
                character_id=str(conversation.character_id),
                model_name=str(system_config.llm.model),
                context_window=int(character.preferred_llm.context_window or system_config.llm.context_window),
                startup_monotonic=None,
            )
            original_list_by_thread = rebuild_assembler.message_repository.list_by_thread
            cutoff_message_id = _determine_cutoff_message_id(
                target_role=target_role,
                target_message_id=str(target_message.id),
                full_history=full_history,
            )
            if not cutoff_message_id:
                output.setdefault("warnings", []).append(
                    "System-prompt rebuild requested, but cutoff could not be determined."
                )
            else:
                def _list_by_thread_cutoff_rebuild(thread_id: str, skip: int = 0, limit: int = 1000):
                    messages = original_list_by_thread(thread_id=thread_id, skip=skip, limit=limit)
                    return _slice_history_up_to_message(messages, cutoff_message_id)

                rebuild_assembler.message_repository.list_by_thread = _list_by_thread_cutoff_rebuild  # type: ignore[assignment]

                loop_step = bool(loop_action)
                loop_kind = (loop_action.get("output") or {}).get("loop_kind") if loop_action else None
                try:
                    rebuilt_components = rebuild_assembler.assemble_prompt(
                        thread_id=str(thread.id),
                        include_memories=not args.no_memories,
                        max_history_messages=args.max_history_messages,
                        primary_user=primary_user,
                        conversation_source=conversation_source,
                        conversation_kind=conversation.conversation_kind,
                        conversation_id=conversation.id,
                        user_id=user_id,
                        include_conversation_context=True,
                        loop_step=loop_step,
                        loop_kind=loop_kind,
                    )
                except Exception as first_error:
                    rebuilt_components = rebuild_assembler.assemble_prompt(
                        thread_id=str(thread.id),
                        include_memories=False,
                        max_history_messages=args.max_history_messages,
                        primary_user=primary_user,
                        conversation_source=conversation_source,
                        conversation_kind=conversation.conversation_kind,
                        conversation_id=conversation.id,
                        user_id=user_id,
                        include_conversation_context=False,
                        loop_step=loop_step,
                        loop_kind=loop_kind,
                    )
                    output.setdefault("warnings", []).append(
                        f"System-prompt rebuild retried without retrieval context: {first_error}"
                    )

                rebuilt_messages = rebuild_assembler.format_for_api(rebuilt_components)
                rebuilt_system_prompt = ""
                if rebuilt_messages and rebuilt_messages[0].get("role") == "system":
                    rebuilt_system_prompt = rebuilt_messages[0].get("content") or ""

                original_system_prompt = ""
                if messages_for_llm and messages_for_llm[0].get("role") == "system":
                    original_system_prompt = messages_for_llm[0].get("content") or ""

                if rebuilt_system_prompt and messages_for_llm and messages_for_llm[0].get("role") == "system":
                    messages_for_llm = list(messages_for_llm)
                    messages_for_llm[0] = {"role": "system", "content": rebuilt_system_prompt}
                    replay_source["system_prompt_rebuilt"] = True
                    replay_source["system_prompt_rebuild_cutoff_message_id"] = cutoff_message_id
                    output["system_prompt_comparison"] = {
                        "original_length": len(original_system_prompt),
                        "rebuilt_length": len(rebuilt_system_prompt),
                        "changed": original_system_prompt != rebuilt_system_prompt,
                        "original_system_prompt": original_system_prompt,
                        "rebuilt_system_prompt": rebuilt_system_prompt,
                    }
                else:
                    output.setdefault("warnings", []).append(
                        "System-prompt rebuild requested, but no system message was available for replacement."
                    )

        model = args.model or character.preferred_llm.model or system_config.llm.model
        temperature = (
            args.temperature
            if args.temperature is not None
            else (
                character.preferred_llm.temperature
                if character.preferred_llm.temperature is not None
                else system_config.llm.temperature
            )
        )
        max_tokens = (
            args.max_tokens
            if args.max_tokens is not None
            else (
                character.preferred_llm.max_tokens
                if character.preferred_llm.max_tokens is not None
                else system_config.llm.max_response_tokens
            )
        )

        output["resolved"] = {
            "character_id": str(conversation.character_id),
            "conversation_id": str(conversation.id),
            "thread_id": str(thread.id),
            "target_message_id": str(target_message.id),
            "target_message_role_db": str(target_message.role),
            "target_role_used": target_role,
            "target_message_created_at": target_message.created_at,
            "target_message_content": target_message.content,
            "conversation_source": conversation_source,
            "primary_user": primary_user,
            "user_id": user_id,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": str(system_config.llm.provider),
            "base_url": str(system_config.llm.base_url),
        }
        output["replay_source"] = replay_source
        output["messages_for_llm"] = messages_for_llm

        # Analyze payload presence/quality on original target output when raw event exists.
        if matched_event_for_output is not None:
            original_raw = (
                matched_event_for_output.get("raw_content")
                or matched_event_for_output.get("display_content")
                or ""
            )
            output["original_target_payload_analysis"] = _analyze_payload(str(original_raw))
            output["original_target_payload_analysis"]["source_event_type"] = matched_event_for_output.get("type")
            output["original_target_payload_analysis"]["source_event_timestamp"] = matched_event_for_output.get("timestamp")
            output["original_target_display_analysis"] = _analyze_display_format(str(original_raw))
            output["original_target_display_analysis"]["source_event_type"] = matched_event_for_output.get("type")
            output["original_target_display_analysis"]["source_event_timestamp"] = matched_event_for_output.get("timestamp")

        llm_result: Dict[str, Any] = {"invoked": False}
        if not args.no_invoke:
            llm_client = create_llm_client(system_config.llm)
            started = datetime.utcnow()
            health_ok = await llm_client.health_check()
            llm_result["health_check_ok"] = bool(health_ok)
            response = await llm_client.generate_with_history(
                messages=messages_for_llm,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )
            finished = datetime.utcnow()
            llm_result.update(
                {
                    "invoked": True,
                    "started_at": started,
                    "finished_at": finished,
                    "duration_ms": int((finished - started).total_seconds() * 1000),
                    "response": {
                        "content": response.content,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                    },
                    "payload_analysis": _analyze_payload(response.content or ""),
                    "display_analysis": _analyze_display_format(response.content or ""),
                }
            )
        else:
            llm_result["note"] = "Invocation skipped (--no-invoke)."

        output["llm"] = llm_result
        output["status"] = "ok"
        return 0
    except Exception as exc:
        output["status"] = "error"
        output["error"] = str(exc)
        output["traceback"] = traceback.format_exc()
        return 1
    finally:
        out_path = _build_output_path(Path(args.output_dir), args.message_id)
        out_path.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")
        print(f"Wrote debug bundle: {out_path}")
        db.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a target message turn by rebuilding (or reusing exact captured) messages_for_llm "
            "and optionally invoking the configured LLM directly."
        )
    )
    parser.add_argument("message_id", help="Target message ID (assistant preferred; user supported)")
    parser.add_argument(
        "--target-role",
        choices=["auto", "assistant", "user"],
        default="auto",
        help="How to interpret target message id (default: auto from DB role)",
    )
    parser.add_argument(
        "--force-reconstruct",
        action="store_true",
        help="Ignore debug-captured payloads and always rebuild via prompt assembly",
    )
    parser.add_argument(
        "--rebuild-system-prompt",
        action="store_true",
        help=(
            "Rebuild only the system prompt with current builder/code and replace message[0] "
            "while preserving the rest of replay transcript messages."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/debug/llm_replay_outputs",
        help="Directory for debug output JSON files",
    )
    parser.add_argument("--no-invoke", action="store_true", help="Build payload only; skip LLM call")
    parser.add_argument("--no-memories", action="store_true", help="Disable memory retrieval in assembly")
    parser.add_argument(
        "--max-history-messages",
        type=int,
        default=10000,
        help="History fetch cap before cut-off is applied",
    )
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--primary-user", default=None, help="Override primary user")
    parser.add_argument("--conversation-source", default=None, help="Override conversation source")
    parser.add_argument("--user-id", default=None, help="Override user id scope")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
