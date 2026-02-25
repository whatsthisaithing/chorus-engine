"""Unified ENS-owned LLM invocation service for chat and analysis calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chorus_engine.ens.assistant_result import normalize_assistant_result
from chorus_engine.llm.request_debug_context import (
    reset_request_debug_context,
    set_request_debug_context,
)
from chorus_engine.ens.tool_registry import (
    TOOL_CHORUS_CONTROL,
    TOOL_IMAGE_GENERATE,
    native_tool_definitions,
    requires_approval_default,
)

NON_DETERMINISTIC_METADATA_KEYS = {
    "trace_id",
    "signal_id",
    "decision_id",
    "action_id",
    "timestamp",
    "created_at",
    "updated_at",
}

_IN_INVOKER_CONTEXT: ContextVar[bool] = ContextVar("ens_in_invoker_context", default=False)
logger = logging.getLogger(__name__)


def in_invoker_context() -> bool:
    return bool(_IN_INVOKER_CONTEXT.get())


@dataclass
class EffectiveLLMConfig:
    """Resolved model/sampling configuration with system/character precedence."""

    provider: str
    engine: str
    model_id: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    context_window: Optional[int]
    top_p: Optional[float]
    top_k: Optional[int]
    repeat_penalty: Optional[float]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]


@dataclass
class InvocationRequest:
    invocation_kind: str
    idempotency_key: str
    model_id: str
    provider: str = "local"
    engine: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None
    surface_id: Optional[str] = None
    character_id: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    vision_images: Optional[List[str]] = None
    vision_image_mime_type: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    native_tool_policy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMInvocationService:
    """Provider-agnostic wrapper around the existing llm_client abstraction."""

    def __init__(self, app_state: Dict[str, Any]) -> None:
        self.app_state = app_state

    @staticmethod
    def _engine_from_client(llm_client: Any) -> str:
        name = llm_client.__class__.__name__.lower()
        if "ollama" in name:
            return "ollama"
        if "lmstudio" in name:
            return "lmstudio"
        if "kobold" in name:
            return "koboldcpp"
        return "unknown"

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        def _clean(value: Any) -> Any:
            if isinstance(value, dict):
                cleaned: Dict[str, Any] = {}
                for k, v in value.items():
                    key = str(k)
                    if key in NON_DETERMINISTIC_METADATA_KEYS:
                        continue
                    cleaned[key] = _clean(v)
                return cleaned
            if isinstance(value, list):
                return [_clean(item) for item in value]
            return value

        return _clean(metadata or {})

    @staticmethod
    def _provider_native_tool_capabilities(engine: Optional[str]) -> Dict[str, bool]:
        normalized = str(engine or "").strip().lower()
        if normalized in {"ollama", "lmstudio"}:
            return {
                "supports_native_tools": True,
                "supports_response_format_json_schema": True,
                "supports_sentinel_retry": True,
                "supports_tool_choice": True,
            }
        if normalized == "koboldcpp":
            return {
                "supports_native_tools": False,
                "supports_response_format_json_schema": False,
                "supports_sentinel_retry": True,
                "supports_tool_choice": False,
            }
        return {
            "supports_native_tools": False,
            "supports_response_format_json_schema": False,
            "supports_sentinel_retry": True,
            "supports_tool_choice": False,
        }

    def resolve_provider_capabilities(self, *, engine: Optional[str]) -> Dict[str, bool]:
        fallback = self._provider_native_tool_capabilities(engine)
        llm_cfg = getattr(self.app_state.get("system_config"), "llm", None)
        cap_map = getattr(llm_cfg, "provider_capabilities", None) if llm_cfg else None
        key = str(engine or "").strip().lower()
        if isinstance(cap_map, dict) and key in cap_map:
            entry = cap_map.get(key)
            if entry is not None:
                native = bool(getattr(entry, "supports_native_tools", fallback["supports_native_tools"]))
                schema = bool(
                    getattr(
                        entry,
                        "supports_response_format_json_schema",
                        fallback["supports_response_format_json_schema"],
                    )
                )
                sentinel = bool(getattr(entry, "supports_sentinel_retry", fallback["supports_sentinel_retry"]))
                return {
                    "supports_native_tools": native,
                    "supports_response_format_json_schema": schema,
                    "supports_sentinel_retry": sentinel,
                    "supports_tool_choice": native,
                }
        return fallback

    def _native_tool_transport_enabled(self, *, request: InvocationRequest) -> bool:
        ens_cfg = getattr(self.app_state.get("system_config"), "ens", None)
        if not ens_cfg:
            return False
        if not bool(getattr(ens_cfg, "enabled", False)):
            return False
        if not bool(getattr(ens_cfg, "native_tool_transport_enabled", False)):
            return False
        if bool(getattr(ens_cfg, "native_tool_transport_force_sentinel", False)):
            return False
        if request.invocation_kind != "chat":
            return False
        caps = self.resolve_provider_capabilities(engine=request.engine)
        if not bool(caps.get("supports_native_tools")):
            return False
        llm_client = self.app_state.get("llm_client")
        if str(request.engine or "").strip().lower() == "ollama" and bool(getattr(llm_client, "use_legacy_chat_api", False)):
            return False
        return True

    def native_tool_transport_mode(self, *, engine: Optional[str], invocation_kind: str = "chat") -> str:
        probe = InvocationRequest(
            invocation_kind=invocation_kind,
            idempotency_key="transport_mode_probe",
            model_id="transport_mode_probe",
            provider="local",
            engine=str(engine or "unknown"),
        )
        return "native" if self._native_tool_transport_enabled(request=probe) else "sentinel"

    def _sentinel_fallback_enabled(self) -> bool:
        ens_cfg = getattr(self.app_state.get("system_config"), "ens", None)
        if not ens_cfg:
            return True
        new_flag = bool(getattr(ens_cfg, "native_tool_transport_sentinel_fallback_enabled", True))
        legacy_flag = bool(getattr(ens_cfg, "v3_sentinel_fallback_enabled", False))
        return bool(new_flag and legacy_flag)

    def _native_tool_transport_debug_mode(self) -> str:
        ens_cfg = getattr(self.app_state.get("system_config"), "ens", None)
        if not ens_cfg:
            return "off"
        mode = str(getattr(ens_cfg, "native_tool_transport_debug_override_mode", "off") or "off").strip().lower()
        if mode not in {"off", "chat_control_only", "loop_image_only", "both"}:
            return "off"
        return mode

    @staticmethod
    def _inject_debug_instruction(request: InvocationRequest, instruction: str) -> None:
        if not instruction:
            return

        if isinstance(request.messages, list):
            updated_messages: List[Dict[str, Any]] = []
            injected = False
            for item in request.messages:
                if not isinstance(item, dict):
                    updated_messages.append(item)
                    continue
                copied = dict(item)
                role = str(copied.get("role") or "").strip().lower()
                if not injected and role == "system":
                    content = str(copied.get("content") or "")
                    copied["content"] = f"{content}\n\n{instruction}" if content else instruction
                    injected = True
                updated_messages.append(copied)
            if not injected:
                updated_messages.insert(0, {"role": "system", "content": instruction})
            request.messages = updated_messages
            return

        system_prompt = str(request.system_prompt or "")
        request.system_prompt = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction

    def _apply_native_transport_debug_overrides(
        self,
        *,
        request: InvocationRequest,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        native_plan: Dict[str, Any],
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[Any], Dict[str, Any]]:
        mode = self._native_tool_transport_debug_mode()
        if mode == "off":
            return tools, tool_choice, native_plan

        metadata = request.metadata or {}
        loop_step = bool(metadata.get("loop_id"))
        loop_kind = str(metadata.get("loop_kind") or "").strip().lower()

        if mode in {"chat_control_only", "both"} and not loop_step:
            forced_tools = native_tool_definitions(
                allowed_media_tools=set(),
                include_control=True,
                include_cold_recall=False,
            )
            forced_choice = {"type": "function", "function": {"name": TOOL_CHORUS_CONTROL}}
            self._inject_debug_instruction(
                request,
                "TEST OVERRIDE: Call `chorus.control` exactly once with {\"action\":\"CONTINUE\"}.",
            )
            debug_plan = dict(native_plan or {})
            debug_plan.update(
                {
                    "debug_override_applied": True,
                    "debug_override_mode": mode,
                    "debug_override_policy": "chat_control_only",
                    "loop_kind": loop_kind or None,
                }
            )
            return forced_tools, forced_choice, debug_plan

        if mode in {"loop_image_only", "both"} and loop_step:
            forced_tools = native_tool_definitions(
                allowed_media_tools={TOOL_IMAGE_GENERATE},
                include_control=False,
                include_cold_recall=False,
            )
            forced_choice = {"type": "function", "function": {"name": TOOL_IMAGE_GENERATE}}
            self._inject_debug_instruction(
                request,
                "TEST OVERRIDE: Call `image.generate` exactly once with a simple, safe prompt.",
            )
            debug_plan = dict(native_plan or {})
            debug_plan.update(
                {
                    "debug_override_applied": True,
                    "debug_override_mode": mode,
                    "debug_override_policy": "loop_image_only",
                    "loop_kind": loop_kind or None,
                }
            )
            return forced_tools, forced_choice, debug_plan

        return tools, tool_choice, native_plan

    def _prepare_native_transport(self, request: InvocationRequest) -> tuple[Optional[List[Dict[str, Any]]], Optional[Any], Dict[str, Any]]:
        if not self._native_tool_transport_enabled(request=request):
            return None, None, {"attempted": False, "enabled": False}
        explicit_policy = request.native_tool_policy if isinstance(request.native_tool_policy, dict) else None
        if explicit_policy is not None:
            return self._prepare_native_transport_from_policy(request=request, policy=explicit_policy)
        metadata = request.metadata or {}
        media_gate = (metadata.get("media_gate_snapshot") or {})
        allowed_tools = set(media_gate.get("allowed_tools_final") or [])
        include_control = bool(metadata.get("loop_id") or metadata.get("loop_kind"))
        loop_kind = str(metadata.get("loop_kind") or "").strip().lower()
        loop_stage = str(metadata.get("loop_stage") or "").strip().lower()
        is_narrative_v1_loop_step = bool(metadata.get("loop_id")) and loop_kind == "narrative.v1"

        if is_narrative_v1_loop_step and loop_stage != "beat":
            tools = native_tool_definitions(
                allowed_media_tools=set(),
                include_control=True,
                include_cold_recall=False,
            )
            if not tools:
                return None, None, {"attempted": False, "enabled": True, "reason": "no_tools_available"}
            use_auto_choice = True
            prepared = (
                tools,
                ("auto" if use_auto_choice else {"type": "function", "function": {"name": TOOL_CHORUS_CONTROL}}),
                {
                "attempted": True,
                "enabled": True,
                "include_control": True,
                "tool_count": len(tools),
                "loop_policy": ("narrative_v1_outcome_control_only_auto_choice" if use_auto_choice else "narrative_v1_control_only_required"),
                "loop_stage": loop_stage or "full",
                },
            )
            return self._apply_native_transport_debug_overrides(
                request=request,
                tools=prepared[0],
                tool_choice=prepared[1],
                native_plan=prepared[2],
            )

        if is_narrative_v1_loop_step and loop_stage == "beat":
            include_control = False

        tools = native_tool_definitions(
            allowed_media_tools=allowed_tools,
            include_control=include_control,
            include_cold_recall=(not (is_narrative_v1_loop_step and loop_stage == "beat")),
        )
        if not tools:
            return None, None, {"attempted": False, "enabled": True, "reason": "no_tools_available"}
        tool_choice: Optional[Any] = "auto"
        return self._apply_native_transport_debug_overrides(
            request=request,
            tools=tools,
            tool_choice=tool_choice,
            native_plan={"attempted": True, "enabled": True, "include_control": include_control, "tool_count": len(tools)},
        )

    def _prepare_native_transport_from_policy(
        self,
        *,
        request: InvocationRequest,
        policy: Dict[str, Any],
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[Any], Dict[str, Any]]:
        tools = policy.get("tools")
        if not isinstance(tools, list):
            allowed_media_tools = set(policy.get("allowed_media_tools") or [])
            include_control = bool(policy.get("include_control", False))
            include_cold_recall = bool(policy.get("include_cold_recall", False))
            tools = native_tool_definitions(
                allowed_media_tools=allowed_media_tools,
                include_control=include_control,
                include_cold_recall=include_cold_recall,
            )
        if not tools:
            return None, None, {"attempted": False, "enabled": True, "reason": "no_tools_available", "policy": "explicit"}
        tool_choice = policy.get("tool_choice", "auto")
        native_plan = {
            "attempted": True,
            "enabled": True,
            "tool_count": len(tools),
            "policy": str(policy.get("policy_id") or "explicit"),
            "explicit_policy": True,
        }
        return self._apply_native_transport_debug_overrides(
            request=request,
            tools=tools,
            tool_choice=tool_choice,
            native_plan=native_plan,
        )

    @staticmethod
    def _safe_json_loads(raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _map_native_tool_calls(
        self,
        *,
        tool_calls: Any,
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        provider_control: Optional[Dict[str, Any]] = None
        provider_tool_requests: List[Dict[str, Any]] = []
        counters = {"native_tool_calls_total": 0, "native_control_calls": 0, "native_tool_requests": 0, "native_tool_parse_failures": 0}

        if not isinstance(tool_calls, list):
            return provider_control, provider_tool_requests, counters

        for item in tool_calls:
            if not isinstance(item, dict):
                counters["native_tool_parse_failures"] += 1
                continue
            counters["native_tool_calls_total"] += 1
            function = item.get("function") if isinstance(item.get("function"), dict) else {}
            tool_name = function.get("name") or item.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                counters["native_tool_parse_failures"] += 1
                continue
            tool_name = tool_name.strip()
            args_obj = self._safe_json_loads(function.get("arguments") if function else item.get("arguments"))
            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"native_{uuid.uuid4().hex[:12]}"

            if tool_name == TOOL_CHORUS_CONTROL:
                action = str(args_obj.get("action") or "").strip().upper()
                if action:
                    provider_control = {"action": action, "args": {}}
                    counters["native_control_calls"] += 1
                else:
                    counters["native_tool_parse_failures"] += 1
                continue

            requires_approval = requires_approval_default(tool_name)
            if requires_approval is None:
                requires_approval = bool(item.get("requires_approval", True))
            provider_tool_requests.append(
                {
                    "id": call_id,
                    "tool": tool_name,
                    "requires_approval": bool(requires_approval),
                    "args": args_obj if isinstance(args_obj, dict) else {},
                }
            )
            counters["native_tool_requests"] += 1

        return provider_control, provider_tool_requests, counters

    def request_fingerprint(self, request: InvocationRequest) -> str:
        fingerprint_payload = {
            "invocation_kind": request.invocation_kind,
            "provider": request.provider,
            "engine": request.engine,
            "model_id": request.model_id,
            "messages": request.messages,
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repeat_penalty": request.repeat_penalty,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop,
            "vision_images": request.vision_images,
            "vision_image_mime_type": request.vision_image_mime_type,
            "native_tool_policy": request.native_tool_policy,
            "metadata": self._sanitize_metadata(request.metadata),
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "thread_id": request.thread_id,
            "surface_id": request.surface_id,
            "character_id": request.character_id,
        }
        doc = json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()

    async def invoke(self, request: InvocationRequest) -> Dict[str, Any]:
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")

        request.engine = request.engine or self._engine_from_client(llm_client)
        request_fingerprint = self.request_fingerprint(request)
        timeout_s = 120 if request.invocation_kind == "chat" else 240
        max_retries = 2
        attempts = 0
        started = time.perf_counter()
        last_error: Optional[Dict[str, Any]] = None

        token = _IN_INVOKER_CONTEXT.set(True)
        try:
            for attempt in range(1, max_retries + 2):
                attempts = attempt
                request_debug_token = None
                try:
                    req_tools, req_tool_choice, native_plan = self._prepare_native_transport(request)
                    request.tools = req_tools
                    request.tool_choice = req_tool_choice
                    request_debug_token = set_request_debug_context(
                        {
                            "conversation_id": request.conversation_id,
                            "thread_id": request.thread_id,
                            "character_id": request.character_id,
                            "invocation_kind": request.invocation_kind,
                            "chat_type": (
                                "loopstep"
                                if bool((request.metadata or {}).get("loop_id"))
                                else "normal"
                            ),
                            "loop_id": (request.metadata or {}).get("loop_id"),
                            "loop_kind": (request.metadata or {}).get("loop_kind"),
                            "model_id": request.model_id,
                            "engine": request.engine,
                        }
                    )
                    response = await asyncio.wait_for(
                        self._call_provider(request),
                        timeout=timeout_s,
                    )
                    if request_debug_token is not None:
                        reset_request_debug_context(request_debug_token)
                    output_text = response.get("output_text", "")
                    finish_reason = response.get("finish_reason")
                    provider_tool_calls = response.get("provider_tool_calls")
                    provider_raw_message = response.get("provider_raw_message")
                    completion_flags = self._compute_completion_flags(output_text, finish_reason)
                    if completion_flags:
                        logger.warning(
                            "[LLM_INVOCATION][FLAGGED] kind=%s engine=%s model=%s finish_reason=%s empty=%s flags=%s attempts=%s fingerprint=%s",
                            request.invocation_kind,
                            request.engine,
                            request.model_id,
                            finish_reason,
                            not bool((output_text or "").strip()),
                            completion_flags,
                            attempt,
                            request_fingerprint[:12],
                        )
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    provider_control = None
                    provider_tool_requests = None
                    native_counts = {
                        "native_tool_calls_total": 0,
                        "native_control_calls": 0,
                        "native_tool_requests": 0,
                        "native_tool_parse_failures": 0,
                    }
                    native_attempted = bool(native_plan.get("attempted"))
                    if native_attempted:
                        provider_control, mapped_tool_requests, native_counts = self._map_native_tool_calls(
                            tool_calls=provider_tool_calls,
                        )
                        has_native_content = bool(provider_control) or bool(mapped_tool_requests)
                        if has_native_content:
                            provider_tool_requests = mapped_tool_requests
                        elif not self._sentinel_fallback_enabled():
                            provider_tool_requests = []
                        if has_native_content:
                            logger.info(
                                "[NATIVE_TOOL_TRANSPORT] success engine=%s model=%s control=%s tools=%s",
                                request.engine,
                                request.model_id,
                                bool(provider_control),
                                len(mapped_tool_requests),
                            )
                        else:
                            logger.info(
                                "[NATIVE_TOOL_TRANSPORT] fallback engine=%s model=%s fallback_enabled=%s",
                                request.engine,
                                request.model_id,
                                self._sentinel_fallback_enabled(),
                            )
                    normalized = normalize_assistant_result(
                        raw_content=output_text,
                        provider_control=provider_control,
                        provider_tool_requests=provider_tool_requests,
                        provider_raw={
                            "finish_reason": finish_reason,
                            "raw_message": provider_raw_message,
                            "provider_tool_calls_raw": provider_tool_calls,
                            "native_transport_attempted": native_attempted,
                            "native_transport_plan": dict(native_plan or {}),
                            "native_tool_calls_total": native_counts["native_tool_calls_total"],
                            "native_control_calls": native_counts["native_control_calls"],
                            "native_tool_requests": native_counts["native_tool_requests"],
                            "native_tool_parse_failures": native_counts["native_tool_parse_failures"],
                            "native_tools_requested": bool(req_tools),
                            "requested_tool_choice": req_tool_choice,
                            "requested_tool_names": [
                                str(((tool.get("function") or {}).get("name")) or "")
                                for tool in (req_tools or [])
                                if isinstance(tool, dict)
                            ],
                        },
                    )
                    return {
                        "status": "success",
                        "output_text": output_text,
                        "assistant_result": {
                            "display_text": normalized.display_text,
                            "control": (
                                {
                                    "action": normalized.control.action,
                                    "args": dict(normalized.control.args or {}),
                                }
                                if normalized.control
                                else None
                            ),
                            "tool_requests": [
                                {
                                    "tool_name": item.tool_name,
                                    "payload": dict(item.payload or {}),
                                    "request_id": item.request_id,
                                }
                                for item in normalized.tool_requests
                            ],
                            "payload_present": bool(normalized.payload_present),
                            "payload_parseable": bool(normalized.payload_parseable),
                            "payload_obj": dict(normalized.payload_obj or {}) if normalized.payload_obj else None,
                            "provider_raw": dict(normalized.provider_raw or {}) if normalized.provider_raw else None,
                        },
                        "raw_response_excerpt": response.get("raw_response_excerpt"),
                        "token_usage": response.get("token_usage"),
                        "finish_reason": finish_reason,
                        "output_empty": not bool((output_text or "").strip()),
                        "completion_flags": completion_flags,
                        "cost": response.get("cost"),
                        "provider": request.provider,
                        "engine": request.engine,
                        "model_id": request.model_id,
                        "timing_ms": latency_ms,
                        "attempts": attempts,
                        "replayed": False,
                        "request_fingerprint": request_fingerprint,
                        "native_transport": {
                            "attempted": native_attempted,
                            "plan": dict(native_plan or {}),
                            "requested_tools": req_tools or [],
                            "requested_tool_choice": req_tool_choice,
                            "provider_tool_calls_raw": provider_tool_calls,
                            "provider_raw_message": provider_raw_message,
                        },
                        "error": None,
                    }
                except asyncio.TimeoutError:
                    if request_debug_token is not None:
                        try:
                            reset_request_debug_context(request_debug_token)
                        except Exception:
                            pass
                    last_error = {
                        "code": "timeout",
                        "type": "timeout",
                        "message": f"LLM invocation timed out after {timeout_s}s",
                        "retryable": True,
                    }
                except Exception as exc:
                    if request_debug_token is not None:
                        try:
                            reset_request_debug_context(request_debug_token)
                        except Exception:
                            pass
                    msg = str(exc)
                    retryable = any(token in msg.lower() for token in ["timeout", "temporar", "connection", "503", "502"])
                    last_error = {
                        "code": "llm_error",
                        "type": exc.__class__.__name__,
                        "message": msg,
                        "retryable": retryable,
                    }
                if not last_error or not last_error.get("retryable") or attempt > max_retries:
                    break
                await asyncio.sleep((0.25 * (2 ** (attempt - 1))) + random.uniform(0.01, 0.1))
        finally:
            _IN_INVOKER_CONTEXT.reset(token)

        latency_ms = int((time.perf_counter() - started) * 1000)
        status = "timeout" if (last_error or {}).get("code") == "timeout" else "error"
        return {
            "status": status,
            "output_text": "",
            "raw_response_excerpt": None,
            "token_usage": None,
            "finish_reason": None,
            "output_empty": True,
            "completion_flags": [],
            "cost": None,
            "provider": request.provider,
            "engine": request.engine,
            "model_id": request.model_id,
            "timing_ms": latency_ms,
            "attempts": attempts,
            "replayed": False,
            "request_fingerprint": request_fingerprint,
            "error": last_error,
        }

    async def _call_provider(self, request: InvocationRequest) -> Dict[str, Any]:
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")
        if request.vision_images:
            response = await llm_client.generate_vision(
                prompt=request.prompt or "",
                image_base64_list=request.vision_images,
                image_mime_type=request.vision_image_mime_type or "image/jpeg",
                system_prompt=request.system_prompt,
                model=request.model_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
        elif request.messages is not None:
            kwargs = dict(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                model=request.model_id,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            if isinstance(request.response_format, dict):
                kwargs["response_format"] = request.response_format
            try:
                response = await llm_client.generate_with_history(**kwargs)
            except TypeError as exc:
                # Backward-compat for test doubles or clients that haven't adopted new kwargs yet.
                fallback_keys = [
                    "response_format",
                    "top_p",
                    "top_k",
                    "repeat_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                ]
                retried = False
                for key in fallback_keys:
                    if key in kwargs:
                        kwargs.pop(key, None)
                        retried = True
                if retried:
                    response = await llm_client.generate_with_history(**kwargs)
                else:
                    raise
        else:
            kwargs = dict(
                prompt=request.prompt or "",
                system_prompt=request.system_prompt,
                model=request.model_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            if isinstance(request.response_format, dict):
                kwargs["response_format"] = request.response_format
            try:
                response = await llm_client.generate(**kwargs)
            except TypeError as exc:
                fallback_keys = [
                    "response_format",
                    "top_p",
                    "top_k",
                    "repeat_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                ]
                retried = False
                for key in fallback_keys:
                    if key in kwargs:
                        kwargs.pop(key, None)
                        retried = True
                if retried:
                    response = await llm_client.generate(**kwargs)
                else:
                    raise
        content = response.content or ""
        return {
            "output_text": content,
            "raw_response_excerpt": content[:240],
            "token_usage": getattr(response, "usage", None),
            "finish_reason": getattr(response, "finish_reason", None),
            "cost": None,
            "provider_tool_calls": getattr(response, "tool_calls", None),
            "provider_raw_message": getattr(response, "raw_message", None),
        }

    @staticmethod
    def _compute_completion_flags(output_text: str, finish_reason: Optional[str]) -> List[str]:
        flags: List[str] = []
        empty_output = not bool((output_text or "").strip())
        reason = (finish_reason or "").strip().lower()

        if empty_output:
            flags.append("empty_output")
        if reason == "length":
            flags.append("ended_by_length")
        elif reason and reason not in {"stop"}:
            flags.append(f"ended_by_{reason}")
        if empty_output and reason == "length":
            flags.append("empty_due_to_length")
        return flags

    def resolve_effective_config(
        self,
        *,
        character: Optional[Any],
        invocation_kind: str,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
        context_window_override: Optional[int] = None,
        top_p_override: Optional[float] = None,
        top_k_override: Optional[int] = None,
        repeat_penalty_override: Optional[float] = None,
        presence_penalty_override: Optional[float] = None,
        frequency_penalty_override: Optional[float] = None,
    ) -> EffectiveLLMConfig:
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")
        system_cfg = self.app_state["system_config"].llm
        preferred = getattr(character, "preferred_llm", None) if character is not None else None

        analysis_default_model = getattr(system_cfg, "archivist_model", None) if invocation_kind == "analysis" else None
        model = (
            model_override
            or (getattr(preferred, "model", None) if preferred else None)
            or analysis_default_model
            or system_cfg.model
        )
        temperature = (
            temperature_override
            if temperature_override is not None
            else (getattr(preferred, "temperature", None) if preferred and getattr(preferred, "temperature", None) is not None else system_cfg.temperature)
        )
        max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else (getattr(preferred, "max_tokens", None) if preferred and getattr(preferred, "max_tokens", None) is not None else system_cfg.max_response_tokens)
        )
        context_window = (
            context_window_override
            if context_window_override is not None
            else (getattr(preferred, "context_window", None) if preferred and getattr(preferred, "context_window", None) is not None else system_cfg.context_window)
        )
        top_p = (
            top_p_override
            if top_p_override is not None
            else (getattr(preferred, "top_p", None) if preferred and getattr(preferred, "top_p", None) is not None else getattr(system_cfg, "top_p", None))
        )
        top_k = (
            top_k_override
            if top_k_override is not None
            else (getattr(preferred, "top_k", None) if preferred and getattr(preferred, "top_k", None) is not None else getattr(system_cfg, "top_k", None))
        )
        repeat_penalty = (
            repeat_penalty_override
            if repeat_penalty_override is not None
            else (
                getattr(preferred, "repeat_penalty", None)
                if preferred and getattr(preferred, "repeat_penalty", None) is not None
                else getattr(system_cfg, "repeat_penalty", None)
            )
        )
        presence_penalty = (
            presence_penalty_override
            if presence_penalty_override is not None
            else (
                getattr(preferred, "presence_penalty", None)
                if preferred and getattr(preferred, "presence_penalty", None) is not None
                else getattr(system_cfg, "presence_penalty", None)
            )
        )
        frequency_penalty = (
            frequency_penalty_override
            if frequency_penalty_override is not None
            else (
                getattr(preferred, "frequency_penalty", None)
                if preferred and getattr(preferred, "frequency_penalty", None) is not None
                else getattr(system_cfg, "frequency_penalty", None)
            )
        )
        return EffectiveLLMConfig(
            provider="local",
            engine=self._engine_from_client(llm_client),
            model_id=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
