"""Unified ENS-owned LLM invocation service for chat and analysis calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    stop: Optional[List[str]] = None
    vision_images: Optional[List[str]] = None
    vision_image_mime_type: Optional[str] = None
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
            "stop": request.stop,
            "vision_images": request.vision_images,
            "vision_image_mime_type": request.vision_image_mime_type,
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
                try:
                    response = await asyncio.wait_for(
                        self._call_provider(request),
                        timeout=timeout_s,
                    )
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    return {
                        "status": "success",
                        "output_text": response.get("output_text", ""),
                        "raw_response_excerpt": response.get("raw_response_excerpt"),
                        "token_usage": response.get("token_usage"),
                        "cost": response.get("cost"),
                        "provider": request.provider,
                        "engine": request.engine,
                        "model_id": request.model_id,
                        "timing_ms": latency_ms,
                        "attempts": attempts,
                        "replayed": False,
                        "request_fingerprint": request_fingerprint,
                        "error": None,
                    }
                except asyncio.TimeoutError:
                    last_error = {
                        "code": "timeout",
                        "type": "timeout",
                        "message": f"LLM invocation timed out after {timeout_s}s",
                        "retryable": True,
                    }
                except Exception as exc:
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
            )
        elif request.messages is not None:
            response = await llm_client.generate_with_history(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                model=request.model_id,
            )
        else:
            response = await llm_client.generate(
                prompt=request.prompt or "",
                system_prompt=request.system_prompt,
                model=request.model_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        content = response.content or ""
        return {
            "output_text": content,
            "raw_response_excerpt": content[:240],
            "token_usage": None,
            "cost": None,
        }

    def resolve_effective_config(
        self,
        *,
        character: Optional[Any],
        invocation_kind: str,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
        context_window_override: Optional[int] = None,
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
        return EffectiveLLMConfig(
            provider="local",
            engine=self._engine_from_client(llm_client),
            model_id=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
        )
