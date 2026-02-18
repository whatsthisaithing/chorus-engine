"""Unified ENS-owned LLM control-plane service."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_IN_CONTROL_CONTEXT: ContextVar[bool] = ContextVar("ens_in_control_plane_context", default=False)


def in_control_plane_context() -> bool:
    return bool(_IN_CONTROL_CONTEXT.get())


@dataclass
class ControlPlaneRequest:
    op: str
    idempotency_key: str
    busy_mode: str = "block_with_timeout"  # block_with_timeout|skip_busy
    timeout_s: Optional[float] = None
    model_id: Optional[str] = None
    reason: Optional[str] = None
    provider: str = "local"
    engine: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None
    surface_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMControlPlaneService:
    """Serialized, observable control-plane wrapper around llm_client."""

    def __init__(self, app_state: Dict[str, Any]) -> None:
        self.app_state = app_state
        if "ens_llm_control_plane_mutex" not in self.app_state:
            self.app_state["ens_llm_control_plane_mutex"] = asyncio.Lock()

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
    def _normalize_busy_mode(mode: Optional[str]) -> str:
        value = str(mode or "block_with_timeout").strip().lower()
        if value not in ("block_with_timeout", "skip_busy"):
            return "block_with_timeout"
        return value

    @staticmethod
    def _normalize_op(op: Optional[str]) -> str:
        mapping = {
            "health": "health",
            "status": "health",
            "list_loaded": "list_loaded",
            "list": "list_loaded",
            "ensure_loaded": "ensure_loaded",
            "ensure": "ensure_loaded",
            "reload": "reload",
            "unload": "unload",
            "unload_all": "unload_all",
            "switch": "switch",
            "switch_model": "switch",
        }
        return mapping.get(str(op or "").strip().lower(), str(op or "").strip().lower())

    async def execute(self, request: ControlPlaneRequest) -> Dict[str, Any]:
        llm_client = self.app_state.get("llm_client")
        if not llm_client:
            raise RuntimeError("LLM client not initialized")

        op = self._normalize_op(request.op)
        busy_mode = self._normalize_busy_mode(request.busy_mode)
        timeout_s = float(request.timeout_s) if request.timeout_s is not None else 10.0
        engine = request.engine or self._engine_from_client(llm_client)

        mutex: asyncio.Lock = self.app_state["ens_llm_control_plane_mutex"]
        llm_usage_lock: Optional[asyncio.Lock] = self.app_state.get("llm_usage_lock")
        comfyui_lock: Optional[asyncio.Lock] = self.app_state.get("comfyui_lock")
        skip_shared_locks = bool((request.metadata or {}).get("skip_shared_locks", False))
        started = time.perf_counter()

        # Lock ordering invariant: ENS control mutex -> llm_usage_lock -> comfyui_lock.
        # Never acquire in reverse order.
        if busy_mode == "skip_busy" and mutex.locked():
            return {
                "_ens_action_status": "skipped",
                "reason": "busy",
                "op": op,
                "provider": request.provider,
                "engine": engine,
                "model_id": request.model_id,
                "attempts": 0,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "busy_mode": busy_mode,
                "timeout_s": timeout_s,
            }

        acquired: List[asyncio.Lock] = []
        try:
            ok = await self._acquire_lock(mutex, busy_mode=busy_mode, timeout_s=timeout_s)
            if not ok:
                return {
                    "_ens_action_status": "skipped",
                    "reason": "busy",
                    "op": op,
                    "provider": request.provider,
                    "engine": engine,
                    "model_id": request.model_id,
                    "attempts": 0,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "busy_mode": busy_mode,
                    "timeout_s": timeout_s,
                }
            acquired.append(mutex)

            needs_shared_locks = op in {"ensure_loaded", "reload", "unload", "unload_all", "switch"}
            if skip_shared_locks:
                needs_shared_locks = False
            if needs_shared_locks and llm_usage_lock is not None:
                ok = await self._acquire_lock(llm_usage_lock, busy_mode=busy_mode, timeout_s=timeout_s)
                if not ok:
                    return {
                        "_ens_action_status": "skipped",
                        "reason": "busy",
                        "op": op,
                        "provider": request.provider,
                        "engine": engine,
                        "model_id": request.model_id,
                        "attempts": 0,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        "busy_mode": busy_mode,
                        "timeout_s": timeout_s,
                    }
                acquired.append(llm_usage_lock)
            if needs_shared_locks and comfyui_lock is not None:
                ok = await self._acquire_lock(comfyui_lock, busy_mode=busy_mode, timeout_s=timeout_s)
                if not ok:
                    return {
                        "_ens_action_status": "skipped",
                        "reason": "busy",
                        "op": op,
                        "provider": request.provider,
                        "engine": engine,
                        "model_id": request.model_id,
                        "attempts": 0,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        "busy_mode": busy_mode,
                        "timeout_s": timeout_s,
                    }
                acquired.append(comfyui_lock)

            token = _IN_CONTROL_CONTEXT.set(True)
            try:
                result = await self._execute_op(llm_client, op=op, model_id=request.model_id)
            finally:
                _IN_CONTROL_CONTEXT.reset(token)

            duration_ms = int((time.perf_counter() - started) * 1000)
            result.update(
                {
                    "op": op,
                    "provider": request.provider,
                    "engine": engine,
                    "model_id": request.model_id,
                    "attempts": 1,
                    "duration_ms": duration_ms,
                    "busy_mode": busy_mode,
                    "timeout_s": timeout_s,
                }
            )
            return result
        finally:
            while acquired:
                lock = acquired.pop()
                if lock.locked():
                    lock.release()

    async def _acquire_lock(self, lock: asyncio.Lock, *, busy_mode: str, timeout_s: float) -> bool:
        if busy_mode == "skip_busy":
            if lock.locked():
                return False
            await lock.acquire()
            return True
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    async def _execute_op(self, llm_client: Any, *, op: str, model_id: Optional[str]) -> Dict[str, Any]:
        if op == "health":
            health = await llm_client.health_check()
            return {
                "status": "success",
                "engine_health": bool(health),
            }
        if op == "list_loaded":
            loaded = await llm_client.get_loaded_models()
            return {
                "status": "success",
                "loaded_models": list(loaded or []),
            }
        if op == "ensure_loaded":
            if not model_id:
                raise RuntimeError("model_id required for ensure_loaded")
            loaded = await llm_client.ensure_model_loaded(model_id)
            loaded_models = await llm_client.get_loaded_models()
            return {
                "status": "success",
                "loaded": bool(loaded),
                "loaded_models": list(loaded_models or []),
                "active_model_id": model_id,
            }
        if op == "reload":
            await llm_client.reload_model()
            loaded_models = await llm_client.get_loaded_models()
            return {
                "status": "success",
                "loaded_models": list(loaded_models or []),
                "active_model_id": model_id or self._active_model_hint(loaded_models),
            }
        if op == "unload":
            if not model_id:
                raise RuntimeError("model_id required for unload")
            await llm_client.unload_model(model_id)
            loaded_models = await llm_client.get_loaded_models()
            return {
                "status": "success",
                "loaded_models": list(loaded_models or []),
            }
        if op == "unload_all":
            await llm_client.unload_all_models()
            loaded_models = await llm_client.get_loaded_models()
            return {
                "status": "success",
                "loaded_models": list(loaded_models or []),
            }
        if op == "switch":
            if not model_id:
                raise RuntimeError("model_id required for switch")
            if not hasattr(llm_client, "switch_model"):
                raise RuntimeError("Model switching not supported by provider")
            success = await llm_client.switch_model(model_id)
            if not success:
                raise RuntimeError("Failed to switch model")
            loaded_models = await llm_client.get_loaded_models()
            return {
                "status": "success",
                "switched": True,
                "active_model_id": model_id,
                "loaded_models": list(loaded_models or []),
            }
        raise RuntimeError(f"Unsupported control operation: {op}")

    @staticmethod
    def _active_model_hint(loaded_models: Any) -> Optional[str]:
        items = list(loaded_models or [])
        return str(items[0]) if items else None
