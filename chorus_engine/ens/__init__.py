"""Executive Nervous System package.

Use lazy exports to avoid import-time cycles between ENS runtime/dispatcher
and services that only need lightweight ENS modules (for example tool registry).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import ENSRuntime, ENSContext, ENSOutcome
    from .models import Signal
    from .llm_invocation_service import (
        LLMInvocationService,
        InvocationRequest,
        EffectiveLLMConfig,
        in_invoker_context,
    )
    from .llm_control_plane_service import (
        LLMControlPlaneService,
        ControlPlaneRequest,
        in_control_plane_context,
    )

__all__ = [
    "ENSRuntime",
    "ENSContext",
    "ENSOutcome",
    "Signal",
    "LLMInvocationService",
    "InvocationRequest",
    "EffectiveLLMConfig",
    "in_invoker_context",
    "LLMControlPlaneService",
    "ControlPlaneRequest",
    "in_control_plane_context",
]


def __getattr__(name: str) -> Any:
    if name in {"ENSRuntime", "ENSContext", "ENSOutcome"}:
        mod = import_module("chorus_engine.ens.runtime")
        return getattr(mod, name)
    if name == "Signal":
        mod = import_module("chorus_engine.ens.models")
        return getattr(mod, name)
    if name in {"LLMInvocationService", "InvocationRequest", "EffectiveLLMConfig", "in_invoker_context"}:
        mod = import_module("chorus_engine.ens.llm_invocation_service")
        return getattr(mod, name)
    if name in {"LLMControlPlaneService", "ControlPlaneRequest", "in_control_plane_context"}:
        mod = import_module("chorus_engine.ens.llm_control_plane_service")
        return getattr(mod, name)
    raise AttributeError(f"module 'chorus_engine.ens' has no attribute {name!r}")

