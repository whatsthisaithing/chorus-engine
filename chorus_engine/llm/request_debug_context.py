"""Context helpers for raw LLM request/response debug capture."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Dict, Optional


_REQUEST_DEBUG_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "llm_request_debug_context",
    default=None,
)


def set_request_debug_context(value: Dict[str, Any]) -> Token:
    return _REQUEST_DEBUG_CONTEXT.set(dict(value or {}))


def reset_request_debug_context(token: Token) -> None:
    _REQUEST_DEBUG_CONTEXT.reset(token)


def get_request_debug_context() -> Dict[str, Any]:
    value = _REQUEST_DEBUG_CONTEXT.get()
    return dict(value or {})

