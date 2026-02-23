"""Canonical AssistantResult normalization for ENS.

This module centralizes normalization from provider output into structured
assistant artifacts consumed by ENS. Providers/adapters should not duplicate
this parsing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chorus_engine.services.tool_payload import extract_tool_payload, parse_tool_payload


ALLOWED_CONTROL_ACTIONS = {"WAIT_FOR_USER", "CONTINUE", "COMPLETE", "YIELD"}


@dataclass
class ControlDirective:
    action: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolRequest:
    tool_name: str
    payload: Dict[str, Any]
    request_id: Optional[str] = None


@dataclass
class AssistantResult:
    raw_content: str
    display_text: str
    control: Optional[ControlDirective]
    tool_requests: List[ToolRequest]
    payload_present: bool
    payload_parseable: bool
    payload_obj: Optional[Dict[str, Any]]
    provider_raw: Optional[Dict[str, Any]] = None


def _normalize_control(raw_control: Any) -> Optional[ControlDirective]:
    if not isinstance(raw_control, dict):
        return None
    action = raw_control.get("action")
    if not isinstance(action, str):
        return None
    action = action.strip().upper()
    if action not in ALLOWED_CONTROL_ACTIONS:
        return None
    args = raw_control.get("args")
    if not isinstance(args, dict):
        args = {}
    return ControlDirective(action=action, args=args)


def _normalize_tool_requests(raw_tool_calls: Any) -> List[ToolRequest]:
    if not isinstance(raw_tool_calls, list):
        return []
    normalized: List[ToolRequest] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        payload = item
        tool_name = item.get("tool")
        request_id = item.get("id")

        # Accept wrapped normalized form from LLMInvocationService:
        # {"tool_name": "...", "payload": {...}, "request_id": "..."}
        wrapped_payload = item.get("payload")
        if isinstance(wrapped_payload, dict):
            payload = dict(wrapped_payload)
            tool_name = payload.get("tool")
            request_id = payload.get("id")
            if request_id is None:
                request_id = item.get("request_id")

        if not isinstance(tool_name, str) or not tool_name.strip():
            tool_name = item.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        if request_id is not None and not isinstance(request_id, str):
            request_id = None
        normalized.append(
            ToolRequest(
                tool_name=tool_name.strip(),
                payload=dict(payload),
                request_id=request_id,
            )
        )
    return normalized


def _normalize_provider_raw(provider_raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(provider_raw, dict):
        return dict(provider_raw)
    return None


def _select_structured_tier(
    *,
    provider_control: Optional[Dict[str, Any]],
    provider_tool_requests: Optional[List[Dict[str, Any]]],
    schema_control: Optional[Dict[str, Any]],
    schema_tool_requests: Optional[List[Dict[str, Any]]],
    sentinel_control: Optional[ControlDirective],
    sentinel_tool_requests: List[ToolRequest],
) -> Tuple[Optional[ControlDirective], List[ToolRequest], str]:
    if provider_control is not None or provider_tool_requests is not None:
        return (
            _normalize_control(provider_control) if provider_control is not None else None,
            _normalize_tool_requests(provider_tool_requests) if provider_tool_requests is not None else [],
            "provider_native",
        )
    if schema_control is not None or schema_tool_requests is not None:
        return (
            _normalize_control(schema_control) if schema_control is not None else None,
            _normalize_tool_requests(schema_tool_requests) if schema_tool_requests is not None else [],
            "schema_structured",
        )
    if sentinel_control is not None or sentinel_tool_requests:
        return sentinel_control, sentinel_tool_requests, "sentinel_fallback"
    return None, [], "none"


def assistant_result_from_normalized_dict(raw_content: str, normalized: Dict[str, Any]) -> AssistantResult:
    """Build AssistantResult from normalized invocation payload."""
    control_obj = normalized.get("control")
    tool_requests_obj = normalized.get("tool_requests")
    payload_obj = normalized.get("payload_obj")
    provider_raw = normalized.get("provider_raw")
    return AssistantResult(
        raw_content=raw_content or "",
        display_text=str(normalized.get("display_text") or ""),
        control=_normalize_control(control_obj),
        tool_requests=_normalize_tool_requests(tool_requests_obj),
        payload_present=bool(normalized.get("payload_present")),
        payload_parseable=bool(normalized.get("payload_parseable")),
        payload_obj=dict(payload_obj) if isinstance(payload_obj, dict) else None,
        provider_raw=_normalize_provider_raw(provider_raw),
    )


def normalize_assistant_result(
    *,
    raw_content: str,
    provider_control: Optional[Dict[str, Any]] = None,
    provider_tool_requests: Optional[List[Dict[str, Any]]] = None,
    schema_control: Optional[Dict[str, Any]] = None,
    schema_tool_requests: Optional[List[Dict[str, Any]]] = None,
    provider_raw: Optional[Dict[str, Any]] = None,
) -> AssistantResult:
    """Normalize provider output into canonical AssistantResult.

    Precedence:
    1. provider structured fields (`provider_control`, `provider_tool_requests`)
    2. sentinel fallback JSON payload in `raw_content`
    """
    extraction = extract_tool_payload(raw_content or "")
    payload_obj = parse_tool_payload(extraction.payload_text)

    sentinel_control = _normalize_control((payload_obj or {}).get("control"))
    sentinel_tool_requests = _normalize_tool_requests((payload_obj or {}).get("tool_calls"))
    control, tool_requests, tier_used = _select_structured_tier(
        provider_control=provider_control,
        provider_tool_requests=provider_tool_requests,
        schema_control=schema_control,
        schema_tool_requests=schema_tool_requests,
        sentinel_control=sentinel_control,
        sentinel_tool_requests=sentinel_tool_requests,
    )

    provider_raw_doc = _normalize_provider_raw(provider_raw) or {}
    provider_raw_doc.setdefault("assistant_result_tier", tier_used)
    return AssistantResult(
        raw_content=raw_content or "",
        display_text=extraction.display_text,
        control=control,
        tool_requests=tool_requests,
        payload_present=extraction.payload_text is not None,
        payload_parseable=payload_obj is not None,
        payload_obj=payload_obj if isinstance(payload_obj, dict) else None,
        provider_raw=provider_raw_doc,
    )
