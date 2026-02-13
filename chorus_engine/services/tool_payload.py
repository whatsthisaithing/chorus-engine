"""
Tool payload extraction and validation for in-conversation tool calls.

Expected format (after </assistant_response>):
---CHORUS_TOOL_PAYLOAD_BEGIN---
{ ...json... }
---CHORUS_TOOL_PAYLOAD_END---
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Optional

from chorus_engine.services.json_extraction import extract_json_block


BEGIN_SENTINEL = "---CHORUS_TOOL_PAYLOAD_BEGIN---"
END_SENTINEL = "---CHORUS_TOOL_PAYLOAD_END---"
SUPPORTED_TOOLS_V1 = {"image.generate", "video.generate"}
MOMENT_PIN_COLD_RECALL_TOOL = "moment_pin.cold_recall"


@dataclass
class ToolCall:
    id: str
    tool: str
    requires_approval: bool
    prompt: str
    confidence: float = 1.0


@dataclass
class ToolPayloadExtraction:
    display_text: str
    payload_text: Optional[str]
    had_begin: bool
    had_end: bool


@dataclass
class ColdRecallToolCall:
    id: str
    tool: str
    requires_approval: bool
    pin_id: str
    reason: str


def extract_tool_payload(raw_text: str) -> ToolPayloadExtraction:
    """
    Extract sentinel-wrapped payload and return chat-safe display text.

    Rules:
    - Only first BEGIN is considered.
    - Everything from BEGIN onward is removed from display text.
    - Malformed block (missing END) is discarded silently.
    """
    if raw_text is None:
        return ToolPayloadExtraction(display_text="", payload_text=None, had_begin=False, had_end=False)

    begin_index = raw_text.find(BEGIN_SENTINEL)
    if begin_index == -1:
        return ToolPayloadExtraction(display_text=raw_text, payload_text=None, had_begin=False, had_end=False)

    display_text = raw_text[:begin_index]
    end_index = raw_text.find(END_SENTINEL, begin_index + len(BEGIN_SENTINEL))
    if end_index == -1:
        return ToolPayloadExtraction(display_text=display_text, payload_text=None, had_begin=True, had_end=False)

    payload_text = raw_text[begin_index + len(BEGIN_SENTINEL):end_index].strip()
    return ToolPayloadExtraction(display_text=display_text, payload_text=payload_text, had_begin=True, had_end=True)


def parse_tool_payload(payload_text: Optional[str]) -> Optional[dict[str, Any]]:
    """
    Parse payload JSON with a small rescue strategy.
    """
    if not payload_text:
        return None

    try:
        parsed = json.loads(payload_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    parsed, _mode = extract_json_block(payload_text, expected_root="object")
    if isinstance(parsed, dict):
        return parsed

    return None


def validate_tool_payload(payload: Optional[dict[str, Any]]) -> list[ToolCall]:
    """
    Validate v1 payload schema and return normalized tool calls.

    v1 behavior:
    - version must be 1
    - tool_calls must be a list
    - only supported tools
    - prompt must be a non-empty string
    """
    if not payload or not isinstance(payload, dict):
        return []

    if payload.get("version") != 1:
        return []

    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    calls: list[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue

        call_id = item.get("id")
        tool = item.get("tool")
        requires_approval = bool(item.get("requires_approval", True))
        args = item.get("args")
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        if tool not in SUPPORTED_TOOLS_V1:
            continue
        if not isinstance(args, dict):
            continue
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        try:
            confidence = float(item.get("confidence", 1.0))
        except Exception:
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))

        calls.append(
            ToolCall(
                id=call_id.strip(),
                tool=tool,
                requires_approval=requires_approval,
                prompt=prompt.strip(),
                confidence=confidence,
            )
        )

    return calls


def validate_cold_recall_payload(payload: Optional[dict[str, Any]]) -> Optional[ColdRecallToolCall]:
    """
    Validate and normalize a moment-pin cold recall call.

    Rules:
    - version must be 1
    - exactly one tool_calls item
    - tool must be moment_pin.cold_recall
    - requires_approval must be false
    - args.pin_id must be non-empty string
    - args.reason must be non-empty string
    """
    if not payload or not isinstance(payload, dict):
        return None
    if payload.get("version") != 1:
        return None

    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) != 1:
        return None

    item = raw_calls[0]
    if not isinstance(item, dict):
        return None
    call_id = item.get("id")
    tool = item.get("tool")
    requires_approval = item.get("requires_approval", False)
    args = item.get("args")

    if not isinstance(call_id, str) or not call_id.strip():
        return None
    if tool != MOMENT_PIN_COLD_RECALL_TOOL:
        return None
    if requires_approval is not False:
        return None
    if not isinstance(args, dict):
        return None
    pin_id = args.get("pin_id")
    reason = args.get("reason")
    if not isinstance(pin_id, str) or not pin_id.strip():
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None

    return ColdRecallToolCall(
        id=call_id.strip(),
        tool=tool,
        requires_approval=False,
        pin_id=pin_id.strip(),
        reason=reason.strip(),
    )
