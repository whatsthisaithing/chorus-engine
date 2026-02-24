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
import re
from typing import Any, Optional

from chorus_engine.ens.tool_registry import (
    TOOL_MOMENT_PIN_COLD_RECALL,
    sentinel_media_tools,
)
from chorus_engine.services.json_extraction import extract_json_block


BEGIN_SENTINEL = "---CHORUS_TOOL_PAYLOAD_BEGIN---"
END_SENTINEL = "---CHORUS_TOOL_PAYLOAD_END---"
_RELAXED_BEGIN_SENTINEL_RE = re.compile(r"(?is)-{0,3}\s*CHORUS_TOOL_PAYLOAD_BEGIN---")
_RELAXED_END_SENTINEL_RE = re.compile(r"(?is)-{0,3}\s*CHORUS_TOOL_PAYLOAD_END---")
SUPPORTED_TOOLS_V1 = sentinel_media_tools()
MOMENT_PIN_COLD_RECALL_TOOL = TOOL_MOMENT_PIN_COLD_RECALL


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


_HEADING_FENCED_PATTERN = re.compile(
    r"(?is)(?:\n|^)\s*\*\*[^*\n]*payload[^*\n]*\*\*\s*```json[\s\S]*?```"
)
_FENCED_TOOL_JSON_PATTERN = re.compile(
    r"(?is)(?:\n|^)\s*```json[\s\S]*?\"tool_calls\"[\s\S]*?```"
)
_RAW_TOOL_JSON_PATTERN = re.compile(
    r"(?is)(?:\n|^)\s*\{[\s\S]{0,6000}\"tool_calls\"[\s\S]{0,6000}\}\s*$"
)


def detect_malformed_tool_payload_block(raw_text: str) -> tuple[bool, Optional[str]]:
    """
    Detect likely tool payload leaks that are not wrapped in sentinels.
    """
    if not raw_text:
        return False, None
    if BEGIN_SENTINEL in raw_text or _RELAXED_BEGIN_SENTINEL_RE.search(raw_text):
        return False, None
    if _HEADING_FENCED_PATTERN.search(raw_text):
        return True, "heading_json"
    if _FENCED_TOOL_JSON_PATTERN.search(raw_text):
        return True, "fenced_json"
    if _RAW_TOOL_JSON_PATTERN.search(raw_text) and ("\"image.generate\"" in raw_text or "\"video.generate\"" in raw_text):
        return True, "raw_json"
    return False, None


def strip_malformed_tool_payload_block(raw_text: str) -> tuple[str, bool, Optional[str]]:
    """
    Remove likely malformed tool payload blocks from visible assistant text.
    """
    detected, payload_type = detect_malformed_tool_payload_block(raw_text)
    if not detected:
        return raw_text, False, None

    stripped = raw_text
    stripped = _HEADING_FENCED_PATTERN.sub("", stripped)
    stripped = _FENCED_TOOL_JSON_PATTERN.sub("", stripped)
    # Only strip trailing raw JSON blobs to reduce false positives.
    stripped = _RAW_TOOL_JSON_PATTERN.sub("", stripped)
    stripped = stripped.rstrip()
    return stripped, True, payload_type


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
    begin_end = -1
    if begin_index == -1:
        relaxed_begin = _RELAXED_BEGIN_SENTINEL_RE.search(raw_text)
        if not relaxed_begin:
            return ToolPayloadExtraction(display_text=raw_text, payload_text=None, had_begin=False, had_end=False)
        begin_index = relaxed_begin.start()
        begin_end = relaxed_begin.end()
    else:
        begin_end = begin_index + len(BEGIN_SENTINEL)

    display_text = raw_text[:begin_index]
    end_index = raw_text.find(END_SENTINEL, begin_end)
    end_start = -1
    if end_index == -1:
        relaxed_end = _RELAXED_END_SENTINEL_RE.search(raw_text, begin_end)
        if not relaxed_end:
            return ToolPayloadExtraction(display_text=display_text, payload_text=None, had_begin=True, had_end=False)
        end_start = relaxed_end.start()
    else:
        end_start = end_index

    payload_text = raw_text[begin_end:end_start].strip()
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
