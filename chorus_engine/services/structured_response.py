"""
Structured Response Parsing & Adapters

Parses and normalizes the XML-like structured response format:
<assistant_response><speech>...</speech>...</assistant_response>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import re

from chorus_engine.services.tool_payload import detect_malformed_tool_payload_block


@dataclass
class StructuredSegment:
    channel: str
    text: str


@dataclass
class StructuredResponse:
    segments: List[StructuredSegment]
    is_fallback: bool = False
    parse_error: Optional[str] = None
    had_untagged: bool = False
    unknown_tags: List[str] = field(default_factory=list)
    trailing_text_dropped: bool = False


ALLOWED_CHANNELS_ALL = {"speech", "physicalaction", "innerthought", "narration", "action"}


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def parse_structured_response(
    raw: str,
    allowed_channels: Optional[set[str]] = None,
    required_channels: Optional[set[str]] = None,
) -> StructuredResponse:
    """
    Parse structured response. Returns fallback on any invalid structure.
    Rules enforced:
    - Root <assistant_response> required
    - Only allowed channel tags
    - No attributes in tags
    - No text outside tags
    - No nesting (implicitly enforced by flat matching)
    """
    if allowed_channels is None:
        allowed_channels = set(ALLOWED_CHANNELS_ALL)
    if required_channels is None:
        required_channels = set()

    if not raw or not raw.strip():
        return StructuredResponse(
            segments=[StructuredSegment(channel="speech", text="")],
            is_fallback=True,
            parse_error="empty_response",
            had_untagged=False
        )

    source = raw
    unknown_tags: set[str] = set()
    trailing_text_dropped = False

    # Enforce root boundary when present: anything outside assistant_response is dropped.
    root_match = re.search(r"<assistant_response>([\s\S]*?)</assistant_response>", source)
    if root_match:
        root_start, root_end = root_match.span()
        raw_prefix = source[:root_start]
        raw_suffix = source[root_end:]

        if raw_prefix.strip():
            tail_is_malformed_tool, _payload_type = detect_malformed_tool_payload_block(raw_prefix)
            if not tail_is_malformed_tool:
                trailing_text_dropped = True
        if raw_suffix.strip():
            tail_is_malformed_tool, _payload_type = detect_malformed_tool_payload_block(raw_suffix)
            if not tail_is_malformed_tool:
                trailing_text_dropped = True

        source = root_match.group(1)

    # Lenient normalization:
    # - Scan for known tags in order
    # - Any text outside known tags becomes <speech>
    segments: List[StructuredSegment] = []
    cursor = 0
    had_untagged = False
    parse_error: Optional[str] = None
    pattern = re.compile(r"<([a-z][a-z0-9_]*)>([\s\S]*?)</\1>")
    def add_segment(channel: str, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            segments.append(StructuredSegment(channel=channel, text=cleaned))
    
    for match in pattern.finditer(source):
        start, end = match.span()
        raw_prefix = source[cursor:start]
        if raw_prefix.strip():
            had_untagged = True
            add_segment("speech", _strip_tags(raw_prefix))
        
        channel = match.group(1)
        text = match.group(2)
        if channel in allowed_channels:
            # Strip any tag-like text inside to avoid leaking raw markup
            add_segment(channel, _strip_tags(text))
        else:
            had_untagged = True
            parse_error = parse_error or f"unknown_channel:{channel}"
            unknown_tags.add(channel)
            # Unknown tags are dropped.
        
        cursor = end
    
    raw_tail = source[cursor:]
    if raw_tail.strip():
        had_untagged = True
        tail_is_malformed_tool, _payload_type = detect_malformed_tool_payload_block(raw_tail)
        if tail_is_malformed_tool:
            parse_error = parse_error or "tool_like_tail_suppressed"
        else:
            add_segment("speech", _strip_tags(raw_tail))

    if not segments:
        had_untagged = True
        if root_match:
            parse_error = parse_error or "no_valid_segments"
            segments = [StructuredSegment(channel="speech", text="")]
        else:
            segments = [StructuredSegment(channel="speech", text=_strip_tags(source).strip())]
    
    # Ensure required channels exist (note only for diagnostics)
    present = {s.channel for s in segments}
    missing_required = required_channels - present
    if missing_required:
        parse_error = parse_error or f"missing_required:{','.join(sorted(missing_required))}"
        had_untagged = True

    return StructuredResponse(
        segments=segments,
        is_fallback=had_untagged or (parse_error is not None),
        parse_error=parse_error,
        had_untagged=had_untagged,
        unknown_tags=sorted(unknown_tags),
        trailing_text_dropped=trailing_text_dropped,
    )


def serialize_structured_response(segments: List[StructuredSegment]) -> str:
    parts = ["<assistant_response>"]
    for seg in segments:
        parts.append(f"<{seg.channel}>{seg.text}</{seg.channel}>")
    parts.append("</assistant_response>")
    return "".join(parts)


def template_rules(template: str) -> tuple[set[str], set[str]]:
    """
    Returns (allowed_channels, required_channels) for a template.
    """
    if template == "A":
        return {"speech", "physicalaction", "innerthought"}, {"speech"}
    if template == "B":
        return {"speech", "narration"}, {"narration"}
    if template == "C":
        return {"speech"}, {"speech"}
    if template == "D":
        return {"speech", "action"}, {"action"}
    return set(ALLOWED_CHANNELS_ALL), {"speech"}


def to_plain_text(
    segments: List[StructuredSegment],
    include_physicalaction: bool = False,
) -> str:
    """
    Extracts plain text for TTS:
    - speech, narration, action always
    - physicalaction optional
    - innerthought excluded
    """
    allowed = {"speech", "narration", "action"}
    if include_physicalaction:
        allowed.add("physicalaction")
    parts = [s.text for s in segments if s.channel in allowed and s.text]
    return "\n\n".join(parts).strip()


def to_discord_text(segments: List[StructuredSegment]) -> str:
    """
    Convert structured response to Discord-friendly text:
    - speech/narration/action as plain text
    - physicalaction italicized
    - innerthought dropped
    """
    lines = []
    for seg in segments:
        if not seg.text:
            continue
        if seg.channel == "innerthought":
            continue
        if seg.channel == "physicalaction":
            lines.append(f"*{seg.text}*")
        else:
            lines.append(seg.text)
    return "\n\n".join(lines).strip()
