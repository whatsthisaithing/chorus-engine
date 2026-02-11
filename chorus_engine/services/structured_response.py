"""
Structured Response Parsing & Adapters

Parses and normalizes the XML-like structured response format:
<assistant_response><speech>...</speech>...</assistant_response>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re


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
    # Lenient normalization:
    # - Scan for known tags in order
    # - Any text outside known tags becomes <speech>
    segments: List[StructuredSegment] = []
    cursor = 0
    had_untagged = False
    parse_error: Optional[str] = None
    pattern = re.compile(r"<([a-z]+)>([\s\S]*?)</\1>")
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
            raw_full = source[start:end]
            add_segment("speech", _strip_tags(raw_full))
        
        cursor = end
    
    raw_tail = source[cursor:]
    if raw_tail.strip():
        had_untagged = True
        add_segment("speech", _strip_tags(raw_tail))

    if not segments:
        had_untagged = True
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
        had_untagged=had_untagged
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
