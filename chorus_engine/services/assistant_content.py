"""Assistant content normalization/routing for FrameLines v2 cutover."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chorus_engine.services.structured_response import (
    StructuredSegment,
    parse_structured_response,
    serialize_structured_response,
)


FORMAT_FRAMELINES_V2 = "framelines_v2"
FORMAT_LEGACY_XML_V1 = "legacy_xml_v1"
FORMAT_TAGXML_V1 = "tagxml_v1"  # backward-compatible alias
FORMAT_MARKDOWN_V1 = "markdown_v1"
FORMAT_PLAIN_V0 = "plain_v0"

_FRAME_MARKER_RE = re.compile(r"^\[\[([A-Z])\]\]\s+(.+?)\s*$")
_EXACT_END_RE = re.compile(r"^\[\[E\]\]\s*$")
_SECONDARY_MARKER_RE = re.compile(r"\s(\[\[[A-Z]\]\])\s")
_INLINE_MARKER_RE = re.compile(r"\[\[([A-Z])\]\]")
_MARKDOWN_END_RE = re.compile(r"^\s*---CHORUS_END---\s*$")
_MARKDOWN_END_TOKEN = "---CHORUS_END---"
_MARKDOWN_END_TEXT_ONLY_RE = re.compile(r"^\s*CHORUS_END\s*$")
_MARKDOWN_DASH_ONLY_RE = re.compile(r"^\s*---+\s*$")


@dataclass
class TemplateContext:
    template_id: str
    allowed_markers: set[str]
    required_markers: set[str]
    default_marker: str


@dataclass
class ThinkingCaptureState:
    in_reasoning: bool = False
    active_tag: Optional[str] = None
    boundary_buffer: str = ""


@dataclass
class ThinkingDeltaResult:
    visible_delta: str
    reasoning_delta: str
    state: ThinkingCaptureState


@dataclass
class ParsedFrameLines:
    canonical_text: str
    segments: List[StructuredSegment]
    diagnostics: Dict[str, Any]
    visible_text: str
    post_end_tail: str
    had_end_marker: bool


@dataclass
class ParsedMarkdown:
    canonical_text: str
    diagnostics: Dict[str, Any]
    visible_text: str
    post_end_tail: str
    had_end_marker: bool


def template_context(template_id: str) -> TemplateContext:
    tid = str(template_id or "C").strip().upper() or "C"
    if tid == "A":
        return TemplateContext(
            template_id="A",
            allowed_markers={"S", "A", "T"},
            required_markers={"S"},
            default_marker="S",
        )
    if tid == "B":
        return TemplateContext(
            template_id="B",
            allowed_markers={"S", "N"},
            required_markers={"N"},
            default_marker="N",
        )
    if tid == "D":
        return TemplateContext(
            template_id="D",
            allowed_markers={"S", "A"},
            required_markers={"A"},
            default_marker="A",
        )
    return TemplateContext(
        template_id="C",
        allowed_markers={"S"},
        required_markers={"S"},
        default_marker="S",
    )


def detect_assistant_output_format(raw_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    fmt = str(meta.get("assistant_output_format") or "").strip().lower()
    if fmt == FORMAT_TAGXML_V1:
        return FORMAT_LEGACY_XML_V1
    if fmt in {FORMAT_FRAMELINES_V2, FORMAT_LEGACY_XML_V1, FORMAT_MARKDOWN_V1, FORMAT_PLAIN_V0}:
        return fmt
    text = str(raw_text or "")
    if "<assistant_response>" in text:
        return FORMAT_LEGACY_XML_V1
    if "[[E]]" in text and any(line.strip().startswith("[[") for line in text.splitlines()):
        return FORMAT_FRAMELINES_V2
    if "---CHORUS_END---" in text:
        return FORMAT_MARKDOWN_V1
    return FORMAT_PLAIN_V0


def _effective_template(template_id: Optional[str]) -> str:
    return str(template_id or "C").strip().upper() or "C"


def _action_channel_for_template(template_id: str) -> str:
    return "action" if _effective_template(template_id) == "D" else "physicalaction"


def _marker_to_channel(marker: str, template_id: str) -> str:
    if marker == "S":
        return "speech"
    if marker == "T":
        return "innerthought"
    if marker == "N":
        return "narration"
    if marker == "A":
        return _action_channel_for_template(template_id)
    return "speech"


def _channel_to_marker(channel: str) -> str:
    ch = str(channel or "").strip().lower()
    if ch == "speech":
        return "S"
    if ch == "innerthought":
        return "T"
    if ch in {"physicalaction", "action"}:
        return "A"
    if ch == "narration":
        return "N"
    return "S"


def _safe_lines(text: str) -> List[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return [line.rstrip() for line in normalized.split("\n") if line.strip()]


def _repair_markdown_terminator_layout(raw_text: str) -> Tuple[str, bool]:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    repaired = False

    def _nearest_nonempty(lines: List[str], start: int, step: int) -> Optional[int]:
        i = start
        while 0 <= i < len(lines):
            if lines[i].strip():
                return i
            i += step
        return None

    # Repair marker layouts such as:
    # ---
    # CHORUS_END
    # ...or...
    # CHORUS_END
    # ---
    # ...or...
    # CHORUS_END
    if _MARKDOWN_END_TOKEN not in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if not _MARKDOWN_END_TEXT_ONLY_RE.match(line):
                continue
            prev_idx = _nearest_nonempty(lines, i - 1, -1)
            next_idx = _nearest_nonempty(lines, i + 1, +1)
            prev_dash = prev_idx is not None and _MARKDOWN_DASH_ONLY_RE.match(lines[prev_idx] or "")
            next_dash = next_idx is not None and _MARKDOWN_DASH_ONLY_RE.match(lines[next_idx] or "")
            if not prev_dash and not next_dash:
                start_idx = i
                end_idx = i
            else:
                start_idx = prev_idx if prev_dash else i
                end_idx = next_idx if next_dash else i
            lines = lines[:start_idx] + [_MARKDOWN_END_TOKEN] + lines[end_idx + 1 :]
            text = "\n".join(lines)
            repaired = True
            break

    idx = text.find(_MARKDOWN_END_TOKEN)
    if idx < 0:
        return text, repaired

    # Ensure the terminator starts on its own line.
    if idx > 0 and text[idx - 1] != "\n":
        text = text[:idx] + "\n" + text[idx:]
        idx += 1
        repaired = True

    # Ensure exactly one newline immediately before the marker when not at BOF.
    if idx > 0:
        run_start = idx
        while run_start > 0 and text[run_start - 1] == "\n":
            run_start -= 1
        newline_count = idx - run_start
        if newline_count != 1:
            text = text[:run_start] + "\n" + text[idx:]
            idx = run_start + 1
            repaired = True

    # Ensure nothing trails on the same line as the terminator.
    end_idx = idx + len(_MARKDOWN_END_TOKEN)
    if end_idx < len(text) and text[end_idx] != "\n":
        text = text[:end_idx] + "\n" + text[end_idx:]
        repaired = True

    return text, repaired


def extract_visible_and_tail(raw_text: str, *, format_id: str) -> Tuple[str, str, bool]:
    text = str(raw_text or "")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fmt = str(format_id or "").strip().lower()
    if fmt == FORMAT_MARKDOWN_V1:
        end_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if _MARKDOWN_END_RE.match(line):
                end_idx = i
                break
        if end_idx is None:
            return text.strip(), "", False
        visible = "\n".join(lines[:end_idx]).strip()
        tail = "\n".join(lines[end_idx + 1 :]).strip()
        return visible, tail, True
    if fmt == FORMAT_FRAMELINES_V2:
        end_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if _EXACT_END_RE.match(line.strip()):
                end_idx = i
                break
        if end_idx is None:
            return text.strip(), "", False
        visible = "\n".join(lines[:end_idx]).strip()
        tail = "\n".join(lines[end_idx + 1 :]).strip()
        return visible, tail, True
    if fmt == FORMAT_LEGACY_XML_V1:
        close_idx = text.find("</assistant_response>")
        if close_idx < 0:
            return text.strip(), "", False
        cutoff = close_idx + len("</assistant_response>")
        visible = text[:cutoff].strip()
        tail = text[cutoff:].strip()
        return visible, tail, True
    return text.strip(), "", False


def _split_multi_marker_line(line: str, allowed_markers: set[str], max_splits: int = 4) -> List[str]:
    if not line.startswith("[["):
        return [line]
    chunks: List[str] = []
    current = line
    split_count = 0
    while split_count < max_splits:
        m = _SECONDARY_MARKER_RE.search(current)
        if not m:
            break
        marker_token = m.group(1)
        marker = marker_token[2:3]
        if marker not in allowed_markers:
            break
        left = current[: m.start()].rstrip()
        right = current[m.start() + 1 :].lstrip()
        if not left or not right:
            break
        chunks.append(left)
        current = right
        split_count += 1
    chunks.append(current)
    return chunks


def _split_embedded_marker_switches(
    start_marker: str,
    content: str,
    allowed_markers: set[str],
) -> Tuple[List[Tuple[str, str]], int, int]:
    marker = str(start_marker or "").strip().upper()
    remaining = str(content or "")
    frames: List[Tuple[str, str]] = []
    split_count = 0
    dangling_count = 0

    while True:
        m = _INLINE_MARKER_RE.search(remaining)
        if not m:
            tail = remaining.strip()
            if tail:
                frames.append((marker, tail))
            break

        next_marker = m.group(1)
        if next_marker not in allowed_markers:
            # Keep unknown marker-like text as literal content.
            tail = remaining.strip()
            if tail:
                frames.append((marker, tail))
            break

        head = remaining[: m.start()].strip()
        if head:
            frames.append((marker, head))
        split_count += 1
        marker = next_marker
        remaining = remaining[m.end() :].lstrip()

        if not remaining.strip():
            dangling_count += 1
            break

    return frames, split_count, dangling_count


def parse_framelines_v2(raw_text: str, *, template_id: str, allow_repairs: bool = True) -> ParsedFrameLines:
    ctx = template_context(template_id)
    lines = _safe_lines(raw_text)
    diagnostics: Dict[str, Any] = {
        "template_id": ctx.template_id,
        "repairs": [],
        "dropped_lines": 0,
        "coerced_lines": 0,
        "split_lines": 0,
        "fallback_inserted": False,
    }

    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _EXACT_END_RE.match(line):
            end_idx = i
            break
    visible_lines = lines if end_idx is None else lines[:end_idx]
    post_end_tail = "" if end_idx is None else "\n".join(lines[end_idx + 1 :]).strip()
    had_end_marker = end_idx is not None

    parsed_frames: List[Tuple[str, str]] = []
    pending = list(visible_lines)
    while pending:
        line = pending.pop(0).strip()
        if not line:
            continue
        split_lines = _split_multi_marker_line(line, ctx.allowed_markers)
        if len(split_lines) > 1:
            diagnostics["split_lines"] = int(diagnostics["split_lines"]) + 1
        for split_line in split_lines:
            m = _FRAME_MARKER_RE.match(split_line)
            if m:
                marker = m.group(1)
                content = m.group(2).strip()
                if marker in ctx.allowed_markers and content:
                    split_frames, embedded_splits, dangling = _split_embedded_marker_switches(
                        marker,
                        content,
                        ctx.allowed_markers,
                    )
                    if split_frames:
                        parsed_frames.extend(split_frames)
                        if embedded_splits:
                            diagnostics["split_lines"] = int(diagnostics["split_lines"]) + embedded_splits
                            diagnostics["repairs"].append("split_embedded_marker_switch")
                        if dangling:
                            diagnostics["repairs"].append("removed_dangling_marker_token")
                        continue
            if not allow_repairs:
                diagnostics["dropped_lines"] = int(diagnostics["dropped_lines"]) + 1
                continue
            lowered = split_line.lower()
            near_map = {
                "[[speech]]": "S",
                "[[thought]]": "T",
                "[[think]]": "T",
                "[[action]]": "A",
                "[[narration]]": "N",
            }
            repaired = None
            for k, v in near_map.items():
                if lowered.startswith(k + " "):
                    repaired = (v, split_line[len(k) :].strip())
                    break
            if repaired and repaired[0] in ctx.allowed_markers and repaired[1]:
                parsed_frames.append(repaired)
                diagnostics["coerced_lines"] = int(diagnostics["coerced_lines"]) + 1
                diagnostics["repairs"].append("near_miss_marker")
                continue
            if split_line.startswith("[[") and "]]" in split_line:
                diagnostics["dropped_lines"] = int(diagnostics["dropped_lines"]) + 1
                diagnostics["repairs"].append("dropped_invalid_marker_line")
                continue
            parsed_frames.append((ctx.default_marker, split_line))
            diagnostics["coerced_lines"] = int(diagnostics["coerced_lines"]) + 1
            diagnostics["repairs"].append("coerced_unmarked_line")

    present = {marker for marker, _ in parsed_frames}
    if ctx.required_markers - present:
        fallback_text = "Sorry, I had trouble formatting that response."
        parsed_frames.insert(0, (sorted(ctx.required_markers)[0], fallback_text))
        diagnostics["fallback_inserted"] = True

    canonical_lines = [f"[[{m}]] {t.strip()}" for m, t in parsed_frames if t.strip()]
    canonical_lines.append("[[E]]")
    canonical_text = "\n".join(canonical_lines).strip()
    segments = [
        StructuredSegment(channel=_marker_to_channel(marker, ctx.template_id), text=text)
        for marker, text in parsed_frames
        if text
    ]
    visible_text = "\n".join(visible_lines).strip()
    if not had_end_marker:
        diagnostics["repairs"].append("missing_end_marker_appended")
    diagnostics["had_end_marker"] = had_end_marker
    diagnostics["post_end_tail_present"] = bool(post_end_tail)
    return ParsedFrameLines(
        canonical_text=canonical_text,
        segments=segments,
        diagnostics=diagnostics,
        visible_text=visible_text,
        post_end_tail=post_end_tail,
        had_end_marker=had_end_marker,
    )


def parse_markdown_v1(raw_text: str) -> ParsedMarkdown:
    repaired_text, repaired_inline = _repair_markdown_terminator_layout(raw_text)
    visible, tail, had_end = extract_visible_and_tail(repaired_text, format_id=FORMAT_MARKDOWN_V1)
    diagnostics = {
        "format": FORMAT_MARKDOWN_V1,
        "missing_end_marker": (not had_end),
        "post_end_tail_present": bool(tail),
        "inline_terminator_repaired": bool(repaired_inline),
    }
    return ParsedMarkdown(
        canonical_text=visible.strip(),
        diagnostics=diagnostics,
        visible_text=visible.strip(),
        post_end_tail=tail,
        had_end_marker=had_end,
    )


def tagxml_to_framelines_v2(raw_text: str, *, template_id: str) -> ParsedFrameLines:
    parsed = parse_structured_response(raw_text or "")
    lines: List[str] = []
    for seg in parsed.segments:
        marker = _channel_to_marker(seg.channel)
        lines.append(f"[[{marker}]] {seg.text}")
    lines.append("[[E]]")
    canonical = "\n".join([line for line in lines if line.strip()])
    return parse_framelines_v2(canonical, template_id=template_id, allow_repairs=True)


def _segments_to_markdown(segments: Iterable[StructuredSegment], *, template_id: str) -> str:
    template = _effective_template(template_id)
    lines: List[str] = []
    for seg in segments:
        text = str(seg.text or "").strip()
        if not text:
            continue
        ch = str(seg.channel or "").strip().lower()
        if ch in {"physicalaction", "action"}:
            if template in {"A", "D"}:
                lines.append(f"**{text}**")
            else:
                lines.append(text)
        elif ch == "innerthought":
            if template == "A":
                lines.append(f"*{text}*")
            else:
                lines.append(text)
        elif ch == "narration":
            if template == "B":
                lines.append(f"> {text}")
            else:
                lines.append(text)
        else:
            lines.append(text)
    return "\n".join(lines).strip()


def tagxml_to_markdown_v1(raw_text: str, *, template_id: str) -> ParsedMarkdown:
    parsed = parse_structured_response(raw_text or "")
    canonical = _segments_to_markdown(parsed.segments, template_id=template_id)
    return parse_markdown_v1(canonical)


def framelines_to_markdown_v1(raw_text: str, *, template_id: str) -> ParsedMarkdown:
    parsed = parse_framelines_v2(raw_text, template_id=template_id, allow_repairs=True)
    canonical = _segments_to_markdown(parsed.segments, template_id=template_id)
    return parse_markdown_v1(canonical)


def plain_to_markdown_v1(raw_text: str, *, template_id: str) -> ParsedMarkdown:
    _ = template_id
    return parse_markdown_v1(raw_text)


def plain_to_framelines_v2(raw_text: str, *, template_id: str) -> ParsedFrameLines:
    ctx = template_context(template_id)
    lines = _safe_lines(raw_text)
    seeded = "\n".join([f"[[{ctx.default_marker}]] {line}" for line in lines] + ["[[E]]"])
    return parse_framelines_v2(seeded, template_id=template_id, allow_repairs=True)


def normalize_to_framelines_v2(
    raw_text: str,
    *,
    template_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ParsedFrameLines:
    fmt = detect_assistant_output_format(raw_text, metadata)
    if fmt == FORMAT_FRAMELINES_V2:
        return parse_framelines_v2(raw_text, template_id=template_id, allow_repairs=True)
    if fmt == FORMAT_LEGACY_XML_V1:
        return tagxml_to_framelines_v2(raw_text, template_id=template_id)
    if fmt == FORMAT_MARKDOWN_V1:
        return markdown_to_framelines_v2(raw_text, template_id=template_id)
    return plain_to_framelines_v2(raw_text, template_id=template_id)


def _markdown_to_segments(raw_text: str, *, template_id: str) -> Tuple[List[StructuredSegment], Dict[str, Any]]:
    template = _effective_template(template_id)
    md = parse_markdown_v1(raw_text)
    lines = _safe_lines(md.canonical_text)
    segments: List[StructuredSegment] = []
    lossy = False
    action_channel = _action_channel_for_template(template)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            text = stripped.lstrip(">").strip()
            if template == "B":
                segments.append(StructuredSegment(channel="narration", text=text))
            else:
                lossy = True
                segments.append(StructuredSegment(channel="speech", text=text))
            continue
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            text = stripped[2:-2].strip()
            if template in {"A", "D"}:
                segments.append(StructuredSegment(channel=action_channel, text=text))
            else:
                lossy = True
                segments.append(StructuredSegment(channel="speech", text=text))
            continue
        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            text = stripped[1:-1].strip()
            if template == "A":
                segments.append(StructuredSegment(channel="innerthought", text=text))
            else:
                lossy = True
                segments.append(StructuredSegment(channel="speech", text=text))
            continue
        segments.append(StructuredSegment(channel="speech", text=stripped))
    return segments, {
        "conversion_lossy": lossy,
        "source_format": FORMAT_MARKDOWN_V1,
        "target_template": template,
        "post_end_tail_present": bool(md.post_end_tail),
    }


def markdown_to_framelines_v2(raw_text: str, *, template_id: str) -> ParsedFrameLines:
    segments, diag = _markdown_to_segments(raw_text, template_id=template_id)
    canonical = segments_to_framelines_v2(segments, template_id=template_id)
    parsed = parse_framelines_v2(canonical, template_id=template_id, allow_repairs=True)
    parsed.diagnostics.update(diag)
    return parsed


def markdown_to_tagxml_v1(raw_text: str, *, template_id: str) -> str:
    segments, _diag = _markdown_to_segments(raw_text, template_id=template_id)
    return serialize_structured_response(segments)


def normalize_to_markdown_v1(
    raw_text: str,
    *,
    template_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ParsedMarkdown:
    fmt = detect_assistant_output_format(raw_text, metadata)
    if fmt == FORMAT_MARKDOWN_V1:
        return parse_markdown_v1(raw_text)
    if fmt == FORMAT_LEGACY_XML_V1:
        return tagxml_to_markdown_v1(raw_text, template_id=template_id)
    if fmt == FORMAT_FRAMELINES_V2:
        return framelines_to_markdown_v1(raw_text, template_id=template_id)
    return plain_to_markdown_v1(raw_text, template_id=template_id)


def normalize_to_legacy_xml_v1(
    raw_text: str,
    *,
    template_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> StructuredResponse:
    fmt = detect_assistant_output_format(raw_text, metadata)
    if fmt == FORMAT_LEGACY_XML_V1:
        return parse_structured_response(raw_text or "")
    if fmt == FORMAT_FRAMELINES_V2:
        segments = parse_framelines_v2(raw_text, template_id=template_id, allow_repairs=True).segments
        return parse_structured_response(serialize_structured_response(segments))
    if fmt == FORMAT_MARKDOWN_V1:
        return parse_structured_response(markdown_to_tagxml_v1(raw_text, template_id=template_id))
    # plain -> speech wrapper
    return parse_structured_response(serialize_structured_response([StructuredSegment(channel="speech", text=str(raw_text or "").strip())]))


def normalize_to_mode(
    raw_text: str,
    *,
    target_mode: str,
    template_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mode = str(target_mode or FORMAT_MARKDOWN_V1).strip().lower()
    if mode == FORMAT_MARKDOWN_V1:
        parsed = normalize_to_markdown_v1(raw_text, template_id=template_id, metadata=metadata)
        return {
            "format": FORMAT_MARKDOWN_V1,
            "canonical_text": parsed.canonical_text,
            "post_end_tail": parsed.post_end_tail,
            "had_end_marker": parsed.had_end_marker,
            "diagnostics": dict(parsed.diagnostics or {}),
        }
    if mode == FORMAT_LEGACY_XML_V1:
        parsed = normalize_to_legacy_xml_v1(raw_text, template_id=template_id, metadata=metadata)
        canonical = serialize_structured_response(parsed.segments)
        return {
            "format": FORMAT_LEGACY_XML_V1,
            "canonical_text": canonical,
            "post_end_tail": "",
            "had_end_marker": "</assistant_response>" in canonical,
            "diagnostics": {
                "is_fallback": bool(parsed.is_fallback),
                "parse_error": parsed.parse_error,
                "had_untagged": bool(parsed.had_untagged),
            },
        }
    parsed = normalize_to_framelines_v2(raw_text, template_id=template_id, metadata=metadata)
    return {
        "format": FORMAT_FRAMELINES_V2,
        "canonical_text": parsed.canonical_text,
        "post_end_tail": parsed.post_end_tail,
        "had_end_marker": parsed.had_end_marker,
        "diagnostics": dict(parsed.diagnostics or {}),
    }


def render_for_ui(canonical_text: str, *, format_id: str, template_id: str) -> str:
    fmt = str(format_id or "").strip().lower()
    if fmt == FORMAT_MARKDOWN_V1:
        return normalize_to_markdown_v1(canonical_text, template_id=template_id).canonical_text
    if fmt == FORMAT_LEGACY_XML_V1:
        return serialize_structured_response(normalize_to_legacy_xml_v1(canonical_text, template_id=template_id).segments)
    return framelines_to_tagxml_render(canonical_text, template_id=template_id)


def framelines_to_tagxml_render(text: str, *, template_id: str) -> str:
    parsed = parse_framelines_v2(text, template_id=template_id, allow_repairs=True)
    return serialize_structured_response(parsed.segments)


def framelines_to_segments(text: str, *, template_id: str) -> List[StructuredSegment]:
    return parse_framelines_v2(text, template_id=template_id, allow_repairs=True).segments


def segments_to_framelines_v2(segments: Iterable[StructuredSegment], *, template_id: str) -> str:
    lines: List[str] = []
    for seg in segments:
        marker = _channel_to_marker(seg.channel)
        text = str(seg.text or "").strip()
        if not text:
            continue
        lines.append(f"[[{marker}]] {text}")
    lines.append("[[E]]")
    return "\n".join(lines).strip()


class ThinkingCaptureProcessor:
    """Delta-based thinking capture/suppression processor."""

    _OPEN_TAGS = ("think", "analysis", "thinking")

    def __init__(self) -> None:
        self.state = ThinkingCaptureState()

    def process_delta(self, delta: str) -> ThinkingDeltaResult:
        text = self.state.boundary_buffer + str(delta or "")
        visible: List[str] = []
        reasoning: List[str] = []
        idx = 0
        while idx < len(text):
            if self.state.in_reasoning:
                close_tag = f"</{self.state.active_tag}>"
                cidx = text.find(close_tag, idx)
                if cidx == -1:
                    reasoning.append(text[idx:])
                    idx = len(text)
                    break
                reasoning.append(text[idx:cidx])
                idx = cidx + len(close_tag)
                self.state.in_reasoning = False
                self.state.active_tag = None
                continue

            nearest_open: Optional[Tuple[int, str]] = None
            for tag in self._OPEN_TAGS:
                token = f"<{tag}>"
                pos = text.find(token, idx)
                if pos != -1 and (nearest_open is None or pos < nearest_open[0]):
                    nearest_open = (pos, tag)
            if nearest_open is None:
                visible.append(text[idx:])
                idx = len(text)
                break
            open_pos, open_tag = nearest_open
            visible.append(text[idx:open_pos])
            idx = open_pos + len(f"<{open_tag}>")
            self.state.in_reasoning = True
            self.state.active_tag = open_tag

        if not self.state.in_reasoning:
            # Keep a tiny boundary buffer to tolerate split open tags in future deltas.
            keep = min(16, len("".join(visible)))
            visible_text = "".join(visible)
            self.state.boundary_buffer = visible_text[-keep:] if keep else ""
            emit_visible = visible_text[:-keep] if keep else visible_text
        else:
            emit_visible = "".join(visible)
            self.state.boundary_buffer = ""
        return ThinkingDeltaResult(
            visible_delta=emit_visible,
            reasoning_delta="".join(reasoning),
            state=self.state,
        )

    def finalize(self) -> ThinkingDeltaResult:
        trailing = self.state.boundary_buffer
        self.state.boundary_buffer = ""
        if self.state.in_reasoning:
            return ThinkingDeltaResult(visible_delta="", reasoning_delta=trailing, state=self.state)
        return ThinkingDeltaResult(visible_delta=trailing, reasoning_delta="", state=self.state)


def capture_and_strip_thinking(raw_text: str) -> Tuple[str, str, Dict[str, Any]]:
    processor = ThinkingCaptureProcessor()
    step = processor.process_delta(raw_text or "")
    end = processor.finalize()
    visible = (step.visible_delta or "") + (end.visible_delta or "")
    reasoning = (step.reasoning_delta or "") + (end.reasoning_delta or "")
    diag = {
        "reasoning_suppressed": bool(reasoning.strip()),
        "reasoning_unclosed": bool(processor.state.in_reasoning),
        "reasoning_chars": len(reasoning),
    }
    return visible, reasoning, diag
