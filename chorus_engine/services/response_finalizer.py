"""Response finalization helpers shared across assistant output paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

from chorus_engine.services.structured_response import (
    StructuredResponse,
    StructuredSegment,
    parse_structured_response,
    serialize_structured_response,
)


_XML_ROOT_OPEN = "<assistant_response>"
_XML_ROOT_CLOSE = "</assistant_response>"
_FRAME_TERMINATOR = "E:"


def _truncate_at_terminator(raw_text: str) -> tuple[str, Optional[str], bool]:
    text = str(raw_text or "")
    idx_xml = text.find(_XML_ROOT_CLOSE)
    idx_frame = text.find(_FRAME_TERMINATOR)

    candidates: List[tuple[int, int, str]] = []
    if idx_xml >= 0:
        candidates.append((idx_xml, idx_xml + len(_XML_ROOT_CLOSE), _XML_ROOT_CLOSE))
    if idx_frame >= 0:
        candidates.append((idx_frame, idx_frame, _FRAME_TERMINATOR))

    if not candidates:
        return text, None, False

    start_idx, end_idx, terminator = sorted(candidates, key=lambda item: item[0])[0]
    cutoff = end_idx if terminator == _XML_ROOT_CLOSE else start_idx
    cutoff = max(0, min(cutoff, len(text)))
    truncated = text[:cutoff]
    return truncated, terminator, (cutoff < len(text))


class ContractAdapter(Protocol):
    def parse(self, raw_text: str, **kwargs: Any) -> Any:
        ...

    def canonicalize(self, parsed: Any) -> str:
        ...

    def diagnostics(self, parsed: Any) -> Dict[str, Any]:
        ...


@dataclass
class XmlParsedContract:
    raw_text: str
    terminated_text: str
    structured: StructuredResponse
    diagnostics_payload: Dict[str, Any] = field(default_factory=dict)


class XmlContractAdapter:
    """Adapter for the existing XML structured response contract."""

    def parse(
        self,
        raw_text: str,
        *,
        allowed_channels: Optional[Set[str]] = None,
        required_channels: Optional[Set[str]] = None,
    ) -> XmlParsedContract:
        raw = str(raw_text or "")
        terminated_text, terminator, truncated = _truncate_at_terminator(raw)
        structured = parse_structured_response(
            terminated_text,
            allowed_channels=allowed_channels,
            required_channels=required_channels,
        )
        missing_end_marker = (_XML_ROOT_OPEN in terminated_text) and (_XML_ROOT_CLOSE not in terminated_text)
        diag = {
            "adapter": "xml",
            "terminator": terminator,
            "truncated_after_terminator": bool(truncated),
            "multi_root": raw.count(_XML_ROOT_OPEN) > 1,
            "missing_end_marker": bool(missing_end_marker),
            "trailing_text_dropped": bool(structured.trailing_text_dropped),
            "unknown_tags": list(structured.unknown_tags or []),
            "had_untagged": bool(structured.had_untagged),
            "parse_error": structured.parse_error,
            "is_fallback": bool(structured.is_fallback),
        }
        return XmlParsedContract(
            raw_text=raw,
            terminated_text=terminated_text,
            structured=structured,
            diagnostics_payload=diag,
        )

    def canonicalize(self, parsed: XmlParsedContract) -> str:
        return serialize_structured_response(parsed.structured.segments)

    def diagnostics(self, parsed: XmlParsedContract) -> Dict[str, Any]:
        return dict(parsed.diagnostics_payload or {})


@dataclass
class FrameLinesParsedContract:
    raw_text: str
    terminated_text: str
    lines: List[str]
    diagnostics_payload: Dict[str, Any] = field(default_factory=dict)


class FrameLinesAdapter:
    """Stub adapter for future frame-line contract migration."""

    def parse(self, raw_text: str, **kwargs: Any) -> FrameLinesParsedContract:
        _ = kwargs
        raw = str(raw_text or "")
        terminated_text, terminator, truncated = _truncate_at_terminator(raw)
        lines = [line.rstrip() for line in terminated_text.splitlines()]
        diag = {
            "adapter": "frame_lines",
            "terminator": terminator,
            "truncated_after_terminator": bool(truncated),
            "multi_root": False,
            "missing_end_marker": False,
            "stub": True,
        }
        return FrameLinesParsedContract(
            raw_text=raw,
            terminated_text=terminated_text,
            lines=lines,
            diagnostics_payload=diag,
        )

    def canonicalize(self, parsed: FrameLinesParsedContract) -> str:
        return "\n".join(parsed.lines).strip()

    def diagnostics(self, parsed: FrameLinesParsedContract) -> Dict[str, Any]:
        return dict(parsed.diagnostics_payload or {})


@dataclass
class FinalizedResponse:
    text: str
    adapter_name: str
    diagnostics: Dict[str, Any]
    segments: List[StructuredSegment] = field(default_factory=list)
    is_fallback: bool = False
    parse_error: Optional[str] = None
    had_untagged: bool = False
    unknown_tags: List[str] = field(default_factory=list)
    trailing_text_dropped: bool = False


class ResponseFinalizer:
    """Facade that finalizes assistant output using contract adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[str, ContractAdapter] = {
            "xml": XmlContractAdapter(),
            "frame_lines": FrameLinesAdapter(),
        }

    def finalize(
        self,
        raw_text: str,
        *,
        adapter_name: str = "xml",
        allowed_channels: Optional[Set[str]] = None,
        required_channels: Optional[Set[str]] = None,
    ) -> FinalizedResponse:
        adapter = self._adapters.get(adapter_name, self._adapters["xml"])
        parsed = adapter.parse(
            raw_text,
            allowed_channels=allowed_channels,
            required_channels=required_channels,
        )
        canonical_text = adapter.canonicalize(parsed)
        diagnostics = adapter.diagnostics(parsed)

        if isinstance(parsed, XmlParsedContract):
            return FinalizedResponse(
                text=canonical_text,
                adapter_name=adapter_name,
                diagnostics=diagnostics,
                segments=list(parsed.structured.segments or []),
                is_fallback=bool(parsed.structured.is_fallback),
                parse_error=parsed.structured.parse_error,
                had_untagged=bool(parsed.structured.had_untagged),
                unknown_tags=list(parsed.structured.unknown_tags or []),
                trailing_text_dropped=bool(parsed.structured.trailing_text_dropped),
            )

        return FinalizedResponse(
            text=canonical_text,
            adapter_name=adapter_name,
            diagnostics=diagnostics,
        )
