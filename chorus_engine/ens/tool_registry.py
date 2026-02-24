"""Centralized ENS tool registry for native transport, validation defaults, and prompt docs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


TOOL_IMAGE_GENERATE = "image.generate"
TOOL_VIDEO_GENERATE = "video.generate"
TOOL_MOMENT_PIN_COLD_RECALL = "moment_pin.cold_recall"
TOOL_CHORUS_CONTROL = "chorus.control"
TOOL_SCENE_CAPTURE_GENERATE = "scene_capture.generate"


@dataclass(frozen=True)
class ENSToolSpec:
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    requires_approval_default: Optional[bool]
    supports_native_transport: bool
    supports_sentinel_transport: bool
    prompt_doc: str


_IMAGE_VIDEO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "minLength": 1},
        "negative_prompt": {"type": "string"},
        "seed": {"type": "integer"},
        "workflow_id": {"type": ["string", "integer"]},
        "trigger_words": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompt"],
    "additionalProperties": True,
}


_REGISTRY: Dict[str, ENSToolSpec] = {
    TOOL_IMAGE_GENERATE: ENSToolSpec(
        name=TOOL_IMAGE_GENERATE,
        description="Generate an image from a prompt.",
        parameters_schema=_IMAGE_VIDEO_SCHEMA,
        requires_approval_default=True,
        supports_native_transport=True,
        supports_sentinel_transport=True,
        prompt_doc="Use when the user explicitly asks for image generation.",
    ),
    TOOL_VIDEO_GENERATE: ENSToolSpec(
        name=TOOL_VIDEO_GENERATE,
        description="Generate a video from a prompt.",
        parameters_schema=_IMAGE_VIDEO_SCHEMA,
        requires_approval_default=True,
        supports_native_transport=True,
        supports_sentinel_transport=True,
        prompt_doc="Use when the user explicitly asks for video generation.",
    ),
    TOOL_MOMENT_PIN_COLD_RECALL: ENSToolSpec(
        name=TOOL_MOMENT_PIN_COLD_RECALL,
        description="Load archival transcript context for an injected moment pin.",
        parameters_schema={
            "type": "object",
            "properties": {
                "pin_id": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["pin_id", "reason"],
            "additionalProperties": False,
        },
        requires_approval_default=False,
        supports_native_transport=True,
        supports_sentinel_transport=True,
        prompt_doc="Use only when transcript precision is required for an injected moment pin.",
    ),
    TOOL_CHORUS_CONTROL: ENSToolSpec(
        name=TOOL_CHORUS_CONTROL,
        description="Emit ENS loop control action for progression state.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["CONTINUE", "YIELD", "COMPLETE"],
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        requires_approval_default=None,
        supports_native_transport=True,
        supports_sentinel_transport=False,
        prompt_doc="Use in loop steps to select next loop action.",
    ),
    TOOL_SCENE_CAPTURE_GENERATE: ENSToolSpec(
        name=TOOL_SCENE_CAPTURE_GENERATE,
        description="Internal scene capture execution handle.",
        parameters_schema={
            "type": "object",
            "properties": {
                "media_type": {"type": "string", "enum": ["image", "video"]},
                "prompt": {"type": "string"},
            },
            "required": ["media_type", "prompt"],
            "additionalProperties": True,
        },
        requires_approval_default=True,
        supports_native_transport=False,
        supports_sentinel_transport=False,
        prompt_doc="Internal use only.",
    ),
}


def get_tool_spec(name: str) -> Optional[ENSToolSpec]:
    return _REGISTRY.get(str(name or "").strip())


def iter_tool_specs() -> Iterable[ENSToolSpec]:
    return _REGISTRY.values()


def sentinel_media_tools() -> set[str]:
    return {
        spec.name
        for spec in _REGISTRY.values()
        if spec.supports_sentinel_transport and spec.name in {TOOL_IMAGE_GENERATE, TOOL_VIDEO_GENERATE}
    }


def native_tool_definitions(
    *,
    allowed_media_tools: Optional[set[str]] = None,
    include_control: bool = False,
    include_cold_recall: bool = True,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    allowed_media = set(allowed_media_tools or set())
    for spec in _REGISTRY.values():
        if not spec.supports_native_transport:
            continue
        if spec.name == TOOL_CHORUS_CONTROL and not include_control:
            continue
        if spec.name == TOOL_MOMENT_PIN_COLD_RECALL and not include_cold_recall:
            continue
        if spec.name in {TOOL_IMAGE_GENERATE, TOOL_VIDEO_GENERATE} and spec.name not in allowed_media:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters_schema,
                },
            }
        )
    return tools


def requires_approval_default(tool_name: str) -> Optional[bool]:
    spec = get_tool_spec(tool_name)
    return spec.requires_approval_default if spec else None


def prompt_doc_lines(tool_names: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for name in tool_names:
        spec = get_tool_spec(name)
        if not spec:
            continue
        lines.append(f"- `{name}`: {spec.prompt_doc}")
    return lines
