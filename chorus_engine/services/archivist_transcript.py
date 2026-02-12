"""
Archivist Transcript Builder

Builds a cleaned transcript for analysis by filtering at the message level
before formatting. Removes visual context payloads, image generation tool
wrappers, deduplicates repeated blocks, and optionally truncates very long
assistant outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Any, Optional
import json


VISUAL_CONTEXT_PATTERN = re.compile(r"\[VISUAL CONTEXT:.*?\]", re.DOTALL)
IMAGE_GEN_BLOCK_PATTERN = re.compile(
    r"(?s)\*{0,2}(IMAGE|VIDEO) BEING GENERATED:.*?(?=\n\n|\Z)"
)
CRITICAL_LINE_PATTERN = re.compile(r"^CRITICAL:.*$", re.MULTILINE)
IMAGE_GEN_LINE_PATTERNS = [
    re.compile(r"^The image IS .*generated.*$", re.MULTILINE),
    re.compile(r"^The video IS .*generated.*$", re.MULTILINE),
]
TOOL_PAYLOAD_PATTERN = re.compile(
    r"---CHORUS_TOOL_PAYLOAD_BEGIN---[\s\S]*?(?:---CHORUS_TOOL_PAYLOAD_END---|\Z)"
)


@dataclass
class ArchivistFilterStats:
    messages_in: int = 0
    messages_out: int = 0
    visual_blocks_removed: int = 0
    image_gen_blocks_removed: int = 0
    duplicate_blocks_removed: int = 0
    assistant_truncated: int = 0
    chars_before: int = 0
    chars_after: int = 0
    markers_added: int = 0
    seen_blocks: set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_in": self.messages_in,
            "messages_out": self.messages_out,
            "visual_blocks_removed": self.visual_blocks_removed,
            "image_gen_blocks_removed": self.image_gen_blocks_removed,
            "duplicate_blocks_removed": self.duplicate_blocks_removed,
            "assistant_truncated": self.assistant_truncated,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "markers_added": self.markers_added,
        }


def _normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _remove_visual_context(content: str, stats: ArchivistFilterStats) -> str:
    blocks = VISUAL_CONTEXT_PATTERN.findall(content)
    if not blocks:
        return content

    unique_count = 0
    for block in blocks:
        normalized = _normalize_block(block)
        if normalized in stats.seen_blocks:
            stats.duplicate_blocks_removed += 1
            continue
        stats.seen_blocks.add(normalized)
        unique_count += 1

    stats.visual_blocks_removed += len(blocks)
    content = VISUAL_CONTEXT_PATTERN.sub("", content).strip()
    if unique_count > 0:
        marker = f"[VISUAL_CONTEXT_PRESENT: {unique_count}]"
        stats.markers_added += 1
        if content:
            content = f"{content}\n{marker}"
        else:
            content = marker
    return content


def _remove_image_gen_blocks(content: str, stats: ArchivistFilterStats) -> str:
    content = TOOL_PAYLOAD_PATTERN.sub("", content).strip()
    blocks = IMAGE_GEN_BLOCK_PATTERN.findall(content)
    if not blocks:
        return content

    matches = list(IMAGE_GEN_BLOCK_PATTERN.finditer(content))
    unique_count = 0
    for match in matches:
        block = match.group(0)
        normalized = _normalize_block(block)
        if normalized in stats.seen_blocks:
            stats.duplicate_blocks_removed += 1
            continue
        stats.seen_blocks.add(normalized)
        unique_count += 1

    stats.image_gen_blocks_removed += len(matches)
    content = IMAGE_GEN_BLOCK_PATTERN.sub("", content).strip()

    # Remove stray tool wrapper lines if any remain
    content = CRITICAL_LINE_PATTERN.sub("", content)
    for pattern in IMAGE_GEN_LINE_PATTERNS:
        content = pattern.sub("", content)
    content = content.strip()

    if unique_count > 0:
        marker = "[IMAGE_GENERATION_OCCURRED]"
        stats.markers_added += 1
        if content:
            content = f"{content}\n{marker}"
        else:
            content = marker
    return content


def _truncate_assistant_monologue(content: str, stats: ArchivistFilterStats) -> str:
    max_len = 4000
    keep_len = 1000
    if len(content) <= max_len:
        return content
    truncated = content[:keep_len].rstrip()
    truncated += f"\n[ASSISTANT_LONG_OUTPUT_TRUNCATED length={len(content)}]"
    stats.assistant_truncated += 1
    stats.markers_added += 1
    return truncated


def filter_archivist_messages(messages: Iterable, role_attr: str = "role") -> tuple[List[Dict[str, str]], ArchivistFilterStats]:
    """
    Filter messages at the message level and return cleaned message dicts.
    """
    stats = ArchivistFilterStats()
    cleaned: List[Dict[str, str]] = []

    for msg in messages:
        stats.messages_in += 1
        role_val = getattr(msg, role_attr)
        role = role_val.value if hasattr(role_val, "value") else str(role_val)
        content = msg.content or ""

        stats.chars_before += len(content)

        # Only apply tool payload removal to assistant messages
        if role.lower() == "assistant":
            content = _remove_image_gen_blocks(content, stats)
            content = _truncate_assistant_monologue(content, stats)

        # Apply visual context filtering to all roles (typically user)
        content = _remove_visual_context(content, stats)

        stats.chars_after += len(content)

        if content.strip():
            cleaned.append({
                "role": role,
                "content": content.strip()
            })

    stats.messages_out = len(cleaned)
    return cleaned, stats


def filter_archivist_messages_by_role(
    messages: Iterable[Dict[str, str]],
    roles: Iterable[str]
) -> List[Dict[str, str]]:
    """
    Filter already-cleaned archivist messages by role (case-insensitive).
    """
    role_set = {role.lower() for role in roles}
    return [
        msg for msg in messages
        if (msg.get("role") or "").strip().lower() in role_set
    ]


def format_archivist_transcript(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def format_archivist_transcript_json(
    messages: List[Dict[str, str]],
    name_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Serialize messages as a JSON array of {role, name?, content}.
    Roles are normalized to: user | assistant | system | tool.
    """
    allowed_roles = {"user", "assistant", "system", "tool"}
    payload: List[Dict[str, str]] = []

    for msg in messages:
        role_raw = (msg.get("role") or "").strip().lower()
        role = role_raw if role_raw in allowed_roles else "tool"
        entry: Dict[str, str] = {
            "role": role,
            "content": msg.get("content", "")
        }
        if name_map and role in name_map and name_map[role]:
            entry["name"] = name_map[role]
        payload.append(entry)

    return json.dumps(payload, ensure_ascii=False)
