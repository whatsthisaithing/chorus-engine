"""Metadata patch policy for ENS-owned message metadata mutations."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

CANONICAL_SURFACE_IDS = {"web", "discord", "telegram", "voice", "unknown"}

SYSTEM_METADATA_KEYS = {
    # Moderation (note: `system.hidden` is currently reserved metadata only;
    # message visibility is still controlled by messages.deleted_at soft-delete)
    "system.hidden",
    "system.soft_deleted",
    "system.delete_reason",
    "system.edited",
    # Provenance/correlation
    "system.surface_id",
    "system.source",
    "system.speaker_external_id",
    "system.external_thread_id",
    "system.external_message_id",
    "system.ingest_kind",
    "system.client_message_id",
    # Tool/media linkage
    "system.tool_call_id",
    "system.image_id",
    "system.video_id",
    "system.audio_id",
}

PROVENANCE_WRITE_ONCE_KEYS = {
    "system.ingest_kind",
    "system.surface_id",
    "system.source",
    "system.speaker_external_id",
    "system.external_thread_id",
    "system.external_message_id",
    "system.client_message_id",
}


def canonicalize_surface_id(surface_id: Any) -> str:
    value = str(surface_id or "").strip().lower()
    if value in CANONICAL_SURFACE_IDS:
        return value
    return "unknown"


def sanitize_metadata_patch(
    *,
    existing_metadata: Dict[str, Any] | None,
    patch: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    existing = existing_metadata or {}
    incoming = patch or {}
    accepted: Dict[str, Any] = {}
    rejected: List[Dict[str, str]] = []

    for raw_key, value in incoming.items():
        key = str(raw_key)
        if key.startswith("system."):
            if key not in SYSTEM_METADATA_KEYS:
                rejected.append({"key": key, "reason": "system_key_not_allowlisted"})
                continue
            if key in PROVENANCE_WRITE_ONCE_KEYS and key in existing and existing.get(key) != value:
                rejected.append({"key": key, "reason": "write_once_provenance_key"})
                continue
            accepted[key] = value
            continue

        if key.startswith("adapters."):
            parts = key.split(".", 2)
            if len(parts) < 3 or not parts[2]:
                rejected.append({"key": key, "reason": "invalid_adapter_key"})
                continue
            surface_id = canonicalize_surface_id(parts[1])
            normalized_key = f"adapters.{surface_id}.{parts[2]}"
            accepted[normalized_key] = value
            continue

        rejected.append({"key": key, "reason": "must_use_system_or_adapters_namespace"})

    return accepted, rejected
