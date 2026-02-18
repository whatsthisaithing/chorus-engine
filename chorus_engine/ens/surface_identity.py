"""Canonical surface identity helpers for ENS ingress/egress."""

from __future__ import annotations

from typing import Any

CANONICAL_SURFACE_IDS = {"web", "discord", "telegram", "voice", "sms", "unknown"}


def canonicalize_surface_id(surface_id: Any) -> str:
    value = str(surface_id or "").strip().lower()
    if value in CANONICAL_SURFACE_IDS:
        return value
    return "unknown"
