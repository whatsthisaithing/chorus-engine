"""Registry for ENS loop-kind plugins."""

from __future__ import annotations

from typing import Dict

from chorus_engine.ens.loop_plugins.contracts import LoopKindPlugin
from chorus_engine.ens.loop_plugins.narrative_v1 import GenericLoopPlugin, NarrativeV1LoopPlugin


_GENERIC = GenericLoopPlugin()
_NARRATIVE_V1 = NarrativeV1LoopPlugin()

_PLUGINS_BY_KIND: Dict[str, LoopKindPlugin] = {
    _GENERIC.kind(): _GENERIC,
    _NARRATIVE_V1.kind(): _NARRATIVE_V1,
}


def get_loop_plugin(loop_kind: str) -> LoopKindPlugin:
    key = str(loop_kind or "").strip()
    if not key:
        return _GENERIC
    return _PLUGINS_BY_KIND.get(key, _GENERIC)


def has_loop_plugin(loop_kind: str) -> bool:
    key = str(loop_kind or "").strip()
    return key in _PLUGINS_BY_KIND


def registered_loop_kinds() -> list[str]:
    return sorted(_PLUGINS_BY_KIND.keys())

