"""Deterministic loop working-memory compression helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashing/replay stability."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _trim_list(values: List[Any], *, cap: int) -> List[Any]:
    if cap <= 0:
        return []
    return list(values[-cap:])


def build_step_memory_payload(
    *,
    step_index_after: int,
    control_action: Optional[str],
    state_after: str,
    display_text: str,
    tool_requests_allowed: List[str],
    tool_requests_blocked: List[str],
    finish_reason: Optional[str],
    assistant_result_tier: str,
) -> Dict[str, Any]:
    """Build deterministic per-step working-memory payload."""
    safe_display = str(display_text or "")
    return {
        "schema_version": 1,
        "step_index": int(step_index_after),
        "facts": {
            "display_text_sha256": hashlib.sha256(safe_display.encode("utf-8")).hexdigest(),
            "display_text_preview": safe_display[:256],
        },
        "goals": {},
        "decisions": {
            "control_action": (str(control_action) if control_action else None),
            "state_after": str(state_after or ""),
            "assistant_result_tier": str(assistant_result_tier or "unknown"),
            "provider_finish_reason": (str(finish_reason) if finish_reason else None),
        },
        "tool_results": {
            "allowed_tool_names": sorted(str(x) for x in (tool_requests_allowed or [])),
            "blocked_tool_names": sorted(str(x) for x in (tool_requests_blocked or [])),
            "allowed_count": len(tool_requests_allowed or []),
            "blocked_count": len(tool_requests_blocked or []),
        },
        "scratch": [],
    }


def fold_memory_payloads(
    *,
    prior_folded_json: Optional[Dict[str, Any]],
    selected_payloads: List[Dict[str, Any]],
    list_cap: int = 64,
) -> Dict[str, Any]:
    """Deterministically fold prior artifact + newly eligible step payloads."""
    base: Dict[str, Any] = copy.deepcopy(prior_folded_json or {})
    if not isinstance(base, dict):
        base = {}
    base.setdefault("schema_version", 1)
    base.setdefault("facts", {})
    base.setdefault("goals", {})
    base.setdefault("decisions", {})
    base.setdefault("tool_results", {})
    base.setdefault("scratch", [])

    for payload in selected_payloads:
        doc = payload if isinstance(payload, dict) else {}

        for section in ("facts", "goals", "decisions"):
            incoming = doc.get(section) if isinstance(doc.get(section), dict) else {}
            target = base.get(section) if isinstance(base.get(section), dict) else {}
            target.update(incoming)
            base[section] = target

        incoming_tools = doc.get("tool_results") if isinstance(doc.get("tool_results"), dict) else {}
        merged_tools = base.get("tool_results") if isinstance(base.get("tool_results"), dict) else {}
        merged_tools.update(incoming_tools)
        base["tool_results"] = merged_tools

        scratch_items = doc.get("scratch") if isinstance(doc.get("scratch"), list) else []
        merged_scratch = base.get("scratch") if isinstance(base.get("scratch"), list) else []
        merged_scratch.extend(scratch_items)
        base["scratch"] = _trim_list(merged_scratch, cap=list_cap)

    # Normalize list-valued entries in tool_results.
    tool_results = base.get("tool_results") if isinstance(base.get("tool_results"), dict) else {}
    for key, value in list(tool_results.items()):
        if isinstance(value, list):
            tool_results[key] = _trim_list(sorted(value), cap=list_cap)
    base["tool_results"] = tool_results

    return json.loads(canonical_json(base))
