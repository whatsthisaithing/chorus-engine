"""ENS v3 arbitration selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from chorus_engine.models.ens import ENSSignalQueue


@dataclass
class ArbitrationSelection:
    selected: Optional[ENSSignalQueue]
    reason_trace: Dict[str, Any]
    tie_break: Dict[str, Any]


def _stable_order(rows: List[ENSSignalQueue]) -> List[ENSSignalQueue]:
    return sorted(rows, key=lambda r: (int(r.created_at_us or 0), str(r.signal_id)))


class ArbitrationEngine:
    """Deterministic v3-lite arbitration selector.

    Policy in this slice:
    - USER preempts SYSTEM/LOOP.
    - budgets/cooldowns are explicit pass-through stubs.
    - non-user fairness rotates by surface first, relationship second.
    - deterministic tie-break is always (created_at_us, signal_id).
    """

    def select(
        self,
        candidates: List[ENSSignalQueue],
        *,
        last_non_user_surface_id: Optional[str],
        last_non_user_relationship_id: Optional[str],
        floor_locks_by_relationship: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ArbitrationSelection:
        ordered = _stable_order(list(candidates or []))
        floor_locks_by_relationship = dict(floor_locks_by_relationship or {})
        if not ordered:
            return ArbitrationSelection(
                selected=None,
                reason_trace={
                    "selection": "arbitration_v3",
                    "candidate_count": 0,
                    "budgets_check": "pass_stub",
                    "cooldowns_check": "pass_stub",
                },
                tie_break={"applied": False},
            )

        tier_counts = {"user": 0, "system": 0, "loop": 0}
        for row in ordered:
            tier = str(row.priority_tier or "system")
            if tier not in tier_counts:
                tier = "system"
            tier_counts[tier] += 1

        user_rows = [r for r in ordered if str(r.priority_tier) == "user"]
        if user_rows:
            selected = user_rows[0]
            return ArbitrationSelection(
                selected=selected,
                reason_trace={
                    "selection": "arbitration_v3",
                    "phase": "user_preemption",
                    "candidate_count": len(ordered),
                    "tier_counts": tier_counts,
                    "selected_signal_id": selected.signal_id,
                    "selected_priority_tier": selected.priority_tier,
                    "budgets_check": "pass_stub",
                    "cooldowns_check": "pass_stub",
                    "fairness_applied": False,
                },
                tie_break={
                    "applied": len(user_rows) > 1,
                    "created_at_us": selected.created_at_us,
                    "signal_id": selected.signal_id,
                },
            )

        non_user_rows = [r for r in ordered if str(r.priority_tier) in ("system", "loop")]
        fairness_pool = list(non_user_rows)
        fairness_applied = False
        fairness_notes: Dict[str, Any] = {}
        attention_lock_applied = False

        if last_non_user_surface_id:
            different_surface = [
                r for r in fairness_pool if str(r.surface_id or "") != str(last_non_user_surface_id)
            ]
            if different_surface:
                fairness_pool = _stable_order(different_surface)
                fairness_applied = True
                fairness_notes["surface_rotation_applied"] = True
                fairness_notes["last_non_user_surface_id"] = last_non_user_surface_id
            else:
                fairness_notes["surface_rotation_applied"] = False
                fairness_notes["surface_rotation_reason"] = "single_surface_available"

        if fairness_pool and last_non_user_relationship_id:
            different_relationship = [
                r for r in fairness_pool if str(r.relationship_id or "") != str(last_non_user_relationship_id)
            ]
            if different_relationship:
                fairness_pool = _stable_order(different_relationship)
                fairness_applied = True
                fairness_notes["relationship_rotation_applied"] = True
                fairness_notes["last_non_user_relationship_id"] = last_non_user_relationship_id
            else:
                fairness_notes["relationship_rotation_applied"] = False
                fairness_notes["relationship_rotation_reason"] = "single_relationship_available"

        weighted_pool = _stable_order(fairness_pool or non_user_rows)
        if weighted_pool:
            if any(floor_locks_by_relationship.get(str(r.relationship_id or "")) for r in weighted_pool):
                attention_lock_applied = True

            def _attention_lock_penalty(row: ENSSignalQueue) -> int:
                lock = floor_locks_by_relationship.get(str(row.relationship_id or ""))
                if not lock:
                    return 0
                active_surface_id = str(lock.get("active_surface_id") or "")
                if not active_surface_id:
                    return 0
                return 0 if str(row.surface_id or "") == active_surface_id else 1

            weighted_pool = sorted(
                weighted_pool,
                key=lambda r: (
                    _attention_lock_penalty(r),
                    int(r.created_at_us or 0),
                    str(r.signal_id),
                ),
            )

        selected = weighted_pool[0]
        return ArbitrationSelection(
            selected=selected,
            reason_trace={
                "selection": "arbitration_v3",
                "phase": "non_user_fairness_then_tie_break",
                "candidate_count": len(ordered),
                "tier_counts": tier_counts,
                "selected_signal_id": selected.signal_id,
                "selected_priority_tier": selected.priority_tier,
                "selected_surface_id": selected.surface_id,
                "selected_relationship_id": selected.relationship_id,
                "last_non_user_surface_id": last_non_user_surface_id,
                "last_non_user_relationship_id": last_non_user_relationship_id,
                "budgets_check": "pass_stub",
                "cooldowns_check": "pass_stub",
                "fairness_applied": fairness_applied,
                "attention_lock_applied": attention_lock_applied,
                **fairness_notes,
            },
            tie_break={
                "applied": len(weighted_pool) > 1,
                "created_at_us": selected.created_at_us,
                "signal_id": selected.signal_id,
            },
        )
