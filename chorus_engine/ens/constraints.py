"""Constraint evaluation for ENS Slice 0/1."""

from typing import Any, Dict, List


def evaluate_constraints(signal_type: str, app_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return active constraints for the current signal."""
    constraints: List[Dict[str, Any]] = []
    llm_client = app_state.get("llm_client")
    if signal_type in ("user.message", "chat.simple") and not llm_client:
        constraints.append(
            {
                "type": "model_unavailable",
                "reason": "LLM client not initialized",
            }
        )
    return constraints
