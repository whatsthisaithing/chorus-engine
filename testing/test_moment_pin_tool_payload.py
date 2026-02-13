"""
Basic tests for moment-pin tool payload validation.

Usage:
    python testing/test_moment_pin_tool_payload.py
"""

from chorus_engine.services.tool_payload import (
    validate_cold_recall_payload,
    MOMENT_PIN_COLD_RECALL_TOOL,
)


def test_valid_cold_recall_payload():
    payload = {
        "version": 1,
        "tool_calls": [
            {
                "id": "call_1",
                "tool": MOMENT_PIN_COLD_RECALL_TOOL,
                "requires_approval": False,
                "args": {"pin_id": "1234", "reason": "Need exact quote"},
            }
        ],
    }
    call = validate_cold_recall_payload(payload)
    assert call is not None
    assert call.pin_id == "1234"


def test_reject_requires_approval_true():
    payload = {
        "version": 1,
        "tool_calls": [
            {
                "id": "call_1",
                "tool": MOMENT_PIN_COLD_RECALL_TOOL,
                "requires_approval": True,
                "args": {"pin_id": "1234", "reason": "Need exact quote"},
            }
        ],
    }
    assert validate_cold_recall_payload(payload) is None


def test_reject_chained_calls():
    payload = {
        "version": 1,
        "tool_calls": [
            {
                "id": "call_1",
                "tool": MOMENT_PIN_COLD_RECALL_TOOL,
                "requires_approval": False,
                "args": {"pin_id": "1234", "reason": "Need exact quote"},
            },
            {
                "id": "call_2",
                "tool": "image.generate",
                "requires_approval": True,
                "args": {"prompt": "test"},
            },
        ],
    }
    assert validate_cold_recall_payload(payload) is None


if __name__ == "__main__":
    test_valid_cold_recall_payload()
    test_reject_requires_approval_true()
    test_reject_chained_calls()
    print("moment pin tool payload tests: OK")
