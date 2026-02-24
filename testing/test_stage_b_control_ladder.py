from chorus_engine.ens.assistant_result import AssistantResult, ControlDirective
from chorus_engine.ens.dispatcher import ENSDispatcher


def _dispatcher(helpers) -> ENSDispatcher:
    return ENSDispatcher(helpers.app_module.app_state)


def test_stage_b_content_parse_extracts_single_action(helpers):
    dispatcher = _dispatcher(helpers)
    result = dispatcher._extract_stage_b_action_from_content('{"action":"continue"}')
    assert result["success"] is True
    assert result["ambiguous"] is False
    assert result["action"] == "CONTINUE"


def test_stage_b_content_parse_marks_ambiguity_for_continue_and_yield(helpers):
    dispatcher = _dispatcher(helpers)
    result = dispatcher._extract_stage_b_action_from_content(
        "```json\n{\"action\":\"CONTINUE\"}\n```\n```json\n{\"action\":\"YIELD\"}\n```"
    )
    assert result["success"] is False
    assert result["ambiguous"] is True
    assert "ambiguous_actions" in str(result["reason"] or "")


def test_stage_b_native_rung_marks_multiple_control_calls_ambiguous(helpers):
    dispatcher = _dispatcher(helpers)
    assistant_result = AssistantResult(
        raw_content="",
        display_text="",
        control=ControlDirective(action="CONTINUE", args={}),
        tool_requests=[],
        payload_present=False,
        payload_parseable=False,
        payload_obj=None,
        provider_raw={
            "provider_tool_calls_raw": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "chorus.control", "arguments": '{"action":"CONTINUE"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "chorus.control", "arguments": '{"action":"YIELD"}'},
                },
            ]
        },
    )
    result = dispatcher._evaluate_stage_b_native_rung(assistant_result)
    assert result["success"] is False
    assert result["ambiguous"] is True
    assert "ambiguous_native_control_actions" in str(result["reason"] or "")

