from chorus_engine.ens.assistant_result import AssistantResult, ToolRequest
from chorus_engine.ens.dispatcher import _extract_validated_media_tool_calls_from_retry_assistant_result


def _assistant_result(
    *,
    display_text: str = "",
    payload_obj=None,
    tool_requests=None,
) -> AssistantResult:
    return AssistantResult(
        raw_content=display_text,
        display_text=display_text,
        control=None,
        tool_requests=list(tool_requests or []),
        payload_present=payload_obj is not None,
        payload_parseable=payload_obj is not None,
        payload_obj=payload_obj,
        provider_raw={},
    )


def test_retry_extraction_prefers_normalized_tool_requests():
    result = _assistant_result(
        tool_requests=[
            ToolRequest(
                tool_name="image.generate",
                payload={
                    "id": "native-1",
                    "tool": "image.generate",
                    "requires_approval": True,
                    "args": {"prompt": "native prompt"},
                },
            )
        ]
    )
    payload_obj, validated, meta = _extract_validated_media_tool_calls_from_retry_assistant_result(
        assistant_result=result,
        allowed_tools_set={"image.generate"},
    )
    assert payload_obj is None
    assert len(validated) == 1
    assert validated[0].tool == "image.generate"
    assert meta.get("source") == "normalized"


def test_retry_extraction_falls_back_to_payload_obj():
    payload = {
        "version": 1,
        "tool_calls": [
            {
                "id": "p1",
                "tool": "image.generate",
                "requires_approval": True,
                "args": {"prompt": "payload prompt"},
            }
        ],
    }
    result = _assistant_result(payload_obj=payload)
    payload_obj, validated, meta = _extract_validated_media_tool_calls_from_retry_assistant_result(
        assistant_result=result,
        allowed_tools_set={"image.generate"},
    )
    assert isinstance(payload_obj, dict)
    assert len(validated) == 1
    assert validated[0].prompt == "payload prompt"
    assert meta.get("source") == "payload_obj"


def test_retry_extraction_falls_back_to_content_json():
    content_json = (
        '{"version":1,"tool_calls":[{"id":"c1","tool":"image.generate","requires_approval":true,'
        '"args":{"prompt":"content prompt"}}]}'
    )
    result = _assistant_result(display_text=content_json)
    payload_obj, validated, meta = _extract_validated_media_tool_calls_from_retry_assistant_result(
        assistant_result=result,
        allowed_tools_set={"image.generate"},
    )
    assert isinstance(payload_obj, dict)
    assert len(validated) == 1
    assert validated[0].prompt == "content prompt"
    assert meta.get("source") == "content_json"
