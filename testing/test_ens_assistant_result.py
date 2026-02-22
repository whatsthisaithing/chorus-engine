from chorus_engine.ens.assistant_result import normalize_assistant_result


def test_assistant_result_parses_sentinel_tool_calls_and_control():
    raw = (
        "Visible response text.\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"c1","tool":"image.generate","args":{"prompt":"sunset"}}],'
        '"control":{"action":"yield","args":{"reason":"step_done"}}}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    result = normalize_assistant_result(raw_content=raw)

    assert result.display_text.strip() == "Visible response text."
    assert result.payload_present is True
    assert result.payload_parseable is True
    assert result.control is not None
    assert result.control.action == "YIELD"
    assert len(result.tool_requests) == 1
    assert result.tool_requests[0].tool_name == "image.generate"


def test_assistant_result_prefers_provider_structured_fields_over_sentinel():
    raw = (
        "Visible response text.\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"c1","tool":"image.generate","args":{"prompt":"old"}}],'
        '"control":{"action":"YIELD"}}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    result = normalize_assistant_result(
        raw_content=raw,
        provider_control={"action": "CONTINUE", "args": {"k": "v"}},
        provider_tool_requests=[{"tool_name": "video.generate", "id": "p1", "args": {"prompt": "new"}}],
    )

    assert result.control is not None
    assert result.control.action == "CONTINUE"
    assert len(result.tool_requests) == 1
    assert result.tool_requests[0].tool_name == "video.generate"


def test_assistant_result_ignores_malformed_control():
    raw = (
        "text\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[],"control":{"action":"DROP_TABLES"}}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    result = normalize_assistant_result(raw_content=raw)
    assert result.control is None


def test_assistant_result_parses_relaxed_sentinel_markers_without_leaking_payload():
    raw = (
        "Visible text first.\n"
        "CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"c2","tool":"image.generate","args":{"prompt":"mountain"}}]}\n'
        "CHORUS_TOOL_PAYLOAD_END---"
    )
    result = normalize_assistant_result(raw_content=raw)
    assert result.display_text.strip() == "Visible text first."
    assert result.payload_present is True
    assert result.payload_parseable is True
    assert len(result.tool_requests) == 1
    assert result.tool_requests[0].tool_name == "image.generate"
