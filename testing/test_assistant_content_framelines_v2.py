from chorus_engine.services.assistant_content import (
    FORMAT_FRAMELINES_V2,
    FORMAT_LEGACY_XML_V1,
    FORMAT_PLAIN_V0,
    ThinkingCaptureProcessor,
    detect_assistant_output_format,
    normalize_to_markdown_v1,
    normalize_to_framelines_v2,
)


def test_detect_format_prefers_metadata():
    fmt = detect_assistant_output_format("<assistant_response><speech>x</speech></assistant_response>", {"assistant_output_format": "framelines_v2"})
    assert fmt == FORMAT_FRAMELINES_V2


def test_detect_format_sniffs_tagxml():
    fmt = detect_assistant_output_format("<assistant_response><speech>x</speech></assistant_response>", {})
    assert fmt == FORMAT_LEGACY_XML_V1


def test_detect_format_sniffs_plain():
    fmt = detect_assistant_output_format("hello there", {})
    assert fmt == FORMAT_PLAIN_V0


def test_tagxml_action_and_physicalaction_map_to_a_marker():
    raw = (
        "<assistant_response>"
        "<physicalaction>waves</physicalaction>"
        "<action>steps back</action>"
        "</assistant_response>"
    )
    parsed = normalize_to_framelines_v2(raw, template_id="A")
    assert "[[A]] waves" in parsed.canonical_text
    assert "[[A]] steps back" in parsed.canonical_text
    assert parsed.canonical_text.strip().endswith("[[E]]")


def test_parse_preserves_post_end_tail():
    raw = "[[S]] hello\n[[E]]\n---CHORUS_TOOL_PAYLOAD_BEGIN---\n{}\n---CHORUS_TOOL_PAYLOAD_END---"
    parsed = normalize_to_framelines_v2(raw, template_id="C")
    assert parsed.post_end_tail.startswith("---CHORUS_TOOL_PAYLOAD_BEGIN---")
    assert parsed.canonical_text == "[[S]] hello\n[[E]]"


def test_thinking_capture_handles_split_tags_and_unclosed():
    proc = ThinkingCaptureProcessor()
    a = proc.process_delta("abc<th")
    b = proc.process_delta("ink>hidden")
    c = proc.process_delta("</think>z<analysis>k")
    d = proc.finalize()
    visible = a.visible_delta + b.visible_delta + c.visible_delta + d.visible_delta
    reasoning = a.reasoning_delta + b.reasoning_delta + c.reasoning_delta + d.reasoning_delta
    assert "abcz" in visible
    assert "hidden" in reasoning
    assert "k" in reasoning


def test_parse_repairs_embedded_marker_switches_and_dangling_marker_tokens():
    raw = (
        "[[A]] she waves [[S]] hello there\n"
        "[[A]] steps closer [[S]]\n"
        "[[E]]"
    )
    parsed = normalize_to_framelines_v2(raw, template_id="D")
    lines = parsed.canonical_text.splitlines()
    assert "[[A]] she waves" in lines
    assert "[[S]] hello there" in lines
    assert "[[A]] steps closer" in lines
    assert not any(line.strip() == "[[S]]" for line in lines)
    assert parsed.canonical_text.strip().endswith("[[E]]")


def test_parse_uses_marker_first_lines_not_blank_line_paragraphs():
    raw = "[[S]] first line\n\n[[A]] second line\n[[E]]"
    parsed = normalize_to_framelines_v2(raw, template_id="A")
    assert parsed.canonical_text == "[[S]] first line\n[[A]] second line\n[[E]]"


def test_markdown_inline_terminator_is_repaired_to_own_line():
    raw = "Final sentence here. ---CHORUS_END---"
    parsed = normalize_to_markdown_v1(raw, template_id="A")
    assert parsed.canonical_text == "Final sentence here."
    assert parsed.had_end_marker is True
    assert parsed.diagnostics.get("missing_end_marker") is False
    assert parsed.diagnostics.get("inline_terminator_repaired") is True


def test_markdown_trailing_text_after_inline_terminator_goes_to_tail():
    raw = "Hello there. ---CHORUS_END--- extra tail text"
    parsed = normalize_to_markdown_v1(raw, template_id="C")
    assert parsed.canonical_text == "Hello there."
    assert parsed.post_end_tail == "extra tail text"
    assert parsed.had_end_marker is True


def test_markdown_split_terminator_repaired_when_dash_precedes_text_only_marker():
    raw = "Visible text.\n---\n\nCHORUS_END"
    parsed = normalize_to_markdown_v1(raw, template_id="D")
    assert parsed.canonical_text == "Visible text."
    assert parsed.had_end_marker is True
    assert parsed.diagnostics.get("missing_end_marker") is False
    assert parsed.diagnostics.get("inline_terminator_repaired") is True


def test_markdown_split_terminator_repaired_when_dash_follows_text_only_marker():
    raw = "Visible text.\nCHORUS_END\n---"
    parsed = normalize_to_markdown_v1(raw, template_id="D")
    assert parsed.canonical_text == "Visible text."
    assert parsed.had_end_marker is True
    assert parsed.diagnostics.get("missing_end_marker") is False
    assert parsed.diagnostics.get("inline_terminator_repaired") is True


def test_markdown_text_only_terminator_repaired_without_dashes():
    raw = "Visible text.\nCHORUS_END"
    parsed = normalize_to_markdown_v1(raw, template_id="D")
    assert parsed.canonical_text == "Visible text."
    assert parsed.had_end_marker is True
    assert parsed.diagnostics.get("missing_end_marker") is False
    assert parsed.diagnostics.get("inline_terminator_repaired") is True


def test_markdown_strips_dash_only_line_before_exact_terminator():
    raw = "Visible text.\n---\n---CHORUS_END---"
    parsed = normalize_to_markdown_v1(raw, template_id="A")
    assert parsed.canonical_text == "Visible text."
    assert parsed.had_end_marker is True
    assert parsed.post_end_tail == ""


def test_markdown_strips_dash_only_line_after_exact_terminator():
    raw = "Visible text.\n---CHORUS_END---\n---"
    parsed = normalize_to_markdown_v1(raw, template_id="A")
    assert parsed.canonical_text == "Visible text."
    assert parsed.had_end_marker is True
    assert parsed.post_end_tail == ""
