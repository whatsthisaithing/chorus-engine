from chorus_engine.models.ens import ENSActionResult, ENSDecision
from chorus_engine.models.conversation import Conversation, Message, MessageRole


class _LLMResponse:
    def __init__(self, content: str):
        self.content = content


class _LLMClientWithPayload:
    base_url = "http://test-llm"

    def __init__(self, payload_text: str):
        self.payload_text = payload_text

    async def health_check(self):
        return True

    async def generate_with_history(self, messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        return _LLMResponse(self.payload_text)


def _latest_decision(db):
    return db.query(ENSDecision).order_by(ENSDecision.created_at.desc()).first()


def test_slice25_adjudicate_idempotency_key_uses_user_message_id(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "Tool candidate.\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"img1","tool":"image.generate","requires_approval":true,"args":{"prompt":"cat portrait"}}]}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "please make an image", "metadata": {"client_message_id": "slice25-idem-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    user_message_id = body["user_message"]["id"]

    decision = _latest_decision(db)
    assert decision is not None
    adjudicate = (
        db.query(ENSActionResult)
        .filter(
            ENSActionResult.decision_id == decision.decision_id,
            ENSActionResult.kind == "tool_payload.adjudicate",
        )
        .first()
    )
    assert adjudicate is not None
    assert adjudicate.idempotency_key == f"gate:adjudicate:{decision.session_id}:{user_message_id}"


def test_slice25_actions_have_matching_action_results(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "No tool payload this turn."
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "hello", "metadata": {"client_message_id": "slice25-parity-1"}},
    )
    assert resp.status_code == 200, resp.text

    decision = _latest_decision(db)
    assert decision is not None
    planned_actions = decision.actions_json or []
    planned_action_ids = {item["action_id"] for item in planned_actions}
    assert planned_action_ids

    results = db.query(ENSActionResult).filter(ENSActionResult.decision_id == decision.decision_id).all()
    result_action_ids = {item.action_id for item in results}
    assert planned_action_ids == result_action_ids

    persist = next((r for r in results if r.kind == "tool_call.persist_pending"), None)
    assert persist is not None
    assert persist.status == "skipped"
    assert (persist.output_json or {}).get("reason") == "no_tool_calls"


def test_ens_normalizes_speech_only_response_to_structured_content(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "<speech>Of course! What kind of tests are we diving into today?</speech>"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "test response format", "metadata": {"client_message_id": "slice25-structured-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assistant_id = body["assistant_message"]["id"]

    assistant_row = (
        db.query(Message)
        .filter(Message.id == assistant_id, Message.role == MessageRole.ASSISTANT)
        .first()
    )
    assert assistant_row is not None
    content = assistant_row.content or ""
    assert "Of course! What kind of tests are we diving into today?" in content
    metadata = assistant_row.meta_data or {}
    assert metadata.get("assistant_output_format") == "markdown_v1"
    assert "Of course! What kind of tests are we diving into today?" in str(metadata.get("render_content") or "")


def test_ens_drops_unknown_structured_tags_and_trailing_text(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "<assistant_response><speech>Hello there.</speech><system_notes>drop me</system_notes></assistant_response>\n"
        "---\n"
        "<system_notes>drop this too</system_notes>"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "hello", "metadata": {"client_message_id": "slice25-structured-drop-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assistant_id = body["assistant_message"]["id"]

    assistant_row = (
        db.query(Message)
        .filter(Message.id == assistant_id, Message.role == MessageRole.ASSISTANT)
        .first()
    )
    assert assistant_row is not None
    content = assistant_row.content or ""
    assert content.strip() == "Hello there."

    metadata = assistant_row.meta_data or {}
    assert metadata.get("assistant_output_format") == "markdown_v1"
    render_content = str(metadata.get("render_content") or "")
    assert "Hello there." in render_content
    assert "<system_notes>" not in render_content
    structured = metadata.get("structured_response") or {}
    diag = structured.get("adapter_diagnostics") or {}
    parse_diag = diag.get("parse") or {}
    assert parse_diag.get("missing_end_marker") is True or parse_diag.get("had_end_marker") is True


def test_slice25_explicit_image_request_respects_disable_confirmation_setting(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "Sure.\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"img-explicit","tool":"image.generate","requires_approval":true,"args":{"prompt":"selfie portrait"}}]}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    assert conversation is not None
    conversation.image_confirmation_disabled = "true"
    db.commit()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "please send a selfie", "metadata": {"client_message_id": "slice25-disable-confirm-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body.get("pending_tool_calls") or []) == 1
    pending = body["pending_tool_calls"][0]
    assert pending["tool"] == "image.generate"
    assert pending["classification"] == "explicit_request"
    assert pending["needs_confirmation"] is False


def test_slice25_malformed_non_sentinel_payload_is_stripped_and_not_accepted(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "<assistant_response><speech>Goodbye for now.</speech></assistant_response>\n\n"
        "**Image generation payload:**\n\n"
        "```json\n"
        "{\n"
        "  \"version\": 1,\n"
        "  \"tool_calls\": [\n"
        "    {\n"
        "      \"id\": \"selfie_generation_20260215\",\n"
        "      \"tool\": \"image.generate\",\n"
        "      \"requires_approval\": true,\n"
        "      \"args\": {\"prompt\": \"A portrait\"}\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "bye for now", "metadata": {"client_message_id": "slice25-malformed-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending_tool_calls"] == []

    assistant_id = body["assistant_message"]["id"]
    assistant_row = (
        db.query(Message)
        .filter(Message.id == assistant_id, Message.role == MessageRole.ASSISTANT)
        .first()
    )
    assert assistant_row is not None
    content = assistant_row.content or ""
    assert "Image generation payload" not in content
    assert "```json" not in content

    metadata = assistant_row.meta_data or {}
    structured = metadata.get("structured_response") or {}
    assert "Image generation payload" in (structured.get("raw_response") or "")

    decision = _latest_decision(db)
    adjudicate = (
        db.query(ENSActionResult)
        .filter(
            ENSActionResult.decision_id == decision.decision_id,
            ENSActionResult.kind == "tool_payload.adjudicate",
        )
        .first()
    )
    assert adjudicate is not None
    output = adjudicate.output_json or {}
    assert output.get("tool_call_count") == 0
    assert output.get("malformed_tool_payload_non_sentinel") is True
    assert output.get("tool_parse_status") == "malformed_non_sentinel"


def test_slice25_iteration_not_triggered_by_generic_another_without_recent_media_context(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "<assistant_response><speech>Copy that.</speech></assistant_response>"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    # Old media turn (outside recent window of 5 messages)
    old_assistant = Message(
        thread_id=thread_id,
        role=MessageRole.ASSISTANT,
        content="<assistant_response><speech>old media turn</speech></assistant_response>",
        meta_data={"image_id": "old-image-1"},
    )
    db.add(old_assistant)
    db.commit()

    # Push old media out of recent window.
    fillers = []
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        fillers.append(
            Message(
                thread_id=thread_id,
                role=role,
                content=f"filler {i}",
            )
        )
    db.add_all(fillers)
    db.commit()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "I noticed another bug and fixed it.", "metadata": {"client_message_id": "slice25-another-bug-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("pending_tool_calls") == []

    decision = _latest_decision(db)
    assert decision is not None
    media_gate = (
        db.query(ENSActionResult)
        .filter(
            ENSActionResult.decision_id == decision.decision_id,
            ENSActionResult.kind == "media.gating.evaluate",
        )
        .first()
    )
    assert media_gate is not None
    snapshot = (media_gate.output_json or {}).get("media_gate_snapshot") or {}
    assert snapshot.get("turn_classification") != "iterate_media"
    assert snapshot.get("is_iteration_request") is False
    assert snapshot.get("requested_media_type") == "none"


def test_slice25_acknowledgement_turn_blocks_proactive_media_offer(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _LLMClientWithPayload(
        "<assistant_response><speech>Glad you liked it.</speech></assistant_response>\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"ack-offer","tool":"image.generate","requires_approval":true,"args":{"prompt":"forest dawn"}}]}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={
            "message": "Oh... that's my kind of place. Very nice.",
            "metadata": {"client_message_id": "slice25-ack-offer-block-1"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("pending_tool_calls") == []

    decision = _latest_decision(db)
    assert decision is not None

    media_gate = (
        db.query(ENSActionResult)
        .filter(
            ENSActionResult.decision_id == decision.decision_id,
            ENSActionResult.kind == "media.gating.evaluate",
        )
        .first()
    )
    assert media_gate is not None
    snapshot = (media_gate.output_json or {}).get("media_gate_snapshot") or {}
    assert snapshot.get("requested_media_type") == "none"
    assert snapshot.get("offer_allowed") is False
    assert snapshot.get("media_tool_calls_allowed") is False
    assert snapshot.get("allowed_tools_final") == []

    adjudicate = (
        db.query(ENSActionResult)
        .filter(
            ENSActionResult.decision_id == decision.decision_id,
            ENSActionResult.kind == "tool_payload.adjudicate",
        )
        .first()
    )
    assert adjudicate is not None
    output = adjudicate.output_json or {}
    assert output.get("tool_call_count") == 0

