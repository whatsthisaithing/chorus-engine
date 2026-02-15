from chorus_engine.models.ens import ENSToolCallRequest
from chorus_engine.models.conversation import Conversation


class _ToolPayloadResponse:
    def __init__(self, content: str):
        self.content = content


class _ToolPayloadLLMClient:
    base_url = "http://test-llm"

    def __init__(self, payload_text: str):
        self.payload_text = payload_text

    async def health_check(self):
        return True

    async def generate_with_history(self, messages, temperature=None, max_tokens=None, model=None):
        return _ToolPayloadResponse(self.payload_text)


def test_slice2_parsing_returns_pending_and_persists_tool_call(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _ToolPayloadLLMClient(
        "Here is that image.\n"
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
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "please make an image", "metadata": {"client_message_id": "slice2-tool-1"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assistant_message"]["content"].strip() == "Here is that image."
    assert len(body["pending_tool_calls"]) == 1
    pending = body["pending_tool_calls"][0]
    assert pending["tool"] == "image.generate"
    assert pending["id"].startswith("tc:")

    tool_rows = db.query(ENSToolCallRequest).all()
    assert len(tool_rows) == 1
    assert tool_rows[0].status == "pending"
    assert tool_rows[0].tool_name == "image.generate"


def test_slice2_idempotent_replay_no_duplicate_tool_call_request(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _ToolPayloadLLMClient(
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
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()
    payload = {"message": "same", "metadata": {"client_message_id": "slice2-replay-1"}}

    r1 = client.post(f"/threads/{thread_id}/messages", json=payload)
    r2 = client.post(f"/threads/{thread_id}/messages", json=payload)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    b1 = r1.json()
    b2 = r2.json()
    assert b1["assistant_message"]["id"] == b2["assistant_message"]["id"]
    assert b1["user_message"]["id"] == b2["user_message"]["id"]

    tool_rows = db.query(ENSToolCallRequest).all()
    assert len(tool_rows) == 1


def test_slice2_parsing_flag_off_returns_no_pending_tool_calls(client, helpers):
    helpers.app_module.app_state["llm_client"] = _ToolPayloadLLMClient(
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
        slice2_tool_parsing_ownership=False,
        slice2_tool_dispatch_ownership=False,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "same", "metadata": {"client_message_id": "slice2-flag-off"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending_tool_calls"] == []


def test_slice2_cold_recall_chained_with_media_is_rejected(client, db, helpers):
    helpers.app_module.app_state["llm_client"] = _ToolPayloadLLMClient(
        "Attempting mixed tools.\n"
        "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
        '{"version":1,"tool_calls":[{"id":"cold1","tool":"moment_pin.cold_recall","requires_approval":false,"args":{"pin_id":"p1","reason":"need quote"}},{"id":"img1","tool":"image.generate","requires_approval":true,"args":{"prompt":"cat portrait"}}]}\n'
        "---CHORUS_TOOL_PAYLOAD_END---"
    )
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "mix tools", "metadata": {"client_message_id": "slice2-chain-reject"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending_tool_calls"] == []
    assert db.query(ENSToolCallRequest).count() == 0


def test_slice2_prompt_override_blocked_when_tool_not_pending(db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=True,
    )
    row = ENSToolCallRequest(
        tool_call_id="tc:test:1",
        session_id="sess-1",
        assistant_message_id="msg-1",
        tool_name="image.generate",
        args_json={"prompt": "original prompt"},
        status="completed",
        idempotency_key="tool:pending:sess-1:msg-1:img1",
        result_ref={"success": True, "image_id": 1, "prompt": "original prompt"},
    )
    db.add(row)
    db.commit()

    import asyncio
    import pytest

    with pytest.raises(RuntimeError, match="Prompt override allowed only while tool call is pending"):
        asyncio.run(
            helpers.app_module._ens_execute_tool_call(
                db,
                {"tool_call_id": "tc:test:1", "thread_id": "thread-1", "prompt": "override prompt"},
            )
        )


def test_slice2_disable_confirmation_flag_propagates_through_ens_tool_execution(db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    row = ENSToolCallRequest(
        tool_call_id="tc:test:disable-confirm:1",
        session_id="sess-disable-confirm",
        assistant_message_id="msg-disable-confirm",
        tool_name="image.generate",
        args_json={"thread_id": thread_id, "prompt": "portrait", "negative_prompt": "blur"},
        status="pending",
        idempotency_key="tool:pending:disable-confirm",
        result_ref=None,
    )
    db.add(row)
    db.commit()

    captured = {}

    async def _fake_generate_image(thread_id, request, db):
        _ = db
        captured["thread_id"] = thread_id
        captured["disable_future_confirmations"] = request.disable_future_confirmations
        return {
            "success": True,
            "image_id": 99,
            "file_path": "/images/test.png",
            "thumbnail_path": "/images/test_thumb.png",
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "generation_time": 0.01,
        }

    helpers.app_module.generate_image = _fake_generate_image

    import asyncio

    result = asyncio.run(
        helpers.app_module._ens_execute_tool_call(
            db,
            {
                "tool_call_id": "tc:test:disable-confirm:1",
                "thread_id": thread_id,
                "disable_future_confirmations": True,
            },
        )
    )
    assert result["success"] is True
    assert captured["thread_id"] == thread_id
    assert captured["disable_future_confirmations"] is True

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    assert conversation is not None
    assert conversation.image_confirmation_disabled == "true"
