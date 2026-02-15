from chorus_engine.models.conversation import Message, MessageRole
from chorus_engine.models.ens import ENSToolCallRequest


async def _fake_scene_preview(_db, params):
    media_type = params.get("media_type") or "image"
    return {
        "tool_call_id": f"tc:scene:{media_type}:fixed",
        "client_capture_id": "cap_fixed",
        "prompt": f"{media_type} prompt",
        "negative_prompt": "no blur",
        "reasoning": "test-preview",
        "needs_trigger": False,
        "type": "video_scene_capture" if media_type == "video" else "scene_capture",
    }


async def _fake_scene_preview_with_capture_id(_db, params):
    media_type = params.get("media_type") or "image"
    capture_id = params.get("client_capture_id") or "missing"
    return {
        "tool_call_id": f"tc:scene:{media_type}:{capture_id}",
        "client_capture_id": capture_id,
        "prompt": f"{media_type} prompt {capture_id}",
        "negative_prompt": "no blur",
        "reasoning": "test-preview",
        "needs_trigger": False,
        "type": "video_scene_capture" if media_type == "video" else "scene_capture",
    }


class _ScenePreviewLLMResponse:
    def __init__(self, content: str, model: str = "test-model"):
        self.content = content
        self.model = model


class _ScenePreviewLLMClient:
    base_url = "http://test-llm"

    async def health_check(self):
        return True

    async def generate(self, prompt, system_prompt=None, model=None, **kwargs):
        _ = (prompt, system_prompt, model, kwargs)
        return _ScenePreviewLLMResponse(
            '{"prompt":"scene prompt from llm","negative_prompt":"none","reasoning":"test"}'
        )


def test_scene_preview_persists_tool_call_no_comfy(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_scene_capture_ownership=True,
    )
    helpers.app_module.app_state["ens_scene_preview_executor"] = _fake_scene_preview
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(f"/threads/{thread_id}/capture-scene-prompt")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tool_call_id"] == "tc:scene:image:fixed"
    assert body["prompt"] == "image prompt"

    row = db.query(ENSToolCallRequest).filter(ENSToolCallRequest.tool_call_id == body["tool_call_id"]).first()
    assert row is not None
    assert row.status == "pending"
    assert row.tool_name == "scene_capture.generate"
    assert (row.args_json or {}).get("preview", {}).get("prompt") == "image prompt"


def test_scene_preview_real_executor_returns_prompt_and_tool_call_id(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_scene_capture_ownership=True,
    )
    helpers.app_module.app_state["llm_client"] = _ScenePreviewLLMClient()
    _conversation_id, thread_id = helpers.create_conversation_thread()

    db.add(
        Message(
            thread_id=thread_id,
            role=MessageRole.USER,
            content="Capture this scene.",
        )
    )
    db.commit()

    resp = client.post(f"/threads/{thread_id}/capture-scene-prompt")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("prompt") == "scene prompt from llm"
    assert body.get("tool_call_id", "").startswith("tc:scene:image:")
    assert body.get("client_capture_id")

    row = db.query(ENSToolCallRequest).filter(ENSToolCallRequest.tool_call_id == body["tool_call_id"]).first()
    assert row is not None
    assert row.status == "pending"
    assert (row.args_json or {}).get("preview", {}).get("prompt") == "scene prompt from llm"


def test_scene_preview_repeated_requests_generate_distinct_tool_calls(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_scene_capture_ownership=True,
    )
    helpers.app_module.app_state["ens_scene_preview_executor"] = _fake_scene_preview_with_capture_id
    _conversation_id, thread_id = helpers.create_conversation_thread()

    r1 = client.post(f"/threads/{thread_id}/capture-scene-prompt")
    r2 = client.post(f"/threads/{thread_id}/capture-scene-prompt")
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    b1 = r1.json()
    b2 = r2.json()
    assert b1["tool_call_id"] != b2["tool_call_id"]
    assert b1["client_capture_id"] != b2["client_capture_id"]

    rows = db.query(ENSToolCallRequest).all()
    assert len(rows) == 2


def test_scene_confirm_requires_tool_call_id_when_owned(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_scene_capture_ownership=True,
        slice2_scene_capture_legacy_confirm_without_tool_call=False,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(f"/threads/{thread_id}/capture-scene", json={"prompt": "x"})
    assert resp.status_code == 400
    assert "tool_call_id" in resp.text


def test_scene_confirm_replay_idempotent_no_duplicate_scene_message(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_scene_capture_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    row = ENSToolCallRequest(
        tool_call_id="tc:scene:image:replay",
        session_id="sess-1",
        assistant_message_id=None,
        tool_name="scene_capture.generate",
        args_json={"thread_id": thread_id, "media_type": "image", "prompt": "sunset"},
        status="pending",
        idempotency_key="scene:preview:test",
        result_ref=None,
    )
    db.add(row)
    db.commit()

    async def _fake_scene_image_exec(db_session, payload):
        return {
            "success": True,
            "image_id": 11,
            "file_path": "/images/test.png",
            "thumbnail_path": "/images/test_thumb.png",
            "prompt": payload.get("prompt"),
            "negative_prompt": payload.get("negative_prompt"),
            "generation_time": 0.1,
            "scene_message_id": payload["scene_message_id"],
        }

    helpers.app_module._ens_execute_scene_capture_image = _fake_scene_image_exec

    r1 = client.post(
        f"/threads/{thread_id}/capture-scene",
        json={"tool_call_id": "tc:scene:image:replay", "prompt": "sunset"},
    )
    r2 = client.post(
        f"/threads/{thread_id}/capture-scene",
        json={"tool_call_id": "tc:scene:image:replay", "prompt": "sunset"},
    )
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    b1 = r1.json()
    b2 = r2.json()
    assert b1["image_id"] == b2["image_id"] == 11

    scene_messages = db.query(Message).filter(Message.thread_id == thread_id, Message.role == MessageRole.SCENE_CAPTURE).all()
    assert len(scene_messages) == 1

    refreshed = db.query(ENSToolCallRequest).filter(ENSToolCallRequest.tool_call_id == "tc:scene:image:replay").first()
    assert refreshed.status == "completed"
    assert (refreshed.result_ref or {}).get("scene_message_id") == scene_messages[0].id
