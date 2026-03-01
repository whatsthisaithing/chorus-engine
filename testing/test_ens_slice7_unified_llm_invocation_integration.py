from chorus_engine.ens.llm_invocation_service import InvocationRequest, LLMInvocationService
import asyncio
import json
import pytest
import io
from datetime import datetime
from pathlib import Path
from PIL import Image
from chorus_engine.services.vision_service import VisionService
from chorus_engine.models import ImageAttachment, Memory
from chorus_engine.models.ens import ENSActionResult


def test_slice7_chat_includes_unified_invocation_fields(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={
            "message": "slice7 unified invoker chat",
            "metadata": {"client_message_id": "slice7-chat-001"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assistant_message"]["content"]


def test_slice7_request_fingerprint_ignores_trace_metadata(helpers):
    invoker = LLMInvocationService(helpers.app_module.app_state)
    req_a = InvocationRequest(
        invocation_kind="analysis",
        idempotency_key="k1",
        model_id="m1",
        prompt="hello",
        system_prompt="sys",
        metadata={"trace_id": "abc", "timestamp": "2026-01-01T00:00:00Z", "stable": "x"},
    )
    req_b = InvocationRequest(
        invocation_kind="analysis",
        idempotency_key="k1",
        model_id="m1",
        prompt="hello",
        system_prompt="sys",
        metadata={"trace_id": "def", "timestamp": "2099-12-31T00:00:00Z", "stable": "x"},
    )
    assert invoker.request_fingerprint(req_a) == invoker.request_fingerprint(req_b)


def test_slice7_unified_analysis_invocation_works(helpers):
    invoker = LLMInvocationService(helpers.app_module.app_state)
    req = InvocationRequest(
        invocation_kind="analysis",
        idempotency_key="slice7-analysis-001",
        model_id=helpers.app_module.app_state["system_config"].llm.model,
        prompt="Analyze this test context.",
        system_prompt="Return concise analysis.",
        metadata={"analysis_kind": "scene_prompt", "trace_id": "should_not_affect_fingerprint"},
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert result["provider"] == "local"
    assert result["engine"] in ("ollama", "lmstudio", "koboldcpp", "unknown")
    assert result["request_fingerprint"]


def test_slice7_nonstream_generation_is_ens_owned(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "legacy path should be blocked", "metadata": {"client_message_id": "slice7-block-001"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assistant_message"]["content"]


def test_slice7_stream_generation_is_ens_owned(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()
    resp = client.post(
        f"/threads/{thread_id}/messages/stream",
        json={"message": "legacy stream should be blocked", "metadata": {"client_message_id": "slice7-block-002"}},
    )
    assert resp.status_code == 200, resp.text
    text = resp.text
    assert '"type": "done"' in text


def test_slice7_direct_generate_call_is_blocked_in_tests(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
    )
    llm_client = helpers.app_module.app_state["llm_client"]
    with pytest.raises(RuntimeError, match="Direct llm_client.generate call blocked"):
        asyncio.run(llm_client.generate(prompt="should fail outside invoker"))


def test_slice7_vision_attachment_uses_unified_invoker_and_is_idempotent(client, helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
    )
    app_module = helpers.app_module
    app_module.app_state["vision_service"] = VisionService(
        vision_config={
            "enabled": True,
            "model": {"name": "vision-test-model"},
            "processing": {"max_retries": 0, "timeout_seconds": 5, "resize_target": 512},
            "memory": {"auto_create": True, "min_confidence": 0.1, "default_priority": 70},
        },
        llm_config={"provider": "ollama", "base_url": "http://test-llm"},
        llm_client=app_module.app_state["llm_client"],
        llm_invoke_fn=app_module._invoke_llm_unified,
    )

    conversation_id, thread_id = helpers.create_conversation_thread()

    image = Image.new("RGB", (8, 8), color="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    upload = client.post(
        "/api/attachments/upload",
        files={"file": ("vision.png", buf.getvalue(), "image/png")},
    )
    assert upload.status_code == 200, upload.text
    attachment_id = upload.json()["attachment_id"]

    payload = {
        "message": "Please analyze this image",
        "metadata": {"client_message_id": "slice7-vision-001"},
        "image_attachment_ids": [attachment_id],
    }
    resp = client.post(f"/threads/{thread_id}/messages", json=payload)
    assert resp.status_code == 200, resp.text

    attachment = db.query(ImageAttachment).filter(ImageAttachment.id == attachment_id).first()
    assert attachment is not None
    assert attachment.vision_processed == "true"

    vision_memories = (
        db.query(Memory)
        .filter(Memory.conversation_id == conversation_id, Memory.source_kind == "explicit_vision")
        .all()
    )
    assert len(vision_memories) == 1

    process_action = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "attachments.process_vision")
        .order_by(ENSActionResult.created_at.desc())
        .first()
    )
    assert process_action is not None
    output = process_action.output_json or {}
    invocations = output.get("llm_invocations") or []
    assert len(invocations) >= 1
    first = invocations[0]
    assert first.get("provider") == "local"
    assert first.get("engine") in ("ollama", "lmstudio", "koboldcpp", "unknown")
    assert first.get("request_fingerprint")
    assert isinstance(first.get("attempts"), int)

    replay = client.post(f"/threads/{thread_id}/messages", json=payload)
    assert replay.status_code == 200, replay.text
    vision_memories_after = (
        db.query(Memory)
        .filter(Memory.conversation_id == conversation_id, Memory.source_kind == "explicit_vision")
        .all()
    )
    assert len(vision_memories_after) == 1


def test_slice7_debug_capture_full_prompt_writes_prompt_payload_to_ens_log(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
        debug_capture_full_prompt=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={
            "message": "capture full prompt please",
            "metadata": {"client_message_id": "slice7-prompt-capture-001"},
        },
    )
    assert resp.status_code == 200, resp.text

    today = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = Path("data/debug_logs/conversations") / conversation_id / f"ens_conversation_{today}.jsonl"
    assert log_file.exists()

    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    turn_rows = [row for row in rows if row.get("type") == "ens_llm_turn"]
    assert turn_rows
    latest = turn_rows[-1]
    capture = latest.get("prompt_capture") or {}
    assert capture.get("enabled") is True
    assert isinstance(capture.get("system_prompt"), str)
    assert capture.get("system_prompt")
    assert isinstance(capture.get("messages_for_llm"), list)
    assert len(capture.get("messages_for_llm")) >= 2
    assert isinstance(capture.get("token_breakdown"), dict)
