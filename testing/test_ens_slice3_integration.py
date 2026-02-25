import asyncio
from types import SimpleNamespace

from chorus_engine.ens import ENSContext, Signal
from chorus_engine.models.conversation import ImageAttachment, Memory, MemoryType, Message, MessageRole, MomentPin
from chorus_engine.models.ens import ENSActionResult, ENSDecision
from chorus_engine.repositories import MemoryRepository
from chorus_engine.services.continuity_bootstrap_task import ContinuityBootstrapTaskHandler
from chorus_engine.services.heartbeat_service import BackgroundTask, TaskPriority


def test_slice3_explicit_memory_client_id_idempotent(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    payload = {
        "content": "Remember this preference",
        "thread_id": thread_id,
        "client_memory_id": "mem-client-1",
    }
    r1 = client.post(f"/conversations/{conversation_id}/memories", json=payload)
    r2 = client.post(f"/conversations/{conversation_id}/memories", json=payload)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["id"] == r2.json()["id"]

    rows = (
        db.query(Memory)
        .filter(
            Memory.conversation_id == conversation_id,
            Memory.memory_type == MemoryType.EXPLICIT,
            Memory.client_memory_id == "mem-client-1",
        )
        .all()
    )
    assert len(rows) == 1


def test_explicit_memory_create_not_routed_through_slice4_conversation_delete_path(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=False,
        slice4_config_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/conversations/{conversation_id}/memories",
        json={
            "content": "Remember this explicit fact",
            "thread_id": thread_id,
            "tags": ["explicit"],
            "priority": 70,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory_type"] == "explicit"

    rows = (
        db.query(Memory)
        .filter(
            Memory.conversation_id == conversation_id,
            Memory.memory_type == MemoryType.EXPLICIT,
            Memory.content == "Remember this explicit fact",
        )
        .all()
    )
    assert len(rows) == 1


def test_slice3_continuity_refresh_routes_through_ens(client, db, helpers):
    class _FakeContinuityService:
        async def generate_and_save(self, character, conversation_id=None, force=False):
            _ = (character, conversation_id, force)
            return {"skipped": False, "cache": {"ok": True}}

    helpers.app_module.app_state["continuity_service"] = _FakeContinuityService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )

    resp = client.post("/continuity/refresh", json={"character_id": "test_char", "force": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True

    decision_count = (
        db.query(ENSDecision)
        .filter(ENSDecision.signal_type == "continuity.bootstrap_requested")
        .count()
    )
    action_count = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "continuity.bootstrap")
        .count()
    )
    assert decision_count >= 1
    assert action_count >= 1


def test_slice3_continuity_idempotency_key_changes_when_inputs_change(client, db, helpers):
    class _FakeContinuityService:
        async def generate_and_save(self, character, conversation_id=None, force=False):
            _ = (character, conversation_id, force)
            return {"skipped": False, "cache": {"ok": True}}

    helpers.app_module.app_state["continuity_service"] = _FakeContinuityService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )

    # First run should execute and create a success action result.
    r1 = client.post("/continuity/refresh", json={"character_id": "test_char", "force": False})
    assert r1.status_code == 200, r1.text
    success_before = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "continuity.bootstrap", ENSActionResult.status == "success")
        .count()
    )
    assert success_before >= 1

    # Second run with unchanged inputs should replay/skip.
    r2 = client.post("/continuity/refresh", json={"character_id": "test_char", "force": False})
    assert r2.status_code == 200, r2.text
    skipped_mid = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "continuity.bootstrap", ENSActionResult.status == "skipped")
        .count()
    )
    assert skipped_mid >= 1

    # Add a new memory input; watermark should change and force a new success.
    conversation_id, thread_id = helpers.create_conversation_thread()
    mem_repo = MemoryRepository(db)
    mem_repo.create(
        content="new continuity-relevant memory",
        character_id="test_char",
        memory_type=MemoryType.IMPLICIT,
        conversation_id=conversation_id,
        thread_id=thread_id,
        status="approved",
    )

    r3 = client.post("/continuity/refresh", json={"character_id": "test_char", "force": False})
    assert r3.status_code == 200, r3.text
    success_after = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "continuity.bootstrap", ENSActionResult.status == "success")
        .count()
    )
    assert success_after >= success_before + 1


def test_slice3_continuity_heartbeat_task_routes_through_ens(db, helpers):
    class _FailIfCalledContinuityService:
        async def generate_and_save(self, character, conversation_id=None, force=False):
            _ = (character, conversation_id, force)
            raise AssertionError("legacy continuity service should not be called when slice3 ownership is enabled")

    helpers.app_module.app_state["continuity_service"] = _FailIfCalledContinuityService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )

    handler = ContinuityBootstrapTaskHandler()
    task = BackgroundTask(
        id="continuity_task_test_1",
        task_type="continuity_bootstrap",
        priority=TaskPriority.LOW,
        data={"character_id": "test_char"},
    )

    result = asyncio.run(handler.execute(task, helpers.app_module.app_state))
    assert result.success is True
    assert result.data and result.data.get("via_ens") is True

    decision_count = (
        db.query(ENSDecision)
        .filter(ENSDecision.signal_type == "continuity.bootstrap_requested")
        .count()
    )
    action_count = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "continuity.bootstrap")
        .count()
    )
    assert decision_count >= 1
    assert action_count >= 1


def test_slice3_moment_pin_create_routes_through_ens_and_replays(client, db, helpers, monkeypatch):
    class _FakeExtractionResult:
        parsed = {
            "what_happened": "The user shared a key detail.",
            "why_it_mattered": "It impacts future planning.",
            "quote_snippet": "key detail",
            "tags": ["planning"],
            "telemetry_flags": {},
        }
        parse_mode = "json"
        raw_response = "{}"
        error = None

    class _FakeMomentPinExtractionService:
        def __init__(self, db, llm_client, model):
            _ = (db, llm_client, model)

        def build_snapshot(self, conversation_id, selected_message_ids):
            _ = conversation_id
            return "snapshot text", list(selected_message_ids)

        async def extract_moment(self, snapshot_json):
            _ = snapshot_json
            return _FakeExtractionResult()

    monkeypatch.setattr(
        "chorus_engine.ens.dispatcher.MomentPinExtractionService",
        _FakeMomentPinExtractionService,
    )

    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()
    msg1 = Message(thread_id=thread_id, role=MessageRole.USER, content="Message one")
    msg2 = Message(thread_id=thread_id, role=MessageRole.ASSISTANT, content="Message two")
    db.add(msg1)
    db.add(msg2)
    db.commit()
    db.refresh(msg1)
    db.refresh(msg2)

    payload = {"selected_message_ids": [msg1.id, msg2.id]}
    r1 = client.post(f"/conversations/{conversation_id}/moment-pins", json=payload)
    r2 = client.post(f"/conversations/{conversation_id}/moment-pins", json=payload)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["id"] == r2.json()["id"]

    pins = db.query(MomentPin).filter(MomentPin.conversation_id == conversation_id).all()
    assert len(pins) == 1
    assert pins[0].selection_fingerprint

    decisions = db.query(ENSDecision).filter(ENSDecision.signal_type == "pin.create_requested").count()
    actions = db.query(ENSActionResult).filter(ENSActionResult.kind == "pin.create").count()
    assert decisions >= 1
    assert actions >= 1


def test_slice3_manual_analyze_routes_through_ens(client, db, helpers):
    class _FakeMemory:
        def __init__(self):
            self.memory_type = type("T", (), {"value": "fact"})()
            self.content = "The user preferred concise testing loops."
            self.confidence = 0.92
            self.emotional_weight = 0.2
            self.reasoning = "Explicitly stated preference."
            self.durability = "long_term"
            self.pattern_eligible = False

    class _FakeAnalysisService:
        async def analyze_summary_only(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return None

        async def save_summary_only(self, conversation_id, character_id, analysis, manual):
            _ = (conversation_id, character_id, analysis, manual)
            return False

        async def analyze_memories_only(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return None

        async def save_memories_only(self, conversation_id, character_id, analysis):
            _ = (conversation_id, character_id, analysis)
            return False

        async def analyze_conversation(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return SimpleNamespace(
                memories=[_FakeMemory()],
                summary="summary",
                key_topics=[],
                tone="neutral",
                emotional_arc=[],
                participants=[],
                open_questions=[],
            )

        async def save_analysis(self, conversation_id, character_id, analysis, manual):
            _ = (conversation_id, character_id, analysis, manual)
            return True

    helpers.app_module.app_state["analysis_service"] = _FakeAnalysisService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )
    conversation_id, _thread_id = helpers.create_conversation_thread()

    resp = client.post(f"/conversations/{conversation_id}/analyze?force=true")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"
    assert resp.json()["memories_extracted"] == 1
    assert len(resp.json()["memories"]) == 1
    assert resp.json()["memory_counts"].get("fact") == 1

    decisions = db.query(ENSDecision).filter(ENSDecision.signal_type == "analysis.manual_requested").count()
    actions = db.query(ENSActionResult).filter(ENSActionResult.kind == "analysis.execute").count()
    assert decisions >= 1
    assert actions >= 1


def test_slice3_manual_analyze_idempotency_changes_with_new_messages_general_chat(client, db, helpers):
    class _FakeMemory:
        def __init__(self):
            self.memory_type = type("T", (), {"value": "fact"})()
            self.content = "A new memory"
            self.confidence = 0.9
            self.emotional_weight = 0.1
            self.reasoning = "test"
            self.durability = "long_term"
            self.pattern_eligible = False

    class _FakeAnalysisService:
        async def analyze_summary_only(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return None

        async def save_summary_only(self, conversation_id, character_id, analysis, manual):
            _ = (conversation_id, character_id, analysis, manual)
            return False

        async def analyze_memories_only(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return SimpleNamespace(
                memories=[_FakeMemory()],
                summary="",
                key_topics=[],
                tone="",
                emotional_arc=[],
                participants=[],
                open_questions=[],
                processed_through_message_id="cursor-1",
                processed_through_created_at=None,
            )

        async def save_memories_only(self, conversation_id, character_id, analysis):
            _ = (conversation_id, character_id, analysis)
            return True

        async def analyze_conversation(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return None

        async def save_analysis(self, conversation_id, character_id, analysis, manual):
            _ = (conversation_id, character_id, analysis, manual)
            return False

    helpers.app_module.app_state["analysis_service"] = _FakeAnalysisService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )

    from chorus_engine.models.conversation import Conversation

    conversation_id, thread_id = helpers.create_conversation_thread()
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    conv.conversation_kind = "general_chat"
    db.commit()

    m1 = Message(thread_id=thread_id, role=MessageRole.USER, content="one")
    m2 = Message(thread_id=thread_id, role=MessageRole.ASSISTANT, content="two")
    db.add_all([m1, m2])
    db.commit()

    r1 = client.post(f"/conversations/{conversation_id}/analyze?force=true")
    assert r1.status_code == 200, r1.text
    assert r1.json()["status"] == "success"

    key_1 = (
        db.query(ENSActionResult.idempotency_key)
        .filter(ENSActionResult.kind == "analysis.execute")
        .order_by(ENSActionResult.created_at.desc())
        .first()
    )
    assert key_1 is not None
    key_1 = key_1[0]
    assert ":memories:" in key_1
    assert ":None:None:" not in key_1

    m3 = Message(thread_id=thread_id, role=MessageRole.USER, content="three")
    db.add(m3)
    db.commit()

    r2 = client.post(f"/conversations/{conversation_id}/analyze?force=true")
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "success"

    key_2 = (
        db.query(ENSActionResult.idempotency_key)
        .filter(ENSActionResult.kind == "analysis.execute")
        .order_by(ENSActionResult.created_at.desc())
        .first()
    )
    assert key_2 is not None
    key_2 = key_2[0]
    assert key_2 != key_1


def test_slice3_heartbeat_analyze_idempotency_derives_message_ranges_when_missing(db, helpers):
    class _FakeMemory:
        def __init__(self):
            self.memory_type = type("T", (), {"value": "fact"})()
            self.content = "A new memory"
            self.confidence = 0.9
            self.emotional_weight = 0.1
            self.reasoning = "test"
            self.durability = "long_term"
            self.pattern_eligible = False

    class _FakeAnalysisService:
        async def analyze_summary_only(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return None

        async def save_summary_only(self, conversation_id, character_id, analysis, manual):
            _ = (conversation_id, character_id, analysis, manual)
            return False

        async def analyze_memories_only(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return SimpleNamespace(
                memories=[_FakeMemory()],
                summary="",
                key_topics=[],
                tone="",
                emotional_arc=[],
                participants=[],
                open_questions=[],
                processed_through_message_id="cursor-1",
                processed_through_created_at=None,
            )

        async def save_memories_only(self, conversation_id, character_id, analysis):
            _ = (conversation_id, character_id, analysis)
            return True

        async def analyze_conversation(self, conversation_id, character, manual):
            _ = (conversation_id, character, manual)
            return None

        async def save_analysis(self, conversation_id, character_id, analysis, manual):
            _ = (conversation_id, character_id, analysis, manual)
            return False

    helpers.app_module.app_state["analysis_service"] = _FakeAnalysisService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )

    conversation_id, thread_id = helpers.create_conversation_thread()
    runtime = helpers.app_module.app_state["ens_runtime"]

    db.add_all(
        [
            Message(thread_id=thread_id, role=MessageRole.USER, content="one"),
            Message(thread_id=thread_id, role=MessageRole.ASSISTANT, content="two"),
        ]
    )
    db.commit()

    signal = Signal(
        type="analysis.heartbeat_requested",
        scope="SESSION",
        source="external",
        assistant_id="test_char",
        payload={
            "conversation_id": conversation_id,
            "character_id": "test_char",
            "analysis_kind": "memories",
        },
    )
    asyncio.run(runtime.ingest(signal, ENSContext(app_state=helpers.app_module.app_state, surface="web", source="web")))

    key_1 = (
        db.query(ENSActionResult.idempotency_key)
        .filter(ENSActionResult.kind == "analysis.execute")
        .order_by(ENSActionResult.created_at.desc())
        .first()
    )
    assert key_1 is not None
    key_1 = key_1[0]
    assert ":memories:" in key_1
    assert ":None:None:" not in key_1
    assert ":na:na:" not in key_1

    db.add(Message(thread_id=thread_id, role=MessageRole.USER, content="three"))
    db.commit()
    asyncio.run(runtime.ingest(signal, ENSContext(app_state=helpers.app_module.app_state, surface="web", source="web")))

    key_2 = (
        db.query(ENSActionResult.idempotency_key)
        .filter(ENSActionResult.kind == "analysis.execute")
        .order_by(ENSActionResult.created_at.desc())
        .first()
    )
    assert key_2 is not None
    key_2 = key_2[0]
    assert key_2 != key_1


def test_slice3_core_memory_endpoint_writes_yaml_first(client, db, helpers):
    helpers.app_module.app_state["system_config"].debug_ui = False
    resp = client.post(
        "/characters/test_char/core-memories",
        json={"content": "this is a yaml first core fact", "tags": ["t1"], "priority": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory_type"] == "core"
    assert body["character_id"] == "test_char"


def test_slice3_ens_chat_processes_vision_attachments_and_exposes_visual_context_metric(client, db, helpers, tmp_path):
    class _VisionResult:
        model = "vision-test"
        backend = "stub"
        processing_time_ms = 11
        observation = '{"description":"A red scarf on a chair."}'
        confidence = 0.95
        tags = ["chair", "scarf"]

    class _FakeVisionService:
        config = {"memory": {"auto_create": True, "min_confidence": 0.6, "default_priority": 70}}
        model_name = "vision-test"
        backend = "stub"

        async def analyze_image(self, image_path, context=None, character_id=None):
            _ = (image_path, context, character_id)
            return _VisionResult()

    helpers.app_module.app_state["vision_service"] = _FakeVisionService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    img_path = tmp_path / "vision_test_image.png"
    img_path.write_bytes(b"fake-image")
    attachment = ImageAttachment(
        id="att-ens-1",
        message_id="pending",
        conversation_id="pending",
        character_id="pending",
        original_path=str(img_path),
        original_filename="vision_test_image.png",
        mime_type="image/png",
        file_size=10,
        vision_processed="false",
        vision_skipped="false",
        source="web",
    )
    db.add(attachment)
    db.commit()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={
            "message": "What do you see in this image?",
            "metadata": {"client_message_id": "vision-ens-chat-1"},
            "image_attachment_ids": ["att-ens-1"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    user_message_id = body["user_message"]["id"]

    refreshed = db.query(ImageAttachment).filter(ImageAttachment.id == "att-ens-1").first()
    assert refreshed is not None
    assert refreshed.message_id == user_message_id
    assert refreshed.conversation_id == conversation_id
    assert refreshed.character_id == "test_char"
    assert refreshed.vision_processed == "true"
    assert refreshed.vision_observation
    user_row = db.query(Message).filter(Message.id == user_message_id).first()
    assert user_row is not None
    assert "[VISUAL CONTEXT:" not in (user_row.content or "")
    user_meta = user_row.meta_data if isinstance(user_row.meta_data, dict) else {}
    snapshots = user_meta.get("visual_context_snapshots_v1") or []
    assert isinstance(snapshots, list)
    assert len(snapshots) >= 1
    assert "summary" in snapshots[0]

    mem_count = (
        db.query(Memory)
        .filter(
            Memory.conversation_id == conversation_id,
            Memory.source_kind == "explicit_vision",
        )
        .count()
    )
    assert mem_count == 1

    llm_rows = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "llm.invoke.chat")
        .order_by(ENSActionResult.created_at.desc())
        .all()
    )
    assert llm_rows
    output = llm_rows[0].output_json or {}
    assert int(output.get("current_turn_visual_context_count") or 0) >= 1


def test_slice3_ens_history_add_with_attachment_links_and_processes_without_llm(client, db, helpers, tmp_path):
    class _VisionResult:
        model = "vision-test"
        backend = "stub"
        processing_time_ms = 9
        observation = '{"description":"A blue bicycle by a fence."}'
        confidence = 0.93
        tags = ["bicycle"]

    class _FakeVisionService:
        config = {"memory": {"auto_create": True, "min_confidence": 0.6, "default_priority": 70}}
        model_name = "vision-test"
        backend = "stub"

        async def analyze_image(self, image_path, context=None, character_id=None):
            _ = (image_path, context, character_id)
            return _VisionResult()

    helpers.app_module.app_state["vision_service"] = _FakeVisionService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice3_continuity_writes_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    img_path = tmp_path / "vision_history_image.png"
    img_path.write_bytes(b"fake-image-history")
    attachment = ImageAttachment(
        id="att-hist-1",
        message_id="pending",
        conversation_id="pending",
        character_id="pending",
        original_path=str(img_path),
        original_filename="vision_history_image.png",
        mime_type="image/png",
        file_size=18,
        vision_processed="false",
        vision_skipped="false",
        source="web",
    )
    db.add(attachment)
    db.commit()

    resp = client.post(
        f"/threads/{thread_id}/messages/add",
        json={
            "content": "History user message with image",
            "role": "user",
            "metadata": {"client_message_id": "vision-hist-1"},
            "image_attachment_ids": ["att-hist-1"],
        },
    )
    assert resp.status_code == 200, resp.text
    msg_id = resp.json()["id"]

    refreshed = db.query(ImageAttachment).filter(ImageAttachment.id == "att-hist-1").first()
    assert refreshed is not None
    assert refreshed.message_id == msg_id
    assert refreshed.conversation_id == conversation_id
    assert refreshed.character_id == "test_char"
    assert refreshed.vision_processed == "true"

    llm_count = db.query(ENSActionResult).filter(ENSActionResult.kind == "llm.invoke.chat").count()
    assert llm_count == 0

