from chorus_engine.models.ens import ENSActionResult, ENSDecision
from chorus_engine.models.conversation import MessageRole, MemoryType
from chorus_engine.repositories import MessageRepository, MemoryRepository


class _DummyExtractionService:
    async def approve_pending_memory(self, memory_id):
        return True


def test_slice5_continuity_choice_routes_through_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )
    conversation_id, _thread_id = helpers.create_conversation_thread()

    resp = client.post(
        "/continuity/choice",
        json={"conversation_id": conversation_id, "mode": "use", "remember_choice": False},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True

    decision_count = db.query(ENSDecision).filter(ENSDecision.signal_type == "config.conversation.change_requested").count()
    apply_count = db.query(ENSActionResult).filter(ENSActionResult.kind == "config.conversation.apply").count()
    assert decision_count >= 1
    assert apply_count >= 1


def test_slice5_message_metadata_provenance_write_once(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()
    msg_repo = MessageRepository(db)
    message = msg_repo.create(
        thread_id=thread_id,
        role=MessageRole.USER,
        content="hello",
        metadata={"system.surface_id": "web"},
    )

    resp = client.patch(
        f"/messages/{message.id}/metadata",
        json={"metadata": {"system.surface_id": "discord", "system.hidden": True}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "system.hidden" in body["applied_keys"]
    rejected_keys = {item["key"]: item["reason"] for item in body["rejected"]}
    assert rejected_keys.get("system.surface_id") == "write_once_provenance_key"
    assert body["metadata"]["system.surface_id"] == "web"
    assert body["metadata"]["system.hidden"] is True


def test_slice5_memory_approve_routes_through_ens(client, db, helpers):
    helpers.app_module.app_state["extraction_service"] = _DummyExtractionService()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()
    mem_repo = MemoryRepository(db)
    memory = mem_repo.create(
        content="pending memory",
        character_id="test_char",
        memory_type=MemoryType.IMPLICIT,
        conversation_id=conversation_id,
        thread_id=thread_id,
        status="pending",
    )

    resp = client.post(f"/memories/{memory.id}/approve")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    assert body["memory_id"] == memory.id

    decision_count = db.query(ENSDecision).filter(ENSDecision.signal_type == "memory.moderation_requested").count()
    apply_count = db.query(ENSActionResult).filter(ENSActionResult.kind == "memory.moderation.apply").count()
    assert decision_count >= 1
    assert apply_count >= 1
