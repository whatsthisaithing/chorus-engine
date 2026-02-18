from chorus_engine.models.conversation import Conversation, Message, Thread
from chorus_engine.models.ens import ENSSession, SurfaceBinding
from chorus_engine.ens import ENSContext, SignalEnvelope


def test_slice6_surface_binding_create_and_authoritative_reuse(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice6_surface_routing_ownership=True,
    )
    _conversation_id_a, thread_id_a = helpers.create_conversation_thread()
    _conversation_id_b, thread_id_b = helpers.create_conversation_thread()

    payload = {
        "content": "history message one",
        "role": "user",
        "metadata": {
            "external_thread_id": "ext-thread-001",
            "target_hint": "relationship_dm",
            "speaker_external_id": "user-ext-1",
        },
    }
    first = client.post(f"/threads/{thread_id_a}/messages/add", json=payload)
    assert first.status_code == 200, first.text
    assert first.json()["thread_id"] == thread_id_a

    second = client.post(
        f"/threads/{thread_id_b}/messages/add",
        json={
            "content": "history message two",
            "role": "user",
            "metadata": {
                "external_thread_id": "ext-thread-001",
                "target_hint": "relationship_dm",
            },
        },
    )
    assert second.status_code == 200, second.text
    assert second.json()["thread_id"] == thread_id_a

    bindings = db.query(SurfaceBinding).filter(SurfaceBinding.external_thread_id == "ext-thread-001").all()
    assert len(bindings) == 1
    assert bindings[0].surface_id == "web"
    assert bindings[0].thread_id == thread_id_a


def test_slice6_surface_canonicalization_unknown_surface(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice6_surface_routing_ownership=True,
    )

    conversation = Conversation(character_id="test_char", title="Unknown Surface", source="my_custom_surface")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    thread = Thread(conversation_id=conversation.id, title="Main Thread")
    db.add(thread)
    db.commit()
    db.refresh(thread)

    resp = client.post(
        f"/threads/{thread.id}/messages/add",
        json={
            "content": "history from unknown surface",
            "role": "user",
            "metadata": {"external_thread_id": "unknown-ext-thread"},
        },
    )
    assert resp.status_code == 200, resp.text

    binding = (
        db.query(SurfaceBinding)
        .filter(SurfaceBinding.external_thread_id == "unknown-ext-thread")
        .one()
    )
    assert binding.surface_id == "unknown"

    session = db.query(ENSSession).filter(ENSSession.thread_id == thread.id).order_by(ENSSession.created_at.desc()).first()
    assert session is not None
    assert session.surface == "unknown"
    assert session.source == "unknown"


def test_slice6_history_idempotency_prefers_message_external_id(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice6_surface_routing_ownership=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    req = {
        "content": "same payload replay",
        "role": "user",
        "metadata": {
            "external_message_id": "ext-msg-100",
            "client_message_id": "client-msg-100",
        },
    }
    first = client.post(f"/threads/{thread_id}/messages/add", json=req)
    assert first.status_code == 200, first.text

    second = client.post(
        f"/threads/{thread_id}/messages/add",
        json={
            "content": "same payload replay",
            "role": "user",
            "metadata": {
                "external_message_id": "ext-msg-100",
                "client_message_id": "client-msg-999",
            },
        },
    )
    assert second.status_code == 200, second.text
    assert second.json()["id"] == first.json()["id"]

    third = client.post(
        f"/threads/{thread_id}/messages/add",
        json={
            "content": "same payload replay",
            "role": "user",
            "metadata": {
                "external_message_id": "ext-msg-101",
                "client_message_id": "client-msg-100",
            },
        },
    )
    assert third.status_code == 200, third.text
    assert third.json()["id"] != first.json()["id"]

    message_count = db.query(Message).filter(Message.thread_id == thread_id).count()
    assert message_count == 2


def test_slice6_session_signal_with_conversation_hint_reuses_existing_thread(helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice6_surface_routing_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()
    runtime = helpers.app_module.app_state["ens_runtime"]

    signal = SignalEnvelope(
        type="analysis.heartbeat_requested",
        scope="SESSION",
        source="external",
        assistant_id="test_char",
        payload={
            "conversation_id": conversation_id,
            "character_id": "test_char",
            "analysis_kind": "summary",
        },
    )
    _ = helpers.app_module
    import asyncio
    asyncio.run(runtime.ingest(signal, ENSContext(app_state=helpers.app_module.app_state, surface="web", source="web")))

    bindings = db.query(SurfaceBinding).filter(SurfaceBinding.conversation_id == conversation_id).all()
    assert len(bindings) == 1
    assert bindings[0].thread_id == thread_id
    assert bindings[0].external_thread_id == thread_id
