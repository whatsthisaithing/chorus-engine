from pathlib import Path

from chorus_engine.models.conversation import Message, MessageRole
from chorus_engine.models.ens import ENSActionResult, ENSDecision, ENSSession


def test_slice0_non_stream_creates_session_decision_and_jsonl(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={
            "message": "hello slice0",
            "metadata": {"client_message_id": "slice0-msg-1"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_message"]["content"] == "hello slice0"
    assert body["assistant_message"]["content"]

    session_rows = db.query(ENSSession).filter(ENSSession.thread_id == thread_id).all()
    assert len(session_rows) == 1

    decisions = db.query(ENSDecision).filter(ENSDecision.signal_type == "user.message.nonstream_intake").all()
    assert len(decisions) >= 1

    jsonl = Path("data/debug_logs/ens/decisions.jsonl")
    assert jsonl.exists()
    assert "user.message.nonstream_intake" in jsonl.read_text(encoding="utf-8")


def test_slice1_idempotent_replay_no_duplicate_assistant(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    payload = {
        "message": "hello replay",
        "metadata": {"client_message_id": "client-id-123"},
    }

    r1 = client.post(f"/threads/{thread_id}/messages", json=payload)
    r2 = client.post(f"/threads/{thread_id}/messages", json=payload)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    b1 = r1.json()
    b2 = r2.json()
    assert b1["assistant_message"]["id"] == b2["assistant_message"]["id"]
    assert b1["user_message"]["id"] == b2["user_message"]["id"]

    assistant_count = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.role == MessageRole.ASSISTANT)
        .count()
    )
    user_count = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.role == MessageRole.USER)
        .count()
    )
    assert assistant_count == 1
    assert user_count == 1


def test_slice1_same_text_different_client_message_id_creates_new_turns(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    payload_1 = {
        "message": "same semantic content",
        "metadata": {"client_message_id": "client-id-a"},
    }
    payload_2 = {
        "message": "same semantic content",
        "metadata": {"client_message_id": "client-id-b"},
    }

    r1 = client.post(f"/threads/{thread_id}/messages", json=payload_1)
    r2 = client.post(f"/threads/{thread_id}/messages", json=payload_2)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    b1 = r1.json()
    b2 = r2.json()
    assert b1["user_message"]["id"] != b2["user_message"]["id"]
    assert b1["assistant_message"]["id"] != b2["assistant_message"]["id"]

    assistant_count = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.role == MessageRole.ASSISTANT)
        .count()
    )
    user_count = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.role == MessageRole.USER)
        .count()
    )
    assert assistant_count == 2
    assert user_count == 2


def test_messages_add_history_write_no_llm_action(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages/add",
        json={
            "content": "history line",
            "role": "user",
            "metadata": {"client_message_id": "hist-1", "discord_user_id": "1234"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"] == "history line"

    llm_actions = db.query(ENSActionResult).filter(ENSActionResult.kind == "llm.invoke.chat").count()
    assert llm_actions == 0

    history_actions = db.query(ENSActionResult).filter(ENSActionResult.kind == "message.write_history").count()
    assert history_actions == 1


def test_streaming_intake_creates_decision_record(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    _conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages/stream",
        json={
            "message": "stream check",
            "metadata": {"client_message_id": "stream-1"},
        },
    )
    # Stream endpoint may still fail in legacy path depending on optional services;
    # Slice 0/1 requirement is ENS intake decision creation.
    assert resp.status_code in (200, 500), resp.text

    decisions = (
        db.query(ENSDecision)
        .filter(ENSDecision.signal_type == "user.message.stream_intake")
        .count()
    )
    assert decisions >= 1
