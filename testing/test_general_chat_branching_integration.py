from __future__ import annotations

from datetime import datetime, timedelta

from chorus_engine.models.conversation import Conversation, ConversationSegment, Message, MessageRole, Thread


def _seed_general_chat_with_segments(db):
    conversation = Conversation(
        character_id="test_char",
        title="General Chat with Test Character",
        source="web",
        conversation_kind="general_chat",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    thread = Thread(conversation_id=conversation.id, title="Main Thread")
    db.add(thread)
    db.commit()
    db.refresh(thread)

    t0 = datetime.utcnow() - timedelta(hours=3)
    m1 = Message(thread_id=thread.id, role=MessageRole.USER, content="old user", created_at=t0)
    m2 = Message(thread_id=thread.id, role=MessageRole.ASSISTANT, content="old assistant", created_at=t0 + timedelta(minutes=1))
    m3 = Message(thread_id=thread.id, role=MessageRole.USER, content="current user", created_at=t0 + timedelta(hours=2))
    m4 = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="current assistant",
        created_at=t0 + timedelta(hours=2, minutes=1),
    )
    db.add_all([m1, m2, m3, m4])
    db.commit()
    for msg in (m1, m2, m3, m4):
        db.refresh(msg)

    closed = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="idle_break",
        state="closed",
        start_message_id=m1.id,
        end_message_id=m2.id,
        started_at=m1.created_at,
        ended_at=m2.created_at,
        usefulness="useful",
        summary_text="Older useful segment summary",
    )
    open_segment = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="manual_break",
        state="open",
        start_message_id=m3.id,
        started_at=m3.created_at,
        usefulness="unknown",
    )
    db.add_all([closed, open_segment])
    db.commit()
    db.refresh(closed)
    db.refresh(open_segment)
    return conversation, thread, (m1, m2, m3, m4), closed, open_segment


def test_branch_present_mode_closes_open_segment_and_sets_origin_segment(client, db):
    conversation, _thread, messages, _closed, open_segment = _seed_general_chat_with_segments(db)
    m3 = messages[2]

    resp = client.post(
        f"/conversations/{conversation.id}/branch",
        json={"selected_message_ids": [m3.id]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["origin_mode"] == "present"
    assert body["closed_segment_id"] == open_segment.id
    assert body["new_conversation_id"]
    assert body["new_thread_id"]

    db.refresh(open_segment)
    assert open_segment.state == "closed"
    assert open_segment.segment_kind == "branch_break"

    branched = (
        db.query(Conversation)
        .filter(Conversation.id == body["new_conversation_id"])
        .first()
    )
    assert branched is not None
    assert branched.origin_conversation_id == conversation.id
    assert branched.origin_mode == "present"
    # Present mode recap source is always the just-closed open segment.
    assert branched.origin_segment_id == open_segment.id


def test_branch_archival_mode_does_not_close_open_segment(client, db):
    conversation, _thread, messages, closed, open_segment = _seed_general_chat_with_segments(db)
    m1, m2 = messages[0], messages[1]

    resp = client.post(
        f"/conversations/{conversation.id}/branch",
        json={"selected_message_ids": [m1.id, m2.id]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["origin_mode"] == "archival"
    assert body["closed_segment_id"] is None

    db.refresh(open_segment)
    assert open_segment.state == "open"

    branched = (
        db.query(Conversation)
        .filter(Conversation.id == body["new_conversation_id"])
        .first()
    )
    assert branched is not None
    assert branched.origin_mode == "archival"
    assert branched.origin_segment_id == closed.id


def test_branch_idempotency_is_canonical_across_selected_id_order(client, db):
    conversation, _thread, messages, _closed, _open_segment = _seed_general_chat_with_segments(db)
    m1, m2 = messages[0], messages[1]

    r1 = client.post(
        f"/conversations/{conversation.id}/branch",
        json={"selected_message_ids": [m1.id, m2.id]},
    )
    r2 = client.post(
        f"/conversations/{conversation.id}/branch",
        json={"selected_message_ids": [m2.id, m1.id]},
    )
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    b1 = r1.json()
    b2 = r2.json()
    assert b1["new_conversation_id"] == b2["new_conversation_id"]


def test_branch_recap_injected_once_marks_conversation_consumed(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    conversation, _thread, messages, _closed, _open_segment = _seed_general_chat_with_segments(db)
    m1, m2 = messages[0], messages[1]

    branch = client.post(
        f"/conversations/{conversation.id}/branch",
        json={"selected_message_ids": [m1.id, m2.id]},
    )
    assert branch.status_code == 200, branch.text
    branched_id = branch.json()["new_conversation_id"]
    assert branch.json()["origin_mode"] == "archival"

    branched_thread = (
        db.query(Thread)
        .filter(Thread.conversation_id == branched_id)
        .order_by(Thread.created_at.asc())
        .first()
    )
    assert branched_thread is not None

    first_turn = client.post(
        f"/threads/{branched_thread.id}/messages",
        json={"message": "first branched turn"},
    )
    assert first_turn.status_code == 200, first_turn.text

    db.expire_all()
    branched = db.query(Conversation).filter(Conversation.id == branched_id).first()
    assert branched is not None
    assert branched.branch_origin_recap_injected_at is not None

    second_turn = client.post(
        f"/threads/{branched_thread.id}/messages",
        json={"message": "second branched turn"},
    )
    assert second_turn.status_code == 200, second_turn.text

    assistants = (
        db.query(Message)
        .filter(
            Message.thread_id == branched_thread.id,
            Message.role == MessageRole.ASSISTANT,
        )
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    # First assistant response after branch should include recap marker, second should not.
    assert len(assistants) >= 2
    assert bool((assistants[-2].meta_data or {}).get("branch_origin_recap_injected")) is True
    assert bool((assistants[-1].meta_data or {}).get("branch_origin_recap_injected")) is False
