import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from chorus_engine.models.conversation import Conversation, Message, MessageRole, Thread
from chorus_engine.services.conversation_analysis_service import ConversationAnalysis, ConversationAnalysisService


def _make_general_chat_conversation(db):
    conv = Conversation(
        character_id="test_char",
        title="General chat analysis test",
        source="web",
        conversation_kind="general_chat",
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    thread = Thread(conversation_id=conv.id, title="Main Thread")
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return conv, thread


def _make_analysis_service(db, helpers):
    return ConversationAnalysisService(
        db=db,
        llm_client=helpers.app_module.app_state["llm_client"],
        vector_store=None,
        embedding_service=None,
    )


def test_general_chat_summary_analysis_is_skipped(db, helpers):
    conv, thread = _make_general_chat_conversation(db)
    db.add(Message(thread_id=thread.id, role=MessageRole.USER, content="hi"))
    db.commit()

    service = _make_analysis_service(db, helpers)
    character = helpers.app_module.app_state["characters"]["test_char"]
    analysis = asyncio.run(service.analyze_summary_only(conv.id, character, manual=False))
    assert analysis is None


def test_general_chat_incremental_cursor_uses_created_at_and_message_id(db, helpers):
    service = _make_analysis_service(db, helpers)

    cursor_time = datetime.utcnow()
    conversation = SimpleNamespace(
        general_chat_memories_processed_through_created_at=cursor_time,
        general_chat_memories_processed_through_message_id="b",
    )
    messages = [
        SimpleNamespace(id="a", created_at=cursor_time),
        SimpleNamespace(id="b", created_at=cursor_time),
        SimpleNamespace(id="c", created_at=cursor_time),
        SimpleNamespace(id="a", created_at=cursor_time + timedelta(seconds=1)),
    ]

    window, processed_id, processed_created_at = service._slice_general_chat_incremental_window(
        messages=messages,
        conversation=conversation,
        bridge_tail_size=1,
    )

    assert [m.id for m in window] == ["b", "c", "a"]
    assert processed_id == "a"
    assert processed_created_at == cursor_time + timedelta(seconds=1)


def test_save_memories_only_updates_general_chat_cursor_fields(db, helpers):
    conv, _thread = _make_general_chat_conversation(db)
    service = _make_analysis_service(db, helpers)

    processed_time = datetime.utcnow()
    analysis = ConversationAnalysis(
        memories=[],
        summary="",
        key_topics=[],
        tone="",
        emotional_arc="",
        participants=[],
        open_questions=[],
        processed_through_message_id="cursor-msg-1",
        processed_through_created_at=processed_time,
    )

    ok = asyncio.run(service.save_memories_only(conv.id, "test_char", analysis))
    assert ok is True

    db.refresh(conv)
    assert conv.general_chat_memories_processed_through_message_id == "cursor-msg-1"
    assert conv.general_chat_memories_processed_through_created_at == processed_time

