from datetime import datetime, timedelta

from chorus_engine.models.conversation import Conversation, ConversationSummary, Thread
from chorus_engine.services.continuity_bootstrap_service import ContinuityBootstrapService


def _make_summary(db, *, conversation_id: str, thread_id: str, summary: str, created_at: datetime):
    row = ConversationSummary(
        conversation_id=conversation_id,
        thread_id=thread_id,
        summary=summary,
        summary_type="progressive",
        message_range_start=1,
        message_range_end=2,
        message_count=2,
        key_topics=["topic"],
        participants=["user", "assistant"],
        emotional_arc="steady",
        tone="neutral",
        open_questions=[],
        manual="false",
        created_at=created_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _make_conversation_and_thread(db, *, kind: str):
    conv = Conversation(
        character_id="test_char",
        title=f"{kind} conversation",
        source="web",
        conversation_kind=kind,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    thread = Thread(conversation_id=conv.id, title="Main Thread")
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return conv, thread


def test_continuity_summary_loader_excludes_general_chat(db, helpers):
    standard_conv, standard_thread = _make_conversation_and_thread(db, kind="standard")
    general_conv, general_thread = _make_conversation_and_thread(db, kind="general_chat")

    now = datetime.utcnow()
    _make_summary(
        db,
        conversation_id=standard_conv.id,
        thread_id=standard_thread.id,
        summary="standard summary",
        created_at=now - timedelta(minutes=1),
    )
    _make_summary(
        db,
        conversation_id=general_conv.id,
        thread_id=general_thread.id,
        summary="general summary should be ignored",
        created_at=now,
    )

    service = ContinuityBootstrapService(db, helpers.app_module.app_state["llm_client"])
    loaded = service._load_recent_summaries("test_char", limit=10)
    loaded_ids = {row.conversation_id for row in loaded}

    assert standard_conv.id in loaded_ids
    assert general_conv.id not in loaded_ids


def test_continuity_staleness_ignores_general_chat_summaries(db, helpers):
    general_conv, general_thread = _make_conversation_and_thread(db, kind="general_chat")
    _make_summary(
        db,
        conversation_id=general_conv.id,
        thread_id=general_thread.id,
        summary="general summary only",
        created_at=datetime.utcnow(),
    )

    service = ContinuityBootstrapService(db, helpers.app_module.app_state["llm_client"])
    assert service.is_bootstrap_stale("test_char") is False

