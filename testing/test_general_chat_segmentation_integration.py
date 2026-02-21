from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from chorus_engine.models.conversation import Conversation, ConversationSegment, Message, MessageRole, Thread
from chorus_engine.ens.dispatcher import ENSDispatcher
from chorus_engine.services.prompt_assembly import PromptAssemblyService
from chorus_engine.services.conversation_segmentation_service import ConversationSegmentationService


def test_general_chat_idle_break_creates_new_segment(db):
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

    t0 = datetime.utcnow() - timedelta(hours=4)
    user1 = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Hey there",
        created_at=t0,
    )
    assistant1 = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="Hi!",
        created_at=t0 + timedelta(minutes=1),
    )
    user2 = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Back after a while.",
        created_at=t0 + timedelta(hours=3),
    )
    db.add_all([user1, assistant1, user2])
    db.commit()
    db.refresh(user1)
    db.refresh(user2)

    cfg = type("Cfg", (), {
        "enabled": True,
        "idle_soft_minutes": 30,
        "idle_hard_minutes": 120,
        "sleep_break_minutes": 360,
        "density_window_messages": 8,
        "density_low_turn_count_max": 4,
        "density_low_avg_chars_max": 180,
        "pending_question_tail_messages": 2,
        "resume_recent_useful_segments": 3,
        "resume_max_age_hours": 72,
    })()
    service = ConversationSegmentationService(db, cfg)

    first = service.ensure_segment_for_turn(
        conversation=conversation,
        thread_id=thread.id,
        user_message_id=user1.id,
        surface_id="web",
        surface_instance_id="",
    )
    assert first.transitioned is True
    assert first.transition_reason == "manual_break"
    assert first.segment_id is not None

    second = service.ensure_segment_for_turn(
        conversation=conversation,
        thread_id=thread.id,
        user_message_id=user2.id,
        surface_id="web",
        surface_instance_id="",
    )
    assert second.transitioned is True
    assert second.transition_reason == "idle_break"
    assert second.closed_segment_id is not None
    assert second.segment_id is not None


def test_general_chat_idle_break_uses_only_prior_messages_for_replay(db):
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

    t0 = datetime.utcnow() - timedelta(days=2)
    user1 = Message(thread_id=thread.id, role=MessageRole.USER, content="day1 user", created_at=t0)
    assistant1 = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="day1 assistant",
        created_at=t0 + timedelta(minutes=1),
    )
    user2 = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="day2 user",
        created_at=t0 + timedelta(days=1),
    )
    assistant2 = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="day2 assistant",
        created_at=t0 + timedelta(days=1, minutes=1),
    )
    # Future message relative to user2 should not affect replay split decision for user2.
    user3 = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="day3 user",
        created_at=t0 + timedelta(days=2),
    )
    db.add_all([user1, assistant1, user2, assistant2, user3])
    db.commit()
    db.refresh(user1)
    db.refresh(user2)

    cfg = type(
        "Cfg",
        (),
        {
            "enabled": True,
            "idle_soft_minutes": 30,
            "idle_hard_minutes": 120,
            "sleep_break_minutes": 360,
            "density_window_messages": 8,
            "density_low_turn_count_max": 4,
            "density_low_avg_chars_max": 180,
            "pending_question_tail_messages": 2,
            "resume_recent_useful_segments": 3,
            "resume_max_age_hours": 72,
        },
    )()
    service = ConversationSegmentationService(db, cfg)

    first = service.ensure_segment_for_turn(
        conversation=conversation,
        thread_id=thread.id,
        user_message_id=user1.id,
        surface_id="web",
        surface_instance_id="",
    )
    assert first.transitioned is True
    assert first.transition_reason == "manual_break"

    second = service.ensure_segment_for_turn(
        conversation=conversation,
        thread_id=thread.id,
        user_message_id=user2.id,
        surface_id="web",
        surface_instance_id="",
    )
    assert second.transitioned is True
    assert second.transition_reason == "idle_break"


def test_general_chat_prompt_history_is_scoped_to_open_segment(db):
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

    t0 = datetime.utcnow() - timedelta(hours=4)
    msgs = [
        Message(thread_id=thread.id, role=MessageRole.USER, content="old user", created_at=t0),
        Message(thread_id=thread.id, role=MessageRole.ASSISTANT, content="old assistant", created_at=t0 + timedelta(minutes=1)),
        Message(thread_id=thread.id, role=MessageRole.USER, content="new seg user", created_at=t0 + timedelta(hours=2)),
        Message(thread_id=thread.id, role=MessageRole.ASSISTANT, content="new seg assistant", created_at=t0 + timedelta(hours=2, minutes=1)),
    ]
    db.add_all(msgs)
    db.commit()
    for m in msgs:
        db.refresh(m)

    closed_seg = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="idle_break",
        state="closed",
        start_message_id=msgs[0].id,
        end_message_id=msgs[1].id,
        started_at=msgs[0].created_at,
        ended_at=msgs[1].created_at,
        usefulness="useful",
        summary_text="Earlier segment summary",
    )
    open_seg = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="idle_break",
        state="open",
        start_message_id=msgs[2].id,
        started_at=msgs[2].created_at,
        usefulness="unknown",
    )
    db.add_all([closed_seg, open_seg])
    db.commit()

    assembler = PromptAssemblyService(db=db, character_id="test_char")
    full_messages = assembler.message_repository.list_by_thread(thread.id, limit=1000)
    scoped_messages = assembler._scope_messages_to_general_chat_segment(
        messages=full_messages,
        conversation_id=conversation.id,
        active_segment_id=open_seg.id,
    )

    scoped_ids = [m.id for m in scoped_messages]
    assert msgs[0].id not in scoped_ids
    assert msgs[1].id not in scoped_ids
    assert scoped_ids[0] == msgs[2].id
    assert scoped_ids[1] == msgs[3].id


def test_idle_break_promotes_just_closed_segment_for_recap_source(db):
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

    t0 = datetime.utcnow() - timedelta(hours=8)
    user1 = Message(thread_id=thread.id, role=MessageRole.USER, content="old open user", created_at=t0)
    assistant1 = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="old open assistant",
        created_at=t0 + timedelta(minutes=1),
    )
    user2 = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="new turn after idle break",
        created_at=t0 + timedelta(hours=6),
    )
    db.add_all([user1, assistant1, user2])
    db.commit()
    db.refresh(user1)
    db.refresh(assistant1)
    db.refresh(user2)

    older_closed = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="idle_break",
        state="closed",
        start_message_id=user1.id,
        end_message_id=assistant1.id,
        started_at=t0 - timedelta(hours=12),
        ended_at=t0 - timedelta(hours=11),
        usefulness="useful",
        summary_text="Older useful summary",
    )
    open_segment = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="manual_break",
        state="open",
        start_message_id=user1.id,
        started_at=t0,
        usefulness="unknown",
    )
    db.add_all([older_closed, open_segment])
    db.commit()
    db.refresh(older_closed)
    db.refresh(open_segment)

    class _TokenCounter:
        @staticmethod
        def count_tokens(text: str) -> int:
            return len(text.split())

    class _AnalysisService:
        token_counter = _TokenCounter()
        archivist_model = "test-model"

        @staticmethod
        async def analyze_segment_summary_only(**kwargs):
            return SimpleNamespace(
                summary="Newest closed segment summary",
                usefulness="useful",
                key_events=[],
                open_threads=[],
                participants=[],
                summary_prompt_version="test-v1",
                summary_input_hash="hash-1",
            )

    app_state = {
        "analysis_service": _AnalysisService(),
        "characters": {"test_char": SimpleNamespace()},
        "system_config": SimpleNamespace(
            general_chat_segmentation=SimpleNamespace(
                enabled=True,
                idle_soft_minutes=30,
                idle_hard_minutes=120,
                sleep_break_minutes=360,
                density_window_messages=8,
                density_low_turn_count_max=4,
                density_low_avg_chars_max=180,
                pending_question_tail_messages=2,
                resume_recent_useful_segments=3,
                resume_max_age_hours=72,
                summary_max_tokens=800,
                summary_model_override=None,
            ),
            llm=SimpleNamespace(model="test-model", embedding_model="test-embed"),
        ),
        "embedding_service": SimpleNamespace(embed=lambda text: [0.01, 0.02, 0.03]),
        "segment_summary_vector_store": SimpleNamespace(upsert_segment_summary=lambda **kwargs: True),
    }
    dispatcher = ENSDispatcher(app_state)

    result = asyncio.run(
        dispatcher._ensure_segment_for_turn(
            db,
            {
                "conversation_id": conversation.id,
                "thread_id": thread.id,
                "user_message_id": user2.id,
                "character_id": "test_char",
                "surface_id": "web",
                "surface_instance_id": "",
            },
        )
    )

    assert result["transitioned"] is True
    assert result["transition_reason"] == "idle_break"
    assert result["summary_status"] == "generated"
    assert result["closed_segment_id"] is not None
    assert result["resume_source_segment_id"] == result["closed_segment_id"]

    new_open = (
        db.query(ConversationSegment)
        .filter(ConversationSegment.id == result["segment_id"])
        .first()
    )
    assert new_open is not None
    assert new_open.resume_source_segment_id == result["closed_segment_id"]


def test_soft_idle_break_density_is_scoped_to_open_segment_only(db):
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

    # Dense historical turns that should NOT influence soft-density checks for the current segment.
    base = datetime.utcnow() - timedelta(hours=6)
    historical = []
    for i in range(10):
        historical.append(
            Message(
                thread_id=thread.id,
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=("x" * 260) + f" hist-{i}",
                created_at=base + timedelta(minutes=i),
            )
        )
    db.add_all(historical)
    db.commit()

    # Open segment starts later with only two lightweight turns.
    seg_start_user = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="hi",
        created_at=base + timedelta(hours=5),
    )
    seg_start_assistant = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="bye",
        created_at=base + timedelta(hours=5, seconds=10),
    )
    trigger_user = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="new turn after a few minutes",
        created_at=base + timedelta(hours=5, minutes=4),
    )
    db.add_all([seg_start_user, seg_start_assistant, trigger_user])
    db.commit()
    db.refresh(seg_start_user)
    db.refresh(seg_start_assistant)
    db.refresh(trigger_user)

    open_segment = ConversationSegment(
        conversation_id=conversation.id,
        relationship_id=None,
        surface_id="web",
        surface_instance_id="",
        segment_kind="manual_break",
        state="open",
        start_message_id=seg_start_user.id,
        started_at=seg_start_user.created_at,
        usefulness="unknown",
    )
    db.add(open_segment)
    db.commit()

    cfg = type(
        "Cfg",
        (),
        {
            "enabled": True,
            "idle_soft_minutes": 2,
            "idle_hard_minutes": 5,
            "sleep_break_minutes": 360,
            "density_window_messages": 8,
            "density_low_turn_count_max": 4,
            "density_low_avg_chars_max": 180,
            "pending_question_tail_messages": 2,
            "resume_recent_useful_segments": 3,
            "resume_max_age_hours": 72,
        },
    )()
    service = ConversationSegmentationService(db, cfg)

    result = service.ensure_segment_for_turn(
        conversation=conversation,
        thread_id=thread.id,
        user_message_id=trigger_user.id,
        surface_id="web",
        surface_instance_id="",
    )

    assert result.transitioned is True
    assert result.transition_reason == "idle_break"
    assert result.closed_segment_id is not None
