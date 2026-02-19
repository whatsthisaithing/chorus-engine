import asyncio

from chorus_engine.ens.surface_router import SurfaceRouter
from chorus_engine.models.conversation import Conversation, Thread
from chorus_engine.models.ens import SurfaceBinding


def test_slice6_router_general_chat_reuses_persistent_target(db):
    router = SurfaceRouter(db)

    first = router.resolve(
        assistant_id="test_char",
        surface_id="web",
        external_thread_id="ext-gc-001",
        target_hint="general_chat",
    )
    second = router.resolve(
        assistant_id="test_char",
        surface_id="web",
        external_thread_id="ext-gc-002",
        target_hint="general_chat",
    )

    assert first.conversation_id == second.conversation_id
    assert first.thread_id == second.thread_id
    assert first.relationship_id
    assert second.relationship_id == first.relationship_id

    conv = db.query(Conversation).filter(Conversation.id == first.conversation_id).first()
    assert conv is not None
    assert conv.conversation_kind == "general_chat"
    assert conv.relationship_id == first.relationship_id

    binding_count = (
        db.query(SurfaceBinding)
        .filter(SurfaceBinding.conversation_id == first.conversation_id)
        .count()
    )
    assert binding_count == 2


def test_slice6_router_non_general_target_hint_is_ignored_for_routing(db):
    router = SurfaceRouter(db)
    resolved = router.resolve(
        assistant_id="test_char",
        surface_id="web",
        external_thread_id="ext-std-001",
        target_hint="relationship_dm",
    )

    conv = db.query(Conversation).filter(Conversation.id == resolved.conversation_id).first()
    assert conv is not None
    assert conv.conversation_kind == "standard"
    assert resolved.ignored_target_hint == "relationship_dm"


def test_title_autogen_is_disabled_for_general_chat(db, helpers):
    class _FailIfCalledTitleService:
        async def generate_title(self, **kwargs):
            _ = kwargs
            raise AssertionError("title generation should be disabled for general_chat")

    helpers.app_module.app_state["title_service"] = _FailIfCalledTitleService()

    conv = Conversation(
        character_id="test_char",
        title="General Chat with Test Character",
        source="web",
        conversation_kind="general_chat",
        title_auto_generated=1,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    thread = Thread(conversation_id=conv.id, title="Main Thread")
    db.add(thread)
    db.commit()
    db.refresh(thread)

    dispatcher = helpers.app_module.app_state["ens_runtime"].dispatcher
    result = asyncio.run(
        dispatcher._maybe_update_conversation_title(
            db,
            {"thread_id": thread.id, "character_id": "test_char"},
        )
    )
    assert result["updated"] is False
    assert result["reason"] == "general_chat_title_locked"
