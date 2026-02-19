def test_surface_envelope_fields_do_not_implicitly_default_general_chat(helpers):
    app_module = helpers.app_module

    neutral = app_module._build_surface_envelope_fields(
        thread_id="thread-1",
        metadata={},
        conversation_source="web",
        speaker_role="user",
        target_hint_default=None,
    )
    assert neutral["target_hint"] is None
    assert neutral["surface_id"] == "web"

    explicit_general = app_module._build_surface_envelope_fields(
        thread_id="thread-1",
        metadata={},
        conversation_source="web",
        speaker_role="user",
        target_hint_default="general_chat",
    )
    assert explicit_general["target_hint"] == "general_chat"

    metadata_hint_wins = app_module._build_surface_envelope_fields(
        thread_id="thread-1",
        metadata={"target_hint": "relationship_dm"},
        conversation_source="web",
        speaker_role="user",
        target_hint_default="general_chat",
    )
    assert metadata_hint_wins["target_hint"] == "relationship_dm"
