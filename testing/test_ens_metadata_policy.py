from chorus_engine.ens.metadata_policy import (
    PROVENANCE_WRITE_ONCE_KEYS,
    SYSTEM_METADATA_KEYS,
    canonicalize_surface_id,
    sanitize_metadata_patch,
)


def test_system_metadata_keys_v1_contains_locked_keys():
    expected = {
        "system.hidden",
        "system.soft_deleted",
        "system.delete_reason",
        "system.edited",
        "system.surface_id",
        "system.source",
        "system.speaker_external_id",
        "system.external_thread_id",
        "system.external_message_id",
        "system.ingest_kind",
        "system.client_message_id",
        "system.tool_call_id",
        "system.image_id",
        "system.video_id",
        "system.audio_id",
    }
    assert expected.issubset(SYSTEM_METADATA_KEYS)


def test_provenance_write_once_keys_locked():
    expected = {
        "system.ingest_kind",
        "system.surface_id",
        "system.source",
        "system.speaker_external_id",
        "system.external_thread_id",
        "system.external_message_id",
        "system.client_message_id",
    }
    assert expected == PROVENANCE_WRITE_ONCE_KEYS


def test_sanitize_rejects_non_allowlisted_system_keys():
    accepted, rejected = sanitize_metadata_patch(
        existing_metadata={},
        patch={"system.raw_tool_payload": {"foo": "bar"}},
    )
    assert accepted == {}
    assert rejected == [{"key": "system.raw_tool_payload", "reason": "system_key_not_allowlisted"}]


def test_sanitize_enforces_write_once_for_provenance():
    accepted, rejected = sanitize_metadata_patch(
        existing_metadata={"system.external_message_id": "m1"},
        patch={"system.external_message_id": "m2", "system.hidden": True},
    )
    assert accepted == {"system.hidden": True}
    assert {"key": "system.external_message_id", "reason": "write_once_provenance_key"} in rejected


def test_sanitize_normalizes_adapter_surface_namespace():
    accepted, rejected = sanitize_metadata_patch(
        existing_metadata={},
        patch={
            "adapters.DiScOrD.message_id": "123",
            "adapters.custom.foo": "bar",
        },
    )
    assert accepted["adapters.discord.message_id"] == "123"
    assert accepted["adapters.unknown.foo"] == "bar"
    assert rejected == []


def test_sanitize_rejects_unscoped_metadata_keys():
    accepted, rejected = sanitize_metadata_patch(
        existing_metadata={},
        patch={"discord_message_id": "abc"},
    )
    assert accepted == {}
    assert rejected == [{"key": "discord_message_id", "reason": "must_use_system_or_adapters_namespace"}]


def test_canonicalize_surface_id():
    assert canonicalize_surface_id("web") == "web"
    assert canonicalize_surface_id("VOICE") == "voice"
    assert canonicalize_surface_id("slack") == "unknown"

