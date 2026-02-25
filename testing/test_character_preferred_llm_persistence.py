from pathlib import Path

import yaml


def _read_preferred_llm_from_yaml(character_id: str = "test_char") -> dict:
    path = Path("characters") / f"{character_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("preferred_llm") or {}


def _assert_extended_fields(preferred_llm: dict) -> None:
    assert preferred_llm.get("top_p") == 0.55
    assert preferred_llm.get("top_k") == 17
    assert preferred_llm.get("repeat_penalty") == 1.15
    assert preferred_llm.get("presence_penalty") == 0.35
    assert preferred_llm.get("frequency_penalty") == -0.25


def test_character_patch_persists_extended_preferred_llm_fields(client, helpers):
    helpers.set_ens_flags(
        enabled=False,
        slice1_chat_ownership=False,
    )

    resp = client.patch(
        "/characters/test_char",
        json={
            "preferred_llm": {
                "top_p": 0.55,
                "top_k": 17,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.35,
                "frequency_penalty": -0.25,
            }
        },
    )
    assert resp.status_code == 200, resp.text

    preferred_llm = _read_preferred_llm_from_yaml("test_char")
    _assert_extended_fields(preferred_llm)


def test_character_patch_persists_extended_preferred_llm_fields_slice4(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )

    resp = client.patch(
        "/characters/test_char",
        json={
            "preferred_llm": {
                "top_p": 0.55,
                "top_k": 17,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.35,
                "frequency_penalty": -0.25,
            }
        },
    )
    assert resp.status_code == 200, resp.text

    preferred_llm = _read_preferred_llm_from_yaml("test_char")
    _assert_extended_fields(preferred_llm)


def test_character_patch_partially_merges_preferred_llm_fields(client, helpers):
    helpers.set_ens_flags(
        enabled=False,
        slice1_chat_ownership=False,
    )

    seed = client.patch(
        "/characters/test_char",
        json={"preferred_llm": {"model": "seed-model", "max_tokens": 777}},
    )
    assert seed.status_code == 200, seed.text

    resp = client.patch(
        "/characters/test_char",
        json={"preferred_llm": {"top_p": 0.61, "presence_penalty": 0.4}},
    )
    assert resp.status_code == 200, resp.text

    preferred_llm = _read_preferred_llm_from_yaml("test_char")
    assert preferred_llm.get("model") == "seed-model"
    assert preferred_llm.get("max_tokens") == 777
    assert preferred_llm.get("top_p") == 0.61
    assert preferred_llm.get("presence_penalty") == 0.4


def test_character_patch_partially_merges_preferred_llm_fields_slice4(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )

    seed = client.patch(
        "/characters/test_char",
        json={"preferred_llm": {"model": "seed-model", "max_tokens": 777}},
    )
    assert seed.status_code == 200, seed.text

    resp = client.patch(
        "/characters/test_char",
        json={"preferred_llm": {"top_p": 0.61, "presence_penalty": 0.4}},
    )
    assert resp.status_code == 200, resp.text

    preferred_llm = _read_preferred_llm_from_yaml("test_char")
    assert preferred_llm.get("model") == "seed-model"
    assert preferred_llm.get("max_tokens") == 777
    assert preferred_llm.get("top_p") == 0.61
    assert preferred_llm.get("presence_penalty") == 0.4
