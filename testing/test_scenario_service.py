from pathlib import Path

import pytest

from chorus_engine.services.scenario_service import MAX_SCENARIO_TEXT_LEN, ScenarioService


def test_scenario_service_crud_round_trip(tmp_path):
    service = ScenarioService(base_dir=tmp_path / "scenarios")
    character_id = "test_char"

    created = service.create_scenario(
        character_id,
        {
            "title": "Test Scenario",
            "description": "scenario description",
            "scenario_text": "Once upon a test.",
            "tags": ["alpha", "beta"],
        },
    )
    assert created.id
    assert created.title == "Test Scenario"

    listed = service.list_scenarios(character_id)
    assert len(listed) == 1
    assert listed[0].id == created.id

    updated = service.update_scenario(
        character_id,
        created.id,
        {"title": "Updated Scenario", "scenario_text": "Updated text."},
    )
    assert updated.title == "Updated Scenario"
    assert updated.scenario_text == "Updated text."

    loaded = service.get_scenario(character_id, created.id)
    assert loaded is not None
    assert loaded.title == "Updated Scenario"

    duplicate = service.duplicate_scenario(character_id, created.id)
    assert duplicate.id != created.id
    assert duplicate.title.startswith("Updated Scenario")

    assert service.delete_scenario(character_id, created.id) is True
    assert service.get_scenario(character_id, created.id) is None


def test_scenario_service_rejects_overlong_text(tmp_path):
    service = ScenarioService(base_dir=tmp_path / "scenarios")
    with pytest.raises(ValueError):
        service.create_scenario(
            "test_char",
            {
                "title": "Too Long",
                "scenario_text": "x" * (MAX_SCENARIO_TEXT_LEN + 1),
            },
        )


def test_scenario_service_atomic_write_creates_file(tmp_path):
    base_dir = tmp_path / "scenarios"
    service = ScenarioService(base_dir=base_dir)
    created = service.create_scenario(
        "test_char",
        {"title": "Atomic", "scenario_text": "atomic data"},
    )
    scenario_path = Path(base_dir) / "test_char" / f"scn_{created.id}.yaml"
    assert scenario_path.exists()
