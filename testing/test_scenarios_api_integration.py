from pathlib import Path
from io import BytesIO

from PIL import Image


def _enable_scenarios(helpers):
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.scenarios_enabled = True


def test_scenario_feature_gate_blocks_library_endpoints(client, helpers):
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.scenarios_enabled = False

    resp = client.get("/characters/test_char/scenarios")
    assert resp.status_code == 400, resp.text
    assert "disabled" in resp.json()["detail"].lower()


def test_scenario_library_crud_flow(client, helpers):
    _enable_scenarios(helpers)

    create_resp = client.post(
        "/characters/test_char/scenarios",
        json={
            "title": "Library Scenario",
            "description": "Initial state",
            "scenario_text": "The room is quiet and dimly lit.",
            "tags": ["intro", "mood"],
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    created = create_resp.json()
    scenario_id = created["id"]
    assert created["title"] == "Library Scenario"

    list_resp = client.get("/characters/test_char/scenarios")
    assert list_resp.status_code == 200, list_resp.text
    scenarios = list_resp.json()["scenarios"]
    assert any(item["id"] == scenario_id for item in scenarios)

    update_resp = client.patch(
        f"/characters/test_char/scenarios/{scenario_id}",
        json={"title": "Library Scenario Updated", "scenario_text": "New text"},
    )
    assert update_resp.status_code == 200, update_resp.text
    assert update_resp.json()["title"] == "Library Scenario Updated"

    dup_resp = client.post(f"/characters/test_char/scenarios/{scenario_id}/duplicate")
    assert dup_resp.status_code == 200, dup_resp.text
    assert dup_resp.json()["id"] != scenario_id

    del_resp = client.delete(f"/characters/test_char/scenarios/{scenario_id}")
    assert del_resp.status_code == 200, del_resp.text
    assert del_resp.json()["success"] is True


def test_create_conversation_with_library_scenario_snapshot(client, helpers):
    _enable_scenarios(helpers)

    create_scenario = client.post(
        "/characters/test_char/scenarios",
        json={
            "title": "Library Seed",
            "description": "seed",
            "scenario_text": "A storm rolls in.",
        },
    )
    assert create_scenario.status_code == 200, create_scenario.text
    scenario_id = create_scenario.json()["id"]

    conv_resp = client.post(
        "/conversations",
        json={
            "character_id": "test_char",
            "title": "Scenario Conversation",
            "source": "web",
            "scenario_mode": "library",
            "scenario_id": scenario_id,
        },
    )
    assert conv_resp.status_code == 200, conv_resp.text
    body = conv_resp.json()
    assert body["scenario_source"] == "library"
    assert body["scenario_id"] == scenario_id
    assert body["scenario_title"] == "Library Seed"
    assert body["scenario_text"] == "A storm rolls in."


def test_create_conversation_with_custom_scenario_and_save_to_library(client, helpers):
    _enable_scenarios(helpers)

    conv_resp = client.post(
        "/conversations",
        json={
            "character_id": "test_char",
            "title": "Custom Scenario Conversation",
            "source": "web",
            "scenario_mode": "custom",
            "custom_scenario_text": "You are preparing for a midnight launch.",
            "save_custom_to_library": True,
        },
    )
    assert conv_resp.status_code == 200, conv_resp.text
    body = conv_resp.json()
    assert body["scenario_source"] == "custom"
    assert body["scenario_title"] == "Custom Scenario"
    assert body["scenario_text"] == "You are preparing for a midnight launch."
    assert body["scenario_id"] is not None

    list_resp = client.get("/characters/test_char/scenarios")
    assert list_resp.status_code == 200, list_resp.text
    assert any(item["id"] == body["scenario_id"] for item in list_resp.json()["scenarios"])

    expected_path = Path("data/scenarios/test_char")
    assert expected_path.exists()


def test_scenario_image_upload_uses_scenario_images_directory(client, helpers):
    _enable_scenarios(helpers)
    create_scenario = client.post(
        "/characters/test_char/scenarios",
        json={"title": "Image Scenario", "scenario_text": "Image path test"},
    )
    assert create_scenario.status_code == 200, create_scenario.text
    scenario_id = create_scenario.json()["id"]

    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    resp = client.post(
        f"/characters/test_char/scenarios/{scenario_id}/image",
        files={"file": ("scenario.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["image_url"].startswith("/scenario_images/")
    filename = Path(str(body["image_ref"])).name
    assert (Path("data/scenario_images") / filename).exists()
