from chorus_engine.models.ens import ENSActionResult, ENSDecision
from chorus_engine.models.workflow import Workflow
from pathlib import Path


def test_slice4_user_identity_update_routes_through_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )

    payload = {"display_name": "Local Owner", "aliases": ["Me", "Owner"]}
    resp = client.put("/system/user-identity", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["user_identity"]["display_name"] == "Local Owner"

    decision_count = db.query(ENSDecision).filter(ENSDecision.signal_type == "config.system.change_requested").count()
    apply_count = db.query(ENSActionResult).filter(ENSActionResult.kind == "config.system.apply").count()
    assert decision_count >= 1
    assert apply_count >= 1


def test_slice4_user_identity_no_change_returns_skipped(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )
    payload = {"display_name": "No Change", "aliases": ["Alias"]}
    r1 = client.put("/system/user-identity", json=payload)
    r2 = client.put("/system/user-identity", json=payload)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    skipped = (
        db.query(ENSActionResult)
        .filter(
            ENSActionResult.kind == "config.system.apply",
            ENSActionResult.status == "skipped",
        )
        .count()
    )
    assert skipped >= 1


def test_slice4_conversation_toggles_route_through_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )
    conversation_id, _thread_id = helpers.create_conversation_thread()

    p = client.put(f"/conversations/{conversation_id}/privacy", json={"is_private": True})
    assert p.status_code == 200, p.text
    assert p.json()["is_private"] is True

    m = client.patch(
        f"/conversations/{conversation_id}/media-offers",
        json={"allow_image_offers": False, "allow_video_offers": True},
    )
    assert m.status_code == 200, m.text
    assert m.json()["allow_image_offers"] is False
    assert m.json()["allow_video_offers"] is True

    t = client.patch(f"/conversations/{conversation_id}/tts", json={"enabled": True})
    assert t.status_code == 200, t.text
    assert t.json()["tts_enabled"] is True

    actions = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "config.conversation.apply")
        .count()
    )
    assert actions >= 3


def test_slice4_workflow_upload_routes_through_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )

    workflow_data = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "__CHORUS_PROMPT__"},
        }
    }
    resp = client.post("/characters/test_char/workflows/s4wf?workflow_type=image", json=workflow_data)
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True

    rows = db.query(Workflow).filter(Workflow.character_name == "test_char", Workflow.workflow_name == "s4wf").all()
    assert len(rows) == 1

    decision_count = db.query(ENSDecision).filter(ENSDecision.signal_type == "config.workflow.change_requested").count()
    rollback_count = db.query(ENSActionResult).filter(ENSActionResult.kind == "config.workflow.rollback_file_or_db").count()
    assert decision_count >= 1
    assert rollback_count >= 1


def test_slice4_core_memory_reload_routes_through_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )

    resp = client.post("/characters/test_char/reload-core-memories")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "reloaded"
    assert body["character_id"] == "test_char"

    diff_count = db.query(ENSActionResult).filter(ENSActionResult.kind == "config.core_memory.diff").count()
    assert diff_count >= 1


def test_config_drift_detects_out_of_band_character_yaml_edit(client, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )
    path = Path("characters/test_char.yaml")
    original = path.read_text(encoding="utf-8")
    path.write_text(original + "\n# drifted edit\n", encoding="utf-8")

    resp = client.get("/config/drift")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["drifted"] is True
    assert "test_char.yaml" in body["character_changes"]["changed"]


def test_slice4_reload_controls_route_via_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice4_config_ownership=True,
    )

    r2 = client.post("/characters/reload")
    assert r2.status_code == 200, r2.text
    assert r2.json()["success"] is True

    character_decisions = db.query(ENSDecision).filter(ENSDecision.signal_type == "config.character.change_requested").count()
    assert character_decisions >= 1

    r1 = client.post("/system/config/reload")
    assert r1.status_code == 200, r1.text
    assert r1.json()["success"] is True

    system_decisions = db.query(ENSDecision).filter(ENSDecision.signal_type == "config.system.change_requested").count()
    assert system_decisions >= 1
