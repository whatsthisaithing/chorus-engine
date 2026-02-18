from chorus_engine.models.ens import SurfaceEgressIntent


def _base_intent_payload():
    return {
        "surface_id": "discord",
        "surface_instance_id": None,
        "external_thread_id": "discord-thread-1",
        "payload_json": {
            "content_type": "text",
            "text": "hello from outbox",
        },
        "assistant_id": "test_char",
        "trace_json": {"source": "test"},
    }


def test_slice65_create_and_replay_intent(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice65_egress_outbox_ownership=True,
    )
    helpers.app_module.app_state["system_config"].debug_ui = True

    payload = _base_intent_payload()
    first = client.post("/debug/egress/send-intent", json=payload)
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["created"] is True
    assert first_body["replayed"] is False
    assert first_body["status"] == "pending"

    second = client.post("/debug/egress/send-intent", json=payload)
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["created"] is False
    assert second_body["replayed"] is True
    assert second_body["intent_id"] == first_body["intent_id"]
    assert second_body["status"] == "pending"

    rows = db.query(SurfaceEgressIntent).all()
    assert len(rows) == 1


def test_slice65_list_ack_fail_and_terminal_monotonicity(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice65_egress_outbox_ownership=True,
    )
    helpers.app_module.app_state["system_config"].debug_ui = True

    first = client.post("/debug/egress/send-intent", json=_base_intent_payload())
    assert first.status_code == 200, first.text
    first_id = first.json()["intent_id"]

    ack = client.post(f"/egress/intents/{first_id}/ack", json={"metadata": {"provider_msg_id": "abc"}})
    assert ack.status_code == 200, ack.text
    assert ack.json()["status"] == "delivered"

    fail_after_ack = client.post(f"/egress/intents/{first_id}/fail", json={"error": "should_not_apply"})
    assert fail_after_ack.status_code == 200, fail_after_ack.text
    assert fail_after_ack.json()["status"] == "delivered"

    second_payload = _base_intent_payload()
    second_payload["external_thread_id"] = "discord-thread-2"
    second = client.post("/debug/egress/send-intent", json=second_payload)
    assert second.status_code == 200, second.text
    second_id = second.json()["intent_id"]

    fail = client.post(f"/egress/intents/{second_id}/fail", json={"error": "transport_failed"})
    assert fail.status_code == 200, fail.text
    assert fail.json()["status"] == "failed"
    assert fail.json()["attempt_count"] == 1

    fail_again = client.post(f"/egress/intents/{second_id}/fail", json={"error": "second_try"})
    assert fail_again.status_code == 200, fail_again.text
    assert fail_again.json()["status"] == "failed"
    assert fail_again.json()["attempt_count"] == 1

    ack_after_fail = client.post(f"/egress/intents/{second_id}/ack", json={})
    assert ack_after_fail.status_code == 200, ack_after_fail.text
    assert ack_after_fail.json()["status"] == "failed"

    listed = client.get("/egress/intents?surface_id=discord&status=failed&limit=10")
    assert listed.status_code == 200, listed.text
    assert any(item["id"] == second_id for item in listed.json()["intents"])


def test_slice65_does_not_emit_web_chat_outbox(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice65_egress_outbox_ownership=True,
    )
    conversation_id, thread_id = helpers.create_conversation_thread()

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={
            "message": "hello web path",
            "metadata": {"client_message_id": "slice65-web-no-outbox-1"},
        },
    )
    assert resp.status_code == 200, resp.text
    _ = conversation_id

    count = db.query(SurfaceEgressIntent).count()
    assert count == 0
