import asyncio
import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chorus_engine.ens.models import Signal
import chorus_engine.ens.time_utils as time_utils
from chorus_engine.models.conversation import Conversation
from chorus_engine.models.ens import ENSDecision, ENSFloorControlState, ENSSchedulerTick, ENSSignalQueue, ENSToolCallRequest


def test_v3_scheduler_scaffold_enqueue_and_tick_executes_selected_signal(helpers, db):
    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
    )

    queued = asyncio.run(runtime.enqueue_signal(signal))
    assert queued["signal_id"] == signal.signal_id
    assert queued["status"] == "pending"

    outcome = asyncio.run(runtime.scheduler_tick())
    assert outcome is not None
    assert outcome.signal_id == signal.signal_id

    queue_row = (
        db.query(ENSSignalQueue)
        .filter(ENSSignalQueue.signal_id == signal.signal_id)
        .first()
    )
    assert queue_row is not None
    assert queue_row.status == "done"

    tick_row = (
        db.query(ENSSchedulerTick)
        .filter(ENSSchedulerTick.selected_signal_id == signal.signal_id)
        .first()
    )
    assert tick_row is not None
    assert tick_row.reason_trace_json.get("selection") == "priority_then_created_at_then_signal_id"
    assert int(tick_row.reason_trace_json.get("candidate_count") or 0) >= 1
    assert tick_row.reason_trace_json.get("selected_signal_id") == signal.signal_id


def test_v3_scheduler_deterministic_tie_break_for_identical_created_at_us(helpers):
    runtime = helpers.app_module.app_state["ens_runtime"]
    created_at_us = 2_000_000
    signal_ids = [
        "00000000-0000-0000-0000-000000000003",
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]

    for signal_id in signal_ids:
        signal = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="test",
            payload={"kind": "noop"},
        )
        signal.signal_id = signal_id
        signal.created_at_us = created_at_us
        asyncio.run(runtime.enqueue_signal(signal))

    selected = []
    for _ in signal_ids:
        outcome = asyncio.run(runtime.scheduler_tick())
        assert outcome is not None
        selected.append(outcome.signal_id)

    assert selected == sorted(signal_ids)


def test_v3_scheduler_user_preempts_loop_and_system_flood(helpers):
    runtime = helpers.app_module.app_state["ens_runtime"]
    created_at_us = 3_000_000

    for idx in range(5):
        s = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="test",
            payload={"kind": f"sys-{idx}"},
        )
        s.created_at_us = created_at_us + idx
        asyncio.run(runtime.enqueue_signal(s))

    for idx in range(5):
        s = Signal(
            type="loop.progression",
            scope="SESSION",
            source="test",
            payload={"kind": f"loop-{idx}"},
        )
        s.created_at_us = created_at_us + 100 + idx
        asyncio.run(runtime.enqueue_signal(s))

    user_signal = Signal(
        type="user.message",
        scope="SESSION",
        source="test",
        payload={"conversation_id": "conv-preempt", "thread_id": "thread-preempt", "content": "hi"},
    )
    user_signal.created_at_us = created_at_us + 1000
    asyncio.run(runtime.enqueue_signal(user_signal))

    first = asyncio.run(runtime.scheduler_tick())
    assert first is not None
    assert first.signal_id == user_signal.signal_id


def test_v3_scheduler_user_not_starved_by_non_user_backlog(helpers):
    runtime = helpers.app_module.app_state["ens_runtime"]
    created_at_us = 4_000_000

    for idx in range(20):
        s = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="test",
            payload={"kind": f"sys-{idx}"},
        )
        s.created_at_us = created_at_us + idx
        asyncio.run(runtime.enqueue_signal(s))

    user_signal = Signal(
        type="user.message",
        scope="SESSION",
        source="test",
        payload={"conversation_id": "conv-starve", "thread_id": "thread-starve", "content": "priority me"},
    )
    user_signal.created_at_us = created_at_us + 500
    asyncio.run(runtime.enqueue_signal(user_signal))

    outcome = asyncio.run(runtime.scheduler_tick())
    assert outcome is not None
    assert outcome.signal_id == user_signal.signal_id


def test_v3_scheduler_no_surface_reopen_gating_non_active_surface_still_runnable(helpers):
    runtime = helpers.app_module.app_state["ens_runtime"]
    first = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        surface_id="discord",
        payload={"kind": "discord-surface-noop"},
    )
    second = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        surface_id="web",
        payload={"kind": "web-surface-noop"},
    )

    asyncio.run(runtime.enqueue_signal(first))
    asyncio.run(runtime.enqueue_signal(second))
    o1 = asyncio.run(runtime.scheduler_tick())
    o2 = asyncio.run(runtime.scheduler_tick())
    assert o1 is not None and o2 is not None
    assert {o1.signal_id, o2.signal_id} == {first.signal_id, second.signal_id}


def test_v3_scheduler_non_user_fairness_rotates_by_surface(helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.v3_arbitration_enabled = True

    runtime = helpers.app_module.app_state["ens_runtime"]
    created_at_us = 5_000_000
    # Backlog skewed toward one surface.
    for idx in range(3):
        s = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="test",
            surface_id="web",
            payload={"kind": f"web-{idx}"},
        )
        s.created_at_us = created_at_us + idx
        asyncio.run(runtime.enqueue_signal(s))
    for idx in range(3):
        s = Signal(
            type="system.noop",
            scope="SYSTEM",
            source="test",
            surface_id="discord",
            payload={"kind": f"discord-{idx}"},
        )
        s.created_at_us = created_at_us + 10 + idx
        asyncio.run(runtime.enqueue_signal(s))

    selected_surface_ids = []
    for _ in range(4):
        outcome = asyncio.run(runtime.scheduler_tick())
        assert outcome is not None
        row = db.query(ENSSignalQueue).filter(ENSSignalQueue.signal_id == outcome.signal_id).first()
        assert row is not None
        selected_surface_ids.append(row.surface_id)

    # v3.2 behavior: non-user fairness prevents monopolization by one surface.
    assert selected_surface_ids[0] != selected_surface_ids[1]


def test_v32_unresolved_relationship_id_fallback_normalized_once(helpers, db):
    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
    )
    queued = asyncio.run(runtime.enqueue_signal(signal))
    row = db.query(ENSSignalQueue).filter(ENSSignalQueue.queue_id == queued["queue_id"]).first()
    assert row is not None
    assert row.relationship_id == "system/unknown"


def test_v33_attention_lock_weights_non_user_selection_without_gating(helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.v3_arbitration_enabled = True
    ens_cfg.scheduler_attention_lock_seconds = 120

    runtime = helpers.app_module.app_state["ens_runtime"]
    rel = "rel-v33-lock"

    user = Signal(
        type="user.message",
        scope="SESSION",
        source="test",
        relationship_hint=rel,
        surface_id="web",
        payload={"conversation_id": "conv-v33", "thread_id": "thread-v33", "content": "lock web"},
    )
    asyncio.run(runtime.enqueue_signal(user))

    older_other_surface = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        relationship_hint=rel,
        surface_id="discord",
        payload={"kind": "other-surface"},
    )
    older_other_surface.created_at_us = 6_000_000
    asyncio.run(runtime.enqueue_signal(older_other_surface))

    newer_locked_surface = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        relationship_hint=rel,
        surface_id="web",
        payload={"kind": "locked-surface"},
    )
    newer_locked_surface.created_at_us = 6_000_100
    asyncio.run(runtime.enqueue_signal(newer_locked_surface))

    # User preemption first.
    first = asyncio.run(runtime.scheduler_tick())
    assert first is not None
    assert first.signal_id == user.signal_id

    # Then non-user selection should be influenced by lock to prefer web despite created_at order.
    second = asyncio.run(runtime.scheduler_tick())
    assert second is not None
    assert second.signal_id == newer_locked_surface.signal_id

    tick = (
        db.query(ENSSchedulerTick)
        .filter(ENSSchedulerTick.selected_signal_id == newer_locked_surface.signal_id)
        .first()
    )
    assert tick is not None
    assert bool((tick.reason_trace_json or {}).get("attention_lock_applied")) is True

    # No reopen gating: remaining non-active-surface signal still runnable next tick.
    third = asyncio.run(runtime.scheduler_tick())
    assert third is not None
    assert third.signal_id == older_other_surface.signal_id

    lock_state = (
        db.query(ENSFloorControlState)
        .filter(ENSFloorControlState.relationship_id == rel)
        .first()
    )
    assert lock_state is not None
    assert lock_state.active_surface_id == "web"


def test_v33_expired_attention_lock_does_not_apply_weighting(helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.v3_arbitration_enabled = True
    ens_cfg.scheduler_attention_lock_seconds = 120

    runtime = helpers.app_module.app_state["ens_runtime"]
    rel = "rel-v33-expired-lock"

    user = Signal(
        type="user.message",
        scope="SESSION",
        source="test",
        relationship_hint=rel,
        surface_id="web",
        payload={"conversation_id": "conv-v33-expired", "thread_id": "thread-v33-expired", "content": "lock web"},
    )
    asyncio.run(runtime.enqueue_signal(user))
    first = asyncio.run(runtime.scheduler_tick())
    assert first is not None
    assert first.signal_id == user.signal_id

    lock_state = (
        db.query(ENSFloorControlState)
        .filter(ENSFloorControlState.relationship_id == rel)
        .first()
    )
    assert lock_state is not None
    lock_state.attention_lock_until_us = 1
    db.commit()

    older_other_surface = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        relationship_hint=rel,
        surface_id="discord",
        payload={"kind": "older-other-surface"},
    )
    older_other_surface.created_at_us = 7_000_000
    asyncio.run(runtime.enqueue_signal(older_other_surface))

    newer_locked_surface = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        relationship_hint=rel,
        surface_id="web",
        payload={"kind": "newer-web-surface"},
    )
    newer_locked_surface.created_at_us = 7_000_100
    asyncio.run(runtime.enqueue_signal(newer_locked_surface))

    second = asyncio.run(runtime.scheduler_tick())
    assert second is not None
    assert second.signal_id == older_other_surface.signal_id

    tick = (
        db.query(ENSSchedulerTick)
        .filter(ENSSchedulerTick.selected_signal_id == older_other_surface.signal_id)
        .first()
    )
    assert tick is not None
    assert bool((tick.reason_trace_json or {}).get("attention_lock_applied")) is False


def test_v33_floor_state_upsert_recovers_from_insert_race(helpers, db, monkeypatch):
    runtime = helpers.app_module.app_state["ens_runtime"]
    scheduler = runtime.scheduler
    rel = "rel-v33-floor-race"
    signal = Signal(
        type="user.message",
        scope="SESSION",
        source="test",
        relationship_hint=rel,
        surface_id="web",
        payload={"conversation_id": "conv-race", "thread_id": "thread-race", "content": "race"},
    )

    original_commit = db.commit
    call_count = {"n": 0}

    def flaky_commit():
        if call_count["n"] == 0:
            call_count["n"] += 1
            other = Session(bind=db.bind)
            try:
                existing = (
                    other.query(ENSFloorControlState)
                    .filter(ENSFloorControlState.relationship_id == rel)
                    .first()
                )
                if not existing:
                    other.add(
                        ENSFloorControlState(
                            id=str(uuid.uuid4()),
                            relationship_id=rel,
                            active_surface_id="discord",
                            attention_lock_until_us=1,
                            lock_source_signal_id="seed-signal",
                            metadata_json={"mode": "deconfliction_weight_only"},
                        )
                    )
                    other.commit()
            finally:
                other.close()
            raise IntegrityError("INSERT", {}, Exception("simulated unique race"))
        return original_commit()

    monkeypatch.setattr(db, "commit", flaky_commit)
    scheduler.update_floor_state_from_signal(db, signal, attention_lock_seconds=120)

    row = (
        db.query(ENSFloorControlState)
        .filter(ENSFloorControlState.relationship_id == rel)
        .first()
    )
    assert row is not None
    assert row.active_surface_id == "web"
    assert int(row.attention_lock_until_us or 0) > 1
    assert row.lock_source_signal_id == signal.signal_id
    assert dict(row.metadata_json or {}).get("mode") == "deconfliction_weight_only"


def test_v3_scheduler_enqueue_dedupes_by_signal_idempotency_key(helpers, db):
    runtime = helpers.app_module.app_state["ens_runtime"]
    key = "slice31:dedupe:001"
    first = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
        idempotency_key=key,
    )
    second = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop-2"},
        idempotency_key=key,
    )

    first_row = asyncio.run(runtime.enqueue_signal(first))
    second_row = asyncio.run(runtime.enqueue_signal(second))

    assert second_row["queue_id"] == first_row["queue_id"]
    assert second_row["signal_id"] == first_row["signal_id"]
    assert (
        db.query(ENSSignalQueue)
        .filter(ENSSignalQueue.idempotency_key == key)
        .count()
        == 1
    )


def test_v3_scheduler_queue_db_unique_idempotency_key_enforced(db):
    key = "slice31:db-unique:001"
    row1 = ENSSignalQueue(
        queue_id=str(uuid.uuid4()),
        signal_id=str(uuid.uuid4()),
        signal_type="system.noop",
        priority_tier="system",
        created_at_us=1_000_000,
        idempotency_key=key,
        signal_json={"type": "system.noop"},
        status="pending",
    )
    db.add(row1)
    db.commit()

    row2 = ENSSignalQueue(
        queue_id=str(uuid.uuid4()),
        signal_id=str(uuid.uuid4()),
        signal_type="system.noop",
        priority_tier="system",
        created_at_us=1_000_001,
        idempotency_key=key,
        signal_json={"type": "system.noop"},
        status="pending",
    )
    db.add(row2)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    rows = db.query(ENSSignalQueue).filter(ENSSignalQueue.idempotency_key == key).all()
    assert len(rows) == 1
    assert rows[0].signal_id == row1.signal_id


def test_time_utils_monotonic_guard_when_clock_stalls(monkeypatch):
    monkeypatch.setattr(time_utils.time, "time_ns", lambda: 1_234_567_000)
    monkeypatch.setattr(time_utils, "_last_seen_us", 0)

    first = time_utils.next_created_at_us()
    second = time_utils.next_created_at_us()
    third = time_utils.next_created_at_us()

    assert first == 1_234_567
    assert second == first + 1
    assert third == second + 1


def test_v3_scheduler_flag_path_processes_enqueued_signal_when_sync_tick_enabled(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.scheduler_sync_ticks_per_ingress = 1

    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
    )
    outcome = asyncio.run(runtime.ingest(signal))
    assert outcome.signal_id == signal.signal_id
    assert outcome.response_payload == {}


def test_v3_scheduler_flag_path_returns_queued_when_sync_ticks_zero(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.scheduler_sync_ticks_per_ingress = 0

    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
    )
    outcome = asyncio.run(runtime.ingest(signal))
    assert outcome.signal_id == signal.signal_id
    assert outcome.response_payload.get("queued") is True
    assert outcome.response_payload.get("status") == "pending"


def test_v3_scheduler_no_double_emit_on_replay(client, db, helpers):
    class _LLMResponse:
        def __init__(self, content: str):
            self.content = content

    class _LLMClient:
        base_url = "http://test-llm"

        async def health_check(self):
            return True

        async def generate_with_history(self, messages, temperature=None, max_tokens=None, model=None):
            _ = (messages, temperature, max_tokens, model)
            return _LLMResponse(
                "<assistant_response><speech>Here you go.</speech></assistant_response>\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"tool_calls":[{"id":"img1","tool":"image.generate","requires_approval":true,"args":{"prompt":"cat portrait"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            )

    helpers.app_module.app_state["llm_client"] = _LLMClient()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.scheduler_sync_ticks_per_ingress = 1

    conversation_id, thread_id = helpers.create_conversation_thread()
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    assert conversation is not None
    conversation.image_offer_count = 0
    db.commit()

    payload = {"message": "please make an image", "metadata": {"client_message_id": "slice0-no-double-emit"}}
    first = client.post(f"/threads/{thread_id}/messages", json=payload)
    second = client.post(f"/threads/{thread_id}/messages", json=payload)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    b1 = first.json()
    b2 = second.json()
    assert b1["user_message"]["id"] == b2["user_message"]["id"]
    assert b1["assistant_message"]["id"] == b2["assistant_message"]["id"]
    assert sorted(b1.keys()) == sorted(b2.keys())
    assert b1.get("pending_tool_calls") == b2.get("pending_tool_calls")
    assert b1.get("conversation_title_updated") == b2.get("conversation_title_updated")
    assert len(db.query(ENSToolCallRequest).all()) == 1
    assert (
        db.query(ENSDecision)
        .filter(ENSDecision.signal_type == "user.message")
        .join(
            ENSSignalQueue,
            ENSSignalQueue.signal_id == ENSDecision.signal_id,
        )
        .filter(
            ENSSignalQueue.idempotency_key
            == "signal:user.message:"
            + conversation_id
            + ":"
            + thread_id
            + ":slice0-no-double-emit"
        )
        .count()
        == 1
    )


def test_v3_scheduler_tool_call_sentinel_fallback_still_works(client, db, helpers):
    class _LLMResponse:
        def __init__(self, content: str):
            self.content = content

    class _LLMClient:
        base_url = "http://test-llm"

        async def health_check(self):
            return True

        async def generate_with_history(self, messages, temperature=None, max_tokens=None, model=None):
            _ = (messages, temperature, max_tokens, model)
            return _LLMResponse(
                "<assistant_response><speech>Sending one.</speech></assistant_response>\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"tool_calls":[{"id":"img2","tool":"image.generate","requires_approval":true,"args":{"prompt":"forest"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            )

    helpers.app_module.app_state["llm_client"] = _LLMClient()
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice2_tool_parsing_ownership=True,
        slice2_tool_dispatch_ownership=False,
        slice25_media_gating_ownership=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.scheduler_sync_ticks_per_ingress = 1

    _conversation_id, thread_id = helpers.create_conversation_thread()
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"message": "please send an image", "metadata": {"client_message_id": "slice0-sentinel-fallback"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body.get("pending_tool_calls") or []) == 1
    assert body["pending_tool_calls"][0]["tool"] == "image.generate"

    tool_rows = db.query(ENSToolCallRequest).all()
    assert len(tool_rows) == 1
    assert tool_rows[0].status == "pending"


def test_v3_scheduler_ingress_tick_and_parallel_tick_do_not_double_execute(helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.scheduler_sync_ticks_per_ingress = 1

    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
    )

    async def _run():
        ingest_task = asyncio.create_task(runtime.ingest(signal))
        concurrent_tick_task = asyncio.create_task(runtime.scheduler_tick())
        return await asyncio.gather(ingest_task, concurrent_tick_task)

    ingest_outcome, concurrent_tick_outcome = asyncio.run(_run())
    assert ingest_outcome is not None
    seen = [
        o.signal_id
        for o in (ingest_outcome, concurrent_tick_outcome)
        if o is not None and o.signal_id == signal.signal_id
    ]
    assert len(seen) == 1

    queue_row = db.query(ENSSignalQueue).filter(ENSSignalQueue.signal_id == signal.signal_id).first()
    assert queue_row is not None
    assert queue_row.status == "done"

    decision_count = db.query(ENSDecision).filter(ENSDecision.signal_id == signal.signal_id).count()
    assert decision_count == 1


def test_v3_scheduler_recovers_stuck_running_signal(helpers, db):
    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = Signal(
        type="system.noop",
        scope="SYSTEM",
        source="test",
        payload={"kind": "noop"},
    )
    queued = asyncio.run(runtime.enqueue_signal(signal))
    row = db.query(ENSSignalQueue).filter(ENSSignalQueue.queue_id == queued["queue_id"]).first()
    assert row is not None
    row.status = "running"
    row.claimed_at_us = max(1, (time_utils.time.time_ns() // 1000) - 5_000_000)
    db.commit()

    recovered = asyncio.run(runtime.scheduler_recover_stuck_running(ttl_seconds=1))
    assert recovered >= 1

    row = db.query(ENSSignalQueue).filter(ENSSignalQueue.queue_id == queued["queue_id"]).first()
    assert row is not None
    assert row.status == "pending"
    assert row.error_message == "stuck_running_recovered"
    assert row.claimed_at_us is None

