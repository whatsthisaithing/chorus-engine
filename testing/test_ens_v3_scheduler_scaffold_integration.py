import asyncio

from chorus_engine.ens.models import SignalEnvelope
import chorus_engine.ens.time_utils as time_utils
from chorus_engine.models.conversation import Conversation
from chorus_engine.models.ens import ENSDecision, ENSSchedulerTick, ENSSignalQueue, ENSToolCallRequest


def test_v3_scheduler_scaffold_enqueue_and_tick_executes_selected_signal(helpers, db):
    runtime = helpers.app_module.app_state["ens_runtime"]
    signal = SignalEnvelope(
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


def test_v3_scheduler_deterministic_tie_break_for_identical_created_at_us(helpers):
    runtime = helpers.app_module.app_state["ens_runtime"]
    created_at_us = 2_000_000
    signal_ids = [
        "00000000-0000-0000-0000-000000000003",
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]

    for signal_id in signal_ids:
        signal = SignalEnvelope(
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
    signal = SignalEnvelope(
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
    signal = SignalEnvelope(
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
    assert len(db.query(ENSToolCallRequest).all()) == 1


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
    signal = SignalEnvelope(
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
