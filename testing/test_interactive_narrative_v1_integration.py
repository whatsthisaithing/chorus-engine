import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from chorus_engine.models.conversation import Conversation


class _LoopControlLLMClient:
    def __init__(self, actions):
        self._actions = list(actions)
        self._idx = 0
        self.base_url = "http://test-llm"

    async def health_check(self):
        return True

    async def generate(self, prompt, system_prompt=None, model=None, **kwargs):
        _ = (prompt, system_prompt, model, kwargs)
        action = self._actions[min(self._idx, len(self._actions) - 1)]
        self._idx += 1
        payload = {
            "version": 1,
            "control": {"action": action, "args": {}},
            "tool_calls": [],
        }
        content = (
            "Narrative beat.\n"
            "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
            f"{json.dumps(payload)}\n"
            "---CHORUS_TOOL_PAYLOAD_END---"
        )
        return SimpleNamespace(content=content, finish_reason="stop", usage={"total_tokens": 32})

    async def generate_with_history(self, messages, temperature=None, max_tokens=None, model=None):
        _ = (messages, temperature, max_tokens, model)
        return await self.generate("history")

    async def stream_with_history(self, messages, temperature=None, max_tokens=None, model=None):
        _ = (messages, temperature, max_tokens, model)
        yield "noop"

    async def generate_vision(
        self,
        *,
        prompt,
        image_base64_list,
        image_mime_type="image/jpeg",
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        model=None,
    ):
        _ = (
            prompt,
            image_base64_list,
            image_mime_type,
            system_prompt,
            temperature,
            max_tokens,
            model,
        )
        return SimpleNamespace(content="{}", finish_reason="stop", usage={"total_tokens": 8})


def _enable_v3_loop_flags(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
        slice7_unified_llm_invocation=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.v3_scheduler_enabled = True
    ens_cfg.v3_arbitration_enabled = True
    ens_cfg.v3_loop_sessions_enabled = True
    ens_cfg.v3_structured_control_enabled = True
    ens_cfg.v3_assistant_result_enabled = True
    ens_cfg.v3_sentinel_fallback_enabled = True


def _prepare_conversation_relationship(db, conversation_id: str, relationship_id: str = "rel-intn-1"):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    assert conversation is not None
    conversation.relationship_id = relationship_id
    db.commit()
    return conversation


def test_interactive_narrative_feature_gate_blocks_create_when_disabled(client, db, helpers):
    _enable_v3_loop_flags(helpers)
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.interactive_narrative = False

    conversation_id, _thread_id = helpers.create_conversation_thread()
    _prepare_conversation_relationship(db, conversation_id)

    resp = client.post(f"/conversations/{conversation_id}/interactive-narrative/session", json={})
    assert resp.status_code == 400
    assert "disabled" in resp.json()["detail"].lower()


def test_interactive_narrative_session_lifecycle_endpoints(client, db, helpers):
    _enable_v3_loop_flags(helpers)
    helpers.app_module.app_state["llm_client"] = _LoopControlLLMClient(["WAIT_FOR_USER"])
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.interactive_narrative = True

    conversation_id, _thread_id = helpers.create_conversation_thread()
    _prepare_conversation_relationship(db, conversation_id, relationship_id="rel-intn-lifecycle")

    created = client.post(f"/conversations/{conversation_id}/interactive-narrative/session", json={})
    assert created.status_code == 200
    created_body = created.json()
    assert created_body["loop_kind"] == "narrative.v1"
    assert created_body["loop_mode"] == "visible"
    assert created_body["loop_id"]
    assert created_body["progression_enqueued"] is False

    loop_id = created_body["loop_id"]
    fetched = client.get(f"/conversations/{conversation_id}/interactive-narrative/session")
    assert fetched.status_code == 200
    assert fetched.json()["loop_id"] == loop_id

    paused = client.post(f"/interactive-narrative/{loop_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["state"] == "paused"

    resumed = client.post(f"/interactive-narrative/{loop_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["loop_id"] == loop_id
    assert resumed.json()["progression_enqueued"] is False
    assert resumed.json()["state"] in ("running", "waiting_for_user")


def test_interactive_narrative_create_backfills_missing_relationship_id(client, db, helpers):
    _enable_v3_loop_flags(helpers)
    helpers.app_module.app_state["llm_client"] = _LoopControlLLMClient(["WAIT_FOR_USER"])
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.interactive_narrative = True

    conversation_id, _thread_id = helpers.create_conversation_thread()

    # Verify precondition: legacy conversation may have no relationship binding.
    conversation_before = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    assert conversation_before is not None
    assert not conversation_before.relationship_id

    created = client.post(f"/conversations/{conversation_id}/interactive-narrative/session", json={})
    assert created.status_code == 200
    body = created.json()
    assert body["loop_id"]
    assert body["loop_kind"] == "narrative.v1"

    db.expire_all()
    conversation_after = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    assert conversation_after is not None
    assert conversation_after.relationship_id


def test_interactive_narrative_tick_treats_yield_as_wait_for_user(client, db, helpers):
    _enable_v3_loop_flags(helpers)
    helpers.app_module.app_state["llm_client"] = _LoopControlLLMClient(["YIELD"])
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.interactive_narrative = True

    conversation_id, thread_id = helpers.create_conversation_thread()
    _prepare_conversation_relationship(db, conversation_id, relationship_id="rel-intn-yield")

    created = client.post(f"/conversations/{conversation_id}/interactive-narrative/session", json={})
    assert created.status_code == 200
    loop_id = created.json()["loop_id"]

    tick = client.post(f"/interactive-narrative/{loop_id}/tick")
    assert tick.status_code == 200
    body = tick.json()
    assert body["last_step_control_action"] == "WAIT_FOR_USER"
    assert "control_action" not in body
    assert body["state"] == "waiting_for_user"
    assert body["progression_enqueued"] is False

    messages = client.get(f"/threads/{thread_id}/messages")
    assert messages.status_code == 200
    rows = messages.json()
    assistant = next((m for m in rows if m["role"] == "assistant"), None)
    assert assistant is not None
    loop_meta = ((assistant.get("metadata") or {}).get("loop") or {})
    assert loop_meta.get("loop_id") == loop_id
    assert loop_meta.get("control_action") == "WAIT_FOR_USER"


def test_interactive_narrative_continue_cap_forces_wait(client, db, helpers):
    _enable_v3_loop_flags(helpers)
    helpers.app_module.app_state["llm_client"] = _LoopControlLLMClient(
        ["CONTINUE", "CONTINUE", "CONTINUE", "CONTINUE", "CONTINUE"]
    )
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.interactive_narrative = True

    conversation_id, _thread_id = helpers.create_conversation_thread()
    _prepare_conversation_relationship(db, conversation_id, relationship_id="rel-intn-cap")

    created = client.post(f"/conversations/{conversation_id}/interactive-narrative/session", json={})
    assert created.status_code == 200
    loop_id = created.json()["loop_id"]

    first = client.post(f"/interactive-narrative/{loop_id}/tick")
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["last_step_control_action"] == "CONTINUE"
    assert first_body["state"] == "running"

    second = client.post(f"/interactive-narrative/{loop_id}/tick")
    assert second.status_code == 200
    second_body = second.json()
    # Default policy max_consecutive_continue=4, but automatic follow-ups may execute
    # between manual ticks. We assert the cap eventually forces WAIT behavior.
    assert second_body["last_step_control_action"] in ("CONTINUE", "WAIT_FOR_USER")

    # Ensure cap is reached by polling a few ticks.
    final_body = second_body
    for _ in range(4):
        if final_body["last_step_control_action"] == "WAIT_FOR_USER":
            break
        resp = client.post(f"/interactive-narrative/{loop_id}/tick")
        assert resp.status_code == 200
        final_body = resp.json()
    assert final_body["last_step_control_action"] == "WAIT_FOR_USER"
    assert final_body["state"] == "waiting_for_user"
    assert final_body["progression_enqueued"] is False


def test_interactive_narrative_loop_step_debug_capture_writes_prompt_payload(client, db, helpers):
    _enable_v3_loop_flags(helpers)
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.debug_capture_full_prompt = True
    helpers.app_module.app_state["llm_client"] = _LoopControlLLMClient(["WAIT_FOR_USER"])
    character = helpers.app_module.app_state["characters"]["test_char"]
    character.features.interactive_narrative = True

    conversation_id, _thread_id = helpers.create_conversation_thread()
    _prepare_conversation_relationship(db, conversation_id, relationship_id="rel-intn-debug-capture")

    created = client.post(f"/conversations/{conversation_id}/interactive-narrative/session", json={})
    assert created.status_code == 200
    loop_id = created.json()["loop_id"]

    tick = client.post(f"/interactive-narrative/{loop_id}/tick")
    assert tick.status_code == 200

    today = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = Path("data/debug_logs/conversations") / conversation_id / f"ens_conversation_{today}.jsonl"
    assert log_file.exists()

    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    step_rows = [row for row in rows if row.get("type") == "ens_loop_step_turn" and row.get("loop_id") == loop_id]
    assert step_rows
    latest = step_rows[-1]
    capture = latest.get("prompt_capture") or {}
    assert capture.get("enabled") is True
    assert capture.get("mode") in ("messages", "prompt")
    if capture.get("mode") == "messages":
        assert isinstance(capture.get("messages_for_llm"), list)
        assert len(capture.get("messages_for_llm")) >= 2
    else:
        assert isinstance(capture.get("system_prompt"), str)
        assert capture.get("system_prompt")
        assert isinstance(capture.get("prompt"), str)
        assert capture.get("prompt")
