import asyncio
from pathlib import Path
import time

import pytest

from chorus_engine.ens.llm_control_plane_service import ControlPlaneRequest, LLMControlPlaneService
from chorus_engine.models.ens import ENSActionResult


def test_slice75_status_endpoint_routes_through_ens(client, db, helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice75_llm_control_plane_ownership=True,
    )

    resp = client.get("/api/llm/status")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"]
    assert body["model_loaded"] is True

    rows = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "llm.control.execute")
        .order_by(ENSActionResult.created_at.desc())
        .limit(5)
        .all()
    )
    assert rows
    ops = {(r.output_json or {}).get("op") for r in rows if isinstance(r.output_json, dict)}
    assert "health" in ops
    assert "list_loaded" in ops


def test_slice75_switch_model_endpoint_routes_through_ens(client, db, helpers, app):
    _test_app, _helper = app
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice75_llm_control_plane_ownership=True,
    )

    model_path = Path("slice75_test.gguf")
    model_path.write_bytes(b"test")

    resp = client.post("/api/llm/switch-model", json={"model_path": str(model_path.resolve())})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["model_path"].endswith("slice75_test.gguf")

    row = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "llm.control.execute")
        .order_by(ENSActionResult.created_at.desc())
        .first()
    )
    assert row is not None
    output = row.output_json or {}
    assert output.get("op") == "switch"
    assert output.get("model_id", "").endswith("slice75_test.gguf")


def test_slice75_busy_mode_skip_busy_returns_skipped(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice75_llm_control_plane_ownership=True,
    )
    app_state = helpers.app_module.app_state
    service = LLMControlPlaneService(app_state)
    mutex = app_state["ens_llm_control_plane_mutex"]

    async def _run():
        await mutex.acquire()
        try:
            return await service.execute(
                ControlPlaneRequest(
                    op="health",
                    idempotency_key="slice75:busy:001",
                    busy_mode="skip_busy",
                    timeout_s=0.1,
                )
            )
        finally:
            if mutex.locked():
                mutex.release()

    result = asyncio.run(_run())
    assert result.get("_ens_action_status") == "skipped"
    assert result.get("reason") == "busy"


def test_slice75_direct_control_call_is_blocked_in_tests(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice75_llm_control_plane_ownership=True,
    )
    llm_client = helpers.app_module.app_state["llm_client"]
    with pytest.raises(RuntimeError, match="Direct llm_client.health_check call blocked under slice75"):
        asyncio.run(llm_client.health_check())


def test_slice75_skip_shared_locks_avoids_nested_wait(helpers):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice75_llm_control_plane_ownership=True,
    )
    app_state = helpers.app_module.app_state
    app_state["llm_usage_lock"] = asyncio.Lock()
    app_state["comfyui_lock"] = asyncio.Lock()
    service = LLMControlPlaneService(app_state)

    async def _run():
        await app_state["llm_usage_lock"].acquire()
        await app_state["comfyui_lock"].acquire()
        try:
            start = time.perf_counter()
            result = await service.execute(
                ControlPlaneRequest(
                    op="unload_all",
                    idempotency_key="slice75:nested-locks:001",
                    busy_mode="block_with_timeout",
                    timeout_s=0.2,
                    metadata={"skip_shared_locks": True},
                )
            )
            elapsed = time.perf_counter() - start
            return result, elapsed
        finally:
            if app_state["comfyui_lock"].locked():
                app_state["comfyui_lock"].release()
            if app_state["llm_usage_lock"].locked():
                app_state["llm_usage_lock"].release()

    result, elapsed = asyncio.run(_run())
    assert result.get("status") == "success"
    assert elapsed < 0.2


def test_slice75_unload_all_does_not_replay_forever(helpers, db):
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=False,
        nonstream_intake_only=True,
        streaming_intake_only=True,
        slice75_llm_control_plane_ownership=True,
    )
    app_state = helpers.app_module.app_state
    app_state["llm_usage_lock"] = asyncio.Lock()
    app_state["comfyui_lock"] = asyncio.Lock()

    async def _run_twice():
        first = await helpers.app_module._invoke_llm_control_unified(
            op="unload_all",
            reason="slice75_test_first",
            busy_mode="block_with_timeout",
            timeout_s=1.0,
            skip_shared_locks=True,
        )
        second = await helpers.app_module._invoke_llm_control_unified(
            op="unload_all",
            reason="slice75_test_second",
            busy_mode="block_with_timeout",
            timeout_s=1.0,
            skip_shared_locks=True,
        )
        return first, second

    first, second = asyncio.run(_run_twice())
    assert first.get("status") == "success"
    assert second.get("status") == "success"

    rows = (
        db.query(ENSActionResult)
        .filter(ENSActionResult.kind == "llm.control.execute")
        .order_by(ENSActionResult.created_at.desc())
        .all()
    )
    unload_rows = [r for r in rows if isinstance(r.output_json, dict) and r.output_json.get("op") == "unload_all"]
    assert len(unload_rows) >= 2
    assert unload_rows[0].status == "success"
    assert unload_rows[1].status == "success"
    assert unload_rows[0].idempotency_key != unload_rows[1].idempotency_key
