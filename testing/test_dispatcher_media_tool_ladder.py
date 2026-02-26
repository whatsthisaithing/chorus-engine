from types import SimpleNamespace

import pytest

from chorus_engine.ens.assistant_result import normalize_assistant_result
from chorus_engine.ens.dispatcher import ENSDispatcher
from chorus_engine.ens.llm_invocation_service import InvocationRequest


def _dispatcher() -> ENSDispatcher:
    app_state = {"system_config": SimpleNamespace(ens=SimpleNamespace(enabled=True), llm=SimpleNamespace())}
    return ENSDispatcher(app_state)


def _effective() -> SimpleNamespace:
    return SimpleNamespace(
        engine="lmstudio",
        model_id="test-model",
        provider="local",
        max_tokens=512,
        top_p=None,
        top_k=None,
        repeat_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        temperature=0.2,
    )


@pytest.mark.asyncio
async def test_media_tool_ladder_rung3_json_schema_success():
    dispatcher = _dispatcher()
    dispatcher.llm_invoker.resolve_provider_capabilities = lambda engine=None: {
        "supports_response_format_json_schema": True
    }

    async def _invoke(_req):
        return {
            "status": "success",
            "output_text": (
                '{"version":1,"tool_calls":[{"id":"j1","tool":"image.generate","requires_approval":true,'
                '"args":{"prompt":"json schema prompt"}}]}'
            ),
            "assistant_result": None,
        }

    dispatcher.llm_invoker.invoke = _invoke

    request = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="ladder-test-rung3",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        thread_id="thread-1",
        session_id="sess-1",
    )
    assistant = normalize_assistant_result(raw_content="no tool call")

    out = await dispatcher._run_media_tool_ladder(
        thread_id="thread-1",
        request=request,
        effective=_effective(),
        source="web",
        conversation_id="conv-1",
        character_id="char-1",
        media_gate_snapshot={
            "media_tool_calls_allowed": True,
            "explicit_allowed": True,
            "is_iteration_request": False,
            "requested_media_type": "image",
            "allowed_tools_final": ["image.generate"],
        },
        messages=[{"role": "user", "content": "send image"}],
        invocation={"status": "success"},
        raw_content="no tool call",
        assistant_result=assistant,
        payload_obj=None,
        normalized_tool_payload={"version": 1, "tool_calls": []},
        validated_tool_calls=[],
        allowed_tools_set={"image.generate"},
    )

    ladder = out.get("media_tool_ladder") or {}
    rung3 = ladder.get("rung3_json_schema_retry") or {}
    rung4 = ladder.get("rung4_sentinel_repair") or {}
    assert rung3.get("attempted") is True
    assert rung3.get("success") is True
    assert rung4.get("attempted") is False
    assert len(out.get("validated_tool_calls") or []) == 1


@pytest.mark.asyncio
async def test_media_tool_ladder_rung4_sentinel_fallback_success():
    dispatcher = _dispatcher()
    dispatcher.llm_invoker.resolve_provider_capabilities = lambda engine=None: {
        "supports_response_format_json_schema": False
    }

    async def _invoke(_req):
        return {
            "status": "success",
            "output_text": (
                "ok\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"tool_calls":[{"id":"s1","tool":"image.generate","requires_approval":true,"args":{"prompt":"sentinel prompt"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            ),
            "assistant_result": None,
        }

    dispatcher.llm_invoker.invoke = _invoke

    request = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="ladder-test-rung4",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        thread_id="thread-1",
        session_id="sess-1",
    )
    assistant = normalize_assistant_result(raw_content="no tool call")

    out = await dispatcher._run_media_tool_ladder(
        thread_id="thread-1",
        request=request,
        effective=_effective(),
        source="web",
        conversation_id="conv-1",
        character_id="char-1",
        media_gate_snapshot={
            "media_tool_calls_allowed": True,
            "explicit_allowed": True,
            "is_iteration_request": False,
            "requested_media_type": "image",
            "allowed_tools_final": ["image.generate"],
        },
        messages=[{"role": "user", "content": "send image"}],
        invocation={"status": "success"},
        raw_content="no tool call",
        assistant_result=assistant,
        payload_obj=None,
        normalized_tool_payload={"version": 1, "tool_calls": []},
        validated_tool_calls=[],
        allowed_tools_set={"image.generate"},
    )

    ladder = out.get("media_tool_ladder") or {}
    rung3 = ladder.get("rung3_json_schema_retry") or {}
    rung4 = ladder.get("rung4_sentinel_repair") or {}
    assert rung3.get("attempted") is False
    assert rung3.get("reason") == "unsupported_response_format_json_schema"
    assert rung4.get("attempted") is True
    assert rung4.get("success") is True
    assert len(out.get("validated_tool_calls") or []) == 1
