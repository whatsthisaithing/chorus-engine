import asyncio

from chorus_engine.ens.dispatcher import ENSDispatcher
from chorus_engine.ens.llm_invocation_service import InvocationRequest, LLMInvocationService
from chorus_engine.llm.base import LLMResponse


def _enable_native_transport(helpers, *, fallback_enabled: bool = True):
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.enabled = True
    ens_cfg.native_tool_transport_enabled = True
    ens_cfg.native_tool_transport_force_sentinel = False
    ens_cfg.native_tool_transport_sentinel_fallback_enabled = fallback_enabled
    ens_cfg.native_tool_transport_debug_override_mode = "off"
    ens_cfg.v3_sentinel_fallback_enabled = fallback_enabled


def test_narrative_v1_loop_step_forces_control_only_auto_tool_choice(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model)
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="native response",
            model="test-model",
            finish_reason="stop",
            tool_calls=[
                {
                    "id": "tc_control",
                    "type": "function",
                    "function": {"name": "chorus.control", "arguments": '{"action":"CONTINUE"}'},
                },
            ],
            raw_message={"content": "native response"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-map-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "continue and generate an image"}],
        metadata={
            "media_gate_snapshot": {
                "allowed_tools_final": ["image.generate"],
            },
            "loop_id": "loop-1",
            "loop_kind": "narrative.v1",
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert captured.get("tools")
    assert captured.get("tool_choice") == "auto"
    tool_names = [
        str(((tool.get("function") or {}).get("name")) or "")
        for tool in (captured.get("tools") or [])
        if isinstance(tool, dict)
    ]
    assert tool_names == ["chorus.control"]

    assistant = result["assistant_result"]
    assert assistant["control"] is not None
    assert assistant["control"]["action"] == "CONTINUE"
    assert assistant["tool_requests"] == []
    assert assistant["provider_raw"]["assistant_result_tier"] == "provider_native"


def test_non_narrative_loop_step_keeps_policy_tools_and_auto_tool_choice(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model)
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="native response",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "native response"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-map-002",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "continue and generate an image"}],
        metadata={
            "media_gate_snapshot": {
                "allowed_tools_final": ["image.generate"],
            },
            "loop_id": "loop-2",
            "loop_kind": "generic.v1",
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert captured.get("tools")
    assert captured.get("tool_choice") == "auto"
    tool_names = {
        str(((tool.get("function") or {}).get("name")) or "")
        for tool in (captured.get("tools") or [])
        if isinstance(tool, dict)
    }
    assert "chorus.control" in tool_names
    assert "image.generate" in tool_names


def test_cold_recall_trigger_keeps_auto_tool_choice_when_pin_is_injected(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model)
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="native response",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "native response"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-map-cold-force-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "Do you remember the exact wording of that conversation?"}],
        metadata={
            "media_gate_snapshot": {"allowed_tools_final": ["image.generate"]},
            "used_moment_pin_ids": ["pin-1"],
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert captured.get("tool_choice") == "auto"
    tool_names = {
        str(((tool.get("function") or {}).get("name")) or "")
        for tool in (captured.get("tools") or [])
        if isinstance(tool, dict)
    }
    assert "moment_pin.cold_recall" in tool_names


def test_cold_recall_trigger_keeps_auto_tool_choice_without_injected_pin(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model)
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="native response",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "native response"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-map-cold-force-002",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "Do you remember the exact wording of that conversation?"}],
        metadata={
            "media_gate_snapshot": {"allowed_tools_final": ["image.generate"]},
            "used_moment_pin_ids": [],
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert captured.get("tool_choice") == "auto"


def test_native_empty_uses_immediate_sentinel_fallback_when_enabled(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model, tools, tool_choice)
        return LLMResponse(
            content=(
                "visible text\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"tool_calls":[{"id":"s1","tool":"image.generate","requires_approval":true,"args":{"prompt":"fallback image"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            ),
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "with sentinel fallback"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-fallback-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "generate image"}],
        metadata={"media_gate_snapshot": {"allowed_tools_final": ["image.generate"]}},
    )
    result = asyncio.run(invoker.invoke(req))
    assistant = result["assistant_result"]
    assert assistant["provider_raw"]["assistant_result_tier"] == "sentinel_fallback"
    assert len(assistant["tool_requests"]) == 1
    assert assistant["tool_requests"][0]["tool_name"] == "image.generate"


def test_native_empty_without_fallback_keeps_provider_native_empty_result(helpers):
    _enable_native_transport(helpers, fallback_enabled=False)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model, tools, tool_choice)
        return LLMResponse(
            content=(
                "visible text\n"
                "---CHORUS_TOOL_PAYLOAD_BEGIN---\n"
                '{"version":1,"tool_calls":[{"id":"s1","tool":"image.generate","requires_approval":true,"args":{"prompt":"should_not_be_used"}}]}\n'
                "---CHORUS_TOOL_PAYLOAD_END---"
            ),
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "native empty no fallback"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-no-fallback-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "generate image"}],
        metadata={"media_gate_snapshot": {"allowed_tools_final": ["image.generate"]}},
    )
    result = asyncio.run(invoker.invoke(req))
    assistant = result["assistant_result"]
    assert assistant["provider_raw"]["assistant_result_tier"] == "provider_native"
    assert assistant["tool_requests"] == []


def test_debug_override_chat_control_only_forces_control_tool_in_normal_chat(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    helpers.app_module.app_state["system_config"].ens.native_tool_transport_debug_override_mode = "chat_control_only"
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (temperature, max_tokens, model)
        captured["messages"] = messages
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="control test",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "control test"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-debug-chat-control-001",
        model_id="test-model",
        provider="local",
        engine="ollama",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"media_gate_snapshot": {"allowed_tools_final": ["image.generate"]}},
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert captured.get("tool_choice") == {
        "type": "function",
        "function": {"name": "chorus.control"},
    }
    tool_names = [
        str(((tool.get("function") or {}).get("name")) or "")
        for tool in (captured.get("tools") or [])
        if isinstance(tool, dict)
    ]
    assert tool_names == ["chorus.control"]
    assert isinstance(captured.get("messages"), list)
    system_message = next((m for m in captured["messages"] if isinstance(m, dict) and m.get("role") == "system"), None)
    assert system_message is not None
    assert "TEST OVERRIDE" in str(system_message.get("content") or "")


def test_debug_override_loop_image_only_forces_image_tool_on_loop_step(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    helpers.app_module.app_state["system_config"].ens.native_tool_transport_debug_override_mode = "loop_image_only"
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (temperature, max_tokens, model)
        captured["messages"] = messages
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="image test",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "image test"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-debug-loop-image-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "continue"}],
        metadata={
            "media_gate_snapshot": {"allowed_tools_final": ["image.generate", "video.generate"]},
            "loop_id": "loop-debug-1",
            "loop_kind": "narrative.v1",
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert captured.get("tool_choice") == {
        "type": "function",
        "function": {"name": "image.generate"},
    }
    tool_names = [
        str(((tool.get("function") or {}).get("name")) or "")
        for tool in (captured.get("tools") or [])
        if isinstance(tool, dict)
    ]
    assert tool_names == ["image.generate"]
    assert isinstance(captured.get("messages"), list)
    system_message = next((m for m in captured["messages"] if isinstance(m, dict) and m.get("role") == "system"), None)
    assert system_message is not None
    assert "TEST OVERRIDE" in str(system_message.get("content") or "")


def test_provider_capabilities_from_config_are_applied(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm_cfg = helpers.app_module.app_state["system_config"].llm
    llm_cfg.provider_capabilities["lmstudio"].supports_response_format_json_schema = False
    llm_cfg.provider_capabilities["lmstudio"].supports_native_tools = True
    llm_cfg.provider_capabilities["lmstudio"].supports_sentinel_retry = False

    caps = invoker.resolve_provider_capabilities(engine="lmstudio")
    assert caps["supports_native_tools"] is True
    assert caps["supports_response_format_json_schema"] is False
    assert caps["supports_sentinel_retry"] is False


def test_koboldcpp_provider_capabilities_disable_native_tools(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-kobold-disabled-001",
        model_id="test-model",
        provider="local",
        engine="koboldcpp",
        messages=[{"role": "user", "content": "continue"}],
        metadata={"loop_id": "loop-k", "loop_kind": "narrative.v1"},
    )
    tools, tool_choice, plan = invoker._prepare_native_transport(req)
    assert tools is None
    assert tool_choice is None
    assert plan["attempted"] is False


def test_explicit_native_tool_policy_overrides_metadata_heuristics(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    captured = {}

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model)
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return LLMResponse(
            content="policy test",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "policy test"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-policy-override-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "continue"}],
        native_tool_policy={
            "policy_id": "test.control_only",
            "allowed_media_tools": [],
            "include_control": True,
            "include_cold_recall": False,
            "tool_choice": {"type": "function", "function": {"name": "chorus.control"}},
        },
        metadata={
            "media_gate_snapshot": {"allowed_tools_final": ["image.generate"]},
            "loop_id": "loop-x",
            "loop_kind": "generic.v1",
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    tool_names = [
        str(((tool.get("function") or {}).get("name")) or "")
        for tool in (captured.get("tools") or [])
        if isinstance(tool, dict)
    ]
    assert tool_names == ["chorus.control"]
    assert captured.get("tool_choice") == {"type": "function", "function": {"name": "chorus.control"}}


def test_native_transport_plan_reports_prompt_contract_mismatch(helpers):
    _enable_native_transport(helpers, fallback_enabled=True)
    invoker = LLMInvocationService(helpers.app_module.app_state)
    llm = helpers.app_module.app_state["llm_client"]

    async def fake_generate_with_history(messages, temperature=None, max_tokens=None, model=None, tools=None, tool_choice=None):
        _ = (messages, temperature, max_tokens, model, tools, tool_choice)
        return LLMResponse(
            content="policy test",
            model="test-model",
            finish_reason="stop",
            tool_calls=[],
            raw_message={"content": "policy test"},
        )

    llm.generate_with_history = fake_generate_with_history

    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="native-policy-mismatch-001",
        model_id="test-model",
        provider="local",
        engine="lmstudio",
        messages=[{"role": "user", "content": "continue"}],
        native_tool_policy={
            "policy_id": "test.image_only",
            "allowed_media_tools": ["image.generate"],
            "include_control": False,
            "include_cold_recall": False,
            "tool_choice": "auto",
        },
        metadata={
            "tool_transport_mode": "native",
            "prompt_contract_tools": ["image.generate", "moment_pin.cold_recall"],
            "used_moment_pin_ids": ["pin-1"],
            "media_gate_snapshot": {"allowed_tools_final": ["image.generate"]},
        },
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    plan = (result.get("native_transport") or {}).get("plan") or {}
    assert "moment_pin.cold_recall" in (plan.get("docs_without_runtime") or [])


def test_prompt_cold_recall_availability_matches_narrative_v1_loop_policy():
    assert ENSDispatcher._cold_recall_available_for_prompt(
        loop_step=True,
        loop_kind="narrative.v1",
        loop_stage="beat",
    ) is False
    assert ENSDispatcher._cold_recall_available_for_prompt(
        loop_step=True,
        loop_kind="narrative.v1",
        loop_stage=None,
    ) is False
    assert ENSDispatcher._cold_recall_available_for_prompt(
        loop_step=True,
        loop_kind="generic.v1",
        loop_stage="full",
    ) is True


def test_prompt_cold_recall_availability_honors_explicit_policy_override():
    assert ENSDispatcher._cold_recall_available_for_prompt(
        loop_step=True,
        loop_kind="narrative.v1",
        loop_stage="beat",
        native_tool_policy={"include_cold_recall": True},
    ) is True
    assert ENSDispatcher._cold_recall_available_for_prompt(
        loop_step=False,
        loop_kind=None,
        loop_stage=None,
        native_tool_policy={"include_cold_recall": False},
    ) is False
