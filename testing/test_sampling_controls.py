import asyncio
from typing import Any, Dict, List, Optional

import httpx

from chorus_engine.config.models import CharacterConfig, LLMConfig, PreferredLLMConfig, SystemConfig
from chorus_engine.ens.llm_invocation_service import InvocationRequest, LLMInvocationService
from chorus_engine.llm.base import LLMResponse
from chorus_engine.llm.koboldcpp import KoboldCppLLMClient
from chorus_engine.llm.lmstudio import LMStudioLLMClient
from chorus_engine.llm.ollama import OllamaLLMClient


class _FakeHTTPResponse:
    def __init__(self, *, status_code: int, json_data: Dict[str, Any], text: Optional[str] = None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text if text is not None else ""
        self.content = (self.text or "{}").encode("utf-8")
        self.encoding = None
        self.headers = {}
        self.request = httpx.Request("POST", "http://test")

    def json(self) -> Dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("HTTP error", request=self.request, response=httpx.Response(self.status_code))


class _RecordingAsyncClient:
    def __init__(self, responses: List[_FakeHTTPResponse]):
        self.responses = list(responses)
        self.payloads: List[Dict[str, Any]] = []

    async def post(self, _url: str, json: Optional[Dict[str, Any]] = None, content: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None):
        _ = headers
        payload_obj: Dict[str, Any]
        if json is not None:
            payload_obj = dict(json)
        elif content is not None:
            import json as json_module

            payload_obj = json_module.loads(content.decode("utf-8"))
        else:
            payload_obj = {}
        self.payloads.append(payload_obj)
        if self.responses:
            return self.responses.pop(0)
        return _FakeHTTPResponse(
            status_code=200,
            json_data={
                "model": payload_obj.get("model", "test"),
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
            text='{"choices":[{"message":{"content":"ok"}}]}',
        )

    async def get(self, _url: str):
        return _FakeHTTPResponse(status_code=200, json_data={"data": []}, text="{}")

    async def stream(self, *_args, **_kwargs):
        raise NotImplementedError


def test_sampling_clamps_for_system_and_preferred_configs():
    llm = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="test",
        top_p=-0.5,
        top_k=-3,
        repeat_penalty=0.0,
        presence_penalty=9.0,
        frequency_penalty=-9.0,
    )
    assert llm.top_p == 0.0
    assert llm.top_k == 0
    assert llm.repeat_penalty == 0.01
    assert llm.presence_penalty == 2.0
    assert llm.frequency_penalty == -2.0

    preferred = PreferredLLMConfig(
        top_p=2.5,
        top_k=-1,
        repeat_penalty=-4,
        presence_penalty=-3,
        frequency_penalty=3,
    )
    assert preferred.top_p == 1.0
    assert preferred.top_k == 0
    assert preferred.repeat_penalty == 0.01
    assert preferred.presence_penalty == -2.0
    assert preferred.frequency_penalty == 2.0


def test_resolve_effective_sampling_precedence():
    class LMStudioStub:
        pass

    system_config = SystemConfig()
    system_config.llm.top_p = 0.91
    system_config.llm.top_k = 33
    system_config.llm.repeat_penalty = 1.1
    system_config.llm.presence_penalty = 0.2
    system_config.llm.frequency_penalty = 0.3

    character = CharacterConfig(
        id="test_char",
        name="Test Char",
        role="assistant",
        system_prompt="You are a detailed assistant for tests.",
        preferred_llm={
            "top_p": 0.77,
            "top_k": 12,
            "repeat_penalty": 1.02,
            "presence_penalty": -0.2,
            "frequency_penalty": -0.3,
        },
    )
    invoker = LLMInvocationService({"system_config": system_config, "llm_client": LMStudioStub()})

    resolved = invoker.resolve_effective_config(character=character, invocation_kind="chat")
    assert resolved.top_p == 0.77
    assert resolved.top_k == 12
    assert resolved.repeat_penalty == 1.02
    assert resolved.presence_penalty == -0.2
    assert resolved.frequency_penalty == -0.3

    overridden = invoker.resolve_effective_config(
        character=character,
        invocation_kind="chat",
        top_p_override=0.44,
        top_k_override=8,
        repeat_penalty_override=1.2,
        presence_penalty_override=0.9,
        frequency_penalty_override=-0.9,
    )
    assert overridden.top_p == 0.44
    assert overridden.top_k == 8
    assert overridden.repeat_penalty == 1.2
    assert overridden.presence_penalty == 0.9
    assert overridden.frequency_penalty == -0.9


def test_unified_invoker_passes_sampling_kwargs():
    class RecordingClient:
        def __init__(self):
            self.last_kwargs = None

        async def generate_with_history(self, **kwargs):
            self.last_kwargs = dict(kwargs)
            return LLMResponse(content="ok", model="test", finish_reason="stop")

    system_config = SystemConfig()
    client = RecordingClient()
    invoker = LLMInvocationService({"system_config": system_config, "llm_client": client})
    req = InvocationRequest(
        invocation_kind="chat",
        idempotency_key="sampling-pass-through-1",
        model_id="test-model",
        messages=[{"role": "user", "content": "hello"}],
        top_p=0.88,
        top_k=22,
        repeat_penalty=1.05,
        presence_penalty=0.1,
        frequency_penalty=-0.1,
    )
    result = asyncio.run(invoker.invoke(req))
    assert result["status"] == "success"
    assert client.last_kwargs["top_p"] == 0.88
    assert client.last_kwargs["top_k"] == 22
    assert client.last_kwargs["repeat_penalty"] == 1.05
    assert client.last_kwargs["presence_penalty"] == 0.1
    assert client.last_kwargs["frequency_penalty"] == -0.1


def test_lmstudio_forwards_supported_sampling_fields():
    client = LMStudioLLMClient(
        base_url="http://localhost:1234",
        model="test",
        timeout=10,
        temperature=0.7,
        max_tokens=128,
    )
    recorder = _RecordingAsyncClient(
        responses=[
            _FakeHTTPResponse(
                status_code=200,
                json_data={
                    "model": "test",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                },
                text='{"choices":[{"message":{"content":"ok"}}]}',
            )
        ]
    )
    client.client = recorder
    asyncio.run(
        client.generate_with_history(
            messages=[{"role": "user", "content": "hello"}],
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=-0.2,
        )
    )
    payload = recorder.payloads[0]
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    assert payload["repeat_penalty"] == 1.1
    assert payload["presence_penalty"] == 0.2
    assert payload["frequency_penalty"] == -0.2


def test_ollama_openai_omits_unsupported_top_k_and_repeat_penalty():
    client = OllamaLLMClient(
        base_url="http://localhost:11434",
        model="test",
        timeout=10,
        temperature=0.7,
        max_tokens=128,
        use_legacy_chat_api=False,
    )
    recorder = _RecordingAsyncClient(
        responses=[
            _FakeHTTPResponse(
                status_code=200,
                json_data={
                    "model": "test",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                text='{"choices":[{"message":{"content":"ok"}}]}',
            )
        ]
    )
    client.client = recorder
    asyncio.run(
        client.generate_with_history(
            messages=[{"role": "user", "content": "hello"}],
            top_p=0.92,
            top_k=17,
            repeat_penalty=1.15,
            presence_penalty=0.3,
            frequency_penalty=-0.3,
        )
    )
    payload = recorder.payloads[0]
    assert payload["top_p"] == 0.92
    assert payload["presence_penalty"] == 0.3
    assert payload["frequency_penalty"] == -0.3
    assert "top_k" not in payload
    assert "repeat_penalty" not in payload


def test_koboldcpp_retries_without_advanced_sampling_on_invalid_parameter():
    client = KoboldCppLLMClient(
        base_url="http://localhost:5001",
        model="test",
        timeout=10,
        temperature=0.7,
        max_tokens=128,
    )
    recorder = _RecordingAsyncClient(
        responses=[
            _FakeHTTPResponse(
                status_code=400,
                json_data={"error": {"message": "unknown parameter top_k"}},
                text='{"error":{"message":"unknown parameter top_k"}}',
            ),
            _FakeHTTPResponse(
                status_code=200,
                json_data={
                    "model": "test",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {},
                },
                text='{"choices":[{"message":{"content":"ok"}}]}',
            ),
        ]
    )
    client.client = recorder
    result = asyncio.run(
        client.generate_with_history(
            messages=[{"role": "user", "content": "hello"}],
            top_p=0.8,
            top_k=30,
            repeat_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )
    )
    assert result.content == "ok"
    assert len(recorder.payloads) == 2
    first_payload = recorder.payloads[0]
    second_payload = recorder.payloads[1]
    assert "top_k" in first_payload
    assert "repeat_penalty" in first_payload
    assert "top_p" not in second_payload
    assert "top_k" not in second_payload
    assert "repeat_penalty" not in second_payload
    assert "presence_penalty" not in second_payload
    assert "frequency_penalty" not in second_payload
