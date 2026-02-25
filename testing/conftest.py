from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chorus_engine.config.models import CharacterConfig, ENSConfig, SystemConfig
from chorus_engine.db.database import Base
import chorus_engine.db.database as db_module
import chorus_engine.ens.runtime as ens_runtime_module
from chorus_engine.ens import ENSRuntime
from chorus_engine.ens.llm_invocation_service import LLMInvocationService
from chorus_engine.ens.llm_invocation_service import in_invoker_context
from chorus_engine.ens.llm_control_plane_service import in_control_plane_context
from chorus_engine.models.conversation import Conversation, Thread
from chorus_engine.models.relationship import Relationship, RelationshipSurface
from chorus_engine.llm.base import LLMResponse


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyLLMClient:
    base_url = "http://test-llm"

    async def health_check(self):
        return True

    async def generate_with_history(
        self,
        messages,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        repeat_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        model=None,
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        _ = (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty, kwargs)
        _ = (tools, tool_choice)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        return _DummyResponse(f"Echo: {last_user}" if last_user else "Echo")

    async def generate(self, prompt, system_prompt=None, model=None, **kwargs):
        return _DummyResponse(f"Echo: {prompt}")

    async def generate_vision(
        self,
        *,
        prompt,
        image_base64_list,
        image_mime_type="image/jpeg",
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        repeat_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        model=None,
    ):
        _ = (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty)
        return LLMResponse(
            content='{"main_subject":"test subject","objects":["obj1"],"people":{"count":0,"descriptions":[]},"text_content":"","spatial_layout":"center","mood":"neutral","colors":["blue"],"notable_details":["detail"],"confidence":0.9}',
            model=model or "dummy-vision",
            finish_reason="stop",
            usage=None,
        )

    async def stream_with_history(
        self,
        messages,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        repeat_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        model=None,
    ):
        _ = (messages, temperature, max_tokens, top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty, model)
        yield "Echo streamed response"

    async def get_loaded_models(self):
        return []

    async def ensure_model_loaded(self, model):
        _ = model
        return True

    async def unload_all_models(self):
        return None

    async def reload_model(self):
        return None

    async def switch_model(self, model_path):
        _ = model_path
        return True

    async def unload_model(self, model_name):
        return None

    async def close(self):
        return None


class DummyIntentDetector:
    def detect(self, _message, enable_multi_intent=True, debug=False):
        return []


@dataclass
class AppHelper:
    app_module: any
    SessionLocal: any

    def set_ens_flags(
        self,
        *,
        enabled: bool,
        slice1_chat_ownership: bool,
        nonstream_intake_only: bool = True,
        streaming_intake_only: bool = True,
        slice2_tool_parsing_ownership: bool = False,
        slice2_tool_dispatch_ownership: bool = False,
        slice2_scene_capture_ownership: bool = False,
        slice2_scene_capture_legacy_confirm_without_tool_call: bool = False,
        slice25_media_gating_ownership: bool = False,
        slice3_continuity_writes_ownership: bool = False,
        slice4_config_ownership: bool = False,
        slice6_surface_routing_ownership: bool = False,
        slice65_egress_outbox_ownership: bool = False,
        slice7_unified_llm_invocation: bool = False,
        slice75_llm_control_plane_ownership: bool = False,
        debug_capture_full_prompt: bool = False,
    ):
        self.app_module.app_state["system_config"].ens = ENSConfig(
            enabled=enabled,
            slice1_chat_ownership=slice1_chat_ownership,
            nonstream_intake_only=nonstream_intake_only,
            streaming_intake_only=streaming_intake_only,
            slice1_compat_postprocessing_enabled=False,
            slice2_tool_parsing_ownership=slice2_tool_parsing_ownership,
            slice2_tool_dispatch_ownership=slice2_tool_dispatch_ownership,
            slice2_scene_capture_ownership=slice2_scene_capture_ownership,
            slice2_scene_capture_legacy_confirm_without_tool_call=slice2_scene_capture_legacy_confirm_without_tool_call,
            slice25_media_gating_ownership=slice25_media_gating_ownership,
            slice3_continuity_writes_ownership=slice3_continuity_writes_ownership,
            slice4_config_ownership=slice4_config_ownership,
            slice6_surface_routing_ownership=slice6_surface_routing_ownership,
            slice65_egress_outbox_ownership=slice65_egress_outbox_ownership,
            slice7_unified_llm_invocation=slice7_unified_llm_invocation,
            slice75_llm_control_plane_ownership=slice75_llm_control_plane_ownership,
            debug_capture_full_prompt=debug_capture_full_prompt,
        )

    def create_conversation_thread(self) -> tuple[str, str]:
        db = self.SessionLocal()
        try:
            conversation = Conversation(character_id="test_char", title="Test Conversation", source="web")
            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            thread = Thread(conversation_id=conversation.id, title="Main Thread")
            db.add(thread)
            db.commit()
            db.refresh(thread)
            return conversation.id, thread.id
        finally:
            db.close()


@pytest.fixture()
def app(tmp_path, monkeypatch) -> Generator:
    monkeypatch.chdir(tmp_path)

    characters_dir = tmp_path / "characters"
    characters_dir.mkdir(parents=True, exist_ok=True)
    (characters_dir / "test_char.yaml").write_text(
        "\n".join(
            [
                "id: test_char",
                "name: Test Character",
                "role: assistant",
                "system_prompt: You are a concise test assistant.",
            ]
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "test_chorus.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        echo=False,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_module, "engine", engine, raising=True)
    monkeypatch.setattr(db_module, "SessionLocal", TestingSessionLocal, raising=True)
    monkeypatch.setattr(ens_runtime_module, "SessionLocal", TestingSessionLocal, raising=True)
    import chorus_engine.services.semantic_intent_detection as semantic_intent_detection

    monkeypatch.setattr(
        semantic_intent_detection,
        "get_intent_detector",
        lambda embedding_model=None: DummyIntentDetector(),
        raising=True,
    )

    import chorus_engine.api.app as app_module

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    app_module.app.router.lifespan_context = _no_lifespan

    system_config = SystemConfig()
    character = CharacterConfig(
        id="test_char",
        name="Test Character",
        role="assistant",
        system_prompt="You are a concise test assistant.",
        image_generation={"enabled": True},
        video_generation={"enabled": True},
    )

    app_module.app_state.update(
        {
            "system_config": system_config,
            "characters": {"test_char": character},
            "llm_client": DummyLLMClient(),
            "idle_detector": None,
            "heartbeat_service": None,
            "document_manager": None,
            "vision_service": None,
            "title_service": None,
            "ens_runtime": None,
            "llm_invocation_service": None,
            "ens_tool_executor": app_module._ens_execute_tool_call,
            "ens_scene_preview_executor": app_module._ens_scene_preview,
        }
    )
    app_module.app_state["llm_invocation_service"] = LLMInvocationService(app_module.app_state)
    app_module.app_state["ens_runtime"] = ENSRuntime(app_module.app_state)

    helper = AppHelper(app_module=app_module, SessionLocal=TestingSessionLocal)
    yield app_module.app, helper


@pytest.fixture()
def client(app) -> Generator[TestClient, None, None]:
    test_app, _helper = app
    with TestClient(test_app) as c:
        yield c


@pytest.fixture()
def db(app):
    _test_app, helper = app
    db_session = helper.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


@pytest.fixture()
def helpers(app) -> AppHelper:
    _test_app, helper = app
    return helper


@pytest.fixture(autouse=True)
def _slice7_strict_direct_generate_guard(monkeypatch, app):
    _test_app, helper = app
    llm_client = helper.app_module.app_state.get("llm_client")
    if llm_client is None:
        return

    def _slice7_guard_enabled() -> bool:
        cfg = helper.app_module.app_state.get("system_config")
        ens_cfg = getattr(cfg, "ens", None) if cfg else None
        return bool(
            ens_cfg
            and getattr(ens_cfg, "enabled", False)
            and getattr(ens_cfg, "slice7_unified_llm_invocation", False)
        )

    orig_generate = llm_client.generate
    orig_generate_with_history = llm_client.generate_with_history
    orig_stream_with_history = llm_client.stream_with_history
    orig_generate_vision = getattr(llm_client, "generate_vision", None)

    async def guarded_generate(*args, **kwargs):
        if _slice7_guard_enabled() and not in_invoker_context():
            raise RuntimeError("Direct llm_client.generate call blocked under slice7")
        return await orig_generate(*args, **kwargs)

    async def guarded_generate_with_history(*args, **kwargs):
        if _slice7_guard_enabled() and not in_invoker_context():
            raise RuntimeError("Direct llm_client.generate_with_history call blocked under slice7")
        return await orig_generate_with_history(*args, **kwargs)

    async def guarded_stream_with_history(*args, **kwargs):
        if _slice7_guard_enabled() and not in_invoker_context():
            raise RuntimeError("Direct llm_client.stream_with_history call blocked under slice7")
        async for chunk in orig_stream_with_history(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(llm_client, "generate", guarded_generate)
    monkeypatch.setattr(llm_client, "generate_with_history", guarded_generate_with_history)
    monkeypatch.setattr(llm_client, "stream_with_history", guarded_stream_with_history)

    if orig_generate_vision is not None:
        async def guarded_generate_vision(*args, **kwargs):
            if _slice7_guard_enabled() and not in_invoker_context():
                raise RuntimeError("Direct llm_client.generate_vision call blocked under slice7")
            return await orig_generate_vision(*args, **kwargs)
        monkeypatch.setattr(llm_client, "generate_vision", guarded_generate_vision)


@pytest.fixture(autouse=True)
def _slice75_strict_direct_control_guard(monkeypatch, app):
    _test_app, helper = app
    llm_client = helper.app_module.app_state.get("llm_client")
    if llm_client is None:
        return

    def _slice75_guard_enabled() -> bool:
        cfg = helper.app_module.app_state.get("system_config")
        ens_cfg = getattr(cfg, "ens", None) if cfg else None
        return bool(
            ens_cfg
            and getattr(ens_cfg, "enabled", False)
            and getattr(ens_cfg, "slice75_llm_control_plane_ownership", False)
        )

    for method_name in (
        "health_check",
        "get_loaded_models",
        "ensure_model_loaded",
        "unload_model",
        "unload_all_models",
        "reload_model",
        "switch_model",
    ):
        original = getattr(llm_client, method_name, None)
        if original is None:
            continue

        async def _guarded(*args, __orig=original, __method=method_name, **kwargs):
            if _slice75_guard_enabled() and not in_control_plane_context():
                raise RuntimeError(f"Direct llm_client.{__method} call blocked under slice75")
            return await __orig(*args, **kwargs)

        monkeypatch.setattr(llm_client, method_name, _guarded)
