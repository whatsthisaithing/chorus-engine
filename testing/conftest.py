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
from chorus_engine.models.conversation import Conversation, Thread


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyLLMClient:
    base_url = "http://test-llm"

    async def health_check(self):
        return True

    async def generate_with_history(self, messages, temperature=None, max_tokens=None, model=None):
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        return _DummyResponse(f"Echo: {last_user}" if last_user else "Echo")

    async def generate(self, prompt, system_prompt=None, model=None, **kwargs):
        return _DummyResponse(f"Echo: {prompt}")

    async def stream_with_history(self, messages, temperature=None, max_tokens=None, model=None):
        yield "Echo streamed response"

    async def get_loaded_models(self):
        return []

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
            "ens_tool_executor": app_module._ens_execute_tool_call,
            "ens_scene_preview_executor": app_module._ens_scene_preview,
        }
    )
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
