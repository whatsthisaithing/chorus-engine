"""ENS v3 replay runner and signature extraction utilities."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional, Type

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import chorus_engine.ens.runtime as ens_runtime_module
import chorus_engine.ens.models as ens_models_module
from chorus_engine.config.models import CharacterConfig, SystemConfig
from chorus_engine.ens import ENSContext, ENSRuntime
from chorus_engine.ens.decision_store import RunSignature, extract_run_signature, extract_run_signature_from_session
from chorus_engine.models.ens import ENSSignalQueue
import chorus_engine.models.ens as ens_sql_models_module

from .mock_llm_provider import DeterministicMockLLMProvider


def _build_app_state(*, provider: Optional[Any] = None, system_config: Optional[SystemConfig] = None) -> Dict[str, Any]:
    cfg = system_config or SystemConfig()
    cfg.ens.enabled = True
    cfg.ens.v3_scheduler_enabled = True
    cfg.ens.v3_arbitration_enabled = True
    cfg.ens.v3_loop_sessions_enabled = True
    cfg.ens.v3_structured_control_enabled = True
    cfg.ens.v3_assistant_result_enabled = True
    cfg.ens.v3_context_compression_enabled = True
    llm_client = provider or DeterministicMockLLMProvider()
    character = CharacterConfig(
        id="test_char",
        name="Test Character",
        role="assistant",
        system_prompt="You are a deterministic replay mock assistant.",
    )
    return {
        "system_config": cfg,
        "characters": {"test_char": character},
        "llm_client": llm_client,
        "llm_invocation_service": None,
        "ens_tool_executor": None,
        "ens_scene_preview_executor": None,
    }


@contextmanager
def _deterministic_uuid_context() -> Any:
    """Patch uuid4 call sites so replay-generated ids are stable across runs."""
    counter = {"i": 0}
    original_global_uuid4 = uuid.uuid4
    original_signal_uuid4 = ens_models_module.uuid.uuid4
    original_sql_uuid4 = ens_sql_models_module.uuid.uuid4

    def _next_uuid() -> uuid.UUID:
        counter["i"] += 1
        return uuid.UUID(int=counter["i"])

    uuid.uuid4 = _next_uuid
    ens_models_module.uuid.uuid4 = _next_uuid
    ens_sql_models_module.uuid.uuid4 = _next_uuid
    try:
        yield
    finally:
        uuid.uuid4 = original_global_uuid4
        ens_models_module.uuid.uuid4 = original_signal_uuid4
        ens_sql_models_module.uuid.uuid4 = original_sql_uuid4


def run_to_quiescence(
    db_path: str,
    *,
    max_ticks: int = 500,
    max_ms: int = 5000,
    provider: Optional[Any] = None,
    provider_cls: Type[Any] = DeterministicMockLLMProvider,
) -> RunSignature:
    """Run scheduler ticks on db_path until queue quiescence and return signature."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False, "timeout": 30}, echo=False)
    ReplaySessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    original_runtime_session_local = ens_runtime_module.SessionLocal
    ens_runtime_module.SessionLocal = ReplaySessionLocal
    try:
        replay_provider = provider if provider is not None else provider_cls()
        app_state = _build_app_state(provider=replay_provider)
        runtime = ENSRuntime(app_state)
        app_state["ens_runtime"] = runtime

        with _deterministic_uuid_context():
            started = time.perf_counter()
            ticks = 0
            while ticks < max(1, int(max_ticks)):
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                if elapsed_ms > max(1, int(max_ms)):
                    break

                outcome = asyncio.run(
                    runtime.scheduler_tick(ENSContext(app_state=app_state, surface="system", source="replay_v3"))
                )
                ticks += 1

                db = ReplaySessionLocal()
                try:
                    pending = (
                        db.query(ENSSignalQueue)
                        .filter(ENSSignalQueue.status.in_(("pending", "running")))
                        .count()
                    )
                finally:
                    db.close()

                if pending == 0 and outcome is None:
                    break

        db = ReplaySessionLocal()
        try:
            return extract_run_signature_from_session(db)
        finally:
            db.close()
    finally:
        ens_runtime_module.SessionLocal = original_runtime_session_local
        engine.dispose()


__all__ = ["RunSignature", "extract_run_signature", "run_to_quiescence", "DeterministicMockLLMProvider"]
