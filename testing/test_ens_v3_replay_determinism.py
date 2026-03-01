from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chorus_engine.devtools.replay_v3 import run_to_quiescence
from chorus_engine.ens import ENSContext
from chorus_engine.ens.models import Signal
from chorus_engine.models.ens import ENSLoopStepEvent


def _db_path_from_helpers(helpers) -> Path:
    db = helpers.SessionLocal()
    try:
        bind = db.get_bind()
        url = bind.url
        return Path(str(url.database))
    finally:
        db.close()


def _copy_db(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    shutil.copyfile(src, dst)


def _configure_replay_flags(helpers) -> None:
    helpers.set_ens_flags(
        enabled=True,
        slice1_chat_ownership=True,
        nonstream_intake_only=False,
        streaming_intake_only=True,
    )
    ens_cfg = helpers.app_module.app_state["system_config"].ens
    ens_cfg.context_compression_enabled = True
    ens_cfg.loop_memory_compress_every_n_steps = 4
    ens_cfg.loop_memory_keep_last_k_steps = 2
    ens_cfg.scheduler_sync_ticks_per_ingress = 0


def _seed_baseline_workload(helpers) -> None:
    runtime = helpers.app_module.app_state["ens_runtime"]
    ctx = ENSContext(app_state=helpers.app_module.app_state, surface="system", source="replay_seed")
    run_id = str(uuid.uuid4())[:8]

    hidden_loop_signal = Signal(
        type="loop.session.create_requested",
        scope="SESSION",
        source="replay_seed",
        signal_id=f"seed-hidden-loop-{run_id}",
        payload={
            "loop_id": f"seed-loop-hidden-{run_id}",
            "loop_kind": "generic.hidden",
            "relationship_id": f"rel-seed-hidden-{run_id}",
            "conversation_id": f"conv-seed-hidden-{run_id}",
            "surface_id": "web",
            "character_id": "test_char",
        },
        relationship_hint=f"rel-seed-hidden-{run_id}",
        surface_id="web",
    )
    _ = asyncio.run(runtime.ingest(hidden_loop_signal, ctx))

    tool_loop_signal = Signal(
        type="loop.session.create_requested",
        scope="SESSION",
        source="replay_seed",
        signal_id=f"seed-tool-loop-{run_id}",
        payload={
            "loop_id": f"seed-loop-tool-{run_id}",
            "loop_kind": "generic",
            "relationship_id": f"rel-seed-tool-{run_id}",
            "conversation_id": f"conv-seed-tool-{run_id}",
            "surface_id": "web",
            "character_id": "test_char",
            "step_prompt": "MOCK_TOOL",
        },
        relationship_hint=f"rel-seed-tool-{run_id}",
        surface_id="web",
    )
    _ = asyncio.run(runtime.ingest(tool_loop_signal, ctx))


def _seed_preemption_mode_workload(helpers) -> None:
    runtime = helpers.app_module.app_state["ens_runtime"]
    ctx = ENSContext(app_state=helpers.app_module.app_state, surface="system", source="replay_seed")
    run_id = str(uuid.uuid4())[:8]

    hidden_loop_signal = Signal(
        type="loop.session.create_requested",
        scope="SESSION",
        source="replay_seed",
        signal_id=f"seed-preempt-loop-{run_id}",
        payload={
            "loop_id": f"seed-preempt-hidden-{run_id}",
            "loop_kind": "generic.hidden",
            "relationship_id": f"rel-preempt-{run_id}",
            "conversation_id": f"conv-preempt-{run_id}",
            "surface_id": "web",
            "character_id": "test_char",
        },
        relationship_hint=f"rel-preempt-{run_id}",
        surface_id="web",
    )
    _ = asyncio.run(runtime.ingest(hidden_loop_signal, ctx))

    # Newer USER signal on the same relationship to validate mode+ordering determinism.
    user_signal = Signal(
        type="user.message",
        scope="SESSION",
        source="replay_seed",
        signal_id=f"seed-preempt-user-{run_id}",
        payload={
            "conversation_id": f"conv-preempt-{run_id}",
            "thread_id": f"thread-preempt-{run_id}",
            "content": "newer user input for replay determinism",
            "client_message_id": f"client-preempt-{run_id}",
        },
        relationship_hint=f"rel-preempt-{run_id}",
        surface_id="web",
    )
    _ = asyncio.run(runtime.enqueue_signal(user_signal))


def _count_tool_requests(db_path: Path) -> int:
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False, "timeout": 30}, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        rows = db.query(ENSLoopStepEvent).all()
        return sum(int(r.tool_requests_count or 0) for r in rows)
    finally:
        db.close()
        engine.dispose()


def test_v39_replay_determinism_baseline(helpers, tmp_path):
    _configure_replay_flags(helpers)
    _seed_baseline_workload(helpers)

    src = _db_path_from_helpers(helpers)
    run_a = tmp_path / "replay_a.db"
    run_b = tmp_path / "replay_b.db"
    _copy_db(src, run_a)
    _copy_db(src, run_b)

    sig_a = run_to_quiescence(str(run_a), max_ticks=500, max_ms=8000)
    sig_b = run_to_quiescence(str(run_b), max_ticks=500, max_ms=8000)

    assert sig_a == sig_b
    assert len(sig_a.selected_sequence) > 0
    assert any(len(v) > 0 for v in sig_a.compression_artifacts.values())
    assert _count_tool_requests(run_a) > 0


def test_v39_replay_mode_preemption_determinism(helpers, tmp_path):
    _configure_replay_flags(helpers)
    _seed_preemption_mode_workload(helpers)

    src = _db_path_from_helpers(helpers)
    run_a = tmp_path / "replay_preempt_a.db"
    run_b = tmp_path / "replay_preempt_b.db"
    _copy_db(src, run_a)
    _copy_db(src, run_b)

    sig_a = run_to_quiescence(str(run_a), max_ticks=500, max_ms=8000)
    sig_b = run_to_quiescence(str(run_b), max_ticks=500, max_ms=8000)

    assert sig_a.selected_sequence == sig_b.selected_sequence
    assert sig_a.reason_trace_hashes == sig_b.reason_trace_hashes
    assert sig_a.loop_terminals == sig_b.loop_terminals
