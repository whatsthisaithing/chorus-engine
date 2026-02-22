from pathlib import Path


def test_v3_signal_only_guardrail_no_signal_envelope_in_core_modules():
    repo_root = Path(__file__).resolve().parents[1]
    core_files = [
        "chorus_engine/ens/runtime.py",
        "chorus_engine/ens/scheduler.py",
        "chorus_engine/ens/dispatcher.py",
    ]
    for rel_path in core_files:
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        assert "SignalEnvelope" not in text, f"legacy SignalEnvelope reference found in {rel_path}"
