from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _touch_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def test_list_conversation_logs_includes_rotated_and_legacy(client, helpers):
    conversation_id, _thread_id = helpers.create_conversation_thread()
    conv_dir = Path("data/debug_logs/conversations") / conversation_id

    legacy = conv_dir / "conversation.jsonl"
    rotated = conv_dir / "ens_conversation_2026-02-18.jsonl"
    _write_jsonl(legacy, [{"event": "legacy"}])
    _write_jsonl(rotated, [{"event": "rotated"}])

    now = time.time()
    _touch_mtime(legacy, now - 120)
    _touch_mtime(rotated, now - 60)

    response = client.get("/logs/conversations")
    assert response.status_code == 200
    payload = response.json()
    rows = [row for row in payload["conversations"] if row["conversation_id"] == conversation_id]
    assert len(rows) == 1

    row = rows[0]
    assert row["selected_log_file"] == "ens_conversation_2026-02-18.jsonl"
    assert isinstance(row.get("log_files"), list)
    names = {entry["name"] for entry in row["log_files"]}
    assert "conversation.jsonl" in names
    assert "ens_conversation_2026-02-18.jsonl" in names


def test_get_conversation_log_defaults_to_latest_and_supports_file_and_date(client, helpers):
    conversation_id, _thread_id = helpers.create_conversation_thread()
    conv_dir = Path("data/debug_logs/conversations") / conversation_id

    legacy = conv_dir / "ens_conversation.jsonl"
    rotated = conv_dir / "ens_conversation_2026-02-19.jsonl"
    _write_jsonl(legacy, [{"event": "legacy"}])
    _write_jsonl(rotated, [{"event": "rotated"}])

    now = time.time()
    _touch_mtime(legacy, now - 180)
    _touch_mtime(rotated, now - 10)

    default_response = client.get(f"/logs/conversations/{conversation_id}?prettify=true")
    assert default_response.status_code == 200
    default_payload = default_response.json()
    assert default_payload["selected_log_file"] == "ens_conversation_2026-02-19.jsonl"
    assert default_payload["interactions"][0]["event"] == "rotated"

    by_file = client.get(
        f"/logs/conversations/{conversation_id}?prettify=true&file=ens_conversation.jsonl"
    )
    assert by_file.status_code == 200
    by_file_payload = by_file.json()
    assert by_file_payload["selected_log_file"] == "ens_conversation.jsonl"
    assert by_file_payload["interactions"][0]["event"] == "legacy"

    by_date = client.get(
        f"/logs/conversations/{conversation_id}?prettify=true&date=2026-02-19"
    )
    assert by_date.status_code == 200
    assert by_date.json()["selected_log_file"] == "ens_conversation_2026-02-19.jsonl"


def test_get_conversation_log_rejects_invalid_file_path(client, helpers):
    conversation_id, _thread_id = helpers.create_conversation_thread()
    conv_dir = Path("data/debug_logs/conversations") / conversation_id
    _write_jsonl(conv_dir / "ens_conversation_2026-02-19.jsonl", [{"event": "ok"}])

    response = client.get(
        f"/logs/conversations/{conversation_id}?prettify=true&file=../conversation.jsonl"
    )
    assert response.status_code == 400
    assert "Invalid log file name" in response.json()["detail"]


def test_dispatcher_writes_rotated_ens_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from chorus_engine.ens.dispatcher import ENSDispatcher

    dispatcher = ENSDispatcher(app_state={})
    conversation_id = "conv-rotation-test"
    dispatcher._append_conversation_ens_debug_log(conversation_id, {"event": "hello"})

    today = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = Path("data/debug_logs/conversations") / conversation_id / f"ens_conversation_{today}.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "hello"
