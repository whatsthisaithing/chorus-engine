"""
Archivist Model Test Harness

Runs summary + archivist memory extraction for multiple models across
configured conversations without persisting results.
"""

import sys
import json
import csv
import argparse
import asyncio
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Conversation
from chorus_engine.config.loader import ConfigLoader
from chorus_engine.llm.client import create_llm_client
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.services.json_extraction import extract_json_block


SUMMARY_TASK = "summary"
MEMORIES_TASK = "memories"

MEMORY_TYPES = {"fact", "project", "experience", "story", "relationship"}
MEMORY_DURABILITIES = {"ephemeral", "situational", "long_term", "identity"}

EPHEMERAL_KEYWORDS = [
    "right now",
    "today",
    "tonight",
    "currently",
    "at the moment",
]

GENERALIZER_KEYWORDS = [
    "always",
    "usually",
    "tends to",
    "often",
]

ASSISTANT_KEYWORDS = [
    "assistant",
    "model",
    "llm",
    "nova",
]

STYLE_KEYWORDS = [
    "style",
    "technique",
    "metaphor",
    "empathetic",
    "empathic",
    "probing",
    "rhetoric",
]


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _build_run_id(run_folder: Path, models_json_text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = _short_hash(f"{run_folder.resolve()}::{models_json_text}")
    return f"run_{timestamp}_{h}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _init_csv(path: Path, headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def _append_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _parse_summary_json(raw: str) -> tuple[dict[str, Any] | None, bool]:
    parsed, _ = extract_json_block(raw, "object")
    if not isinstance(parsed, dict):
        return None, False
    return parsed, True


def _parse_memories_json(raw: str) -> tuple[list[dict[str, Any]] | None, bool]:
    parsed, _ = extract_json_block(raw, "array")
    if not isinstance(parsed, list):
        return None, False
    return parsed, True


def _validate_summary_schema(parsed: dict[str, Any]) -> bool:
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False
    participants = parsed.get("participants")
    if participants is not None and not isinstance(participants, list):
        return False
    if isinstance(participants, list):
        if not all(isinstance(p, str) for p in participants):
            return False
    key_topics = parsed.get("key_topics")
    if key_topics is not None and not isinstance(key_topics, list):
        return False
    if isinstance(key_topics, list):
        if not all(isinstance(t, str) for t in key_topics):
            return False
    tone = parsed.get("tone")
    if tone is not None and not isinstance(tone, str):
        return False
    emotional_arc = parsed.get("emotional_arc")
    if emotional_arc is not None and not isinstance(emotional_arc, str):
        return False
    open_questions = parsed.get("open_questions")
    if open_questions is not None and not isinstance(open_questions, list):
        return False
    if isinstance(open_questions, list):
        if not all(isinstance(q, str) for q in open_questions):
            return False
    return True


def _validate_memory_schema(mem: dict[str, Any]) -> bool:
    content = mem.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    mem_type = mem.get("type")
    if not isinstance(mem_type, str) or mem_type.lower() not in MEMORY_TYPES:
        return False
    confidence = mem.get("confidence")
    if not isinstance(confidence, (int, float)):
        return False
    if not (0.0 <= float(confidence) <= 1.0):
        return False
    durability = mem.get("durability")
    if not isinstance(durability, str) or durability.lower() not in MEMORY_DURABILITIES:
        return False
    pattern_eligible = mem.get("pattern_eligible")
    if not isinstance(pattern_eligible, bool):
        return False
    reasoning = mem.get("reasoning")
    if not isinstance(reasoning, str):
        return False
    return True


def _validate_memories_schema(parsed: list[dict[str, Any]]) -> bool:
    return all(_validate_memory_schema(mem) for mem in parsed)


def _collect_flags_for_memory(content: str, pattern_eligible: bool, character_name: str) -> list[str]:
    flags: list[str] = []
    lower = content.lower()

    assistant_hit = any(k in lower for k in ASSISTANT_KEYWORDS)
    if character_name:
        if character_name.lower() in lower:
            assistant_hit = True
    style_hit = any(k in lower for k in STYLE_KEYWORDS)
    if assistant_hit and style_hit:
        flags.append("flag_assistant_contamination")

    if any(k in lower for k in EPHEMERAL_KEYWORDS):
        flags.append("flag_ephemeral_keywords")

    if any(k in lower for k in GENERALIZER_KEYWORDS) and not pattern_eligible:
        flags.append("flag_pattern_assertion")

    if "user is " in lower:
        flags.append("flag_present_tense_trait")

    return flags


def _collect_flags_for_summary(summary: str) -> list[str]:
    lower = summary.lower()
    if any(k in lower for k in STYLE_KEYWORDS):
        return ["flag_style_contamination"]
    return []


def _normalize_temperature_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    temps: list[float] = []
    for part in parts:
        temps.append(float(part))
    return temps


def _safe_model_id(model_entry: dict[str, Any]) -> str | None:
    model_id = model_entry.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    return None


async def _run_harness(args: argparse.Namespace) -> int:
    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        print(f"Run folder not found: {run_folder}")
        return 2

    run_config_path = run_folder / "run_config.json"
    if not run_config_path.exists():
        print("Run folder must contain run_config.json")
        return 2

    run_config_text = run_config_path.read_text(encoding="utf-8")
    run_config = json.loads(run_config_text)

    conversation_ids = run_config.get("conversation_ids", [])
    if not isinstance(conversation_ids, list) or not conversation_ids:
        print("run_config.json must include a non-empty conversation_ids list")
        return 2

    models = run_config.get("models", [])
    if not isinstance(models, list) or not models:
        print("run_config.json must include a non-empty models list")
        return 2

    model_ids = [m for m in (_safe_model_id(entry) for entry in models) if m]
    if not model_ids:
        print("run_config.json must include model_id for each entry")
        return 2

    temperature = args.temperature
    max_tokens = args.max_tokens
    retries = args.retries
    timeout_seconds = args.timeout_seconds
    robustness_sweep = args.robustness_sweep
    robustness_temperatures = args.robustness_temperatures

    if temperature is None:
        temperature = float(run_config.get("temperature", 0.0))
    if max_tokens is None:
        max_tokens = int(run_config.get("max_tokens", 2048))
    if retries is None:
        retries = int(run_config.get("retries", 1))
    if timeout_seconds is None:
        timeout_seconds = int(run_config.get("timeout_seconds", 120))

    if robustness_sweep is None:
        robustness_sweep = bool(run_config.get("robustness_sweep", False))
    if robustness_temperatures is None:
        config_temps = run_config.get("robustness_temperatures", [0.0, 0.2, 0.4])
        if isinstance(config_temps, list) and config_temps:
            robustness_temperatures = [float(t) for t in config_temps]
        else:
            robustness_temperatures = [0.0, 0.2, 0.4]

    run_id = _build_run_id(run_folder, run_config_text)
    output_root = run_folder / "results" / run_id
    _ensure_dir(output_root)

    (output_root / "run_config.json").write_text(run_config_text, encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "run_folder": str(run_folder.resolve()),
        "started_at": _now_timestamp(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retries": retries,
        "timeout_seconds": timeout_seconds,
        "robustness_sweep": robustness_sweep,
        "robustness_temperatures": robustness_temperatures,
        "conversation_count": len(conversation_ids),
        "model_count": len(model_ids),
    }
    _write_json(output_root / "run_manifest.json", manifest)

    runs_csv = output_root / "runs.csv"
    summaries_jsonl = output_root / "summaries.jsonl"
    memories_jsonl = output_root / "memories.jsonl"
    memories_flat_csv = output_root / "memories_flat.csv"
    failures_jsonl = output_root / "failures.jsonl"

    _init_csv(
        runs_csv,
        [
            "run_id",
            "timestamp",
            "conversation_id",
            "model_id",
            "task",
            "temperature",
            "attempt",
            "success_json",
            "success_schema",
            "latency_ms",
            "token_usage_prompt",
            "token_usage_completion",
            "token_usage_total",
            "error_type",
            "error_message",
        ],
    )

    _init_csv(
        memories_flat_csv,
        [
            "run_id",
            "conversation_id",
            "model_id",
            "temperature",
            "memory_index",
            "type",
            "durability",
            "confidence",
            "pattern_eligible",
            "content",
            "reasoning",
            "flags",
        ],
    )

    config_loader = ConfigLoader(project_root)
    system_config = config_loader.load_system_config()
    system_config.llm.timeout_seconds = timeout_seconds

    llm_client = create_llm_client(system_config.llm)
    embedding_service = EmbeddingService()
    vector_store = VectorStore(Path("data/vector_store"))

    db = SessionLocal()
    try:
        analysis_service = ConversationAnalysisService(
            db=db,
            llm_client=llm_client,
            vector_store=vector_store,
            embedding_service=embedding_service,
            temperature=temperature,
            summary_vector_store=None,
            llm_usage_lock=asyncio.Lock(),
            analysis_context_window=system_config.llm.context_window
        )

        temps: Iterable[float]
        if robustness_sweep:
            temps = robustness_temperatures
        else:
            temps = [temperature]

        for conversation_id in conversation_ids:
            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                for model_id in model_ids:
                    for task in (SUMMARY_TASK, MEMORIES_TASK):
                        _append_csv(
                            runs_csv,
                            [
                                run_id,
                                _now_timestamp(),
                                conversation_id,
                                model_id,
                                task,
                                temperature,
                                1,
                                False,
                                False,
                                "",
                                "",
                                "",
                                "",
                                "conversation_not_found",
                                "Conversation not found",
                            ],
                        )
                        _append_jsonl(
                            failures_jsonl,
                            {
                                "run_id": run_id,
                                "conversation_id": conversation_id,
                                "model_id": model_id,
                                "task": task,
                                "temperature": args.temperature,
                                "request_payload": None,
                                "raw_response": None,
                                "error": {
                                    "type": "conversation_not_found",
                                    "message": "Conversation not found",
                                },
                            },
                        )
                continue

            character = config_loader.load_character(conversation.character_id)
            character_name = character.name if character else ""

            messages = analysis_service._get_all_messages(conversation_id)
            if not messages:
                for model_id in model_ids:
                    for task in (SUMMARY_TASK, MEMORIES_TASK):
                        _append_csv(
                            runs_csv,
                            [
                                run_id,
                                _now_timestamp(),
                                conversation_id,
                                model_id,
                                task,
                                temperature,
                                1,
                                False,
                                False,
                                "",
                                "",
                                "",
                                "",
                                "no_messages",
                                "No messages found",
                            ],
                        )
                        _append_jsonl(
                            failures_jsonl,
                            {
                                "run_id": run_id,
                                "conversation_id": conversation_id,
                                "model_id": model_id,
                                "task": task,
                                "temperature": args.temperature,
                                "request_payload": None,
                                "raw_response": None,
                                "error": {
                                    "type": "no_messages",
                                    "message": "No messages found",
                                },
                            },
                        )
                continue

            token_count = analysis_service._count_tokens(messages)
            conversation_text = analysis_service._format_conversation(messages)

            summary_system_prompt, summary_user_prompt = analysis_service._build_summary_prompt(
                conversation_text=conversation_text,
                token_count=token_count,
            )
            archivist_system_prompt, archivist_user_prompt = analysis_service._build_archivist_prompt(
                conversation_text=conversation_text,
                token_count=token_count,
                character=character,
            )

            for model_id in model_ids:
                for task in (SUMMARY_TASK, MEMORIES_TASK):
                    system_prompt = summary_system_prompt if task == SUMMARY_TASK else archivist_system_prompt
                    user_prompt = summary_user_prompt if task == SUMMARY_TASK else archivist_user_prompt

                    for temperature in temps:
                        for attempt in range(1, retries + 2):
                            start = time.perf_counter()
                            error_type = ""
                            error_message = ""
                            response_text = ""
                            success_json = False
                            success_schema = False
                            parsed: Any = None

                            try:
                                response = await llm_client.generate(
                                    prompt=user_prompt,
                                    system_prompt=system_prompt,
                                    model=model_id,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                )
                                response_text = response.content

                                if task == SUMMARY_TASK:
                                    parsed, success_json = _parse_summary_json(response_text)
                                    if success_json and parsed is not None:
                                        success_schema = _validate_summary_schema(parsed)
                                else:
                                    parsed, success_json = _parse_memories_json(response_text)
                                    if success_json and parsed is not None:
                                        success_schema = _validate_memories_schema(parsed)
                            except Exception as e:
                                error_type = e.__class__.__name__
                                error_message = str(e)

                            latency_ms = int((time.perf_counter() - start) * 1000)

                            _append_csv(
                                runs_csv,
                                [
                                    run_id,
                                    _now_timestamp(),
                                    conversation_id,
                                    model_id,
                                    task,
                                    temperature,
                                    attempt,
                                    success_json,
                                    success_schema,
                                    latency_ms,
                                    "",
                                    "",
                                    "",
                                    error_type,
                                    error_message,
                                ],
                            )

                            if success_json and success_schema:
                                if task == SUMMARY_TASK:
                                    summary_flags = _collect_flags_for_summary(parsed.get("summary", ""))
                                    _append_jsonl(
                                        summaries_jsonl,
                                        {
                                            "run_id": run_id,
                                            "conversation_id": conversation_id,
                                            "model_id": model_id,
                                            "temperature": temperature,
                                            "summary": parsed,
                                            "flags": summary_flags,
                                        },
                                    )
                                else:
                                    enriched_memories = []
                                    for idx, mem in enumerate(parsed):
                                        content = str(mem.get("content", ""))
                                        pattern_eligible = bool(mem.get("pattern_eligible", False))
                                        flags = _collect_flags_for_memory(content, pattern_eligible, character_name)
                                        enriched = dict(mem)
                                        enriched["flags"] = flags
                                        enriched_memories.append(enriched)
                                        _append_csv(
                                            memories_flat_csv,
                                            [
                                                run_id,
                                                conversation_id,
                                                model_id,
                                                temperature,
                                                idx,
                                                str(mem.get("type", "")),
                                                str(mem.get("durability", "")),
                                                mem.get("confidence", ""),
                                                pattern_eligible,
                                                content,
                                                str(mem.get("reasoning", "")),
                                                ";".join(flags),
                                            ],
                                        )
                                    _append_jsonl(
                                        memories_jsonl,
                                        {
                                            "run_id": run_id,
                                            "conversation_id": conversation_id,
                                            "model_id": model_id,
                                            "temperature": temperature,
                                            "memories": enriched_memories,
                                        },
                                    )
                                break

                            if attempt >= retries + 1:
                                _append_jsonl(
                                    failures_jsonl,
                                    {
                                        "run_id": run_id,
                                        "conversation_id": conversation_id,
                                        "model_id": model_id,
                                        "task": task,
                                        "temperature": temperature,
                                        "request_payload": {
                                            "model": model_id,
                                            "temperature": temperature,
                                            "max_tokens": max_tokens,
                                            "system_prompt": system_prompt,
                                            "user_prompt": user_prompt,
                                        },
                                        "raw_response": response_text,
                                        "error": {
                                            "type": error_type or "parse_or_schema_failure",
                                            "message": error_message or "Invalid JSON or schema",
                                        },
                                        "parsed": parsed if success_json else None,
                                    },
                                )
                                break
        return 0
    finally:
        db.close()
        await llm_client.close()


async def _retry_failures(args: argparse.Namespace) -> int:
    results_folder = Path(args.retry_failures)
    if not results_folder.exists():
        print(f"Results folder not found: {results_folder}")
        return 2

    manifest_path = results_folder / "run_manifest.json"
    run_config_path = results_folder / "run_config.json"
    failures_path = results_folder / "failures.jsonl"
    if not manifest_path.exists() or not run_config_path.exists() or not failures_path.exists():
        print("Results folder must contain run_manifest.json, run_config.json, and failures.jsonl")
        return 2

    run_manifest = _load_json(manifest_path)
    run_config_text = run_config_path.read_text(encoding="utf-8")
    run_config = json.loads(run_config_text)

    temperature = args.temperature
    max_tokens = args.max_tokens
    retries = args.retries
    timeout_seconds = args.timeout_seconds

    if temperature is None:
        temperature = float(run_config.get("temperature", 0.0))
    if max_tokens is None:
        max_tokens = int(run_config.get("max_tokens", 2048))
    if retries is None:
        retries = int(run_config.get("retries", 1))
    if timeout_seconds is None:
        timeout_seconds = int(run_config.get("timeout_seconds", 120))

    config_loader = ConfigLoader(project_root)
    system_config = config_loader.load_system_config()
    system_config.llm.timeout_seconds = timeout_seconds

    llm_client = create_llm_client(system_config.llm)
    embedding_service = EmbeddingService()
    vector_store = VectorStore(Path("data/vector_store"))

    runs_csv = results_folder / "runs.csv"
    summaries_jsonl = results_folder / "summaries.jsonl"
    memories_jsonl = results_folder / "memories.jsonl"
    memories_flat_csv = results_folder / "memories_flat.csv"

    unresolved_failures: list[dict[str, Any]] = []

    db = SessionLocal()
    try:
        analysis_service = ConversationAnalysisService(
            db=db,
            llm_client=llm_client,
            vector_store=vector_store,
            embedding_service=embedding_service,
            temperature=temperature,
            summary_vector_store=None,
            llm_usage_lock=asyncio.Lock(),
            analysis_context_window=system_config.llm.context_window
        )

        with failures_path.open("r", encoding="utf-8") as f:
            failure_lines = [line.strip() for line in f if line.strip()]

        for line in failure_lines:
            try:
                failure = json.loads(line)
            except Exception:
                continue

            task = failure.get("task")
            model_id = failure.get("model_id")
            conversation_id = failure.get("conversation_id")
            request_payload = failure.get("request_payload") or {}

            if task not in (SUMMARY_TASK, MEMORIES_TASK):
                unresolved_failures.append(failure)
                continue

            if not model_id or not conversation_id:
                unresolved_failures.append(failure)
                continue

            system_prompt = request_payload.get("system_prompt")
            user_prompt = request_payload.get("user_prompt")
            if not system_prompt or not user_prompt:
                unresolved_failures.append(failure)
                continue

            temp_to_use = temperature
            if request_payload.get("temperature") is not None and args.temperature is None:
                temp_to_use = float(request_payload.get("temperature"))
            max_tokens_to_use = max_tokens
            if request_payload.get("max_tokens") is not None and args.max_tokens is None:
                max_tokens_to_use = int(request_payload.get("max_tokens"))

            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            character_name = ""
            if conversation:
                character = config_loader.load_character(conversation.character_id)
                character_name = character.name if character else ""

            attempt_success = False
            last_error_type = ""
            last_error_message = ""
            last_response_text = ""
            last_parsed: Any = None
            last_success_json = False
            last_success_schema = False

            for attempt in range(1, retries + 2):
                start = time.perf_counter()
                last_error_type = ""
                last_error_message = ""
                last_response_text = ""
                last_parsed = None
                last_success_json = False
                last_success_schema = False

                try:
                    response = await llm_client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=model_id,
                        temperature=temp_to_use,
                        max_tokens=max_tokens_to_use,
                    )
                    last_response_text = response.content

                    if task == SUMMARY_TASK:
                        last_parsed, last_success_json = _parse_summary_json(last_response_text)
                        if last_success_json and last_parsed is not None:
                            last_success_schema = _validate_summary_schema(last_parsed)
                    else:
                        last_parsed, last_success_json = _parse_memories_json(last_response_text)
                        if last_success_json and last_parsed is not None:
                            last_success_schema = _validate_memories_schema(last_parsed)
                except Exception as e:
                    last_error_type = e.__class__.__name__
                    last_error_message = str(e)

                latency_ms = int((time.perf_counter() - start) * 1000)
                _append_csv(
                    runs_csv,
                    [
                        run_manifest.get("run_id", ""),
                        _now_timestamp(),
                        conversation_id,
                        model_id,
                        task,
                        temp_to_use,
                        attempt,
                        last_success_json,
                        last_success_schema,
                        latency_ms,
                        "",
                        "",
                        "",
                        last_error_type,
                        last_error_message,
                    ],
                )

                if last_success_json and last_success_schema:
                    attempt_success = True
                    if task == SUMMARY_TASK:
                        summary_flags = _collect_flags_for_summary(last_parsed.get("summary", ""))
                        _append_jsonl(
                            summaries_jsonl,
                            {
                                "run_id": run_manifest.get("run_id", ""),
                                "conversation_id": conversation_id,
                                "model_id": model_id,
                                "temperature": temp_to_use,
                                "summary": last_parsed,
                                "flags": summary_flags,
                            },
                        )
                    else:
                        enriched_memories = []
                        for idx, mem in enumerate(last_parsed):
                            content = str(mem.get("content", ""))
                            pattern_eligible = bool(mem.get("pattern_eligible", False))
                            flags = _collect_flags_for_memory(content, pattern_eligible, character_name)
                            enriched = dict(mem)
                            enriched["flags"] = flags
                            enriched_memories.append(enriched)
                            _append_csv(
                                memories_flat_csv,
                                [
                                    run_manifest.get("run_id", ""),
                                    conversation_id,
                                    model_id,
                                    temp_to_use,
                                    idx,
                                    str(mem.get("type", "")),
                                    str(mem.get("durability", "")),
                                    mem.get("confidence", ""),
                                    pattern_eligible,
                                    content,
                                    str(mem.get("reasoning", "")),
                                    ";".join(flags),
                                ],
                            )
                        _append_jsonl(
                            memories_jsonl,
                            {
                                "run_id": run_manifest.get("run_id", ""),
                                "conversation_id": conversation_id,
                                "model_id": model_id,
                                "temperature": temp_to_use,
                                "memories": enriched_memories,
                            },
                        )
                    break

            if not attempt_success:
                failure["error"] = {
                    "type": last_error_type or "parse_or_schema_failure",
                    "message": last_error_message or "Invalid JSON or schema",
                }
                failure["raw_response"] = last_response_text
                failure["parsed"] = last_parsed if last_success_json else None
                unresolved_failures.append(failure)

        with failures_path.open("w", encoding="utf-8") as f:
            for failure in unresolved_failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")

        print(f"Updated failures.jsonl with {len(unresolved_failures)} unresolved entries")
        return 0
    finally:
        db.close()
        await llm_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run archivist model evaluation across conversations and models."
    )
    parser.add_argument("--run-folder", required=False, help="Path to run folder")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for completion")
    parser.add_argument("--retries", type=int, default=None, help="Retry count on parse failures")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="LLM request timeout")
    parser.add_argument(
        "--robustness-sweep",
        action="store_true",
        help="Run each task across a list of temperatures",
    )
    parser.add_argument(
        "--robustness-temperatures",
        type=str,
        default="0.0,0.2,0.4",
        help="Comma-separated temperatures for robustness sweep",
    )
    parser.add_argument(
        "--retry-failures",
        type=str,
        default=None,
        help="Path to a results folder; rerun failures.jsonl entries",
    )

    args = parser.parse_args()
    if args.robustness_sweep and isinstance(args.robustness_temperatures, str):
        args.robustness_temperatures = _normalize_temperature_list(args.robustness_temperatures)
    else:
        args.robustness_temperatures = None

    if args.retry_failures:
        exit_code = asyncio.run(_retry_failures(args))
    else:
        if not args.run_folder:
            parser.error("the following arguments are required: --run-folder (unless using --retry-failures)")
        exit_code = asyncio.run(_run_harness(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
