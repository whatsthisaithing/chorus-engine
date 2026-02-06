"""
Compute metrics from archivist model harness results.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

MEMORY_TYPES = ["fact", "project", "experience", "story", "relationship"]
MEMORY_DURABILITIES = ["ephemeral", "situational", "long_term", "identity"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _stats(values: list[float]) -> tuple[str, str, str, str]:
    if not values:
        return "", "", "", ""
    avg = sum(values) / len(values)
    min_v = min(values)
    max_v = max(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    stddev = math.sqrt(variance)
    return (
        f"{avg:.4f}",
        f"{min_v:.4f}",
        f"{max_v:.4f}",
        f"{stddev:.4f}",
    )


def _normalize_memory(mem: dict[str, Any]) -> dict[str, Any]:
    mem_type = str(mem.get("type", "")).strip().lower()
    durability = str(mem.get("durability", "")).strip().lower()
    confidence = mem.get("confidence")
    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = None
    pattern_eligible = bool(mem.get("pattern_eligible", False))
    flags = mem.get("flags") or []
    if not isinstance(flags, list):
        flags = []
    return {
        "type": mem_type,
        "durability": durability,
        "confidence": confidence_val,
        "pattern_eligible": pattern_eligible,
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from archivist model harness results."
    )
    parser.add_argument(
        "--results-folder",
        required=True,
        help="Path to a harness results folder (run_YYYYMMDD_...)",
    )
    args = parser.parse_args()

    results_folder = Path(args.results_folder)
    if not results_folder.exists():
        raise SystemExit(f"Results folder not found: {results_folder}")

    manifest_path = results_folder / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("run_manifest.json not found in results folder")
    manifest = _read_json(manifest_path)

    run_config_path = results_folder / "run_config.json"
    if not run_config_path.exists():
        raise SystemExit("run_config.json not found in results folder")
    run_config = _read_json(run_config_path)

    conversation_ids = run_config.get("conversation_ids", [])
    model_entries = run_config.get("models", [])
    model_ids = [m.get("model_id") for m in model_entries if isinstance(m, dict) and m.get("model_id")]

    memories_path = results_folder / "memories.jsonl"
    memory_runs = _read_jsonl(memories_path)

    # Aggregate memories per (model_id, conversation_id)
    pair_memories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pair_runs: dict[tuple[str, str], int] = defaultdict(int)

    for entry in memory_runs:
        model_id = entry.get("model_id")
        conversation_id = entry.get("conversation_id")
        if not model_id or not conversation_id:
            continue
        mems = entry.get("memories", [])
        if not isinstance(mems, list):
            continue
        normalized = [_normalize_memory(m) for m in mems]
        pair_memories[(model_id, conversation_id)].extend(normalized)
        pair_runs[(model_id, conversation_id)] += 1

    conversation_rows: list[list[Any]] = []
    conversation_headers = [
        "model_id",
        "conversation_id",
        "run_count",
        "total_memories",
        "avg_confidence",
        "min_confidence",
        "max_confidence",
        "confidence_stddev",
        "pattern_eligible_true",
        "pattern_eligible_false",
    ]
    for mem_type in MEMORY_TYPES:
        conversation_headers.append(f"type_{mem_type}")
    for durability in MEMORY_DURABILITIES:
        conversation_headers.append(f"durability_{durability}")

    for model_id in model_ids:
        for conversation_id in conversation_ids:
            key = (model_id, conversation_id)
            mems = pair_memories.get(key, [])
            if not mems and key not in pair_runs:
                continue

            confidences = [m["confidence"] for m in mems if m["confidence"] is not None]
            avg_conf, min_conf, max_conf, std_conf = _stats(confidences)

            pattern_true = sum(1 for m in mems if m["pattern_eligible"])
            pattern_false = sum(1 for m in mems if not m["pattern_eligible"])

            type_counts = {t: 0 for t in MEMORY_TYPES}
            durability_counts = {d: 0 for d in MEMORY_DURABILITIES}

            for m in mems:
                if m["type"] in type_counts:
                    type_counts[m["type"]] += 1
                if m["durability"] in durability_counts:
                    durability_counts[m["durability"]] += 1

            row = [
                model_id,
                conversation_id,
                pair_runs.get(key, 0),
                len(mems),
                avg_conf,
                min_conf,
                max_conf,
                std_conf,
                pattern_true,
                pattern_false,
            ]
            row.extend(type_counts[t] for t in MEMORY_TYPES)
            row.extend(durability_counts[d] for d in MEMORY_DURABILITIES)
            conversation_rows.append(row)

    conversation_metrics_path = results_folder / "conversation_metrics.csv"
    _write_csv(conversation_metrics_path, conversation_headers, conversation_rows)

    # Model-level summaries
    model_rows: list[list[Any]] = []
    model_headers = [
        "model_id",
        "conversation_count",
        "total_memories",
        "avg_memories_per_conversation",
        "ratio_long_term",
        "ratio_pattern_eligible",
        "avg_confidence",
        "confidence_stddev",
        "conversations_zero_memories",
        "total_flags",
    ]

    for model_id in model_ids:
        # Collect per-conversation aggregates
        conv_keys = [key for key in pair_memories.keys() if key[0] == model_id]
        if not conv_keys:
            continue

        conversation_count = 0
        total_memories = 0
        total_long_term = 0
        total_pattern = 0
        total_flags = 0
        confidence_values: list[float] = []
        zero_memories = 0

        for _, conversation_id in conv_keys:
            key = (model_id, conversation_id)
            mems = pair_memories.get(key, [])
            conversation_count += 1
            if not mems:
                zero_memories += 1
                continue
            total_memories += len(mems)
            for m in mems:
                if m["durability"] == "long_term":
                    total_long_term += 1
                if m["pattern_eligible"]:
                    total_pattern += 1
                confidence = m["confidence"]
                if confidence is not None:
                    confidence_values.append(confidence)
                total_flags += len(m["flags"])

        avg_conf, _, _, std_conf = _stats(confidence_values)
        avg_memories = ""
        ratio_long_term = ""
        ratio_pattern = ""

        if conversation_count > 0:
            avg_memories = f"{(total_memories / conversation_count):.4f}"
        if total_memories > 0:
            ratio_long_term = f"{(total_long_term / total_memories):.4f}"
            ratio_pattern = f"{(total_pattern / total_memories):.4f}"

        model_rows.append(
            [
                model_id,
                conversation_count,
                total_memories,
                avg_memories,
                ratio_long_term,
                ratio_pattern,
                avg_conf,
                std_conf,
                zero_memories,
                total_flags,
            ]
        )

    model_metrics_path = results_folder / "model_summary_metrics.csv"
    _write_csv(model_metrics_path, model_headers, model_rows)

    print(f"Wrote {conversation_metrics_path}")
    print(f"Wrote {model_metrics_path}")
    print(f"Loaded run_id: {manifest.get('run_id', '')}")


if __name__ == "__main__":
    main()
