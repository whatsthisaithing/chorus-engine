# Archivist Model Harness

Evaluate multiple LLM models on the archivist workflow (summary + memory extraction)
across a fixed set of conversations. Outputs are written to a timestamped
subfolder under the run folder.

## Usage

```bash
python utilities/archivist_model_harness/archivist_harness.py --run-folder <PATH>
```

Optional flags:

```bash
python utilities/archivist_model_harness/archivist_harness.py --run-folder <PATH> \
  --temperature 0 \
  --max-tokens 2048 \
  --retries 1 \
  --timeout-seconds 120
```

Robustness sweep:

```bash
python utilities/archivist_model_harness/archivist_harness.py --run-folder <PATH> \
  --robustness-sweep \
  --robustness-temperatures 0.0,0.2,0.4
```

Retry failures from a previous run:

```bash
python utilities/archivist_model_harness/archivist_harness.py --retry-failures <RESULTS_FOLDER>
```

## Metrics Tool

After a run completes, compute summary metrics:

```bash
python utilities/archivist_model_harness/compute_metrics.py --results-folder <PATH>
```

This writes:
- `conversation_metrics.csv`
- `model_summary_metrics.csv`

## Run Folder Contract

`run_config.json`
```json
{
  "conversation_ids": ["id1", "id2"],
  "models": [{ "model_id": "ollama/model_id1" }],
  "temperature": 0.0,
  "max_tokens": 2048,
  "retries": 1,
  "timeout_seconds": 120,
  "robustness_sweep": false,
  "robustness_temperatures": [0.0, 0.2, 0.4]
}
```

## Output

The harness creates `<run-folder>/results/<run_id>/` with:
- `run_manifest.json`
- `runs.csv`
- `summaries.jsonl`
- `memories.jsonl`
- `memories_flat.csv`
- `failures.jsonl`
- a copy of `run_config.json`
