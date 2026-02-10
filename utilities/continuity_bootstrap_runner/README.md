# Continuity Bootstrap Runner

Runs the full continuity bootstrap pipeline (relationship state, arcs, packets)
without persisting anything to SQL.

## Usage

```bash
python utilities/continuity_bootstrap_runner/run_bootstrap.py <character_id>
```

## Clear bootstrap data

Removes continuity bootstrap artifacts (relationship state, arcs, cache).
Continuity preferences are preserved.

```bash
python utilities/continuity_bootstrap_runner/clear_bootstrap.py --character-id nova_custom
python utilities/continuity_bootstrap_runner/clear_bootstrap.py --all --yes
python utilities/continuity_bootstrap_runner/clear_bootstrap.py --character-id nova_custom --dry-run
```

Optional flags:

```bash
python utilities/continuity_bootstrap_runner/run_bootstrap.py <character_id> --output data/exports/continuity_preview.json
python utilities/continuity_bootstrap_runner/run_bootstrap.py <character_id> --include-prompts --include-raw
```

## Output

Creates a JSON file containing:
- Step status (each stage success/failure)
- Relationship State
- Arc candidates and normalized arcs
- Selected core + active arcs
- Arc score breakdowns (confidence, stickiness, recency penalty, final score)
- Internal bootstrap packet + user preview
- Optional prompts and raw responses
