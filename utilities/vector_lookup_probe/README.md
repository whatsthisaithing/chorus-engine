# Vector Lookup Probe

Interactive utility for debugging retrieval behavior used during prompt assembly:

- conversation summary lookup
- memory lookup
- moment pin lookup

It uses the same service paths used by runtime prompt assembly and writes both:

- terminal output
- timestamped session logs under `data/vector_lookup_tests/`:
  - `vector_lookup_probe_YYYYMMDD_HHMMSS.log`
  - `vector_lookup_probe_YYYYMMDD_HHMMSS.jsonl`

## Why this exists

This is designed for retrieval debugging/tuning, especially to inspect:

- ranked results
- similarity/distance/score data
- short content previews
- section-level failures without crashing the whole probe

## Usage

`--character` is required so lookups are scoped to the correct collections.

### Recommended (uses embedded Python when present)

```powershell
run_script.bat utilities/vector_lookup_probe/vector_lookup_probe.py --character nova_custom --conversation-id 2034f17b-2082-4e78-b70c-9c6e4c7cda58
```

### Interactive loop

```powershell
python utilities/vector_lookup_probe/vector_lookup_probe.py --character nova_custom --conversation-id 2034f17b-2082-4e78-b70c-9c6e4c7cda58
```

Then type queries at `query>`.  
Exit with `exit`, `quit`, or `:q`.

### Non-interactive (one or more queries)

```powershell
python utilities/vector_lookup_probe/vector_lookup_probe.py --character nova_custom --query "do you remember when..." --query "social connections"
```

Using `run_script.bat`:

```powershell
run_script.bat utilities/vector_lookup_probe/vector_lookup_probe.py --character nova_custom --query "do you remember when..." --query "social connections"
```

## Arguments

- `--character` (required): character id
- `--user-id` (optional, default `user:local:owner`): user scope for moment pin retrieval
- `--conversation-id` (optional): enables conversation/thread/source scoped behavior for memory retrieval and excludes current conversation from summary context results
- `--top-n` (optional, default `10`): result depth per section
- `--inject-k` (optional, default `3`): selected pin count for moment pin retrieval
- `--preview-chars` (optional, default `220`): preview length for content snippets
- `--query` (optional, repeatable): run without interactive loop

## Output shape

For each query, the probe emits:

1. Summary Lookup
- selected results (after retrieval service filtering)
- raw vector candidates

2. Memory Lookup
- selected memories with similarity + rank score

3. Moment Pin Lookup
- selected pins with similarity + score + reinforcement fields
- raw vector candidates

If a section fails (for example vector query exceptions), that section is reported as:

`[FAILED] <ExceptionType>: <message>`

while the other sections still run.
