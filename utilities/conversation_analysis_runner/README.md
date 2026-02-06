# Conversation Analysis Runner

Runs the full conversation analysis (summary + archivist memory extraction)
without persisting anything to SQL or the vector stores.

## Usage

```bash
python utilities/conversation_analysis_runner/run_analysis.py <conversation_id>
```

Optional output path:

```bash
python utilities/conversation_analysis_runner/run_analysis.py <conversation_id> --output data/exports/custom_output.json
```

Include prompts and/or raw responses:

```bash
python utilities/conversation_analysis_runner/run_analysis.py <conversation_id> --include-prompts --include-raw
```

## Output

Creates a JSON file containing:
- Conversation metadata
- Summary fields (`summary`, `participants`, `emotional_arc`, `open_questions`)
- Extracted memories (with durability + pattern eligibility)
- Optional prompt and raw response sections when flags are enabled
