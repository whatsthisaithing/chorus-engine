# Analysis Reset Utility

Reset conversation summaries and extracted memories so you can re-run analysis.

## Usage
```bash
python utilities/analysis_reset/reset_analysis.py [--character-id <id>] [--dry-run]
```

## Examples
```bash
# Dry run for all characters (no changes)
python utilities/analysis_reset/reset_analysis.py --dry-run

# Reset for a specific character
python utilities/analysis_reset/reset_analysis.py --character-id nova_custom
```

## Behavior
- Deletes `conversation_summaries` for matching conversations.
- Deletes extracted memories (non-CORE and non-EXPLICIT) for matching conversations.
- Resets analysis timestamps on conversations.
- Skips private conversations by default.
- Does NOT delete conversations, threads, or messages.
- Requires typing `YES` to confirm unless `--dry-run` is used.
