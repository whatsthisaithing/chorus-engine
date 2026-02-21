# General Chat Segmentation Backfill

One-time/offline utility to apply relationship v1 segmentation to existing `general_chat` conversations.

What it does in `--apply` mode:
- iterates user turns in each target general chat conversation
- runs segmentation lifecycle to create/close segments idempotently
- generates missing segment summaries
- vectorizes segment summaries into `segment_summaries_<character_id>`
- writes a report under `data/segment_backfill_reports/`

Default mode is dry-run (no writes).

## Usage

Recommended launcher (embedded Python aware):

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_backfill.py
```

Apply changes:

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_backfill.py --apply
```

Filter scope:

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_backfill.py --apply --character-id nova_custom --limit 20
```

Single conversation:

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_backfill.py --apply --conversation-id 2034f17b-2082-4e78-b70c-9c6e4c7cda58
```

## Notes

- v1 keeps segment vectors write-only (prompt assembly does not query segment vectors yet).
- Branch-origin hooks are not auto-closing segments in v1.
- If a conversation has no thread/messages, it is skipped.

## Segment Reset Utility

If you need a clean rebuild, use `run_segment_reset.py`.

Default mode is dry-run. In `--apply` mode it:
- removes `conversation_segments` rows for the target scope
- removes matching segment summary vectors from `segment_summaries_<character_id>`
- optionally rebuilds segments/summaries if `--rebuild` is set

Dry-run preview:

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_segment_reset.py --character-id nova_custom
```

Apply reset only:

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_segment_reset.py --apply --character-id nova_custom
```

Apply reset + rebuild:

```powershell
run_script.bat utilities/general_chat_segmentation_backfill/run_segment_reset.py --apply --rebuild --character-id nova_custom
```
