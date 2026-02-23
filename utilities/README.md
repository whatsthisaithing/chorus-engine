# Chorus Engine Utilities

Helpful scripts for inspecting and managing the Chorus Engine database.

## Scripts

### `inspect_character/`

Character database inspection tool that generates detailed markdown reports.

**Usage:**
```bash
python utilities/inspect_character/generate_report.py <character_id>
```

**Example:**
```bash
python utilities/inspect_character/generate_report.py sarah_v1
```

**Output:**
- Creates a timestamped markdown file in `utilities/inspect_character/reports/`
- Filename format: `<character_id>_YYYYMMDD_HHMMSS.md`
- Can be viewed directly in VS Code or any markdown viewer

**Report Contains:**
- Conversations (ID, title, message count, timestamps)
- Threads (ID, conversation, title, message count)
- Messages (ID, thread, role, content preview, timestamp)
- Memories (ID, type, content, priority, confidence, conversation link)
- Conversation Analyses (ID, summary, tone, manual/auto flag)
- Vector Store Entries (vector ID, type, content, metadata)
- Summary statistics by type

**Features:**
- Clean markdown tables (no encoding issues)
- Type summaries for memories and vectors
- Shows orphaned memories (no conversation_id)
- Indicates manual vs automatic analyses
- Timestamped reports for historical tracking

### `reset_character.py`

⚠️ **Destructive Operation** - Completely resets all data for a character.

Removes all conversations, threads, messages, memories (SQL), and vector store entries for a specific character. This is useful for:
- Starting fresh with a character
- Cleaning up after testing
- Removing all traces of a character's interactions

**Usage:**
```bash
python utilities/reset_character.py <character_id>
```

**Example:**
```bash
python utilities/reset_character.py sarah_v1
```

**What it deletes:**
- All conversations for the character
- All threads in those conversations
- All messages in those threads
- All memories (both SQL and vector store)
- The character's vector store collection

**Safety Features:**
- Shows counts of items to be deleted before proceeding
- Requires typing the character ID to confirm
- Cannot be undone once confirmed

**Warning:** This does NOT delete the character configuration file. It only removes conversation history and memories.

### `conversation_analysis_runner/`

Run the full conversation analysis (summary + archivist memory extraction)
without persisting results to SQL or vector stores.

**Usage:**
```bash
python utilities/conversation_analysis_runner/run_analysis.py <conversation_id>
```

**Optional Output Path:**
```bash
python utilities/conversation_analysis_runner/run_analysis.py <conversation_id> --output data/exports/custom_output.json
```

**Output:**
- JSON file with summary + extracted memories (durability + pattern eligibility)

### `analysis_reset/`

Reset conversation summaries and extracted memories (non-CORE/EXPLICIT).
By default, private conversations are skipped.

**Usage:**
```bash
python utilities/analysis_reset/reset_analysis.py [--character-id <id>] [--dry-run]
```

### `archivist_model_harness/`

Evaluate multiple LLM models on the archivist workflow (summary + memory extraction)
across a fixed set of conversations. Outputs are written under
`<run-folder>/results/<run_id>/`.

**Usage:**
```bash
python utilities/archivist_model_harness/archivist_harness.py --run-folder <PATH>
```

**Output:**
- CSV/JSONL results plus a run manifest and copies of run inputs
 - Optional metrics: run `utilities/archivist_model_harness/compute_metrics.py`

### `vector_lookup_probe/`

Interactive retrieval debugger for:
- conversation summary lookup
- memory lookup
- moment pin lookup

Writes terminal output and timestamped session files in `data/vector_lookup_tests/`.

**Usage:**
```bash
run_script.bat utilities/vector_lookup_probe/vector_lookup_probe.py --character <character_id> [--conversation-id <conversation_id>]
```

See `utilities/vector_lookup_probe/README.md` for full options.

### `general_chat_segmentation_backfill/`

Offline utility to backfill relationship v1 segment boundaries and segment summaries for existing `general_chat` conversations.

**Usage:**
```bash
run_script.bat utilities/general_chat_segmentation_backfill/run_backfill.py --apply
```

Segment reset (DB + vectors, optional rebuild):
```bash
run_script.bat utilities/general_chat_segmentation_backfill/run_segment_reset.py --apply --rebuild --character-id <character_id>
```

See `utilities/general_chat_segmentation_backfill/README.md` for filters, dry-run mode, and reset details.

### `replay_user_message_llm.py`

Replay a target message turn and invoke the LLM directly (no ENS path).
Best-effort exact replay for assistant responses:
- Prefers exact debug-captured `messages_for_llm` when available
- Falls back to prompt reconstruction from DB/thread history

**Usage:**
```bash
run_script.bat utilities/replay_user_message_llm.py <message_id>
```

**Dry-run (build payload only, no model invocation):**
```bash
run_script.bat utilities/replay_user_message_llm.py <message_id> --no-invoke
```

**Force reconstruction (ignore debug-captured payloads):**
```bash
run_script.bat utilities/replay_user_message_llm.py <message_id> --force-reconstruct
```

**Rebuild only the system prompt (preserve transcript messages):**
```bash
run_script.bat utilities/replay_user_message_llm.py <message_id> --rebuild-system-prompt
```

**Output:**
- Timestamped JSON bundle in `data/debug/llm_replay_outputs/`
- Includes resolved replay mode, full `messages_for_llm`, model settings, and raw response/error details
- When `--rebuild-system-prompt` is used, bundle includes `system_prompt_comparison` (original vs rebuilt)
- Includes `original_target_payload_analysis` with payload diagnostics for the original target response:
  - sentinel presence flags (`has_sentinel_begin`, `has_sentinel_end`, `has_sentinel_payload`)
  - parsed payload (`sentinel_payload_parsed`) or parse failure (`sentinel_payload_parse_error`)
  - likely malformed/unwrapped payload hints (`likely_payload_outside_sentinel`, `likely_payload_snippets`)
- Includes `original_target_display_analysis` with display-format diagnostics for the original target response:
  - `<assistant_response>` block count (`assistant_response_block_count`)
  - whether visible text exists outside `<assistant_response>` (`has_display_text_outside_assistant_response`)
  - whether visible text exists inside `<assistant_response>` but outside supported template tags (`has_untagged_display_text_inside_assistant_response`)
  - unknown tags detected inside `<assistant_response>` (`unknown_tags_inside_assistant_response`)
- When invocation runs, `llm.payload_analysis` contains the same diagnostics for the replayed raw response
- When invocation runs, `llm.display_analysis` contains the same display-format diagnostics for the replayed raw response

---

## Future Scripts

Additional utility scripts will be added here:
- Character export/import
- Database cleanup tools
- Memory deduplication utilities
- Conversation merging tools
- Bulk operations helpers
