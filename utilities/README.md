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

---

## Future Scripts

Additional utility scripts will be added here:
- Character export/import
- Database cleanup tools
- Memory deduplication utilities
- Conversation merging tools
- Bulk operations helpers
