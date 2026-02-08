# Conversation Analysis Changes (Unified Archivist)

## Summary

This document describes the changes made to the conversation analysis and memory extraction pipeline. The system now uses a unified, two-step analysis cycle that separates narrative summarization from durable memory extraction, and removes per-message/background LLM extraction.

The updated design follows the ImprovedConversationAnalysis direction and is implemented directly in the production services. This document is intended for maintainers and reviewers.

---

## Goals

- Eliminate per-message and background LLM extraction
- Split summary generation from memory extraction
- Enforce assistant-neutral summaries and memories
- Introduce durability and pattern eligibility for all extracted memories
- Preserve existing triggers and workflows (manual analysis and heartbeat scheduling)

---

## High-Level Changes

### 1. Two-Step Analysis Cycle

Conversation analysis now runs two independent LLM calls:

1. **Conversation Summary**
   - Narrative, assistant-neutral summary
   - Produces `summary`, `key_topics`, `tone`, `participants`, `emotional_arc`, `open_questions`

2. **Archivist Memory Extraction**
   - Durable, assistant-neutral memory extraction
   - Produces memories with `durability`, `pattern_eligible`, and `reasoning`

These two calls are executed in `ConversationAnalysisService.analyze_conversation()`.

### 1.1 Dedicated Archivist Model (System Config)

Conversation analysis can now use a dedicated archivist LLM model configured in system settings (`llm.archivist_model`). When set, this model is used for both the summary and archivist memory extraction steps, decoupling analysis quality and cost from the primary character model.

### 2. Background Extraction Sunset

The following per-message extraction paths were removed or disabled:

- Background LLM extractor (`BackgroundMemoryExtractor`)
- Intent detection implicit fact extraction
- Per-message queueing of memory extraction jobs

Memory extraction now occurs only during analysis cycles.

### 3. Data Model Updates

New columns added:

- `memories.durability` (string, default `situational`)
- `memories.pattern_eligible` (0/1)
- `conversation_summaries.open_questions` (JSON list)

Legacy summary fields (`themes`) remain for backward compatibility but are no longer written for new summaries.

### 4. Storage and Retrieval Rules

- `durability=ephemeral` memories are discarded before storage
- Confidence thresholds:
  - >= 0.9: auto_approved (vectorized)
  - >= 0.7: pending (SQL only)
  - < 0.7: discarded
- Duplicates now use vector similarity (no string-only dedupe)
- Retrieval excludes `ephemeral` durability by default

### 5. Vector Metadata Updates

Memory vectors now include:
- `durability`
- `pattern_eligible`

Conversation summary vectors now include:
- `open_questions`
- `key_topics`
- `tone`

---

## Services Updated

### `ConversationAnalysisService`

- Split into summary and archivist calls
- New prompt templates embedded in code
- JSON parsing for separate responses
- Retry-on-failure with fallback model
- Uses shared LLM usage lock
- Updated debug logs for separate prompts and responses

### `IntentDetectionService`

- Removed implicit fact extraction
- Simplified prompt to intent-only classification

### `MemoryExtractionService`

- Now only handles storage and dedupe
- No prompt construction or LLM calls

### `MemoryRetrievalService`

- Default memory types updated to long-term types
- Ephemeral durability excluded
- Pending memories excluded for non-core types

### `ConversationSummaryVectorStore`

- Added `open_questions` metadata
- Maintains backward compatibility for older fields

---

## API and Utility Updates

- Manual analysis responses now return `open_questions` and memory durability fields
- Conversation search results include `open_questions`
- Added a utility runner to execute analysis without persisting results

---

## Migration Notes

A new Alembic migration adds durability and pattern eligibility to memories and open questions to summaries:

- `011_add_memory_durability_and_summary_open_questions.py`

Existing data is preserved. New fields default safely.

---

## Operational Notes

- Per-message extraction is gone; memory updates are now periodic via analysis cycles
- Analysis is more consistent because summary and memory extraction are decoupled
- Debug logging now records prompts and responses for both steps

---

## Validation Checklist

Recommended checks after deployment:

1. Manual analysis produces summary + archivist results
2. Memories include durability and pattern eligibility
3. Ephemeral memories are not persisted
4. Summary vectors include `open_questions`
5. No background extraction logs occur during normal messaging

---

## Files Touched (Non-Exhaustive)

- `chorus_engine/services/conversation_analysis_service.py`
- `chorus_engine/services/intent_detection_service.py`
- `chorus_engine/services/memory_extraction.py`
- `chorus_engine/services/memory_retrieval.py`
- `chorus_engine/db/conversation_summary_vector_store.py`
- `chorus_engine/models/conversation.py`
- `chorus_engine/repositories/memory_repository.py`
- `chorus_engine/api/app.py`
- `alembic/versions/011_add_memory_durability_and_summary_open_questions.py`

---
