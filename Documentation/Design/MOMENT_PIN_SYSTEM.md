# Moment Pin System Design

**Phase**: Pinned Moments v1  
**Created**: February 13, 2026  
**Status**: Implemented and actively tuned

---

## Overview

The Moment Pin system captures bounded, high-value conversational events and makes them retrievable as a dedicated continuity layer. It fills a gap between durable memory and raw transcript history:

- Durable memory is long-lived and abstract.
- Transcript history is precise but too large and not safely available by default.
- Moment Pins provide compact, curated "episodic anchors" that can escalate to transcript precision when needed.

This document describes the architecture and behavior of the system as implemented.

---

## Core Philosophy

### The Episodic Anchor Principle

Moment Pins represent specific events, not generalized identity facts. They are intentionally bounded to preserve context and reduce drift.

### The Hot/Cold Separation Principle

- **Hot layer**: compact pin summaries injected into prompt context.
- **Cold layer**: archival transcript snapshot, only accessed through controlled tool flow.

This preserves precision without making full transcript access unconstrained.

### The Gated Precision Principle

Exact quote retrieval must be explicit and rule-bound. The assistant can request cold recall only for pins already injected in that turn.

---

## Architecture

### Data Model

`MomentPin` stores:

- Ownership/scope: `user_id`, `character_id`, `conversation_id`
- Snapshot fields: `selected_message_ids`, `transcript_snapshot`
- Hot summary fields: `what_happened`, `why_model`, `why_user`, `quote_snippet`, `tags`
- Retrieval metadata: `reinforcement_score`, `turns_since_reinforcement`, `archived`, `vector_id`
- Diagnostics metadata: `telemetry_flags`

### Storage

- SQL table: `moment_pins`
- Vector collection (character-scoped): `moment_pins_{character_id}`
- Embedding text shape: summary + why + quote + tags

### Services

- `MomentPinExtractionService`
  - Validates selected messages
  - Builds bounded snapshot with +/-1 context window
  - Extracts summary payload via LLM JSON output
- `MomentPinRetrievalService`
  - Semantic query over pin vectors
  - Score formula: `0.8 * similarity + 0.2 * normalized_reinforcement`
  - Injects up to top 3 pins
  - Applies deterministic recent-pin carryover for short-term follow-ups

### Prompt Integration

Prompt assembly injects a dedicated "Moment Pins" block before conversation-context summaries. Injected IDs are tracked as `used_moment_pin_ids` in assistant metadata.

---

## Retrieval and Cold Recall Flow

1. User message arrives.
2. Hot retrieval selects pins and injects summaries.
3. Assistant may answer directly, or emit `moment_pin.cold_recall` sentinel payload.
4. Server validates:
   - exactly one cold recall call
   - no chained/mixed tool payload
   - `pin_id` is in injected IDs for this turn
   - scope/archived checks pass
5. On success, server reruns model once with archival transcript wrapper.
6. Final response is returned without exposing sentinel payload to client.

---

## API Surface

- `POST /conversations/{conversation_id}/moment-pins`
- `GET /conversations/{conversation_id}/moment-pins`
- `GET /characters/{character_id}/moment-pins`
- `GET /moment-pins/{pin_id}`
- `PATCH /moment-pins/{pin_id}`
- `DELETE /moment-pins/{pin_id}`

---

## UI Behavior

- Pin creation from selected messages (max 20).
- Character-level pin manager with scope/filter controls.
- Per-message assistant indicator for recalled pins (from `used_moment_pin_ids`).
- Archive toggle excludes pins from injection and cold recall but keeps them editable.

---

## Key Design Decisions

- Canonical tool name: `moment_pin.cold_recall`
- Cold recall requires no user approval (`requires_approval: false`)
- Soft-deleted source messages do not invalidate existing pin snapshots
- Invalid cold recall payloads are ignored safely and logged
- Recent-pin continuity uses assistant metadata (no new DB turn counter)

---

## Known Limitations

- Extraction quality depends on model output consistency.
- Pin quote snippets can occasionally be imperfectly clipped by extractor output.
- Streaming path supports cold recall semantics but UI currently prioritizes non-streaming endpoint.

---

## Observability

Per-conversation diagnostics include:

- `conversation.jsonl`: request/response context and prompt composition
- `moment_pins.jsonl`: extraction raw output + parse diagnostics

These logs are intended for behavior tuning and root-cause analysis.

---

## Related Documents

- `Documentation/Design/PROMPT_ASSEMBLY_TOKEN_MANAGEMENT.md`
- `Documentation/Design/CONVERSATION_CONTINUITY_BOOTSTRAP.md`
- `Documentation/Design/MEMORY_INTELLIGENCE_SYSTEM.md`

