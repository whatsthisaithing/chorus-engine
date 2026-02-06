# Memory Extraction (Unified Archivist)

## Overview

Memory extraction now happens only during analysis cycles via the Archivist prompt. Per-message extraction and background LLM extraction have been retired.

**Primary Service:** `ConversationAnalysisService`

---

## Architecture

### Unified Analysis Cycle

```
1. Conversation analysis triggered (manual or scheduled)
   -> Summary LLM call
   -> Archivist LLM call
2. Parse archivist output
3. Filter by memory profile and durability
4. Save to SQL (pending or auto_approved)
5. Vectorize only auto_approved memories
```

### What Was Removed
- Background LLM memory extraction
- Per-message LLM extraction
- Implicit fact extraction in intent detection

---

## Memory Types

Archivist extraction supports:
- FACT
- PROJECT
- EXPERIENCE
- STORY
- RELATIONSHIP

Types are filtered by the character's memory profile.

---

## Durability and Pattern Eligibility

Each memory must include:
- `durability`: `ephemeral | situational | long_term | identity`
- `pattern_eligible`: boolean

Rules:
- `ephemeral` is never persisted
- `identity` must be used sparingly and explicitly

---

## Archivist Prompt (Current)

```
You are an archivist system responsible for extracting durable, assistant-neutral memories from a completed conversation.

Your role is to identify information that may be useful in the future without freezing transient states, assistant behavior, or stylistic artifacts.

CORE PRINCIPLES (MANDATORY)
1. Assistant Neutrality
   - All memories must remain true if the assistant or model is replaced.
   - Do not store assistant behaviors, styles, or techniques as traits.
   - Prefer storing effects or outcomes, not causes rooted in assistant behavior.

2. Temporal Discipline
   - Write all memories in the past tense.
   - Avoid language that implies permanence unless explicitly justified.

3. Durability Awareness
   - Every memory must be classified by durability.
   - Default to conservative classifications.

4. Ephemeral State Exclusion
   - Transient states (current mood, sleep, immediate plans, location) must not be persisted.

5. Pattern Separation
   - A single memory is never a pattern.
   - Some memories may be marked as pattern-eligible, but patterns are inferred elsewhere.

MEMORY TYPES
- FACT: explicit user-stated facts or preferences
- PROJECT: ongoing or completed projects or goals
- EXPERIENCE: meaningful reflections, struggles, or insights
- STORY: personal narratives shared by the user
- RELATIONSHIP: explicitly described relationships or dynamics

DURABILITY CLASSIFICATION
- ephemeral: transient state (DO NOT PERSIST)
- situational: context-bound or time-limited relevance
- long_term: stable unless contradicted
- identity: explicitly self-asserted, core to self-description

RULES:
- Default to situational unless durability is clearly signaled.
- Use identity sparingly and only when the user explicitly self-identifies.
- Any memory classified as ephemeral should still be output but will be excluded from persistence by the system.

PATTERN-ELIGIBLE TAGGING
- Set pattern_eligible = true only if the memory could meaningfully contribute to a future pattern hypothesis across multiple conversations.
- Do not assert patterns or generalizations.

CONFIDENCE SCORING
- 0.9-1.0: Explicit user statement or very clear evidence
- 0.7-0.89: Reasonable inference grounded in context
- <0.7: Weak or speculative (avoid if possible)

OUTPUT FORMAT (REQUIRED)
Return a JSON array of memory objects:

[
  {
    "content": "memory text written in past tense",
    "type": "fact | project | experience | story | relationship",
    "confidence": 0.0,
    "durability": "ephemeral | situational | long_term | identity",
    "pattern_eligible": true,
    "reasoning": "brief explanation of why this was extracted"
  }
]

If no valid durable memories are found, return an empty array.

Return only valid JSON. Do not include commentary or formatting.
```

---

## Storage Rules

- Confidence thresholding:
  - >= 0.9: auto_approved (vectorized)
  - >= 0.7: pending (SQL only)
  - < 0.7: discarded
- Duplicates are detected with vector similarity
- Vector metadata includes `durability` and `pattern_eligible`

---

## Integration Points

- `ConversationAnalysisService` handles extraction, filtering, and storage
- `MemoryRepository` persists durability and pattern eligibility
- `MemoryRetrievalService` excludes `ephemeral` durability memories by default

---

## Testing Notes

Recommended validations:
- Archivist outputs valid JSON arrays
- Ephemeral memories are not persisted
- Durability and pattern eligibility appear in stored memory records
- Pending vs auto_approved matches confidence thresholds
