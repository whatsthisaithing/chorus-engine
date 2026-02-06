# Prompt Revision Spec & Pipeline Notes
## Unified Archivist (Conversation Analysis + Memory Extraction)

This document captures the **current, agreed‑upon design direction** for revising Chorus’s conversation analysis and memory extraction prompts and pipelines. It is intended to be handed directly to a coder agent for implementation, with minimal interpretation required.

The goal is to:
- unify fact‑based and experiential memory extraction under a single archivist model
- eliminate real‑time LLM‑based background extraction
- split summarization and memory extraction into distinct responsibilities
- encode durability, assistant‑neutrality, and pattern‑eligibility as first‑class concepts

---

## 1. High‑Level Decisions (Locked In)

### 1.1 Mothball Background LLM Memory Extraction

**Action:**
- Retire the existing real‑time / background LLM memory extraction workflow.
- Do not invoke any LLM for memory extraction on a per‑message basis.

**Rationale:**
- Archivist model will be distinct from the character’s conversational model.
- Per‑message model swapping causes unacceptable VRAM churn and latency.
- Memory semantics must be consistent across fact and non‑fact types.

**Replacement:**
- All long‑term memory extraction (facts and non‑facts) occurs during **analysis cycles** only.
- Same‑conversation recall relies on the active context window.

---

## 2. Unified Analysis Cycle

An **analysis cycle** may be triggered by:
- manual user request ("Analyze Now")
- heartbeat / idle scheduling
- conversation inactivity thresholds
- future automatic triggers

During an analysis cycle, the system performs **two distinct LLM calls**:

1. Conversation Summary Generation
2. Archivist Memory Extraction

These calls may use the same or different models, but must be logically separated.

---

## 3. Conversation Summary Generation

### 3.1 Purpose

The conversation summary is a **descriptive artifact**, not a memory source of truth. Its role is to:
- capture themes, insights, and emotional progression
- preserve narrative understanding of the conversation
- support reflection and review

### 3.2 Summary Prompt Rules (Additions)

Add the following constraints to the conversation analysis prompt:

- Summaries must focus on **themes, outcomes, and shifts**, not assistant technique.
- Do not foreground assistant style, rhetoric, or personality.
- If assistant behavior is mentioned, frame it only in terms of its *effect on the conversation*, not as a trait.
- Summaries should remain valid even if the assistant model is replaced.

**Style note:**
- Narrative and interpretive language is allowed.
- Precision requirements are lower than for memory extraction.

---

## 4. Archivist Memory Extraction

### 4.1 Scope

The archivist extracts **all memory types** in one pass:
- FACT
- PROJECT
- EXPERIENCE
- STORY
- RELATIONSHIP

There is no separate fact‑only extraction pipeline.

### 4.2 Required Output Fields (Per Memory)

Each extracted memory **must** include:

```json
{
  "content": "memory text",
  "type": "fact|project|experience|story|relationship",
  "confidence": 0.0-1.0,
  "durability": "ephemeral|situational|long_term|identity",
  "pattern_eligible": true|false,
  "reasoning": "why this was extracted"
}
```

### 4.3 Durability Rules

- `ephemeral`: transient state (drop by default)
- `situational`: context‑bound, time‑limited relevance
- `long_term`: stable unless contradicted
- `identity`: only when explicitly self‑asserted by the user

**Defaults & Constraints:**
- Default to `situational` unless clearly signaled otherwise.
- `identity` must be used sparingly and conservatively.
- Any memory with `ephemeral` durability should not be persisted unless explicitly overridden.

Durability is stored as a **database column**, not metadata.

---

## 5. Assistant‑Neutrality Rules (Critical)

Add the following hard constraints to the archivist prompt:

- Memories must remain true under assistant or model replacement.
- Do not store assistant behaviors, styles, or techniques as traits.
- Prefer storing **effects and outcomes**, not causes rooted in assistant behavior.
- Use **past tense** to avoid implied permanence.

**Bad:**
> Nova uses metaphorical language extensively.

**Good:**
> In this conversation, metaphorical framing helped the user reframe abstract topics.

---

## 6. Ephemeral State Exclusion

Explicitly instruct the archivist:

- Do not store transient states (sleeping, current mood, immediate plans, current location).
- If such information is detected, classify it as `ephemeral`.
- `ephemeral` memories should be excluded from persistence by default.

---

## 7. Pattern‑Eligible Tagging

### 7.1 Purpose

Pattern‑eligible tagging marks memories that *may* contribute to future pattern hypotheses, without asserting a pattern.

### 7.2 Rules

- A single memory **never constitutes a pattern**.
- Pattern hypotheses require aggregation across multiple conversations and time.
- Pattern‑eligible memories must still stand alone as valid experiential or fact memories.

### 7.3 Usage

- `pattern_eligible = true` is advisory only.
- Pattern hypotheses are computed separately and are not memories.

---

## 8. Confidence Scoring

Confidence continues to indicate extraction certainty, not durability.

Suggested semantics:
- ≥0.9: explicit user statement or very clear experiential evidence
- ≥0.7: reasonable inference grounded in conversation
- <0.7: speculative (requires review)

Confidence should not override durability rules.

---

## 9. Pipeline Notes for Implementation

### 9.1 Services

- BackgroundMemoryExtractor (LLM‑based) should be disabled or removed.
- ConversationAnalysisService remains, but is refactored to:
  - call summary generation
  - call archivist extraction separately

### 9.2 Fallback Model Behavior

- Until archivist model selection is finalized, use:
  - character‑preferred LLM, or
  - system default LLM

Mark outputs from this phase as **provisional** for later review.

### 9.3 Failure Handling

- If archivist output JSON is invalid:
  - retry once
  - fall back to character/system model
  - log failure for evaluation

---

## 10. Explicit Non‑Goals (For This Phase)

- No real‑time LLM memory extraction
- No pattern hypothesis computation
- No new nervous‑system components
- No fine‑tuning yet

These are deferred to later design phases.

---

## 11. Summary

This revision establishes a single, consistent archivist pipeline for memory extraction while preserving narrative‑focused conversation summaries. It eliminates per‑message LLM extraction, enforces assistant‑neutral durable memory, and introduces durability and pattern‑eligibility as foundational concepts — all without introducing new system components or destabilizing existing flows.

