# Multi-Layer Continuity System Design

**Phase**: Cross-Phase Architecture (Phases 3-8 + Pinned Moments v1)  
**Created**: February 13, 2026  
**Status**: Active architecture, implemented in layers

---

## Overview

Chorus Engine continuity is intentionally multi-layered. No single subsystem is asked to solve every continuity problem. Instead, each layer handles a distinct job:

1. Conversation analysis (summary + memory extraction)
2. Explicit memories (user-directed durable facts)
3. Core memories (character backstory and grounding)
4. Moment Pins (episodic anchors with optional transcript precision)
5. Continuity bootstrap (warm-start context at conversation start)

Together, these layers produce rich continuity while preserving safety and debuggability.

---

## Design Goal

Create responses that feel coherent across turns, threads, and conversations without:

- overfitting to stale context,
- overexposing raw transcript data,
- or introducing hidden state that developers cannot reason about.

---

## Layer Model

### Layer 0: Core Memories (Character Ground Truth)

- Source: character configuration and core memory store
- Purpose: stable character identity, role boundaries, persistent framing
- Lifecycle: long-lived, low churn

### Layer 1: Explicit Memories (User-Directed Durables)

- Source: direct user/maintainer creation and edits
- Purpose: deterministic "remember this" facts and preferences
- Lifecycle: long-lived, intentionally curated

### Layer 2: Analysis-Derived Memory and Summaries

- Source: analysis pipeline (summaries + memory extraction)
- Purpose: durable semantic continuity from conversation activity
- Lifecycle: periodically refreshed, retrieval-driven at runtime

### Layer 3: Moment Pins (Episodic Continuity)

- Source: user-selected message sets + extractor synthesis
- Purpose: preserve specific event-level context and meaning
- Lifecycle: retrievable as hot summaries, transcript escalation via controlled cold recall

### Layer 4: Continuity Bootstrap (Warm Start)

- Source: continuity artifacts generated asynchronously
- Purpose: initialize new conversations with relationship stance and active arcs
- Lifecycle: injected at conversation start when enabled by user preference

---

## Runtime Orchestration

At generation time, prompt assembly composes context from multiple layers:

1. system prompt + identity/time headers
2. core and retrieved memories
3. moment pin hot injection (if relevant)
4. conversation context summaries
5. current thread history
6. continuity bootstrap packet (only at start, when opted-in)

This ordering preserves a practical precedence:
- stable identity first,
- then durable context,
- then episodic/follow-up context,
- then immediate thread content.

---

## Why This Composition Works

### Complementary Time Horizons

- Core/explicit memories provide long horizon stability.
- Analysis memories and summaries provide medium horizon continuity.
- Moment pins provide short-to-medium horizon episodic precision.
- Bootstrap handles the "new conversation cold start" boundary.

### Controlled Precision Escalation

Moment pins allow a safe transition from semantic recollection to transcript-anchored precision through gated cold recall.

### Explainability

Each layer has clear source-of-truth, ownership, and debug paths. This avoids opaque behavior where continuity appears from unknown state.

---

## Known Trade-offs

- More moving parts than single-layer memory systems.
- Requires tuning retrieval and ranking interactions across layers.
- Episodic extraction quality depends on model consistency and prompts.
- Some cross-layer edge cases are inevitable and handled through logging + iteration.

---

## Operational Guidance

When continuity behavior seems wrong, inspect by layer:

1. Was the needed context extracted at all?
2. Was it stored in the expected layer?
3. Was retrieval scoped and ranked correctly?
4. Was the layer included in prompt assembly for this turn?
5. Did tool gating block precision escalation?

This diagnostic order keeps debugging targeted and fast.

---

## Related Design Documents

- `Documentation/Design/MEMORY_INTELLIGENCE_SYSTEM.md`
- `Documentation/Design/VECTOR_STORE_SYSTEM.md`
- `Documentation/Design/MOMENT_PIN_SYSTEM.md`
- `Documentation/Design/CONVERSATION_CONTINUITY_BOOTSTRAP.md`
- `Documentation/Design/PROMPT_ASSEMBLY_TOKEN_MANAGEMENT.md`

