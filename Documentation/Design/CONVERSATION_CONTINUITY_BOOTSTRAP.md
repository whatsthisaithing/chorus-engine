# Conversation Continuity Bootstrap System

## Summary

This document describes the design and rationale for the Conversation Continuity Bootstrap system in Chorus Engine. The system restores continuity at the start of a new conversation by injecting a compact, safe context packet that captures relationship stance and ongoing arcs. It reflects the original design intent plus refinements discovered during implementation.

---

## Core Problem

Even with durable memories and conversation summaries, new conversations often feel like partial resets:
- Familiar assistants can feel like strangers again
- Ongoing projects lose momentum
- Emotional or intellectual continuity must be re-earned

This happens because most memory systems are reactive: memories are retrieved only when the user's input triggers them.

Design goal: model assumed context explicitly and safely.

---

## Guiding Principles

1. Continuity is first-class
   - It should be an explicit system concern, not an accidental side effect.
2. Assumed context, not omniscience
   - The assistant should feel familiar without appearing all-knowing or invasive.
3. User agency over continuity
   - Users must always be able to opt out per conversation.
4. Explainable behavior
   - The system should be understandable to users and debuggable by developers.
5. Graceful degradation
   - When information is missing or uncertain, generalize rather than hallucinate.

---

## Goals

- Make conversations feel resumed, not restarted.
- Provide a first-class continuity mechanism beyond reactive memory retrieval.
- Preserve user agency with opt-in/opt-out per conversation.
- Keep continuity safe, non-creepy, and debuggable.
- Allow asynchronous generation that does not block user interaction.

---

## Non-Goals

- This is not a full conversation summary system.
- This does not replace semantic memory retrieval.
- This does not attempt to infer new facts beyond existing summaries and memories.

---

## Core Idea: Conversation Bootstrap

A Conversation Bootstrap is a compact context frame injected only at the start of a conversation. It answers:

1. Who is this user to me?
2. Who am I to this user?
3. What are we in the middle of together?

It is a stance-setting artifact, not a recap.

---

## Key Components

### 1) Relationship State

Defines how the character should show up, not what they should say.

Fields:
- Familiarity level (new, familiar, established, close)
- Tone baseline (2-4 adjectives)
- Interaction contract (working norms)
- Boundaries (safety and role constraints)
- Assistant role frame (one sentence)

Characteristics:
- Small and declarative
- Stable across sessions
- Role- and immersion-aware

### 2) Continuity Arcs

Arcs are ongoing contexts that benefit from continuity, such as:
- Projects
- Recurring themes
- Open questions
- Story threads
- Relationship dynamics
- Constraints

Characteristics:
- Short (1-2 sentences)
- Natural language
- Role- and immersion-aware
- Generalized when uncertain

Arcs are split into two tiers:
- Core Context: 1-3 most important arcs
- Active Arcs: up to 7 additional arcs

### 3) Bootstrap Packets

Two outputs are generated:
- Internal packet (injected into system context)
- User preview (shown for opt-in)

Internal packet ends with explicit guardrails:
- "Use subtly and naturally-only when relevant."
- "Do not quote, paraphrase, or summarize this content to the user."

### 4) Continuity Preference

Per-character preference stored for user choice:
- default mode (ask / use / off)
- skip preview (boolean)

---

## Role- and Immersion-Aware Framing

The same continuity content is framed differently depending on character configuration:

- Assistant: emphasize projects, decisions, constraints.
- Companion: emphasize shared exploration and themes, warmth without dependency.
- Roleplay/unbounded: preserve narrative continuity without locking into specifics.

Immersion settings further constrain allowed language.

---

## User Agency

Continuity is offered, not imposed.

At conversation start, the user can:
- Continue with existing context
- Start fresh (no injection)

Optional user settings allow:
- Remembering the choice
- Skipping the preview prompt

---

## Lifecycle and Scheduling

1. Asynchronous generation
   - Continuity artifacts are generated in heartbeat tasks during idle periods.
2. Staleness detection
   - A bootstrap is stale if any conversation summary or memory was created after the last bootstrap.
3. Gating by analysis queues
   - Continuity generation runs only when there are no pending conversation summaries or memory extractions.
4. Caching
   - Results are cached per character to avoid re-generation.
   - A fingerprint includes prompt version and input timestamps to detect changes.

---

## Internal Packet Format

The internal packet is deterministic and assembled in code to ensure safety:

- Relationship State block
- Core Context (always present)
- Active Arcs (omitted when empty)
- Instruction block

This avoids LLM formatting drift and ensures the instruction is always included.

---

## User Preview Format

User preview is deterministic and uses a neutral "carry forward" framing:

Example structure:
- "If you'd like, I can carry forward this context:"
- Bulleted arc summaries
- "Or we can start fresh-either is fine."

---

## Scoring and Selection

Arcs are scored using:
- Confidence weight (high/medium/low)
- Stickiness weight (high/normal/low)
- Recency penalty based on conversations-ago

Selection favors:
- Highest scores
- Diversity by arc kind

Score breakdown is exposed in the debug utility for tuning.

---

## Prompt Injection Rules

Continuity bootstrap is injected only when:
- It is the first assistant response in the conversation
- The conversation has continuity_mode == "use"
- For multi-user contexts, only when the primary_user matches

Injection is skipped for any subsequent assistant messages in the same conversation.

---

## Data Model

Tables:
- `continuity_relationship_states`
- `continuity_arcs`
- `continuity_bootstrap_cache`
- `continuity_preferences`

Preferences are preserved when clearing bootstrap data.

---

## Services and Responsibilities

### ContinuityBootstrapService
- Generates relationship state, arc candidates, normalized arcs
- Selects core and active arcs with scoring
- Assembles internal packet and user preview
- Saves cache when persisting

### ContinuityBootstrapTaskHandler
- Runs during heartbeat idle time
- Skips if character missing or service unavailable
- Logs success and skip status

### PromptAssemblyService
- Injects internal packet only before first assistant response

---

## API and UI

API endpoints:
- `GET /continuity/preview`
- `POST /continuity/choice`
- `POST /continuity/refresh`

UI behavior:
- Modal appears on new conversations when continuity is available
- Offers Continue/Fresh options
- Can remember preference and skip future previews

---

## Utilities and Debugging

### Continuity Bootstrap Runner
`utilities/continuity_bootstrap_runner/run_bootstrap.py`
- Generates preview without persisting
- Outputs artifacts at every stage (relationship, arcs, scoring, packets)

### Clear Bootstrap Utility
`utilities/continuity_bootstrap_runner/clear_bootstrap.py`
- Deletes relationship state, arcs, and cache
- Preserves preferences

---

## Operational Notes

- Generation uses the standard LLM client and shared usage lock.
- Internal packet and user preview are mojibake-normalized.
- Heartbeat defers tasks when the system is active or GPU load is high.

---

## Validation Checklist

1. Heartbeat generates continuity only when analysis queues are empty.
2. Stale detection triggers after new summaries/memories.
3. Internal packet always includes the instruction block.
4. Injection happens only before the first assistant message.
5. User preview uses "carry forward" framing.
6. Clear utility removes bootstrap artifacts but keeps preferences.

---

## Files Touched (Non-Exhaustive)

- `chorus_engine/services/continuity_bootstrap_service.py`
- `chorus_engine/services/continuity_bootstrap_task.py`
- `chorus_engine/services/prompt_assembly.py`
- `chorus_engine/models/continuity.py`
- `chorus_engine/repositories/continuity_repository.py`
- `chorus_engine/api/app.py`
- `alembic/versions/014_add_continuity_bootstrap.py`
- `utilities/continuity_bootstrap_runner/run_bootstrap.py`
- `utilities/continuity_bootstrap_runner/clear_bootstrap.py`
