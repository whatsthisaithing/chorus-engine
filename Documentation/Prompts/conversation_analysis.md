# Conversation Analysis System (Unified Archivist)

## Overview

The Conversation Analysis System now runs a two-step analysis cycle:

1. Conversation Summary Generation (narrative, assistant-neutral)
2. Archivist Memory Extraction (durable, assistant-neutral memories)

This is a bulk analysis pipeline that processes complete conversations. It replaces prior single-pass analysis and works only during analysis cycles (manual or scheduled), not per-message.

**Location:** `chorus_engine/services/conversation_analysis_service.py`

---

## When Analysis Triggers

### Manual Trigger
- User clicks "Analyze Now" in the UI
- Soft minimums (bypassable with `force=true`):
  - >= 5 messages
  - >= 100 tokens
- Runs synchronously and returns results immediately

### Automatic Triggers (Future)
The service supports automatic analysis based on:
- >= 10,000 tokens
- >= 2,500 tokens + 24h inactive

---

## Analysis Pipeline

### 1. Data Collection
```python
conversation = get_by_id(conversation_id)
messages = get_all_messages(conversation_id)
token_count = count_tokens(messages)
```

**Quality Checks**:
- Skips conversations with < 500 tokens
- Validates conversation exists and has messages

### 2. Two-Step LLM Calls

#### Step A: Conversation Summary
- Purpose: narrative summary, themes, shifts, open questions
- NOT a memory extraction task

**Summary System Prompt (as used):**
```
You are a conversation analysis engine.

Your task is to produce a clear, concise, narrative summary of the conversation provided.

PURPOSE
- Capture the themes, insights, tensions, and shifts that occurred in the conversation.
- Preserve a human-readable understanding of what was explored and why it mattered.
- Support later reflection or review.

SCOPE AND CONSTRAINTS
- This is not a memory extraction task.
- Do not output facts, preferences, or durable memories.
- Do not speculate beyond what occurred in the conversation.

STYLE RULES (CRITICAL)
- Focus on outcomes, themes, and changes, not techniques.
- Do not foreground assistant style, rhetoric, or personality.
- Avoid describing how the assistant responded unless it is necessary to explain an effect on the conversation.
- The summary must remain valid if the assistant or model were replaced.

ALLOWED CONTENT
- Topics discussed
- Emotional or cognitive shifts
- Questions raised or resolved
- Reframes or insights acknowledged by the user
- Open threads or unresolved tensions

DISALLOWED CONTENT
- Assistant-specific traits or behaviors
- Commentary on assistant strategy or skill
- Diagnostic judgments
- New information not present in the conversation

OUTPUT FORMAT
Return a single JSON object:

{
  "summary": "A concise narrative summary of the conversation",
  "key_topics": ["3-8 short topic phrases"],
  "tone": "brief overall tone (1-3 words or short phrase)",
  "participants": ["user", "assistant"],
  "emotional_arc": "brief description of the emotional progression",
  "open_questions": ["optional", "list"]
}

All fields except open_questions are required. Use empty lists/strings when no signal is present.

Return only valid JSON. Do not include commentary or formatting.
```

**User Prompt:**
```
CONVERSATION ({token_count} tokens):
---
{formatted_conversation_messages}
---
```

#### Step B: Archivist Memory Extraction
- Purpose: durable, assistant-neutral memory extraction
- Output is a JSON array of memory objects

**Archivist System Prompt (as used):**
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

**User Prompt:**
```
CONVERSATION ({token_count} tokens):
---
{formatted_conversation_messages}
---
```

### 3. Parsing and Validation
- Summary parser expects a JSON object with `summary` required
- Archivist parser expects a JSON array of memory objects
- Parsing failures trigger one retry, then fallback to the system default model

### 4. Memory Storage Rules
- `durability=ephemeral` memories are discarded
- Confidence thresholds:
  - >= 0.9: `auto_approved` (saved + vectorized)
  - >= 0.7: `pending` (saved only)
  - < 0.7: discarded
- Vector metadata now includes `durability` and `pattern_eligible`

### 5. Summary Storage
- Summary fields stored:
  - `summary`
  - `key_topics`
  - `tone`
  - `participants`
  - `emotional_arc`
  - `open_questions`
- Legacy fields (`themes`) remain in the schema for older data but are no longer written for new analyses

---

## Debug Logging

Logs are written to:
```
data/debug_logs/conversations/{conversation_id}/analysis_{timestamp}.jsonl
```

Entries include:
- Summary system prompt and user prompt
- Summary raw response
- Archivist system prompt and user prompt
- Archivist raw response
- Parsed analysis metadata

---

## API Endpoints

### Manual Analysis
```
POST /conversations/{conversation_id}/analyze?force=false
```

Response includes:
- Summary fields (`summary`, `participants`, `emotional_arc`, `open_questions`)
- Extracted memories with `durability` and `pattern_eligible`

---

## Key Design Principles

1. **Separation of Concerns**
   - Summary and memory extraction are independent steps
2. **Assistant Neutrality**
   - Prompts enforce assistant-agnostic memory and summary content
3. **Durability First**
   - All memories carry durability, and ephemeral is excluded from persistence
4. **Consistency and Safety**
   - Low temperature, strict JSON, retry-on-failure

---

## Differences from Prior Pipeline

| Aspect | Previous | Current |
|---|---|---|
| Prompting | Single combined prompt | Two-step summary + archivist |
| Memory extraction | Mixed with summary | Dedicated archivist prompt |
| Assistant-neutral | Partial | Enforced by prompt |
| Durability | Not present | Required + persisted |
| Pattern eligibility | Not present | Required + persisted |
| Storage | Themes/tone/key_topics | Key topics + tone + open questions + participants |

---

## Configuration

**Service Initialization**:
```python
ConversationAnalysisService(
    db=db_session,
    llm_client=llm_client,
    vector_store=vector_store,
    embedding_service=embedding_service,
    temperature=0.1,
    summary_vector_store=summary_vector_store,
    llm_usage_lock=llm_usage_lock
)
```

---
