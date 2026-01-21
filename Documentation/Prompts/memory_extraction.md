# Memory Extraction Prompt Specification

**Version**: 1.0  
**Date**: December 31, 2025  
**Component**: `MemoryExtractionService`  
**File Location**: `chorus_engine/services/memory_extraction.py`

---

## Overview

The memory extraction prompt instructs the LLM to identify and extract memorable facts about the user from conversation messages. This enables characters to remember personal details across conversations without explicit "remember this" commands.

## Purpose

Memory extraction serves several critical functions:

1. **Automatic Learning**: Characters learn about users naturally through conversation
2. **Personalization**: Build detailed user profiles for tailored responses
3. **Continuity**: Remember facts across different conversations and sessions
4. **Context Building**: Accumulate knowledge over time for richer interactions

## Architecture

### Extraction Flow

```
1. User sends message
   ↓
2. LLM generates response
   ↓
3. Message queued for extraction (background async)
   ↓
4. Extraction prompt built with message content
   ↓
5. LLM analyzes and returns JSON facts
   ↓
6. Deduplicate against existing memories (vector similarity)
   ↓
7. Save to database + vector store
   ↓
8. Available for future retrieval
```

### Where It's Used

- **Service**: `MemoryExtractionService.extract_from_messages()`
- **Background Worker**: `BackgroundExtractionManager._process_task()`
- **Trigger**: After EVERY user message (in background, non-blocking)
- **Model**: Uses character's preferred LLM model (same as chat)

### Processing Mode

**Background/Async**:
- Extraction runs in separate asyncio task
- Does not block user interaction
- Queue-based processing (FIFO)
- Paused during VRAM-intensive operations (image generation)

---

## Prompt Structure

### Template Location

**Method**: `_build_extraction_prompt()` - Line ~277  
**File**: `chorus_engine/services/background_memory_extractor.py`

**Phase 3 Enhancement**: Multi-user memory attribution with username-based content  
**Phase 8 Enhancement**: Memory profile integration and new memory types (project, experience, story, relationship)

**Note**: The background worker (`background_memory_extractor.py`) handles all extraction processing. It builds its own prompt with multi-user context, applies 6 defensive filters, and saves memories through the storage layer (`memory_extraction.py`).

### Method Signature (Phase 3)

```python
def _build_extraction_prompt(
    self, 
    messages: List[Message], 
    character_name: str,
    character: CharacterConfig,
    conversation_source: str = 'web',  # Phase 3: 'web', 'discord', 'slack'
    primary_user: Optional[str] = None,  # Phase 3: Username who invoked bot
    all_users: Optional[List[str]] = None  # Phase 3: All participant usernames
) -> tuple[str, str]:
```

### Prompt Components

1. **System Context**: Defines role as memory extraction system
2. **Character Context**: Names the assistant to prevent confusion
3. **Multi-User Context**: Platform and participant information (Phase 3)
4. **Conversation Text**: Formatted messages to analyze
5. **Task Description**: What to extract and what NOT to extract
6. **Memory Type Instructions**: From character's memory profile (Phase 8)
7. **Examples**: Good and bad extraction examples (including multi-user)
8. **Critical Rules**: Reinforced extraction guidelines
9. **Output Format**: Strict JSON schema requirement

---

## Full Prompt Template (Phase 3 + Phase 8)

### System Prompt

```
You are a memory extraction system. Your job is to identify and extract information about the user from their messages.

IMPORTANT: The assistant in this conversation is named {character_name!r}. DO NOT confuse the assistant name with the user name.

{multi_user_context}

TASK: Extract memories about the USER (not the assistant).

{extraction_instructions}

MEMORY TYPES (extract these types only):
- FACT: Factual information (name, location, job, preferences, opinions)
- PROJECT: Goals, plans, ongoing projects or objectives
- EXPERIENCE: Shared experiences, activities, events (if allowed)
- STORY: Narratives, anecdotes, past stories (if allowed)
- RELATIONSHIP: Relationship dynamics, emotional bonds (if allowed)

What NOT to extract:
- Unknown information or speculation
- Information about the assistant/character
- Conversational filler (greetings, small talk)
- Facts NOT mentioned by the user
- The assistant name as the user name
- Requests or questions from the user
- Physical descriptions unless explicitly stated by user
- Demographic assumptions (age, gender, ethnicity) unless explicitly stated

For each memory, provide a JSON object with:
- "content": Clear statement (e.g., "User name is John" or "fitzycodesthings enjoys hiking")
- "type": One of {allowed_type_names}
- "confidence": Float 0.0-1.0 (0.95 explicit, 0.8 clear, 0.7 inference)
- "reasoning": One sentence explaining extraction
- "emotional_weight": Optional float 0.0-1.0 for emotionally significant moments
- "participants": Optional list of people involved (including user)
- "key_moments": Optional list of significant moments in the memory

EXAMPLES OF GOOD EXTRACTIONS:
- "My name is Sarah" → {{"content": "User name is Sarah", "type": "fact", "confidence": 0.95, "reasoning": "User explicitly stated their name"}}
- "I love hiking" → {{"content": "User enjoys hiking", "type": "fact", "confidence": 0.9, "reasoning": "Direct statement of interest"}}
- "I'm working on building a game" → {{"content": "User is developing a game", "type": "project", "confidence": 0.9, "reasoning": "Ongoing project stated"}}
- "We went to the beach last summer" → {{"content": "User went to beach with companion", "type": "experience", "confidence": 0.85, "reasoning": "Shared experience mentioned", "participants": ["user", "companion"]}}
- "When I was a kid, I broke my arm" → {{"content": "User broke arm as child", "type": "story", "confidence": 0.9, "reasoning": "Past narrative shared", "emotional_weight": 0.6}}

EXAMPLES OF BAD EXTRACTIONS (DO NOT extract these):
- DO NOT extract unknowns
- DO NOT extract vague impressions
- DO NOT extract facts about assistant
- DO NOT confuse assistant name with user name
- DO NOT invent facts not mentioned
- DO NOT make assumptions from greetings
- DO NOT extract requests/actions
- DO NOT extract descriptions from character responses

CRITICAL RULES:
1. If the user has not explicitly mentioned something, DO NOT extract it
2. DO NOT confuse greetings with the user stating their own name
3. DO NOT invent hobbies, interests, or facts that were not discussed
4. DO NOT extract what the user is asking for or requesting - only extract facts about themselves
5. DO NOT extract physical descriptions unless the user explicitly states them about themselves
6. DO NOT infer gender, age, ethnicity, or other demographics from names or greetings
7. DO NOT extract demographic information unless the user EXPLICITLY states it about themselves
8. Only extract information that was ACTUALLY stated or clearly implied in the USER messages
9. DO NOT extract conversation actions like "User greeted", "User asked", "User said hello"
10. If the messages contain ONLY greetings or small talk with NO actual facts, return an empty array []
11. Questions from the user contain NO facts about the user - questions always return []
12. You are NOT answering the user's questions - you are extracting facts the user stated about THEMSELVES

RESPONSE FORMAT:
You MUST return ONLY a valid JSON array. NO explanations, NO apologies, NO markdown formatting, NO text before or after.

VALID responses:
- Facts found: [{{"content": "...", "category": "...", "confidence": 0.95, "reasoning": "..."}}]
- No facts: []

INVALID responses (DO NOT use these):
- "Sorry, I can't extract..."
- Text explanations
- Markdown code blocks
- Empty responses
```

### Multi-User Context (Phase 3)

When `conversation_source != 'web'` and `all_users` is provided:

```
CONVERSATION CONTEXT:
- Platform: discord
- Multiple users are participating in this conversation
- Participants include: fitzycodesthings, alex, sarah
- The primary user who invoked you is: fitzycodesthings.
- When extracting memories, attribute them to the SPECIFIC USER who stated the fact
- Use their username in the memory content (e.g., "fitzycodesthings enjoys sci-fi" not "user enjoys sci-fi")
- If a user talks about someone else, that's a fact about the SPEAKER, not the person mentioned

EXAMPLES FOR MULTI-USER:
- "fitzycodesthings: I love hiking" → {{"content": "fitzycodesthings enjoys hiking", ...}}
- "alex: I prefer Python" → {{"content": "alex prefers Python programming language", ...}}
- "sarah: I think alex is right" → {{"content": "sarah agrees with alex", ...}}
```

When `conversation_source == 'web'` (single-user):

```
CONVERSATION CONTEXT:
- Platform: web
- Single user conversation
- Use "User" in memory content (e.g., "User enjoys sci-fi")
```

### Purpose of Multi-User Context

1. **Username Attribution**: Memories tied to specific users in group chats
2. **Platform Awareness**: System knows context (Discord group vs web 1-on-1)
3. **Primary User**: Identifies who directly addressed the bot
4. **Participant List**: All users in conversation for validation
5. **Retrieval Accuracy**: Enables per-user memory filtering in multi-user contexts

### Filter Integration

**Filter 6** (User Content Validation) uses `all_users` parameter:

```python
# For Discord, check if content starts with any participant username
is_user_content = content_lower.startswith("user")
if not is_user_content and all_users:
    is_user_content = any(
        content_lower.startswith(username.lower()) 
        for username in all_users
    )

if not is_user_content:
    logger.warning(f"[FILTER] Blocked non-user content: {content}")
    continue
```

This ensures Discord memories like "fitzycodesthings enjoys..." pass validation alongside web memories like "User enjoys...".

---

## Extraction Categories

### 1. `personal_info`

**What It Captures**:
- Names (user's name, family, pets)
- Locations (city, country, neighborhood)
- Age, birthday, personal milestones
- Occupation, education
- Living situation (apartment, house, roommates)

**Examples**:
- "User's name is Sarah"
- "User lives in Seattle"
- "User works as a software engineer"
- "User has a cat named Whiskers"

**Badge Color**: Primary blue

---

### 2. `preference`

**What It Captures**:
- Likes and dislikes
- Favorite things (food, music, movies, books)
- Hobbies and interests
- Style preferences
- Routines and habits

**Examples**:
- "User enjoys hiking on weekends"
- "User prefers tea over coffee"
- "User loves sci-fi novels"
- "User dislikes spicy food"

**Badge Color**: Success green

---

### 3. `experience`

**What It Captures**:
- Past events and memories
- Travel experiences
- Life events (graduation, moving, job changes)
- Significant stories shared
- Historical context about user

**Examples**:
- "User visited Japan in 2023"
- "User graduated from MIT"
- "User broke their arm skiing"
- "User used to live in New York"

**Badge Color**: Info cyan

---

### 4. `relationship`

**What It Captures**:
- Family members
- Friends and social connections
- Romantic relationships
- Professional relationships
- Pets and companions

**Examples**:
- "User has a younger sister named Emma"
- "User is married to Alex"
- "User's best friend is moving to Canada"
- "User mentors junior developers at work"

**Badge Color**: Warning orange

---

### 5. `goal`

**What It Captures**:
- Future plans and aspirations
- Things user wants to do
- Projects in progress
- Learning objectives
- Life goals

**Examples**:
- "User wants to learn Spanish"
- "User is training for a marathon"
- "User plans to start a blog"
- "User hopes to visit Iceland someday"

**Badge Color**: Purple

---

### 6. `skill`

**What It Captures**:
- Abilities and talents
- Professional skills
- Languages spoken
- Expertise areas
- Creative abilities

**Examples**:
- "User speaks fluent French"
- "User is skilled at photography"
- "User can play piano"
- "User is learning Python programming"

**Badge Color**: Dark info

---

## Confidence Scoring

### Confidence Levels

**0.95 - Explicit Statement**:
- User directly stated the fact
- No ambiguity or interpretation needed
- Example: "My name is John" → 0.95 confidence

**0.80-0.90 - Clear Implication**:
- Strongly implied by context
- Reasonable to infer with high certainty
- Example: "I taught my third-grade class about..." → 0.85 confidence for "User is a teacher"

**0.70-0.75 - Reasonable Inference**:
- Implied but requires some interpretation
- Could be wrong but likely accurate
- Example: "I've been coding for 10 years" → 0.75 confidence for "User is experienced in programming"

**Below 0.70 - DO NOT EXTRACT**:
- Too speculative
- Vague or uncertain
- Could easily be wrong

### Confidence Examples

```json
[
  {
    "content": "User's name is Sarah",
    "category": "personal_info",
    "confidence": 0.95,
    "reasoning": "User explicitly introduced themselves"
  },
  {
    "content": "User is a software engineer",
    "category": "personal_info",
    "confidence": 0.9,
    "reasoning": "User mentioned their job at Google as an engineer"
  },
  {
    "content": "User is interested in machine learning",
    "category": "preference",
    "confidence": 0.8,
    "reasoning": "User asked detailed questions about ML algorithms"
  }
]
```

---

## Defensive Filter Architecture

### Filter Implementation

**Location**: `chorus_engine/services/background_memory_extractor.py` - Lines ~404-469  
**Execution**: Post-LLM, pre-save filtering in `_parse_extraction_response()`

The background worker implements **6 defensive filters** that catch bad extractions after the LLM generates them but before they're saved to the database. These filters are the **primary defense** against hallucinations.

### Filter List

**Filter 1: Conversation Actions** (Highest Priority)
```python
conversation_actions = [
    "user greeted", "user said hello", "user said hi", "user initiated",
    "user responded", "user asked", "user requested", "user thanked",
    "user confirmed", "user agreed", "user disagreed", "user inquired",
    "user asked about", "user requested a", "user wants to know"
]
```
**Catches**: Meaningless conversation behaviors like "User greeted" or "User asked about..."  
**Result**: `logger.warning(f"[FILTER] Blocked conversation action: {content}")`

**Filter 2: Assistant/Character Facts**
```python
assistant_patterns = [
    "assistant is", "assistant's", "assistant has", "assistant can",
    "character is", "character's", "character has"
]
```
**Catches**: Facts about the AI character instead of the user  
**Result**: `logger.warning(f"[FILTER] Blocked assistant fact: {content}")`

**Filter 3: System Prompt Leaks**
```python
system_prompt_indicators = [
    "is uncensored", "is unrestricted", "is a helpful", "is truthful",
    "is unbiased", "is designed to", "follows instructions"
]
```
**Catches**: Extraction instructions being saved as user facts  
**Result**: `logger.warning(f"[FILTER] Blocked system prompt leak: {content}")`

**Filter 4: Demographic Hallucinations**
```python
demographic_assumptions = [
    "is male", "is female", "is a man", "is a woman",
    "years old", "age is", "ethnicity is", "race is"
]
```
**Catches**: Gender, age, race assumptions from names or greetings  
**Result**: `logger.warning(f"[FILTER] Blocked demographic hallucination: {content}")`

**Filter 5: Unknown Information**
```python
if "unknown" in content_lower or "not mentioned" in content_lower:
```
**Catches**: Memories stating something is unknown  
**Result**: `logger.warning(f"[FILTER] Blocked unknown information: {content}")`

**Filter 6: User Content Validation** (Phase 3 Enhanced)
```python
# Must start with "user" (web) or a known username (Discord)
is_user_content = content_lower.startswith("user")
if not is_user_content and all_users:
    is_user_content = any(
        content_lower.startswith(username.lower()) 
        for username in all_users
    )

if not is_user_content:
    logger.warning(f"[FILTER] Blocked non-user content: {content}")
    continue
```
**Catches**: Content not about a user (web or Discord participants)  
**Phase 3 Addition**: Validates Discord usernames against `all_users` list  
**Result**: `logger.warning(f"[FILTER] Blocked non-user content: {content}")`

### Filter Effectiveness

**Test Results** (Phase 3 - January 2026):

**Web Conversation (single-user)**:
- **"Hello."** → LLM extracted: `"User greeted"` → **Filter 1 blocked** ✅
- **"What are you doing?"** → LLM extracted: `"User asked about current activity"` → **Filter 1 blocked** ✅
- **"Send me a photo."** → LLM extracted: `"User requested to send a photo"` → **Filter 1 blocked** ✅
- **"My name is John"** → LLM extracted: `"User name is John"` → **All filters passed** ✅
- **"Beautiful photo."** → LLM extracted: `"User appreciates photography"` → **All filters passed** (valid opinion) ✅
- **"My favorite is whisky (Irish kind)"** → LLM extracted: `"User prefers Irish whiskey"` → **All filters passed** ✅

**Discord Conversation (multi-user)**:
- **"fitzycodesthings: My favorite books are sci-fi"** → LLM extracted: `"fitzycodesthings enjoys science fiction books"` → **All filters passed** (Filter 6 validates username) ✅
- **"alex: Hello Nova!"** → LLM extracted: `"alex greeted"` → **Filter 1 blocked** ✅
- **"sarah: What are you doing?"** → LLM extracted: `"sarah asked about current activity"` → **Filter 1 blocked** ✅
- **"fitzycodesthings: I work as a developer"** → LLM extracted: `"fitzycodesthings is a software developer"` → **All filters passed** ✅

### Why Filters Are Necessary

Despite explicit prompt instructions (Rules 9-12) forbidding conversation actions, the LLM **still tries to extract them**. The model interprets "User greeted" as a valid "fact" with high confidence (0.95).

**Layered Defense Strategy**:
1. **Layer 1 (Prompt)**: Explicit rules against conversation actions
2. **Layer 2 (Temperature)**: Low temperature (0.1) for deterministic behavior
3. **Layer 3 (Filters)**: Post-processing filters catch what prompt missed

This multi-layered approach achieves near-perfect filtering while accepting that LLMs will occasionally misinterpret instructions.

### Tuning Filters

**Adding Patterns**:
```python
# Add new patterns to existing filter lists
conversation_actions = [
    "user greeted",
    "user mentioned",  # New pattern
    # ...
]
```

**Creating New Filters**:
```python
# Add after existing filters in _parse_extraction_response()
# Filter 7: Example new filter
if "pattern to block" in content_lower:
    logger.warning(f"[FILTER] Blocked new pattern: {content}")
    continue
```

---

## Anti-Hallucination Safeguards

### Common Hallucination Patterns

**1. Greeting Confusion**:
```
USER: Hello Nova!
❌ WRONG: {"content": "User's name is Nova", ...}
✅ RIGHT: [] (no extraction)
```

**2. Invented Interests**:
```
USER: Hi!
❌ WRONG: {"content": "User seems friendly", ...}
✅ RIGHT: [] (no extraction)
```

**3. Assistant Facts**:
```
ASSISTANT: I love helping with coding!
❌ WRONG: {"content": "User loves coding", ...}
✅ RIGHT: [] (facts about assistant, not user)
```

**4. Unknown Information**:
```
USER: What's my favorite color?
ASSISTANT: I don't know yet!
❌ WRONG: {"content": "User's favorite color is unknown", ...}
✅ RIGHT: [] (don't extract unknowns)
```

### Reinforced Rules

The prompt includes **CRITICAL RULES** section (repeated twice):

1. ✅ Only extract explicitly mentioned information
2. ❌ Never confuse character name with user name
3. ❌ Never invent facts not in conversation
4. ✅ Only extract from THIS specific conversation batch

---

## Message Filtering

### What Gets Analyzed

**USER Messages Only**:
- Extraction only processes messages with `role == USER`
- Assistant messages are ignored
- Prevents hallucination of facts from AI responses

**Recent Messages**:
- Currently processes EVERY user message individually
- Each message analyzed separately in background
- No artificial delays or batching

### Privacy Integration

**Privacy Mode Check**:
```python

**Model Performance Variability**:

Different Ollama-compatible models show varying performance characteristics for memory extraction:

**Dolphin-Mistral-Nemo** (tested):
- ✅ Good JSON format compliance
- ⚠️ Attempts to extract conversation actions despite explicit rules
- ⚠️ Occasionally drops letters in contractions ("don' " instead of "don't")
- ✅ Works well with defensive filters (0.1 temperature + post-processing)

**Qwen 2.5** (expected better):
- Expected: Better instruction following
- Expected: More consistent text generation
- Not yet tested for extraction-specific behavior

**General Observation**: While any Ollama-compatible model is supported, performance varies significantly across:
- Memory extraction accuracy
- Conversational quality
- Image prompt generation
- Rule adherence

**Current Solution**: Defensive filters compensate for model limitations, achieving reliable extraction regardless of model quirks.

### Future Exploration (Developer Notes)

**Potential Improvements**:

1. **Dedicated Extraction Model**: Test lighter, instruction-tuned models specifically for extraction (e.g., `qwen2.5:7b-instruct`, `gemma2:9b-instruct`)
2. **Model Comparison Testing**: Systematic evaluation of extraction quality across different models
3. **Two-Model Architecture**: Consider using one model for chat (character voice) and a different model for extraction (accuracy)
4. **VRAM Management**: If using separate extraction model, implement queue/wait logic to prevent conflicts
5. **Fine-Tuning**: Possibility of fine-tuning a small model specifically for user fact extraction

**Trade-offs to Consider**:
- **Model switching overhead**: Loading/unloading models adds latency
- **VRAM constraints**: Two models simultaneously may not fit in available VRAM
- **Maintenance complexity**: Multiple models = more configuration and testing
- **Current effectiveness**: Filters already achieve excellent results, dedicated model may not provide significant gains
def _should_skip_extraction(self, conversation_id: str) -> bool:
    return self.conversation_repo.is_private(conversation_id)
```

**Behavior**:
- If conversation has `is_private = "true"`, skip extraction entirely
- Private messages never analyzed
- Privacy toggle is per-conversation
- User controls when memories are created

---

## Deduplication Strategy

### Vector Similarity Check

**Method**: `check_for_duplicates()` - Line ~119  
**Threshold**: 0.85 cosine similarity

**Process**:
```
1. Generate embedding for new memory content
2. Search character's existing memories
3. If similarity > 0.85:
   - Consider it a duplicate or reinforcement
   - Update existing memory or skip
4. If similarity < 0.85:
   - Save as new memory
```

### Deduplication Scenarios

**Exact Duplicate**:
```
Existing: "User's name is John"
New: "User's name is John"
→ Skip, already recorded
```

**Reinforcement**:
```
Existing: "User enjoys hiking"
New: "User loves weekend hiking trips"
→ Update confidence, add detail to existing
```

**Distinct But Related**:
```
Existing: "User lives in Seattle"
New: "User works in downtown Seattle"
→ Both saved (different facts, related context)
```

**Different Facts**:
```
Existing: "User's name is John"
New: "User is a teacher"
→ Both saved (completely different information)
```

---

## JSON Output Format

### Schema

```json
[
  {
    "content": "string (required)",
    "category": "personal_info | preference | experience | relationship | goal | skill (required)",
    "confidence": "float 0.0-1.0 (required)",
    "reasoning": "string (required)"
  }
]
```

### Validation

**Required Fields**: All four fields must be present

**Strict Parsing**:
- Expects valid JSON array
- No markdown code blocks (```json)
- No explanatory text outside array
- Empty array `[]` if no facts found

**Error Handling**:
```python
try:
    import json
    memories = json.loads(response)
except json.JSONDecodeError:
    # Try to extract JSON from markdown
    # Fall back to empty list if fails
    return []
```

---

## Complete Prompt Examples

### Example 1: Single-User Web Extraction (Full Memory Profile)

**Character**: Nova (Full immersion - all memory types enabled)  
**Context**: Web conversation, single user  
**User Message**: "I'm working on a sci-fi novel about time loops. My favorite author is Ted Chiang."

**System Prompt**:
```
You are a memory extraction system. Your job is to identify and extract information about the user from their messages.

IMPORTANT: The assistant in this conversation is named 'Nova'. DO NOT confuse the assistant name with the user name.

CONVERSATION CONTEXT:
- Platform: web
- Single user conversation
- Use "User" in memory content (e.g., "User enjoys sci-fi")

TASK: Extract memories about the USER (not the assistant).

MEMORY PROFILE - Extract ALL memory types (full immersion):
You should extract comprehensive memories across all types:
- FACT: Basic factual information
- PROJECT: Goals, plans, ongoing work
- EXPERIENCE: Shared experiences and activities
- STORY: Personal narratives and anecdotes
- RELATIONSHIP: Emotional bonds and dynamics

Be thorough in capturing the richness of the user's experiences and relationships.

MEMORY TYPES (extract these types only):
- FACT: Factual information (name, location, job, preferences, opinions)
- PROJECT: Goals, plans, ongoing projects or objectives
- EXPERIENCE: Shared experiences, activities, events (if allowed)
- STORY: Narratives, anecdotes, past stories (if allowed)
- RELATIONSHIP: Relationship dynamics, emotional bonds (if allowed)

What NOT to extract:
- Unknown information or speculation
- Information about the assistant/character
- Conversational filler (greetings, small talk)
- Facts NOT mentioned by the user
- The assistant name as the user name
- Requests or questions from the user
- Physical descriptions unless explicitly stated by user
- Demographic assumptions (age, gender, ethnicity) unless explicitly stated

For each memory, provide a JSON object with:
- "content": Clear statement (e.g., "User name is John" or "fitzycodesthings enjoys hiking")
- "type": One of ['fact', 'project', 'experience', 'story', 'relationship']
- "confidence": Float 0.0-1.0 (0.95 explicit, 0.8 clear, 0.7 inference)
- "reasoning": One sentence explaining extraction
- "emotional_weight": Optional float 0.0-1.0 for emotionally significant moments
- "participants": Optional list of people involved (including user)
- "key_moments": Optional list of significant moments in the memory

EXAMPLES OF GOOD EXTRACTIONS:
- "My name is Sarah" → {"content": "User name is Sarah", "type": "fact", "confidence": 0.95, "reasoning": "User explicitly stated their name"}
- "I love hiking" → {"content": "User enjoys hiking", "type": "fact", "confidence": 0.9, "reasoning": "Direct statement of interest"}
- "I'm working on building a game" → {"content": "User is developing a game", "type": "project", "confidence": 0.9, "reasoning": "Ongoing project stated"}
- "We went to the beach last summer" → {"content": "User went to beach with companion", "type": "experience", "confidence": 0.85, "reasoning": "Shared experience mentioned", "participants": ["user", "companion"]}
- "When I was a kid, I broke my arm" → {"content": "User broke arm as child", "type": "story", "confidence": 0.9, "reasoning": "Past narrative shared", "emotional_weight": 0.6}

EXAMPLES OF BAD EXTRACTIONS (DO NOT extract these):
- DO NOT extract unknowns
- DO NOT extract vague impressions
- DO NOT extract facts about assistant
- DO NOT confuse assistant name with user name
- DO NOT invent facts not mentioned
- DO NOT make assumptions from greetings
- DO NOT extract requests/actions
- DO NOT extract descriptions from character responses

CRITICAL RULES:
1. If the user has not explicitly mentioned something, DO NOT extract it
2. DO NOT confuse greetings with the user stating their own name
3. DO NOT invent hobbies, interests, or facts that were not discussed
4. DO NOT extract what the user is asking for or requesting - only extract facts about themselves
5. DO NOT extract physical descriptions unless the user explicitly states them about themselves
6. DO NOT infer gender, age, ethnicity, or other demographics from names or greetings
7. DO NOT extract demographic information unless the user EXPLICITLY states it about themselves
8. Only extract information that was ACTUALLY stated or clearly implied in the USER messages
9. DO NOT extract conversation actions like "User greeted", "User asked", "User said hello"
10. If the messages contain ONLY greetings or small talk with NO actual facts, return an empty array []
11. Questions from the user contain NO facts about the user - questions always return []
12. You are NOT answering the user's questions - you are extracting facts the user stated about THEMSELVES

RESPONSE FORMAT:
You MUST return ONLY a valid JSON array. NO explanations, NO apologies, NO markdown formatting, NO text before or after.

VALID responses:
- Facts found: [{"content": "...", "category": "...", "confidence": 0.95, "reasoning": "..."}]
- No facts: []

INVALID responses (DO NOT use these):
- "Sorry, I can't extract..."
- Text explanations
- Markdown code blocks
- Empty responses

If no memorable facts are present, return exactly: []

CRITICAL: Returning an empty array [] is the CORRECT and EXPECTED response when messages contain no facts. Simple greetings like "Hello" MUST return []. Questions MUST return [].
```

**User Content** (passed separately):
```
I'm working on a sci-fi novel about time loops. My favorite author is Ted Chiang.
```

**Expected LLM Response**:
```json
[
  {
    "content": "User is writing a science fiction novel about time loops",
    "type": "project",
    "confidence": 0.95,
    "reasoning": "User explicitly stated their ongoing creative project"
  },
  {
    "content": "User's favorite author is Ted Chiang",
    "type": "fact",
    "confidence": 0.95,
    "reasoning": "User directly stated their literary preference"
  }
]
```

---

### Example 2: Multi-User Discord Extraction

**Character**: Nova  
**Context**: Discord group chat  
**Participants**: fitzycodesthings, alex, sarah  
**Primary User**: fitzycodesthings  
**Messages**: 
- "fitzycodesthings: I love sci-fi books, especially hard science fiction"
- "alex: I prefer fantasy novels"

**System Prompt**:
```
You are a memory extraction system. Your job is to identify and extract information about the user from their messages.

IMPORTANT: The assistant in this conversation is named 'Nova'. DO NOT confuse the assistant name with the user name.

CONVERSATION CONTEXT:
- Platform: discord
- Multiple users are participating in this conversation
- Participants include: fitzycodesthings, alex, sarah
- The primary user who invoked you is: fitzycodesthings.
- When extracting memories, attribute them to the SPECIFIC USER who stated the fact
- Use their username in the memory content (e.g., "fitzycodesthings enjoys sci-fi" not "user enjoys sci-fi")
- If a user talks about someone else, that's a fact about the SPEAKER, not the person mentioned

EXAMPLES FOR MULTI-USER:
- "fitzycodesthings: I love hiking" → {"content": "fitzycodesthings enjoys hiking", ...}
- "alex: I prefer Python" → {"content": "alex prefers Python programming language", ...}
- "sarah: I think alex is right" → {"content": "sarah agrees with alex", ...}

TASK: Extract memories about the USER (not the assistant).

MEMORY PROFILE - Extract ALL memory types (full immersion):
You should extract comprehensive memories across all types:
- FACT: Basic factual information
- PROJECT: Goals, plans, ongoing work
- EXPERIENCE: Shared experiences and activities
- STORY: Personal narratives and anecdotes
- RELATIONSHIP: Emotional bonds and dynamics

Be thorough in capturing the richness of the user's experiences and relationships.

MEMORY TYPES (extract these types only):
- FACT: Factual information (name, location, job, preferences, opinions)
- PROJECT: Goals, plans, ongoing projects or objectives
- EXPERIENCE: Shared experiences, activities, events (if allowed)
- STORY: Narratives, anecdotes, past stories (if allowed)
- RELATIONSHIP: Relationship dynamics, emotional bonds (if allowed)

What NOT to extract:
- Unknown information or speculation
- Information about the assistant/character
- Conversational filler (greetings, small talk)
- Facts NOT mentioned by the user
- The assistant name as the user name
- Requests or questions from the user
- Physical descriptions unless explicitly stated by user
- Demographic assumptions (age, gender, ethnicity) unless explicitly stated

For each memory, provide a JSON object with:
- "content": Clear statement (e.g., "User name is John" or "fitzycodesthings enjoys hiking")
- "type": One of ['fact', 'project', 'experience', 'story', 'relationship']
- "confidence": Float 0.0-1.0 (0.95 explicit, 0.8 clear, 0.7 inference)
- "reasoning": One sentence explaining extraction
- "emotional_weight": Optional float 0.0-1.0 for emotionally significant moments
- "participants": Optional list of people involved (including user)
- "key_moments": Optional list of significant moments in the memory

EXAMPLES OF GOOD EXTRACTIONS:
- "My name is Sarah" → {"content": "User name is Sarah", "type": "fact", "confidence": 0.95, "reasoning": "User explicitly stated their name"}
- "I love hiking" → {"content": "User enjoys hiking", "type": "fact", "confidence": 0.9, "reasoning": "Direct statement of interest"}
- "I'm working on building a game" → {"content": "User is developing a game", "type": "project", "confidence": 0.9, "reasoning": "Ongoing project stated"}
- "We went to the beach last summer" → {"content": "User went to beach with companion", "type": "experience", "confidence": 0.85, "reasoning": "Shared experience mentioned", "participants": ["user", "companion"]}
- "When I was a kid, I broke my arm" → {"content": "User broke arm as child", "type": "story", "confidence": 0.9, "reasoning": "Past narrative shared", "emotional_weight": 0.6}

EXAMPLES OF BAD EXTRACTIONS (DO NOT extract these):
- DO NOT extract unknowns
- DO NOT extract vague impressions
- DO NOT extract facts about assistant
- DO NOT confuse assistant name with user name
- DO NOT invent facts not mentioned
- DO NOT make assumptions from greetings
- DO NOT extract requests/actions
- DO NOT extract descriptions from character responses

CRITICAL RULES:
1. If the user has not explicitly mentioned something, DO NOT extract it
2. DO NOT confuse greetings with the user stating their own name
3. DO NOT invent hobbies, interests, or facts that were not discussed
4. DO NOT extract what the user is asking for or requesting - only extract facts about themselves
5. DO NOT extract physical descriptions unless the user explicitly states them about themselves
6. DO NOT infer gender, age, ethnicity, or other demographics from names or greetings
7. DO NOT extract demographic information unless the user EXPLICITLY states it about themselves
8. Only extract information that was ACTUALLY stated or clearly implied in the USER messages
9. DO NOT extract conversation actions like "User greeted", "User asked", "User said hello"
10. If the messages contain ONLY greetings or small talk with NO actual facts, return an empty array []
11. Questions from the user contain NO facts about the user - questions always return []
12. You are NOT answering the user's questions - you are extracting facts the user stated about THEMSELVES

RESPONSE FORMAT:
You MUST return ONLY a valid JSON array. NO explanations, NO apologies, NO markdown formatting, NO text before or after.

VALID responses:
- Facts found: [{"content": "...", "category": "...", "confidence": 0.95, "reasoning": "..."}]
- No facts: []

INVALID responses (DO NOT use these):
- "Sorry, I can't extract..."
- Text explanations
- Markdown code blocks
- Empty responses

If no memorable facts are present, return exactly: []

CRITICAL: Returning an empty array [] is the CORRECT and EXPECTED response when messages contain no facts. Simple greetings like "Hello" MUST return []. Questions MUST return [].
```

**User Content** (passed separately):
```
fitzycodesthings: I love sci-fi books, especially hard science fiction
alex: I prefer fantasy novels
```

**Expected LLM Response**:
```json
[
  {
    "content": "fitzycodesthings loves science fiction books, particularly hard science fiction",
    "type": "fact",
    "confidence": 0.95,
    "reasoning": "fitzycodesthings explicitly stated their literary preference with specificity"
  },
  {
    "content": "alex prefers fantasy novels",
    "type": "fact",
    "confidence": 0.95,
    "reasoning": "alex directly stated their reading preference"
  }
]
```

**Post-Filter Results**: Both memories pass all 6 defensive filters because:
- ✅ Not conversation actions
- ✅ Not about assistant
- ✅ Not system prompt leaks
- ✅ No demographic assumptions
- ✅ No "unknown" statements
- ✅ Start with participant usernames (fitzycodesthings, alex)

---

### Example 3: Empty Result (Greeting Only)

**User Message**: "Hello Nova!"

**Expected LLM Response**:
```json
[]
```

**Reasoning**: Simple greeting with no extractable facts. Even if LLM tries to extract "User greeted", Filter 1 blocks it.

---

## LLM Parameters

### Model Selection

**Default**: Uses character's preferred model (same as chat)  
**Reason**: Consistency in extraction quality per character  
**Override**: Extracted memories use character-specific models

**Example**:
- Nova (qwen2.5:14b-instruct) extrbackground_memory_extractor.py`  
**Method**: `_build_extraction_prompt()` - Line ~238

**Filters**: Same file, `_parse_extraction_response()` - Lines ~404-469phin-mistral-nemo

### Temperature

**Setting**: 0.1 (significantly lower than chat's 0.9)  
**Reason**: Highly deterministic extraction with minimal creative interpretation  
**Trade-off**: Very consistent JSON output, strict rule adherence  
**Tuning History**: Started at 0.2, lowered to 0.1 to reduce conversation action extractions

### Context Window

**Input**: Only the messages being extracted (not full history)  
**Typical**: 1-3 messages per extraction call  
**Limit**: No hard limit, processes what's queued

---

## Background Processing

### Queue System

**Manager**: `BackgroundExtractionManager`  
**Queue Type**: `asyncio.Queue` (FIFO)  
**Worker**: Single worker task processes sequentially

**Flow**:
```
1. User message arrives
2. Response generated
3. Message added to extraction queue
4. Worker picks up task (when available)
5. Extraction runs in background
6. Results saved asynchronously
```

### Pause/Resume (VRAM Protection)

**New in Phase 5.1**: Prevents race conditions during image generation

**Pause Trigger**:
- Image generation starts
- LLM models unload for VRAM

**Resume Trigger**:
- Image generation completes
- Models reload

**Behavior While Paused**:
- Worker checks pause flag every 500ms
- Tasks remain in queue
- No memory extraction during pause
- Resumes automatically when safe

---

## Performance Characteristics

### Timing

**Extraction Speed**: ~1-3 seconds per message  
**Doesn't Block**: User interaction continues normally  
**Queue Depth**: Typically 0-2 tasks (fast processing)

### Resource Usage

**VRAM**: Uses same LLM as chat (~4-14GB depending on model)  
**CPU**: Minimal (JSON parsing, database writes)  
**Network**: Local Ollama API calls (fast)

### Throughput

**Messages/Second**: Processes as fast as LLM generates  
**Typical Load**: 1 extraction per user message  
**No Backlog**: Queue clears quickly in normal usage

---

## Database Storage

### Memory Table Schema

```sql
CREATE TABLE memories (
    id VARCHAR PRIMARY KEY,
    conversation_id VARCHAR (optional),
    thread_id VARCHAR (optional),
    character_id VARCHAR NOT NULL,
    memory_type VARCHAR NOT NULL,  -- 'implicit' for extracted
    content TEXT NOT NULL,
    created_at TIMESTAMP,
    priority INTEGER DEFAULT 80,
    tags TEXT[],
    confidence FLOAT,
    category VARCHAR,
    status VARCHAR DEFAULT 'active',
    source_messages TEXT[]
)
```

### Vector Store

**Collection**: Per-character collection in ChromaDB  
**Embedding**: `all-MiniLM-L6-v2` (384 dimensions)  
**Metadata**: Stores memory_type, confidence, category

---

## Editing the Extraction Prompt

### Location

**File**: `chorus_engine/services/memory_extraction.py`  
**Method**: `_build_extraction_prompt()` - Line ~271

### When to Edit

**Add Categories**:
- Define new category in prompt
- Update JSON schema
- Add to database enum
- Update UI badge colors

**Adjust Examples**:
- Add more good/bad examples
- Clarify edge cases
- Address new hallucination patterns

**Change Extraction Philosophy**:
- Modify what counts as "extractable"
- Adjust confidence thresholds
- Change category definitions

**Improve Anti-Hallucination**:
- Add more "DO NOT" rules
- Provide clearer counter-examples
- Reinforce critical guidelines

### Testing Changes

After editing prompt:

1. **Test with various conversations**: Casual chat, technical discussion, personal sharing
2. **Check for hallucinations**: Verify no invented facts
3. **Validate JSON format**: Ensure valid output
4. **Review confidence scores**: Appropriate for content
5. **Test edge cases**: Greetings, character mentions, questions

---

## Integration Points

### 1. API Message Endpoints

**Files**: `chorus_engine/api/app.py`

**Non-Streaming** (Line ~1304):
```python
await extraction_manager.queue_extraction(
    conversation_id=conversation.id,
    character_id=character_id,
    messages=[msg_obj],
    model=model,
    character_name=character.name
)
```

**Streaming** (Line ~1539):
```python
await extraction_manager.queue_extraction(
    conversation_id=conversation.id,
    character_id=character_id,
    messages=[msg_obj],
    model=model,
    character_name=character.name
)
```

### 2. Background Worker

**File**: `chorus_engine/services/background_extraction.py`  
**Method**: `_process_task()` - Line ~149

Calls extraction service with character's model:
```python
extracted_memories = await extraction_service.extract_from_messages(
    messages=task.messages,
    character_id=task.character_id,
    conversation_id=task.conversation_id,
    model=task.model,
    character_name=task.character_name
)
```

### 3. Memory Repository

**File**: `chorus_engine/repositories/memory_repository.py`

Saves extracted memories:
```python
memory = Memory(
    conversation_id=conversation_id,
    thread_id=thread_id,
    character_id=character_id,
    memory_type=MemoryType.IMPLICIT,
    content=extracted.content,
    confidence=extracted.confidence,
    category=extracted.category,
    priority=80  # Default for implicit
)
```

### 4. Vector Store

**File**: `chorus_engine/db/vector_store.py`

Embeds and stores for retrieval:
```python
await vector_store.add_memory(
    memory_id=memory.id,
    character_id=character_id,
    content=memory.content,
    memory_type=MemoryType.IMPLICIT,
    metadata={...}
)
```

---

## UI Visibility

### Memory Panel

**Location**: Web UI sidebar, "Memories" button  
**Filter**: Can filter by category (shows icons/badges)  
**Display**: Shows extracted memories with confidence stars

### Pending Memories (Future)

**Feature**: Manual approval queue  
**Status**: Infrastructure ready, UI minimal  
**Purpose**: Review before saving to vector store

---

## Troubleshooting

### No Memories Extracted

**Check**:
1. Is extraction manager running? (logs show "Background extraction worker STARTED")
2. Is conversation in privacy mode? (private conversations skip extraction)
3. Are user messages being sent? (extraction only on user messages)
4. Check LLM response (logs show extraction JSON)

### Wrong Information Extracted

**Fixes**:
1. Add counter-examples to prompt
2. Lower confidence threshold for inclusion
3. Improve clarity in "what NOT to extract"
4. Test specific conversation patterns

### Hallucination Issues

**Diagnosis**:
1. Check logs for extracted content
2. Identify hallucination pattern
3. Add explicit rule to CRITICAL RULES
4. Add bad example to prevent pattern

**Example Fix**:
```
Problem: Extracting assistant name as user name
Fix: Add to prompt:
"CRITICAL: When user says 'Hello Nova', Nova is the ASSISTANT name, NOT the user's name"
```

### JSON Parsing Failures

**Causes**:
- LLM added markdown code blocks
- LLM added explanatory text
- Malformed JSON

**Solution**: `_parse_extraction_response()` handles cleanup:
- Strips markdown fences
- Extracts array from text
- Falls back to empty list on error

---

## Phase 8: Whole-Conversation Analysis

### Overview

In addition to per-message extraction (Phase 4/6.5), Phase 8 adds whole-conversation analysis that runs when conversations complete. This extracts thematic memories that require full conversation context.

**Service**: `ConversationAnalysisService`  
**File**: `chorus_engine/services/conversation_analysis_service.py`  
**Trigger**: Manual (Analyze Now button) or automatic (token thresholds, completion detection)

### Memory Types Extracted

- **Project**: Ongoing activities and goals mentioned throughout conversation
- **Experience**: Shared moments and collaborations that happened in this conversation
- **Story**: Emotionally significant narratives user shared from their past
- **Relationship**: Trust developments and vulnerability moments

### Analysis Prompt

**Method**: `_build_analysis_prompt()` - Line ~275  
**Temperature**: Character's default  
**Model**: Character's preferred LLM

```
You are analyzing a complete conversation to extract comprehensive memories and create a summary.

CONVERSATION ({token_count} tokens):
---
{formatted_conversation_messages}
---

MEMORY TYPES:
FACT: Factual information (name, preferences, simple statements)
PROJECT: Goals, plans, ongoing work, future intentions
EXPERIENCE: Shared activities, events, interactions
STORY: Narratives, anecdotes, personal stories
RELATIONSHIP: Emotional bonds, dynamics, connection evolution

EXTRACTION GUIDELINES:
1. Extract ALL significant information across enabled memory types
2. Look for patterns, themes, and relationships
3. Identify key moments and emotional turning points
4. Note participants and their roles
5. Be thorough - this is a complete conversation analysis

OUTPUT FORMAT (JSON):
{
  "memories": [
    {
      "content": "Clear, specific memory statement",
      "type": "fact|project|experience|story|relationship",
      "confidence": 0.0-1.0,
      "reasoning": "Why this is significant",
      "emotional_weight": 0.0-1.0 (optional),
      "participants": ["person1", "person2"] (optional),
      "key_moments": ["moment1", "moment2"] (optional)
    }
  ],
  "summary": "2-3 sentence conversation summary",
  "themes": ["theme1", "theme2", "theme3"],
  "tone": "overall emotional tone",
  "emotional_arc": ["start: emotion", "middle: emotion", "end: emotion"],
  "participants": ["all people mentioned"],
  "key_topics": ["topic1", "topic2", "topic3"]
}

Analyze the conversation and respond with the JSON object:
```

### Type Instructions

The prompt includes dynamic memory type instructions based on the character's memory profile (immersion level):

```python
def _build_type_instructions(profile: Dict[str, bool]) -> str:
    """Build memory type instructions based on profile."""
    type_defs = {
        "fact": "FACT: Factual information (name, preferences, simple statements)",
        "project": "PROJECT: Goals, plans, ongoing work, future intentions",
        "experience": "EXPERIENCE: Shared activities, events, interactions",
        "story": "STORY: Narratives, anecdotes, personal stories",
        "relationship": "RELATIONSHIP: Emotional bonds, dynamics, connection evolution"
    }
    
    enabled_types = []
    for type_name, enabled in profile.items():
        type_key = type_name.replace("extract_", "").rstrip("s")
        if enabled and type_key in type_defs:
            enabled_types.append(type_defs[type_key])
    
    return "MEMORY TYPES:\n" + "\n".join(enabled_types)
```

**Example** (Minimal Immersion):
```
MEMORY TYPES:
FACT: Factual information (name, preferences, simple statements)
PROJECT: Goals, plans, ongoing work, future intentions
```

**Example** (Unbounded Immersion):
```
MEMORY TYPES:
FACT: Factual information (name, preferences, simple statements)
PROJECT: Goals, plans, ongoing work, future intentions
EXPERIENCE: Shared activities, events, interactions
STORY: Narratives, anecdotes, personal stories
RELATIONSHIP: Emotional bonds, dynamics, connection evolution
```

### Response Parsing

**Method**: `_parse_analysis_response()` - Line ~355

```python
# Parse memories
for mem_data in data.get("memories", []):
    memory = AnalyzedMemory(
        content=mem_data["content"],
        memory_type=MemoryType(mem_data["type"]),
        confidence=float(mem_data["confidence"]),
        reasoning=mem_data.get("reasoning", ""),
        emotional_weight=float(mem_data["emotional_weight"]) if mem_data.get("emotional_weight") else None,
        participants=mem_data.get("participants"),
        key_moments=mem_data.get("key_moments")
    )
    memories.append(memory)

# Build analysis
analysis = ConversationAnalysis(
    memories=memories,
    summary=data.get("summary", ""),
    themes=data.get("themes", []),
    tone=data.get("tone", ""),
    emotional_arc=data.get("emotional_arc", ""),
    participants=data.get("participants", []),
    key_topics=data.get("key_topics", [])
)
```

### Deduplication

Before saving extracted memories, the system checks for duplicates:

```python
def _is_duplicate_memory(character_id: str, content: str) -> bool:
    """Check if memory already exists (simple content match)."""
    existing = memory_repo.list_by_character(character_id)
    content_lower = content.lower().strip()
    
    # Check first 100 memories (performance optimization)
    for mem in existing[:100]:
        if mem.content.lower().strip() == content_lower:
            return True
    
    return False
```

This prevents re-saving the same memories when running multiple analyses on the same conversation.

### UI Integration

**Frontend**: `web/js/app.js` - Lines 760-880

- **Analyze Now Button**: Manual trigger with loading state
- **Analysis Results Modal**: Display themes, tone, emotional arc, memory counts
- **Analysis History**: View all past analyses with timestamps

**API Endpoints**:
- `POST /conversations/{id}/analyze` - Manual analysis with force parameter
- `GET /conversations/{id}/analyses` - Analysis history with optional memories

---

## Future Enhancements

### Potential Improvements

1. **Batch Processing**: Analyze multiple messages together for context
2. **Whole-Conversation Extraction**: Thematic/relational analysis after conversation ends
3. **Confidence Calibration**: Learn optimal confidence thresholds over time
4. **Multi-Turn Context**: Consider earlier messages when extracting
5. **Entity Linking**: Connect related memories (person + location + experience)
6. **Temporal Tracking**: Track how facts change over time
7. **Correction Mechanism**: User can mark incorrect extractions

### Research Directions

- **Few-Shot Learning**: Include conversation-specific examples
- **Active Learning**: Prompt user for confirmation on low-confidence facts
- **Relationship Graphs**: Build knowledge graph of user's life
- **Semantic Clustering**: Group related memories automatically

---

## Related Documentation

- **Memory Retrieval**: `chorus_engine_memory_retrieval_algorithm_v_1.md`
- **Character Continuity**: `Documentation/Development/PHASE_4_COMPLETE.md`
- **Privacy Mode**: `Documentation/Development/PRIVACY_FIX.md`
- **Background Processing**: `chorus_engine/services/background_extraction.py`
