# Conversation Analysis System

## Overview

The Conversation Analysis System performs comprehensive analysis of complete conversations to extract memories, generate summaries, and identify themes. This is a **bulk analysis** system that processes entire conversations as a unit, unlike the real-time memory extraction that happens during individual messages.

**Location:** `chorus_engine/services/conversation_analysis_service.py`

---

## When Analysis Triggers

### Manual Trigger
- User clicks **"Analyze Now"** button in the UI
- Soft minimums (can be bypassed with `force=true`):
  - ≥5 messages
  - ≥100 tokens
- Runs synchronously and returns detailed results immediately

### Automatic Triggers (Future)
The service is designed to support automatic analysis based on:
- **≥10,000 tokens** - Comprehensive conversation threshold
- **≥2,500 tokens + 24h inactive** - Shorter but complete conversation

*Note: Automatic triggers are not currently implemented in app.py but the service is designed for them.*

---

## Analysis Pipeline

### 1. Data Collection
```python
conversation = get_by_id(conversation_id)
messages = get_all_messages(conversation_id)  # All threads
token_count = count_tokens(messages)
```

**Quality Checks:**
- Skips conversations with < 500 tokens
- Counts tokens across all messages in all threads
- Validates conversation exists and has messages

### 2. Prompt Assembly

The analysis prompt contains:

#### A. Formatted Conversation
```
USER: [message content]

ASSISTANT: [message content]

USER: [message content]
...
```
- All messages across all threads
- Chronologically ordered
- Role labels in uppercase

#### B. Memory Type Definitions
Based on character's memory profile (from `MemoryProfileService`):

```
MEMORY TYPES:
FACT: Factual information (name, preferences, simple statements)
PROJECT: Goals, plans, ongoing work, future intentions  
EXPERIENCE: Shared activities, events, interactions
STORY: Narratives, anecdotes, personal stories
RELATIONSHIP: Emotional bonds, dynamics, connection evolution
```

Only includes types enabled for the character.

#### C. Extraction Guidelines
```
EXTRACTION GUIDELINES:
1. Extract ALL significant information across enabled memory types
2. Look for patterns, themes, and relationships
3. Identify key moments and emotional turning points
4. Note participants and their roles
5. Be thorough - this is a complete conversation analysis
```

#### D. Output Schema
```json
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
```

---

## Actual Prompt (As Used)

The service builds a single prompt string and sends it to the LLM. This is the exact template used in `conversation_analysis_service.py`:

```
You are analyzing a complete conversation to extract comprehensive memories and create a summary.

CONVERSATION ({token_count} tokens):
---
{formatted_conversation_messages}
---

{type_instructions}

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

**Notes**:
- `{type_instructions}` is dynamically generated based on the character’s memory profile.
- `{formatted_conversation_messages}` is the full conversation with `USER:` / `ASSISTANT:` role labels in chronological order.

### 3. LLM Call

```python
response = await llm_client.generate(
    prompt=analysis_prompt,
    model=character.preferred_llm.model,  # Uses character's preferred model
    temperature=0.1,  # Low temperature for consistency
    max_tokens=4000   # Large enough for comprehensive analysis
)
```

**Configuration:**
- **Temperature:** `0.1` (very low for consistent, structured output)
- **Max Tokens:** `4000` (allows detailed analysis of long conversations)
- **Model:** Uses character's preferred LLM model if specified

### 4. Response Parsing

```python
# Extract JSON from response
json_start = response.find("{")
json_end = response.rfind("}") + 1
json_str = response[json_start:json_end]
data = json.loads(json_str)
```

**Parsing Steps:**
1. Find JSON object boundaries in response
2. Parse JSON structure
3. Convert memory types to `MemoryType` enum
4. Validate confidence scores (0.0-1.0)
5. Build `ConversationAnalysis` object
6. Skip invalid memories with warning

**Fallback:** If JSON parsing fails, returns `None` and logs error.

### 5. Memory Storage

For each extracted memory:

#### A. Duplicate Check
```python
if is_duplicate_memory(character_id, content):
    skip_memory()
```
- Compares against first 100 existing memories (performance optimization)
- Case-insensitive content match
- Skips saving if duplicate found

#### B. Embedding Generation
```python
embedding = embedding_service.embed(memory.content)
vector_id = str(uuid.uuid4())
```

#### C. Vector Store
```python
vector_store.add_memories(
    character_id=character_id,
    memory_ids=[vector_id],
    contents=[content],
    embeddings=[embedding],
    metadatas=[{
        "character_id": character_id,
        "conversation_id": conversation_id,
        "type": memory_type,
        "confidence": confidence
    }]
)
```

#### D. Database Record
```python
# Determine status based on confidence
if confidence >= 0.9:
    status = "auto_approved"
elif confidence >= 0.7:
    status = "approved"  # Approved for confident extractions
else:
    status = "pending"  # Below 0.7 requires review

memory = create_memory(
    character_id=character_id,
    content=content,
    memory_type=memory_type,
    vector_id=vector_id,
    conversation_id=conversation_id,
    status=status,
    confidence=confidence,
    emotional_weight=emotional_weight,
    participants=participants,
    key_moments=key_moments
)
```

**Memory Status Rules:**
- **auto_approved:** confidence ≥ 0.9 (high confidence)
- **approved:** confidence ≥ 0.7 (good confidence)
- **pending:** confidence < 0.7 (needs review)

### 6. Summary Storage

```python
summary = ConversationSummary(
    conversation_id=conversation_id,
    summary=analysis.summary,
    message_range_start=0,
    message_range_end=message_count - 1,
    message_count=message_count,
    key_topics=analysis.key_topics,
    participants=analysis.participants,
    emotional_arc=analysis.emotional_arc,  # JSON string
    tone=analysis.tone,
    manual="true" if manual else "false"
)
```

**Summary Fields:**
- `summary`: 2-3 sentence overview
- `message_range_start/end`: Range of analyzed messages
- `message_count`: Total messages analyzed
- `key_topics`: List of main topics discussed
- `participants`: All people mentioned in conversation
- `emotional_arc`: JSON array of emotional progression
- `tone`: Overall emotional tone
- `manual`: Whether manually triggered vs automatic

### 7. Conversation Update

```python
conversation.last_analyzed_at = datetime.utcnow()
db.commit()
```

### 8. Debug Logging

Creates detailed debug log at:
```
data/debug_logs/conversations/{conversation_id}/analysis_{timestamp}.jsonl
```

**Log Entries:**
```jsonl
{"type": "metadata", "conversation_id": "...", "timestamp": "...", "token_count": 1234}
{"type": "prompt", "content": "...full prompt..."}
{"type": "response", "content": "...LLM response..."}
{"type": "analysis", "memory_count": 5, "themes": [...], "tone": "...", "participants": [...]}
```

---

## Key Design Principles

### 1. Completeness Over Speed
- Analyzes **entire conversation** as one unit
- Low temperature (0.1) for consistency
- Large token budget (4000) for detailed output
- Thorough extraction across all memory types

### 2. Quality Control
- Token count minimums prevent analysis of trivial conversations
- Confidence scoring determines memory approval status
- Duplicate detection prevents memory redundancy
- Invalid memories skipped with warnings

### 3. Character-Aware
- Uses character's preferred LLM model
- Respects character's memory profile (enabled types)
- Links memories to specific character
- Character-scoped vector store

### 4. Comprehensive Data
- Extracts multiple memory types simultaneously
- Identifies themes, tone, emotional arc
- Tracks participants and key topics
- Preserves context (conversation_id linkage)

### 5. Auditability
- Debug logs capture full analysis pipeline
- Timestamp and token count recorded
- Manual vs automatic analysis tracked
- Reasoning field explains why memories extracted

---

## Differences from Real-Time Memory Extraction

| Aspect | Real-Time | Conversation Analysis |
|--------|-----------|----------------------|
| **Scope** | Single message/exchange | Entire conversation |
| **Timing** | During response generation | After conversation completion |
| **Context** | Recent messages only | All messages, all threads |
| **Volume** | 1-2 memories per exchange | Comprehensive bulk extraction |
| **Purpose** | Immediate context capture | Holistic understanding |
| **Temperature** | 0.7 (normal) | 0.1 (very consistent) |
| **Output** | Quick extraction | Detailed analysis with summary |

---

## API Endpoints

### Manual Analysis
```
POST /conversations/{conversation_id}/analyze?force=false
```

**Request Query Params:**
- `force` (bool): Bypass soft minimums if true

**Response:**
```json
{
  "status": "success",
  "analysis_type": "manual",
  "memories_extracted": 12,
  "memory_counts": {
    "fact": 5,
    "experience": 4,
    "relationship": 3
  },
  "memories": [
    {
      "type": "fact",
      "content": "User loves hiking",
      "confidence": 0.95,
      "reasoning": "Explicitly stated preference"
    }
  ],
  "summary": {
    "text": "Conversation about outdoor activities...",
    "themes": ["nature", "exercise", "wellness"],
    "tone": "enthusiastic",
    "key_topics": ["hiking", "camping", "photography"]
  }
}
```

**Warning Response (under minimums):**
```json
{
  "status": "warning",
  "message": "Conversation might be too short (4 messages, ~85 tokens)",
  "can_force": true,
  "message_count": 4,
  "estimated_tokens": 85
}
```

---

## Configuration

### Service Initialization
```python
ConversationAnalysisService(
    db=db_session,
    llm_client=llm_client,
    vector_store=vector_store,
    embedding_service=embedding_service,
    temperature=0.1  # Configurable but defaults to 0.1
)
```

### Character Memory Profile
Defined in `MemoryProfileService.get_extraction_profile()`:
```python
{
    "extract_facts": True,
    "extract_projects": True, 
    "extract_experiences": True,
    "extract_stories": True,
    "extract_relationships": True
}
```

Disabled types are not included in analysis instructions.

---

## Future Enhancements

### Potential Automatic Triggers
1. **Token-based:**
   - Trigger at 10,000 tokens (comprehensive)
   - Trigger at 2,500 tokens if 24h inactive

2. **Time-based:**
   - Daily analysis of active conversations
   - Weekly analysis of dormant conversations

3. **Event-based:**
   - On conversation closure
   - After significant exchanges (user-defined)

### Incremental Analysis
- Analyze conversation segments (e.g., last 5,000 tokens)
- Update existing summaries rather than replace
- Track analyzed ranges to avoid re-processing

### Advanced Features
- Sentiment tracking over time
- Relationship evolution visualization
- Topic clustering across conversations
- Memory importance scoring refinement

---

## Example Workflow

```
User has conversation with 50 messages (8,500 tokens)
    ↓
User clicks "Analyze Now" button
    ↓
System validates: conversation exists, has sufficient content
    ↓
Prompt assembled: messages + memory types + guidelines + schema
    ↓
LLM analyzes: extracts 15 memories, identifies 4 themes, creates summary
    ↓
Response parsed: validates JSON, converts types, builds analysis object
    ↓
Memories processed:
  - 12 memories saved (3 were duplicates)
  - 8 auto-approved (conf ≥ 0.9)
  - 3 approved (conf ≥ 0.7)
  - 1 pending review (conf < 0.7)
    ↓
Summary saved: overview, themes, emotional arc, participants
    ↓
Conversation marked: last_analyzed_at = now
    ↓
Debug log written: complete analysis pipeline captured
    ↓
API returns: success + memory counts + detailed breakdown
```

---

## Debugging

### Debug Logs Location
```
data/debug_logs/conversations/{conversation_id}/analysis_{timestamp}.jsonl
```

### Log Contents
1. **Metadata:** Conversation ID, timestamp, token count
2. **Prompt:** Complete prompt sent to LLM
3. **Response:** Raw LLM response
4. **Analysis:** Parsed results summary

### Common Issues

**No memories extracted:**
- Check if memory types are enabled in character profile
- Verify conversation has meaningful content
- Review LLM response in debug log for errors

**All memories pending:**
- LLM assigning low confidence scores
- May need to adjust confidence thresholds
- Check reasoning field for LLM's uncertainty

**Parsing failures:**
- LLM returned non-JSON response
- Check temperature (should be 0.1)
- Verify model supports structured output

**Duplicate skipping:**
- Similar memories already exist
- Duplicate check is case-insensitive
- Only checks first 100 memories (performance)

---

## Best Practices

1. **Wait for sufficient content** before manual analysis (>500 tokens)
2. **Review pending memories** periodically to improve confidence thresholds
3. **Check debug logs** when extraction yields unexpected results
4. **Disable unused memory types** in character profile for focused extraction
5. **Use force=true** sparingly - soft minimums exist for quality

---

## Integration Points

### Services Used
- `LLMClient` - Analysis generation
- `EmbeddingService` - Memory embeddings
- `VectorStore` - Memory storage and retrieval
- `MemoryProfileService` - Character memory configuration
- `TokenCounter` - Token counting and validation

### Repositories Used
- `ConversationRepository` - Conversation data
- `MessageRepository` - Message retrieval
- `MemoryRepository` - Memory CRUD operations
- `ThreadRepository` - Thread management

### Models Used
- `Conversation` - Conversation metadata
- `Message` - Message content and roles
- `Memory` - Extracted memory records
- `ConversationSummary` - Analysis results
- `MemoryType` - Enum for memory classification
