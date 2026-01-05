# Background Memory Extraction System Design

**Phase**: Phase 4.1 (Implicit Memory Extraction), Phase 7.5 (Optimization)  
**Created**: November 25, 2025  
**Status**: Complete with Phase 7.5 Optimization (Uses Character Model)

---

## Overview

The Background Memory Extraction System provides non-blocking, intelligent extraction of implicit memories from conversations. Rather than requiring users to explicitly tell the character what to remember, the system uses LLM-powered analysis to automatically extract facts, preferences, projects, experiences, and relationships from natural conversation flow—all happening in the background without interrupting the user experience.

This design document captures the philosophy, architecture, and design decisions behind Chorus Engine's automatic memory extraction approach.

---

## Core Philosophy

### The Extract-Continuously Principle

**Central Insight**: Memory extraction should be invisible, automatic, and never block conversation flow.

**The Problem with Manual Memory**:
- ❌ Users forget to add memories
- ❌ Typing memories manually is tedious
- ❌ "Remember X" commands feel unnatural
- ❌ Memory becomes sparse and incomplete

**The Automatic Solution**:
```
User: "I'm working on building a game with my friend Alex"
                          ↓
System: [Extracts in background]:
  1. "User is developing a game"
  2. "User has a friend named Alex"
                          ↓
Character: "That sounds exciting! What kind of game are you building?"
                          ↓
Memories available for future retrieval (no user action needed)
```

**Why Continuous Extraction Works**:
- No user action required
- Natural conversation flow uninterrupted
- Comprehensive memory coverage
- Background processing = zero latency impact

---

### The Background-Only Principle

**Central Insight**: Memory extraction must never block conversation response time.

**Architecture**:
```
User sends message
       ↓
Generate response IMMEDIATELY (no extraction yet)
       ↓
Response sent to user (fast)
       ↓
Queue extraction task (async)
       ↓
Background worker processes task (separate thread)
       ↓
Extraction happens after response (invisible to user)
       ↓
Memories saved and available for next message
```

**Why Background Processing Works**:
- Conversation feels instant
- Extraction runs in parallel
- Failures don't affect user experience
- Queue absorbs traffic spikes

---

### The Character-Model-Reuse Principle (Phase 7.5)

**Central Insight**: Character's loaded LLM model can extract memories without VRAM overhead.

**The Problem (Phase 4.1)**:
- Character model: Qwen2.5:14B (in VRAM for conversation)
- Extraction used separate call → could trigger model reload
- Model reload: 10-30 seconds VRAM thrashing

**The Solution (Phase 7.5)**:
```python
# Use character's already-loaded model for extraction
await llm_client.generate(
    prompt=extraction_prompt,
    system_prompt=extraction_system_prompt,
    temperature=0.1,  # Low temperature for consistent extraction
    model=character_model  # Same model already loaded
)
```

**Why Character Model Reuse Works**:
- No model swapping (VRAM stays stable)
- No reload latency (model already in memory)
- Character-consistent extraction (same model = same understanding)
- Minimal overhead (just one extra LLM call per message)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   User Sends Message                         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Generate Response      │
        │  (Priority: Fast!)      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Send Response to User  │
        │  (User sees response    │
        │   immediately)          │
        └────────────┬────────────┘
                     │
                     │ (After response sent)
                     │
        ┌────────────▼────────────┐
        │  Queue Extraction Task  │
        │  - conversation_id      │
        │  - character_id         │
        │  - messages             │
        │  - character_model      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Background Worker      │
        │  (AsyncIO Task)         │
        │  Processes queue FIFO   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Build Extraction Prompt│
        │  Filter to user messages│
        │  Add memory profile     │
        │  Format for LLM         │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Call LLM               │
        │  (Character's model)    │
        │  Temperature: 0.1       │
        │  Returns JSON array     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Parse JSON Response    │
        │  Extract memories with  │
        │  confidence scores      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Deduplication Check    │
        │  Vector similarity      │
        │  (ChromaDB query)       │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Save to Database       │
        │  Status: auto_approved  │
        │  or pending             │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Add to Vector Store    │
        │  (If auto-approved)     │
        │  Generate embedding     │
        └─────────────────────────┘
```

### Background Worker Lifecycle

```python
class BackgroundMemoryExtractor:
    async def start(self):
        """Start worker task."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def _worker_loop(self):
        """Main worker loop."""
        while self._running:
            try:
                task = await self._task_queue.get()
                await self._process_task(task)
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                # Continue processing (don't crash worker)
    
    async def stop(self):
        """Stop worker gracefully."""
        self._running = False
        if self._worker_task:
            await self._worker_task
```

---

## Key Components

### 1. BackgroundMemoryExtractor Service

**Purpose**: Async worker that processes memory extraction tasks.

**Key Methods**:
```python
class BackgroundMemoryExtractor:
    def __init__(
        self,
        llm_client: LLMClient,
        extraction_service: MemoryExtractionService,
        temperature: float = 0.1,
        llm_usage_lock: Optional[asyncio.Lock] = None
    ):
        """Initialize with character's LLM client."""
    
    async def start(self):
        """Start background worker task."""
    
    async def stop(self):
        """Stop worker gracefully."""
    
    async def queue_extraction(
        self,
        conversation_id: str,
        character_id: str,
        messages: List[Message],
        model: str,  # Character's model name
        character_name: str,
        character: CharacterConfig  # For memory profile
    ):
        """Queue extraction task (called after response sent)."""
    
    async def _worker_loop(self):
        """Main worker loop (processes queue)."""
    
    async def _process_task(self, task: ExtractionTask):
        """Process single extraction task."""
```

**Phase 7.5 Key Change**: Uses `model` parameter to call LLM with character's already-loaded model.

---

### 2. MemoryExtractionService

**Purpose**: LLM-powered memory extraction logic.

**Key Methods**:
```python
class MemoryExtractionService:
    async def extract_from_messages(
        self,
        messages: List[Message],
        character_id: str,
        conversation_id: str,
        model: Optional[str] = None,
        character_name: Optional[str] = None
    ) -> List[ExtractedMemory]:
        """Extract memories from messages using LLM."""
    
    async def check_for_duplicates(
        self,
        memory_content: str,
        character_id: str
    ) -> Optional[Memory]:
        """Check if memory already exists (vector similarity)."""
    
    async def save_extracted_memory(
        self,
        extracted: ExtractedMemory,
        character_id: str,
        conversation_id: str
    ) -> Optional[Memory]:
        """Save memory with confidence-based auto-approval."""
```

**Extraction Logic**:
```python
# Filter to user messages only (no assistant messages)
user_messages = [m for m in messages if m.role == "user"]

# Build extraction prompt
prompt = self._build_extraction_prompt(user_messages, character_name)

# Call LLM with low temperature for consistency
response = await self.llm_client.generate(
    prompt=prompt,
    system_prompt=extraction_system_prompt,
    temperature=0.1,  # Low = consistent extraction
    model=model  # Character's model
)

# Parse JSON array of extracted memories
memories = self._parse_extraction_response(response.content)
```

---

### 3. Extraction Prompt Design

**System Prompt**:
```
You are a memory extraction assistant. Analyze conversations and extract
memorable facts about the user.

EXTRACT:
- User name, background, demographics (if stated)
- Preferences and dislikes
- Current projects and goals
- Past experiences and stories
- Relationships with people/places
- Skills and expertise

DO NOT EXTRACT:
- Speculation or assumptions
- Information about the assistant/character
- Conversational filler
- Questions from the user
- Physical descriptions (unless explicit)
```

**User Prompt** (formatted messages):
```
Analyze this conversation and extract facts worth remembering.

For each fact, provide a JSON object with:
- "content": Clear statement (e.g., "User name is John")
- "type": One of [fact, project, experience, story, relationship]
- "confidence": Float 0.0-1.0 (0.95 explicit, 0.8 clear, 0.7 inference)
- "reasoning": One sentence explaining extraction

Conversation:
User: I'm working on building a game with my friend Alex
Assistant: That sounds exciting! What kind of game?
User: It's a 2D platformer inspired by Celeste

Return JSON array: [{...}, {...}]
If no facts, return: []
```

**LLM Response**:
```json
[
  {
    "content": "User is developing a game",
    "type": "project",
    "confidence": 0.9,
    "reasoning": "User explicitly stated they are working on building a game"
  },
  {
    "content": "User has a friend named Alex",
    "type": "relationship",
    "confidence": 0.95,
    "reasoning": "User mentioned collaborating with friend Alex"
  },
  {
    "content": "User's game is a 2D platformer",
    "type": "project",
    "confidence": 0.95,
    "reasoning": "User explicitly described game genre"
  },
  {
    "content": "User's game is inspired by Celeste",
    "type": "project",
    "confidence": 0.95,
    "reasoning": "User stated Celeste as inspiration"
  }
]
```

---

### 4. Confidence-Based Auto-Approval

**Confidence Thresholds**:
```python
# Confidence ≥ 0.9: Auto-approved (saved to DB + vector store)
# 0.7 ≤ Confidence < 0.9: Pending (saved to DB, user reviews)
# Confidence < 0.7: Discarded (not saved)
```

**Why This Works**:
- High-confidence facts (0.9+) are obvious and safe to auto-approve
- Medium-confidence facts (0.7-0.89) need user review
- Low-confidence facts (<0.7) are too uncertain to save

**Example**:
```
"User name is John" → confidence: 0.95 → auto_approved
"User might like hiking" → confidence: 0.75 → pending
"User probably lives in NYC" → confidence: 0.6 → discarded
```

---

### 5. Semantic Deduplication

**Purpose**: Prevent saving the same fact multiple times.

**Process**:
```python
async def check_for_duplicates(self, memory_content: str, character_id: str):
    """Check vector store for similar memories."""
    # Query ChromaDB for similar memories
    results = await self.vector_store.query(
        character_id=character_id,
        query_text=memory_content,
        limit=5
    )
    
    # Check similarity threshold
    for result in results:
        if result.similarity >= 0.85:  # 85% similar = duplicate
            return result.memory
    
    return None  # No duplicate found
```

**Duplicate Handling**:
```python
if duplicate_found:
    # Option 1: Reinforce (increase confidence)
    duplicate.confidence = max(duplicate.confidence, extracted.confidence)
    duplicate.updated_at = datetime.now()
    
    # Option 2: Update content if new is more specific
    if len(extracted.content) > len(duplicate.content):
        duplicate.content = extracted.content
    
    return duplicate
```

**Why Semantic Deduplication Works**:
- Catches rephrased duplicates ("User likes coffee" vs. "User enjoys coffee")
- Prevents memory bloat
- Updates memories with new information
- Uses existing vector store infrastructure

---

## Extraction Flow

### Complete Message Flow

```
1. User: "I'm working on building a game"
   ↓
2. LLM generates response: "That sounds exciting! What kind?"
   ↓
3. Response sent to user (FAST - no extraction yet)
   ↓
4. Queue extraction task:
   - conversation_id
   - character_id
   - messages: [last 5 messages]
   - model: "qwen2.5:14b-instruct" (character's model)
   - character_name: "Nova"
   ↓
5. Background worker picks up task
   ↓
6. Filter to user messages: ["I'm working on building a game"]
   ↓
7. Build extraction prompt with memory profile
   ↓
8. Acquire LLM usage lock (prevent model unload during extraction)
   ↓
9. Call LLM with character's model:
   - temperature: 0.1 (low for consistency)
   - max_tokens: 1000
   ↓
10. Parse JSON response:
    [{"content": "User is developing a game", "type": "project", "confidence": 0.9, ...}]
   ↓
11. For each extracted memory:
    a. Check for duplicates (vector similarity)
    b. If duplicate: update or reinforce
    c. If new and confidence ≥ 0.7: save to database
    d. If confidence ≥ 0.9: add to vector store (auto-approved)
    e. If confidence 0.7-0.89: save to database (pending review)
   ↓
12. Write debug log (data/debug_logs/{conversation_id}_{timestamp}.json)
   ↓
13. Release LLM usage lock
   ↓
14. Memory available for next message retrieval
```

**Typical Latency**:
- Steps 1-3: <500ms (conversation response - fast path)
- Steps 4-14: 2-5 seconds (background, invisible to user)

---

## Design Decisions & Rationale

### Decision: Extract After Response, Not Before

**Alternatives Considered**:
1. **Extract before generating response**
   - ❌ Adds 2-5 seconds latency to every message
   - ❌ Terrible user experience (slow conversation)

2. **Extract inline during response generation**
   - ❌ Complex (two LLM calls in parallel)
   - ❌ VRAM contention
   - ❌ Still adds latency

3. **Extract after response sent** (chosen) ✅
   - ✅ Response immediate (fast)
   - ✅ Extraction invisible (background)
   - ✅ Simple architecture (async queue)

**Why Background-After Works**:
- User sees response instantly
- Extraction happens while user is reading/typing
- Failures don't affect conversation
- Memories available for next message

---

### Decision: User Messages Only, Not Assistant Messages

**Alternatives Considered**:
1. **Extract from all messages** (user + assistant)
   - ❌ Assistant can hallucinate facts
   - ❌ Assistant speculates ("Maybe you should...")
   - ❌ Pollutes memory with incorrect facts

2. **Extract from user messages only** (chosen) ✅
   - ✅ User statements are ground truth
   - ✅ No hallucination risk
   - ✅ Cleaner, more accurate memories
   - ✅ Less LLM token usage

**Example Problem**:
```
User: "What should I name my cat?"
Assistant: "How about Whiskers? Many people like that name."
```
If extracting from assistant: ❌ "User has a cat named Whiskers" (WRONG)
If extracting from user only: ✅ No extraction (correct - no fact stated)

---

### Decision: Confidence-Based Auto-Approval Instead of Manual Review

**Alternatives Considered**:
1. **Manual review for all memories**
   - ❌ User burden (tons of confirmations)
   - ❌ Interrupts conversation flow
   - ❌ Most memories are obvious

2. **Auto-approve everything**
   - ❌ Low-quality memories saved
   - ❌ Speculation and guesses saved

3. **Confidence-based auto-approval** (chosen) ✅
   - ✅ High-confidence (0.9+): auto-approve
   - ✅ Medium-confidence (0.7-0.89): pending review
   - ✅ Low-confidence (<0.7): discard
   - ✅ Balances automation with quality

**Statistics** (observed):
- ~70% of extractions are high-confidence (auto-approved)
- ~20% are medium-confidence (pending)
- ~10% are low-confidence (discarded)

**Result**: 70% of memories saved automatically, 20% need review (manageable), 10% correctly discarded.

---

### Decision: Use Character's Model Instead of Separate Extraction Model

**Alternatives Considered (Phase 4.1)**:
1. **Separate extraction model** (e.g., always use Qwen2.5:14B)
   - ❌ Model reload: 10-30 seconds
   - ❌ VRAM thrashing (character model → extraction model → character model)
   - ❌ Inconsistent (extraction model may understand differently)

2. **Use character's model** (Phase 7.5) ✅
   - ✅ No model reload (already in VRAM)
   - ✅ Consistent understanding (same model = same interpretation)
   - ✅ Fast (just one extra LLM call)
   - ✅ VRAM stable (no swapping)

**Phase 7.5 Key Change**:
```python
# OLD (Phase 4.1): Separate extraction call (could trigger reload)
response = await extraction_llm_client.generate(...)

# NEW (Phase 7.5): Use character's model explicitly
response = await llm_client.generate(
    ...,
    model=character_model  # Same model already loaded
)
```

---

### Decision: Semantic Deduplication Instead of Exact Match

**Alternatives Considered**:
1. **Exact string match**
   - ❌ Misses rephrases ("User likes coffee" vs. "User enjoys coffee")
   - ❌ Misses synonyms ("User name is John" vs. "User's name is John")

2. **Simple keyword match**
   - ❌ Too many false positives
   - ❌ Doesn't capture semantic meaning

3. **Vector similarity** (chosen) ✅
   - ✅ Catches semantic duplicates
   - ✅ Threshold tunable (0.85 = 85% similar)
   - ✅ Reuses existing vector store infrastructure
   - ✅ Proven reliable in Phase 3

**Threshold Selection**:
- 0.90+: Too strict (misses obvious duplicates)
- 0.85: Sweet spot (catches duplicates, few false positives) ✅
- 0.80: Too loose (false positives)

---

## Known Limitations

### 1. LLM Quality Dependent

**Limitation**: Extraction quality depends on LLM's capabilities.

**Why**: Smaller models may miss nuances or misclassify facts.

**Example**:
```
User: "I'm learning Rust"
Weak model: No extraction (doesn't recognize as fact)
Strong model: ✅ "User is learning Rust" (type: skill, confidence: 0.9)
```

**Mitigation**: Use strong models (Qwen2.5:14B, Llama3:8B+)

**Future**: Fine-tune specialized extraction model.

---

### 2. Confidence Scores Not Calibrated

**Limitation**: Confidence scores are LLM's self-assessment, not statistically calibrated.

**Why**: LLMs aren't trained to output calibrated probabilities.

**Example**: Model might output 0.9 confidence but only be correct 70% of the time.

**Mitigation**:
- Conservative thresholds (0.9 = high confidence)
- User can review pending memories (0.7-0.89)

**Future**: Collect statistics and recalibrate thresholds.

---

### 3. No Cross-Message Context Aggregation

**Limitation**: Each extraction analyzes only recent messages (typically last 5).

**Why**: Context window limits (8K tokens).

**Example**:
```
Message 1: "I'm learning Rust"
Message 10: "My Rust project compiles now!"
```
Extraction at Message 10 may not connect to Message 1.

**Workaround**: Deduplication catches related facts.

**Future**: Conversation summarization + extraction (Phase 8 planning).

---

### 4. No Fact Update/Deletion Detection

**Limitation**: System doesn't detect when facts become outdated.

**Why**: No temporal reasoning or change detection.

**Example**:
```
Day 1: "I'm learning Rust" → Saved
Day 30: "I've switched to Go" → Saved separately (both exist)
```
System doesn't mark "learning Rust" as outdated.

**Workaround**: Higher confidence on newer fact takes precedence in retrieval.

**Future**: Temporal weighting, fact revision detection.

---

## Performance Characteristics

**Queue Processing**: O(1), FIFO queue (asyncio.Queue)

**Extraction Latency**:
- Prompt building: ~10ms
- LLM call: 1-3 seconds (model/hardware dependent)
- JSON parsing: ~10ms
- Deduplication check: ~50ms (vector query)
- Database save: ~10ms
- **Total**: 2-5 seconds (background, invisible to user)

**VRAM Usage** (Phase 7.5):
- Character model already loaded: 8GB (14B model)
- Extraction reuses same model: 0GB additional
- **Total**: 8GB (same as without extraction)

**VRAM Usage** (Phase 4.1 - before optimization):
- Potential model reload: 10-30 seconds VRAM thrashing
- Phase 7.5 eliminated this issue

**CPU Usage**: Minimal (asyncio task, single LLM call per message)

**Storage**:
- Memory record: ~500 bytes
- Vector embedding: 1536 bytes (384-dim float32)
- Debug log: 1-5KB per extraction
- **Per message**: ~2KB

---

## Testing & Validation

### Unit Tests

```python
def test_extract_from_simple_message():
    messages = [Message(role="user", content="My name is John")]
    extracted = await extraction_service.extract_from_messages(messages, "nova", "conv_1")
    assert len(extracted) == 1
    assert "name is John" in extracted[0].content
    assert extracted[0].confidence >= 0.9

def test_duplicate_detection():
    # Save initial memory
    memory1 = await extraction_service.save_extracted_memory(
        ExtractedMemory(content="User likes coffee", confidence=0.9, ...),
        "nova", "conv_1"
    )
    
    # Try to save duplicate
    memory2 = await extraction_service.save_extracted_memory(
        ExtractedMemory(content="User enjoys coffee", confidence=0.85, ...),
        "nova", "conv_1"
    )
    
    # Should detect as duplicate
    assert memory2.id == memory1.id  # Same memory object

def test_confidence_thresholds():
    # High confidence: auto-approved
    memory1 = await extraction_service.save_extracted_memory(
        ExtractedMemory(content="User name is John", confidence=0.95, ...),
        "nova", "conv_1"
    )
    assert memory1.status == "auto_approved"
    
    # Medium confidence: pending
    memory2 = await extraction_service.save_extracted_memory(
        ExtractedMemory(content="User might like skiing", confidence=0.75, ...),
        "nova", "conv_1"
    )
    assert memory2.status == "pending"
    
    # Low confidence: discarded
    memory3 = await extraction_service.save_extracted_memory(
        ExtractedMemory(content="User probably lives in NYC", confidence=0.65, ...),
        "nova", "conv_1"
    )
    assert memory3 is None  # Not saved
```

### Integration Tests

```python
@pytest.mark.integration
async def test_end_to_end_extraction():
    # Send message
    response = await client.post("/conversations/conv_1/messages", json={
        "content": "I'm working on building a game"
    })
    assert response.status_code == 200
    
    # Wait for background extraction
    await asyncio.sleep(3)
    
    # Check memories were extracted
    memories = await memory_repo.get_for_character("nova")
    assert any("developing a game" in m.content for m in memories)
```

---

## Migration Guide

### Enabling Background Extraction

**Phase 4.1+**: Enabled by default, no configuration needed.

**Verification**:
```
1. Send message to character
2. Check logs: "[BACKGROUND MEMORY] Queued extraction task..."
3. Wait 3 seconds
4. Check logs: "[BACKGROUND MEMORY] Extracted N memories"
5. View memories in UI
```

### Phase 7.5 Upgrade (Character Model Reuse)

**Automatic**: No code changes needed.

**Behavior Change**:
- Before: Extraction might trigger model reload (10-30s VRAM spike)
- After: Extraction reuses character's model (0s overhead)

**Verification**:
- Check logs: No "Loading model..." during extraction
- VRAM usage stable during extraction

---

## Future Enhancements

### High Priority

**1. Fact Update Detection**
- Detect when facts change ("was learning Rust" → "now learning Go")
- Mark old memories as outdated
- Update instead of creating duplicates

**2. Cross-Message Context Aggregation**
- Summarize long conversations
- Extract from summary + recent messages
- Better long-term context understanding

**3. Confidence Recalibration**
- Track accuracy of confidence scores
- Adjust thresholds based on statistics
- Provide calibrated probabilities

### Medium Priority

**4. Memory Revision Workflow**
- User can edit extracted memories
- Approve/reject pending memories in batch
- Flag incorrect memories for review

**5. Extraction Trigger Customization**
- Extract every N messages (default: 1)
- Extract only for important conversations
- Extract on-demand via API

**6. Multi-Turn Fact Extraction**
- Combine facts from multiple messages
- "User is learning Rust" + "User's Rust project compiles" → "User is learning Rust and has a working project"

### Low Priority

**7. Entity Linking**
- Link extracted facts to entities (people, places, projects)
- Graph-based memory representation
- Relationship inference

**8. Fact Source Tracking**
- Show which messages contributed to each memory
- Trace memory provenance
- Audit trail for corrections

---

## Conclusion

The Background Memory Extraction System represents Chorus Engine's commitment to invisible, automatic memory management. By extracting facts from natural conversation in the background using the character's already-loaded model, the system provides comprehensive memory coverage without sacrificing conversation responsiveness or user experience.

Key achievements:
- **Continuous Extraction**: Every message analyzed automatically
- **Zero Latency Impact**: Extraction happens after response sent
- **Character Model Reuse**: No VRAM overhead (Phase 7.5)
- **Confidence-Based Auto-Approval**: 70% of memories approved automatically
- **Semantic Deduplication**: Prevents duplicate facts via vector similarity
- **Background Worker**: Async queue-based architecture, graceful error handling

The system has proven successful through:
- Nova: Remembers user's creative projects and preferences
- Alex: Remembers technical discussions and code snippets
- Both characters: Comprehensive memory coverage without user effort
- Phase 7.5: Eliminated VRAM overhead via character model reuse

Future enhancements (fact update detection, cross-message aggregation, confidence recalibration) build naturally on this foundation. The background extraction approach provides the perfect balance of automation and quality control.

**Status**: Production-ready, battle-tested across conversations, Phase 7.5 optimization complete, recommended pattern for all implicit memory extraction needs.

---

**Document Version**: 1.1  
**Last Updated**: January 4, 2026  
**Author**: System Design Documentation  
**Phases**: Phase 4.1 (Foundation), Phase 7.5 (Character Model Optimization)
