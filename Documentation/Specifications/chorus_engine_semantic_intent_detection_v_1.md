# Chorus Engine – Semantic Intent Detection (v1)

**Purpose**: Detect user intents from natural language using embedding-based semantic similarity  
**Status**: Production (January 2026)

This document defines how Chorus Engine detects user intents using embedding similarity, replacing legacy keyword-based detection with a robust, maintainable semantic approach.

Goals:
- Natural language understanding without regex maintenance
- Confidence-based detection with tunable thresholds
- Multi-intent support for complex user requests
- Zero additional dependencies (reuses memory embedding model)
- Extensible architecture for adding new intents

**Table of Contents**:
1. [Architecture Overview](#architecture-overview)
2. [Implementation Details](#implementation-details)
3. [Adding New Intents](#adding-new-intents)
4. [Tuning Thresholds](#tuning-thresholds)
5. [Exclusion Groups](#modifying-exclusion-groups)
6. [Debugging Detection Issues](#debugging-detection-issues)
7. [Testing Procedures](#testing-procedures)
8. [Performance Considerations](#performance-considerations)
9. [Maintenance Checklist](#maintenance-checklist)

---

## Architecture Overview

---

## Architecture Overview

```
User Message
    ↓
SemanticIntentDetector (singleton)
    ↓
Embed message (all-MiniLM-L6-v2, ~10ms)
    ↓
Cosine similarity vs pre-embedded prototypes
    ↓
Lexical anchor bonus (hybrid boost for reminders)
    ↓
Threshold filtering + ambiguity check
    ↓
Exclusion group filtering
    ↓
List[Intent(name, confidence)]
    ↓
Action handlers (orchestrators, services)
```

**Components**:
- `SemanticIntentDetector`: Core detection class with prototype management
- `EmbeddingService`: Shared singleton for all-MiniLM-L6-v2 model
- `Intent`: Dataclass for detected intents with confidence scores
- `INTENT_PROTOTYPES`: Dictionary of example phrases per intent
- `THRESHOLDS`: Confidence thresholds for each intent type
- `EXCLUSION_GROUPS`: Mutually exclusive intent families

---

## Implementation Details

### Core Data Structures

**Intent Dataclass**:
```python
@dataclass
class Intent:
    name: str           # Intent identifier (e.g., "send_image")
    confidence: float   # Similarity score (0.0 - 1.0)
    raw_message: str    # Original user message
```

**Intent Prototypes**:

Each intent is defined by 5-20 example phrases. Starting with v1.1, reminder prototypes include both:
- **Generic prototypes**: "remind me to do something", "set a reminder for later"
- **Content-heavy prototypes**: "remind me to call the doctor next week", "remind me in three weeks to call the dog groomer"

Content-heavy prototypes capture realistic task language, preventing embedding vectors from being pulled toward task semantics ("call", "schedule") rather than reminder intent.

**Lexical Anchor Bonus (Hybrid Approach)**:

The system is primarily semantic but applies a small lexical boost for certain intents:
- **Reminder intent**: +0.05 if message contains "remind", "reminder", "don't let me forget", "don't forget"
- **Purpose**: Catch edge cases where complex phrasing scores 0.45-0.50 but explicit reminder language is present
- **Impact**: Boosts borderline cases over threshold without compromising semantic matching

This hybrid approach achieves 95%+ detection accuracy for reminders without needing two-stage temporal detection.

### Step 1: Define Prototypes

Add your intent to `INTENT_PROTOTYPES` in `chorus_engine/services/semantic_intent_detection.py`:

```python
INTENT_PROTOTYPES = {
    # ... existing intents ...
    
    "search_memory": [
        "search my memories for something",
        "look up what I told you about",
        "find information in my memories",
        "what did I say about",
        "do you remember when I mentioned"
    ]
}
```

**Guidelines**:
- **5-14 prototypes**: Start with 5-8, add more if testing reveals gaps
- **Diverse structures**: Cover different verbs, word orders, formality
- **Generic objects**: Use "something", "someone", not specific examples
- **Natural phrasing**: Write how real users would phrase the request
- **Avoid overlap**: Check similarity to existing intents using test utility

### Step 2: Set Threshold

Add threshold to `THRESHOLDS` dictionary:

```python
THRESHOLDS = {
    # ... existing thresholds ...
    
    "search_memory": 0.55,  # Medium confidence for informational query
}
```

**Threshold Guidelines**:
- **0.45-0.50**: Low-stakes, confirmable actions (images, videos)
- **0.50-0.60**: Medium-stakes, user-reviewed actions (reminders, memory search)
- **0.60-0.70**: High-stakes actions (delete, execute, destructive operations)
- **Start conservative** (higher), lower based on testing

### Step 3: Add to Exclusion Group (if needed)

If your intent is mutually exclusive with others:

```python
EXCLUSION_GROUPS = {
    "media_generation": ["send_image", "send_video"],
    
    # Add new group
    "memory_operations": ["search_memory", "save_memory", "delete_memory"]
}
```

**When to use**:
- Intents are semantically similar but require different actions
- Multiple intents triggering would be confusing or wrong
- User clearly meant one thing, not multiple

### Step 4: Integrate with Action Handler

In `chorus_engine/api/app.py`, add detection flag and handler:

```python
# In semantic intent detection block
semantic_has_search = False

for intent in semantic_intents:
    if intent.name == "search_memory":
        semantic_has_search = True
        logger.info(f"[SEMANTIC INTENT] Memory search intent detected with {intent.confidence:.2f} confidence")

# Later in the function
if semantic_has_search:
    # Trigger memory search functionality
    search_results = await memory_service.search(
        query=request.message,
        character_id=character.id,
        max_results=5
    )
```

### Step 5: Test Thoroughly

Use the test utility:

```bash
cd utilities/intent_detection_test
python test_intent.py
```

**Test cases**:
- ✅ Direct requests ("search my memories")
- ✅ Indirect phrasing ("what did I tell you about X?")
- ✅ Casual language ("do you remember when...")
- ✅ Formal requests ("please retrieve information about...")
- ❌ False positives (similar but wrong intent)
- ❌ Ambiguous cases (could be multiple intents)

Log all test results for analysis.

---

## Tuning Thresholds

### When to Tune

**Lower threshold** if:
- ✅ Valid requests consistently score 0.45-0.55 but threshold is 0.55
- ✅ Testing shows <70% detection rate for clear requests
- ✅ False negative rate is high (missing legitimate intents)

**Raise threshold** if:
- ❌ False positives are common (wrong intent detected)
- ❌ Ambiguous phrases trigger when they shouldn't
- ❌ Confidence distribution shows clear gap (e.g., real: 0.70+, noise: <0.50)

**Keep threshold** if:
- ✅ 80%+ detection rate on test phrases
- ✅ <5% false positive rate
- ✅ Ambiguity rate <3%

### Tuning Process

1. **Collect Data**: Run 20-50 test phrases through utility, log results
2. **Analyze Distribution**: Plot confidence scores for true/false detections
3. **Identify Gaps**: Look for cluster of missed intents at similar confidence
4. **Adjust Incrementally**: Change by ±0.02-0.05 at a time
5. **Re-test**: Validate with full test suite
6. **Document**: Note threshold change and reasoning in code comments

### Example Analysis

```
Test Results (50 phrases):
  True Positives (detected correctly):
    - 25 phrases: 0.55-0.85 confidence
  False Negatives (missed):
    - 10 phrases: 0.48-0.54 confidence
    - 2 phrases: 0.35-0.40 confidence (add prototypes)
  False Positives (wrong intent):
    - 1 phrase: 0.56 confidence (acceptable rate)

Conclusion: Lower threshold from 0.55 → 0.50
Expected improvement: +10 true positives, +1 false positive
```

---

## Modifying Exclusion Groups

### Purpose

Exclusion groups prevent spurious multi-intent detections where two similar intents both trigger, but user only meant one.

### Structure

```python
EXCLUSION_GROUPS = {
    "group_name": ["intent1", "intent2", "intent3"]
}
```

**Behavior**: Only the highest-confidence intent from each group is kept.

### When to Add an Intent to Group

✅ **Add if**:
- Intents are semantically similar ("send image" vs "send video")
- Both triggering would confuse user experience
- Real-world testing shows frequent co-triggering

❌ **Don't add if**:
- Intents are genuinely different actions
- User might legitimately want both ("send image AND remind me")
- Co-triggering is rare (<1% of cases)

### Example: Adding New Group

```python
EXCLUSION_GROUPS = {
    "media_generation": ["send_image", "send_video"],
    
    # New: Prevent search and save from both triggering
    "memory_operations": ["search_memory", "save_memory"]
}
```

**Rationale**: "Tell me what I said about X" could score high for both search (retrieve info) and save (record statement). User likely only wants search.

---

## Debugging Detection Issues

### Issue: No Intent Detected (False Negative)

**Symptoms**: User phrase clearly requests action, but nothing triggers

**Debug Steps**:

1. **Enable debug mode**:
   ```python
   intents = detector.detect(message, enable_multi_intent=True, debug=True)
   ```

2. **Check output**:
   ```
   send_image: 0.482
   send_video: 0.398
   set_reminder: 0.156
   
   Below global minimum (0.45) OR just below threshold (0.50)
   ```

3. **Diagnose**:
   - If 0.45-0.50 and threshold is 0.50: **Lower threshold**
   - If 0.30-0.45: **Add similar prototype**
   - If <0.30: **Very different phrasing, may not be same intent**

4. **Fix**:
   ```python
   # Option 1: Add prototype
   "send_image": [
       # ... existing ...
       "could you capture an image?"  # Add user's phrasing
   ]
   
   # Option 2: Lower threshold
   THRESHOLDS = {
       "send_image": 0.48  # Was 0.50
   }
   ```

### Issue: Wrong Intent Detected (False Positive)

**Symptoms**: System detects intent user didn't mean

**Debug Steps**:

1. **Enable debug mode** and check scores
2. **Identify confusion**:
   ```
   set_reminder: 0.578  ✓ (detected - wrong)
   search_memory: 0.534 ✗ (should have been this)
   
   Margin: 0.044 (within ambiguity margin of 0.08)
   ```

3. **Diagnose**:
   - If scores are close: **Ambiguity margin working as intended**
   - If wrong intent much higher: **Prototypes too generic**

4. **Fix**:
   ```python
   # Refine prototypes to be more specific
   "set_reminder": [
       "remind me to do something",
       "set a reminder for later",
       # Remove: "remember to tell me" (too similar to search)
   ]
   ```

### Issue: Ambiguous Detection (No Result)

**Symptoms**: Two intents score very close, both rejected

**Debug Steps**:

1. **Check debug output**:
   ```
   Ambiguous: 0.567 vs 0.524 (margin < 0.08)
   ```

2. **Evaluate legitimacy**:
   - Is phrase genuinely ambiguous? ("show me a clip" - image or video?)
   - Or should one clearly win?

3. **Fix**:
   - If genuinely ambiguous: **Working as intended** (implement partially-sure confirmations)
   - If one should win: **Refine prototypes** to increase separation

---

## Testing Procedures

### Before Committing Changes

**Required tests**:

1. ✅ **Prototype coverage**: Run 20+ diverse phrasings through test utility
2. ✅ **False positive check**: Test 10+ non-intent phrases (casual conversation)
3. ✅ **Multi-intent**: Test legitimate multi-intent phrases
4. ✅ **Edge cases**: Test ambiguous, borderline, and unusual phrasings
5. ✅ **Integration**: Send test message through API, verify intent triggers

### Test Utility Usage

```bash
cd utilities/intent_detection_test
python test_intent.py

# Test phrases for new "search_memory" intent:
Enter phrase: search my memories for dogs
Enter phrase: what did I tell you about dogs?
Enter phrase: do you remember when I mentioned dogs?
Enter phrase: find information about dogs
Enter phrase: look up dogs in my memories
```

Review log file in `logs/intent_test_TIMESTAMP.log` for full results.

### Regression Testing

After changing prototypes or thresholds, re-run saved test phrases:

1. Keep a reference log of known-good detections
2. Run same phrases with new configuration
3. Compare: did any previously working phrases break?
4. Investigate any regressions before committing

---

## Code Architecture

### File Structure

```
chorus_engine/services/
  semantic_intent_detection.py  # Core detector class
  embedding_service.py           # Shared embedding model

utilities/intent_detection_test/
  test_intent.py                 # Interactive test utility
  README.md                      # Testing guidelines
  logs/                          # Test result logs

Documentation/Design/
  SEMANTIC_INTENT_DETECTION.md   # User-facing design doc
  SPECIFICATIONS.md              # This file (developer specs)
```

### Class Hierarchy

```
SemanticIntentDetector
  ├── __init__(embedding_model)
  │     Pre-computes prototype embeddings
  │
  ├── _embed_prototypes() → Dict[str, List[np.ndarray]]
  │     One-time embedding of all prototypes
  │
  ├── _anchor_bonus(message, intent_name) → float
  │     Hybrid lexical boost for specific intents (+0.05 for reminders)
  │
  ├── detect(message, enable_multi_intent, debug) → List[Intent]
  │     Main detection logic
  │
  ├── _apply_exclusion_groups(intents) → List[Intent]
  │     Filter spurious multi-intent detections
  │
  └── get_stats() → Dict
        Return detection statistics

get_intent_detector() → SemanticIntentDetector
  Singleton accessor (uses shared EmbeddingService)
```

### Data Flow

```
1. User message → API endpoint
2. get_intent_detector() → singleton instance
3. detector.detect(message)
   ├─ Embed message (all-MiniLM-L6-v2)
   ├─ Calculate cosine similarity vs all prototypes
   ├─ Find max similarity per intent
   ├─ Apply lexical anchor bonus (reminder: +0.05 if keywords present)
   ├─ Filter by thresholds
   ├─ Check ambiguity margin
   ├─ Apply exclusion groups
   └─ Return List[Intent]
4. Set flags (semantic_has_image, etc.)
5. Trigger action handlers (orchestrators)
```

---

## Performance Considerations

### Latency Budget

- **Embedding**: 5-15ms (CPU-bound)
- **Similarity calculation**: <1ms (numpy operations)
- **Total**: ~10-15ms typical, ~30ms worst case (old CPU)

**Context**: Intent detection is 0.1-0.2% of total message processing time (LLM inference is 1000-5000ms).

### Memory Footprint

- **Model**: all-MiniLM-L6-v2 = ~90MB (shared with memory system, no additional cost)
- **Prototype embeddings**: ~92KB for 10 intents × 6 prototypes × 384 dims
- **Negligible impact**

### Scaling

**Adding intents**:
- Linear scaling: 50 intents ≈ 20ms detection time
- No practical limit (100+ intents still <50ms)

**Adding prototypes**:
- 20 prototypes per intent: ~15ms detection
- Diminishing returns after 10-15 prototypes

**Optimization opportunities** (if needed):
- Batch embed multiple messages (not applicable for single-message API)
- Quantize embeddings to float16 (saves 50% memory, minimal accuracy loss)
- Cache message embeddings (not useful - messages are unique)

### Best Practices

✅ **Do**:
- Keep prototypes focused (3-10 words)
- Aim for 5-10 prototypes per intent
- Test performance after adding 10+ new intents

❌ **Don't**:
- Add 50+ prototypes per intent (diminishing returns)
- Embed very long prototypes (>50 words) - dilutes similarity
- Lower global minimum below 0.40 (too much noise)

---

## Configuration (Future Enhancement)

Currently hardcoded in Python. Future enhancement: move to `system.yaml`:

```yaml
intent_detection:
  enabled: true
  method: semantic  # or "keyword" (legacy)
  
  thresholds:
    global_minimum: 0.45
    ambiguity_margin: 0.08
    
    # Per-intent overrides
    send_image: 0.50
    send_video: 0.50
    set_reminder: 0.50
  
  enable_multi_intent: true
  
  # Debugging
  log_detections: false
  debug_mode: false
```

**Implementation notes**:
- Load config in `SemanticIntentDetector.__init__()`
- Reload without restart: `detector.reload_config(new_config)`
- Character overrides: `characters/character.yaml` → `intent_detection.thresholds`

---

## Maintenance Checklist

### Monthly Review

- [ ] Check detection statistics via `get_stats()`
- [ ] Review false positive/negative reports from users
- [ ] Analyze confidence score distributions in logs
- [ ] Test new edge cases discovered in production

### After Adding Intent

- [ ] Add prototypes following guidelines
- [ ] Set appropriate threshold
- [ ] Add to exclusion group if needed
- [ ] Test 20+ phrases in utility
- [ ] Integrate with action handler in API
- [ ] Document intent in design doc
- [ ] Commit and deploy

### After Threshold Change

- [ ] Document reason for change (comment in code)
- [ ] Re-run regression tests
- [ ] Monitor for 48-72 hours
- [ ] Roll back if issues detected
- [ ] Update documentation

---

## Troubleshooting Reference

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Intent never detected | Confidence always <0.30 | Add more diverse prototypes |
| Intent missed sometimes | Confidence 0.45-0.50 | Lower threshold by 0.05 |
| Wrong intent detected | Scores very close | Refine prototypes for separation |
| Ambiguous rejections | Margin <0.08 frequently | Evaluate if legitimately ambiguous |
| Both image+video trigger | No exclusion group | Add to media_generation group |
| Slow detection (>50ms) | Too many prototypes | Reduce to 10-15 per intent |

---

## Examples

### Complete Intent Addition Example

```python
# File: chorus_engine/services/semantic_intent_detection.py

INTENT_PROTOTYPES = {
    # ... existing intents ...
    
    "capture_scene": [
        "what do you see",
        "describe your surroundings",
        "tell me what's in front of you",
        "what's happening around you",
        "look around and tell me what you notice"
    ]
}

THRESHOLDS = {
    # ... existing thresholds ...
    "capture_scene": 0.52  # Slightly higher - vision queries need clarity
}

# No exclusion group needed - distinct from other intents

# File: chorus_engine/api/app.py

semantic_has_capture = False

for intent in semantic_intents:
    if intent.name == "capture_scene":
        semantic_has_capture = True
        logger.info(f"[SEMANTIC INTENT] Scene capture intent detected with {intent.confidence:.2f} confidence")

# Later...
if semantic_has_capture:
    # Trigger vision/camera system
    scene_description = await vision_service.capture_and_describe()
    # Include in AI response
```

---

## Summary

This specification provides complete guidance for:
- ✅ Adding new intents with proper prototypes and thresholds
- ✅ Tuning detection parameters based on testing
- ✅ Debugging false positives and false negatives
- ✅ Testing procedures and regression prevention
- ✅ Performance considerations and scaling

**Key principles**:
1. Start conservative (higher thresholds)
2. Test extensively with diverse phrasings
3. Monitor production statistics
4. Iterate based on real data
5. Document all changes

For questions or issues, refer to the main design document (`SEMANTIC_INTENT_DETECTION.md`) or the planning document (`Private/InternalPlanning/SEMANTIC_INTENT_DETECTION.md`).
