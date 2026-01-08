# Semantic Intent Detection System

**Status**: Production (January 2026)  
**Purpose**: Detect user intents from natural language using embedding-based semantic similarity

---

## Overview

The semantic intent detection system replaces traditional keyword-based matching with a robust embedding-based approach. Instead of maintaining complex regex patterns, the system compares user messages against carefully crafted "intent prototypes" using cosine similarity.

**Key Advantages:**
- Handles natural language variation and paraphrasing automatically
- No maintenance burden of regex patterns
- Provides confidence scores for better decision-making
- Supports multi-intent detection (e.g., "send me a photo and remind me later")
- Zero additional dependencies (reuses memory embedding model)

---

## How It Works

### 1. Intent Prototypes

Each intent is defined by 5-14 carefully crafted example phrases that capture different ways users might express that intent:

```python
"send_image": [
    "send me a photo",
    "show me an image",
    "generate a picture",
    "create an image of something",
    "take a photo and send it",
    "can you make me a picture"
]
```

These prototypes are pre-embedded once at startup using the same `all-MiniLM-L6-v2` model used for memory embeddings.

### 2. Detection Process

When a user sends a message:

1. **Embed the message** using the shared embedding model (~5-15ms)
2. **Compare to all prototypes** using cosine similarity
3. **Calculate max similarity** for each intent (best matching prototype)
4. **Apply lexical anchor bonus** - hybrid boost for certain intents (+0.05 if reminder keywords present)
5. **Apply thresholds** - only return intents above their confidence threshold
6. **Check ambiguity** - reject if two intents are too close
7. **Return ranked list** of detected intents with confidence scores

**Hybrid Approach**: The system uses primarily semantic matching but applies a small lexical bonus (+0.05) for reminder intents when explicit keywords like "remind", "reminder", or "don't let me forget" are present. This hybrid approach catches edge cases where complex phrasing with specific task content might otherwise score slightly below threshold.

### 3. Example Detection

```
User: "Could you send me a picture of yourself?"

Similarity Scores:
  send_image: 0.78  ✓ (above 0.50 threshold)
  send_video: 0.42  ✗ (below 0.50 threshold)
  set_reminder: 0.15 ✗ (well below threshold)

Result: [Intent(send_image, 0.78)]
```

---

## Supported Intents

### send_image
**Purpose**: User requesting image generation  
**Threshold**: 0.50 (confirmable action)  
**Examples**:
- "Send me a photo of a sunset"
- "Can you show me what you look like?"
- "Generate an image of a forest"

**Action**: Triggers image generation workflow with confirmation dialog

### send_video
**Purpose**: User requesting video generation  
**Threshold**: 0.50 (confirmable action)  
**Examples**:
- "Send me a video of you dancing"
- "Show me a video clip"
- "Create a video showing the ocean"

**Action**: Triggers video generation workflow with confirmation dialog

### set_reminder
**Purpose**: User requesting a future reminder  
**Threshold**: 0.50 (confirmable action)  
**Prototypes**: 20 examples including both generic and content-heavy phrases  
**Examples**:
- "Remind me tomorrow to call mom"
- "Set a reminder for next week"
- "Don't let me forget to send that email"
- "Remind me in three weeks to call the dog groomer" (content-heavy)
- "Remind me that I need to schedule an appointment" (content-heavy)

**Lexical Boost**: +0.05 confidence if message contains "remind", "reminder", or "don't let me forget"

**Action**: Currently logged only (reminder system not yet implemented)

---

## Confidence Thresholds

Thresholds determine how confident the system must be before triggering an intent:

- **0.50**: Confirmable actions (image, video, reminder)  
  - Lower bar is acceptable because user can cancel in confirmation dialog
  
- **0.45**: Global minimum  
  - Nothing triggers below this regardless of intent type

- **0.08**: Ambiguity margin  
  - If two intents are within 0.08 of each other, detection is rejected as ambiguous

### Why These Values?

Initial testing showed:
- 0.50 catches most natural variations while avoiding false positives
- 0.60 was too conservative, missing valid paraphrases
- 0.45 minimum prevents random noise from triggering intents

---

## Multi-Intent Detection

The system can detect multiple intents in a single message:

```
User: "Take a photo of the sunset and remind me in an hour to send it to Mom"

Detected: [
    Intent(send_image, 0.82),
    Intent(set_reminder, 0.71)
]
```

**Exclusion Groups** prevent spurious multi-intent detections:
- `media_generation`: [send_image, send_video]
  - If both image and video are detected, only the highest confidence one is kept
  - Prevents "send me a video" from also triggering image generation

---

## Architecture

### Component Flow

```
User Message
    ↓
SemanticIntentDetector
    ↓
Embedding (all-MiniLM-L6-v2, ~10ms)
    ↓
Cosine Similarity vs Prototypes
    ↓
Threshold Filtering
    ↓
Exclusion Group Filtering
    ↓
[Intent(name, confidence), ...]
    ↓
Action Handlers (image/video orchestrators, etc.)
```

### Shared Resources

- **EmbeddingService**: Singleton service managing the `all-MiniLM-L6-v2` model
- **Model Reuse**: Same model used for both memory embeddings and intent detection
- **Zero Additional Cost**: No extra VRAM or startup time

### Performance

- **Latency**: 10-15ms typical (CPU-bound, embedding + similarity calculation)
- **Memory**: ~90KB for prototype embeddings (negligible)
- **Context**: Intent detection overhead is trivial compared to LLM inference (1000-5000ms)

---

## Integration Points

### API Endpoints

**Both `/threads/{thread_id}/messages` endpoints** (regular and streaming):
1. User message saved to database
2. **Semantic intent detection** runs (~10ms)
3. Flags set: `semantic_has_image`, `semantic_has_video`
4. Image/video orchestrators triggered if flags are true
5. Confirmation prompts shown to user
6. LLM generates response (potentially referencing upcoming media)

### Logs

```
[SEMANTIC INTENT] Detected 2 intent(s): send_image(0.78), set_reminder(0.65)
[SEMANTIC INTENT] Image generation intent detected with 0.78 confidence
[SEMANTIC INTENT] Reminder intent detected with 0.65 confidence
```

---

## Testing

### Interactive Test Utility

Located at `utilities/intent_detection_test/test_intent.py`:

```bash
cd utilities/intent_detection_test
python test_intent.py
```

**Features**:
- Type test phrases interactively
- See real-time detection results with confidence scores
- All tests logged to `logs/intent_test_TIMESTAMP.log`
- Statistics summary on exit (CTRL+C)

**Example Session**:
```
Enter phrase: send me a selfie
--------------------------------------------------------------------
send_image: 0.725
send_video: 0.412
set_reminder: 0.189

✓ Detected 1 intent(s):
  • send_image: 0.7250
```

### Adding Test Cases

To validate new prototypes or threshold changes:
1. Run test utility
2. Try diverse phrasings (formal, casual, indirect)
3. Check for false positives (wrong intent) and false negatives (missed intent)
4. Adjust prototypes or thresholds as needed
5. Re-test to confirm improvements

---

## Future Enhancements

### Two-Stage Temporal Detection (Deferred)

An optional enhancement for reminder detection:
- **Stage 1**: Semantic similarity check (current system)
- **Stage 2**: Temporal expression detection using dateparser
- **Trigger**: If semantic score is 0.45-0.50 AND temporal expression found, boost to trigger

**Status**: Deferred - current content-heavy prototypes + lexical anchor bonus achieve 95%+ detection accuracy without additional complexity.

### Partially-Sure Confirmations

For intents scoring just below threshold (e.g., 0.45-0.50 for 0.50 threshold):
- Show confirmation popup: "Did you want me to [action]?"
- If user confirms, treat as detected
- Catches edge cases without lowering global threshold
- User feedback improves future tuning

### Context-Aware Detection

Currently analyzes each message independently. Future enhancement:
- Consider previous messages in conversation
- Example: "I want to remember this" → "do it" (should trigger reminder with context)

### User Feedback Loop

Capture corrections from users:
- "No, I didn't want a reminder" → log false positive
- Analyze patterns in misclassifications
- Suggest new prototypes or threshold adjustments

---

## Comparison to Keyword Detection (Legacy)

| Aspect | Keyword (Legacy) | Semantic (Current) |
|--------|------------------|-------------------|
| **Matching** | Exact keywords + regex | Semantic similarity |
| **Paraphrasing** | ❌ Breaks easily | ✅ Handles naturally |
| **Maintenance** | ⚠️ Manual regex updates | ✅ Add prototypes |
| **Confidence** | ❌ Binary (yes/no) | ✅ 0-1 scores |
| **Multi-intent** | ❌ Not supported | ✅ Full support |
| **False Positives** | ⚠️ Common ("video game") | ✅ Rare |
| **Dependencies** | ✅ None | ✅ Reuses memory model |
| **Latency** | ✅ Instant | ✅ ~10ms (negligible) |

**Status**: Keyword detection disabled but preserved in codebase (marked `# TODO: Remove`). Will be removed after stable operation period.

---

## Troubleshooting

### Intent Not Detected (False Negative)

**Symptoms**: User clearly requests action, but system doesn't detect it

**Diagnosis**:
1. Check logs for confidence score (might be just below threshold)
2. Run phrase through test utility to see actual scores
3. Check if phrase is paraphrased very differently from prototypes

**Solutions**:
- Add similar prototype to cover that phrasing style
- Lower threshold slightly (0.50 → 0.48) if consistently close
- Implement partially-sure confirmations for near-threshold cases

### Wrong Intent Detected (False Positive)

**Symptoms**: System detects unintended intent

**Diagnosis**:
1. Check confidence score (might be just above threshold)
2. Check if user's phrase is genuinely ambiguous
3. Look for keyword overlap causing confusion

**Solutions**:
- Raise threshold for that intent
- Add to exclusion group if spurious multi-intent
- Refine prototypes to be more specific

### Ambiguous Detection

**Symptoms**: No intent detected, but user clearly made a request

**Diagnosis**:
1. Check logs for "Ambiguous" message
2. Two intents likely scored very close (< 0.08 margin)

**Solutions**:
- Review the specific phrase - is it genuinely ambiguous?
- Adjust prototypes to create more separation
- Consider if both intents are actually valid (multi-intent scenario)

---

## Development Guidelines

### When to Add a New Intent

**Good candidates**:
- Clear user action request (generate, create, remind, search, etc.)
- Confirmable or low-stakes (user can cancel/undo)
- Distinct from existing intents (not ambiguous)

**Poor candidates**:
- Conversational responses ("that's interesting", "I agree")
- Ambiguous phrasing overlapping with existing intents
- High-stakes actions without confirmation (delete, execute, etc.)

### Prototype Design Best Practices

1. **Diversity**: Cover different verbs, structures, formality levels
2. **Brevity**: Keep prototypes 3-10 words (focused phrases)
3. **Realism**: Use phrases real users would say
4. **Avoid Specifics**: "send me a photo" not "send me a 1920x1080 photo of a cat"
5. **Test Coverage**: Validate with real user phrases in test utility

### Threshold Tuning Process

1. **Start conservative** (0.55-0.60)
2. **Collect real data** (logs from production or test utility)
3. **Analyze misses**: False negatives scoring 0.45-0.55? Lower threshold
4. **Analyze false positives**: Wrong intents above threshold? Raise it
5. **Iterate**: Small adjustments (±0.05) and re-test

---

## Statistics and Monitoring

The detector tracks statistics accessible via `get_stats()`:

```python
{
    "total_detections": 1547,
    "successful_detections": 1289,
    "ambiguous_detections": 34,
    "no_intent_detected": 224,
    "multi_intent_detections": 15
}
```

**Key Metrics**:
- **Success rate**: successful / total (target: >80%)
- **Ambiguity rate**: ambiguous / total (target: <5%)
- **No-intent rate**: no_intent / total (acceptable: 10-20%, users often don't request actions)

---

## Summary

The semantic intent detection system provides robust, maintainable intent recognition using embedding similarity. It handles natural language variation gracefully, requires minimal maintenance, and integrates seamlessly with existing systems through the shared embedding model.

**Key Takeaways**:
- ✅ Production-ready replacement for keyword detection
- ✅ Handles paraphrasing and natural language variation
- ✅ Zero additional dependencies or performance cost
- ✅ Easy to extend with new intents
- ✅ Provides confidence scores for better decision-making
- ✅ Supports multi-intent scenarios

**Next Steps**:
- Monitor detection accuracy in production
- Collect user feedback on false positives/negatives
- Add partially-sure confirmations for edge cases
- Remove legacy keyword detection code after stable operation
