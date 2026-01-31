# Vision System - Character Visual Perception

**Status**: Production (January 2026)  
**Purpose**: Enable characters to perceive and interpret visual content naturally within conversations

---

## Overview

The Vision System gives characters the ability to "see" images uploaded by users and discuss visual content naturally in conversations. Rather than treating images as external tools, vision is integrated as a core perceptual capability that enhances character interactions.

**Key Advantages:**
- Two-stage architecture preserves character personality (observation → interpretation)
- Automatic VRAM management (Ollama/LM Studio handle model swapping)
- Visual memories enable long-term recall of images
- Works seamlessly across web UI and Discord bridge
- Supports multiple images per message
- Backend flexibility (Ollama or LM Studio)

---

## How It Works

### Two-Stage Architecture

The system separates **perception** from **interpretation** to maintain character agency:

**Stage 1 - Vision Model Observes (Perception Layer)**
- Vision model (Qwen3-VL) analyzes image objectively
- Produces structured observation: subjects, objects, text, mood, spatial layout
- No personality, just factual description
- ~5-15 seconds processing time

**Stage 2 - Character Interprets (Personality Layer)**
- Character's LLM receives observation in message context
- Interprets through their unique personality
- Responds naturally as if they "saw" the image

**Example Flow:**
```
User uploads cat photo 
→ Vision: "A tabby cat sitting on red cushion, eyes closed, content expression"
→ Nova (playful): "Aww, what a cozy little philosopher! That cat has achieved peak relaxation."
→ Marcus (analytical): "Interesting. The closed eyes and relaxed posture suggest REM sleep stage..."
```

This separation ensures characters maintain distinct personalities while sharing the same vision capability.

---

## Supported Backends

### Ollama (Primary)
- **Models**: qwen3-vl (2b, 4b, 8b, 14b, 30b variants)
- **Auto-management**: Automatically swaps LLM ↔ vision model as needed
- **VRAM**: Handles allocation transparently
- **Processing**: 5-12 seconds typical (qwen3-vl:4b)

### LM Studio (Alternative)
- **Models**: Same qwen3-vl variants via OpenAI-compatible API
- **Auto-management**: Same transparent model swapping
- **VRAM**: Handles allocation like Ollama
- **Processing**: Similar performance to Ollama

**Backend Selection**: Automatically detected from `llm.provider` in system.yaml. No separate configuration needed.

---

## Image Processing Pipeline

### 1. Upload & Validation
- **Supported formats**: JPEG, PNG, WebP, GIF
- **Size limit**: 10MB (web UI), 50MB (Discord bridge - auto-resized anyway)
- **Storage**: `data/images/conversation_id/`
- **Database**: Full metadata in `image_attachments` table

### 2. Vision Analysis
- **Preprocessing**: Auto-resize to 1024px (web) or 2048px (Discord) max dimension
- **Inference**: Vision model analyzes image
- **Output**: Structured JSON observation with confidence score
- **Cache**: Results stored permanently in database

### 3. Character Integration
- **Context injection**: Observation added to message for LLM
- **Memory creation**: Auto-creates visual memory (category="visual")
- **Retrieval**: Visual memories participate in semantic search

---

## Visual Memories

Every analyzed image automatically creates a memory:

**Memory Properties:**
- **Type**: EXPLICIT (user deliberately showed image)
- **Category**: "visual" (new category type)
- **Priority**: 70 (moderately important)
- **Confidence**: Based on vision model confidence (0.60+ threshold)
- **Content**: Human-readable summary, not raw JSON

**Example Memory:**
```
"User showed me an image of a sunset over mountains. 
The image conveyed a peaceful, serene mood."
```

**Retrieval:**
- Query: "remember that sunset photo?" → Retrieves visual memory
- Characters can naturally reference past images in conversation
- Visual memories persist across conversations

---

## Discord Bridge Integration

### Simplified Approach

**Design Philosophy**: Bot invocation = intent to analyze images

When a user @mentions the bot (or replies/DMs):
1. **Current message**: All images in triggering message are processed
2. **Recent history**: Last 5-10 messages are synced, including any images
3. **Cache sharing**: Multiple bots share image cache (no duplicate analysis)
4. **Limit**: Max 5 images per message (configurable)

**No Complex Intent Detection**: Unlike the original plan for trigger phrase detection, the implementation uses a simpler approach - if the bot is invoked, process all images. This avoids false negatives and matches user expectations.

**Multi-Bot Efficiency:**
- Discord attachment ID → Chorus attachment ID mapping cached in database
- NovaBot uploads image → cached
- MarcusBot sees same image → reuses cached analysis
- One vision analysis per unique Discord image

### Processing Flow

```
User: "@NovaBot look at this" [image.jpg]
  ↓
1. Bot detects @mention trigger
2. Downloads image from Discord
3. Checks cache (not found)
4. Resizes to 2048px max
5. Uploads to Chorus API
6. Vision analysis triggered
7. Observation added to message context
8. Character responds with vision context
9. Cache mapping stored for other bots
```

---

## VRAM Management

### Automatic Backend Management

**Ollama & LM Studio** handle model swapping transparently:
- No manual VRAM configuration needed
- Backends intelligently unload LLM when vision model needed
- Vision model auto-unloads after processing
- Total cycle: ~5-15 seconds for complete swap + analysis

**Manual Management (ComfyUI Only)**:
- Vision models unload when ComfyUI workflows run
- Uses existing `llm.unload_during_image_generation` setting
- Ensures maximum VRAM for image/audio generation

### Model Size Tiers

| VRAM | Model | Use Case |
|------|-------|----------|
| 8-10 GB | qwen3-vl:4b | Basic perception, OCR, object detection |
| 12-16 GB | qwen3-vl:8b | **Recommended** - Best balance |
| 20-24 GB | qwen3-vl:14b | Better spatial reasoning, detail |
| 32 GB+ | qwen3-vl:30b | Maximum capability, future video support |

---

## Configuration

### System Settings (system.yaml)

```yaml
vision:
  enabled: true
  
  model:
    name: qwen3-vl:4b  # For Ollama
    # name: qwen/qwen3-vl-4b  # For LM Studio
    load_timeout_seconds: 60
    
  processing:
    max_retries: 2
    timeout_seconds: 30
    resize_target: 768
    max_file_size_mb: 10
    
  memory:
    auto_create: true
    category: visual
    default_priority: 70
    min_confidence: 0.6
    
  cache:
    enabled: true
    allow_reanalysis: true
```

### Character-Specific Overrides

Characters can have custom vision settings in their YAML:

```yaml
vision:
  enabled: true  # This character can "see"
  auto_memory: true  # Override global memory creation
```

**Future Enhancement**: Character-specific observation styles (technical, artistic, emotional)

---

## Web UI Features

### Image Upload
- **Button**: "+" icon on left of message input
- **Drag-and-drop**: Drop images onto conversation area or input
- **Preview**: Thumbnail with filename before sending
- **Auto-expanding textarea**: Grows upward as you type

### Image Display
- **In conversation**: Images shown below user message
- **Click to expand**: Full-screen modal with vision details
- **Vision metadata**: Model used, confidence score, processing time
- **Re-analysis**: Future feature to analyze with different model

### Settings Panel
- Complete vision configuration in System Settings modal
- Enable/disable vision system
- Model selection and processing settings
- Memory integration controls
- Cache management options

---

## Known Limitations

### Current Limitations

1. **No Video Support**: Only static images (video planned for future phase)
2. **Sequential Processing**: Multiple images processed one-at-a-time (not parallel)
3. **No Cloud APIs**: Local models only (OpenAI/Anthropic in future)
4. **Format Restrictions**: Limited to JPG, PNG, WebP, GIF
5. **No Re-analysis UI**: Can't re-analyze existing images with different model (code exists, UI pending)

### Design Trade-offs

**Two-Stage vs Direct Vision**: We chose two-stage (vision → character) over giving LLMs direct vision access:
- **Pro**: Maintains character personality separation
- **Pro**: Can swap vision models without retraining characters
- **Pro**: Vision observations are auditable and cacheable
- **Con**: Adds 5-15 seconds latency
- **Con**: Loses some nuance vs direct vision

**Auto-analyze vs Intent Detection**: We simplified from complex intent detection to "bot triggered = analyze":
- **Pro**: Zero false negatives (never miss an image user wanted analyzed)
- **Pro**: Simpler code, easier to maintain
- **Pro**: Matches user expectations (why share image if not for analysis?)
- **Con**: May process some social sharing images unnecessarily
- **Con**: No granular control over when vision runs


### Optimization Opportunities

1. **Parallel processing**: Process multiple images concurrently
2. **Smaller models**: Use 2b variant for faster processing
3. **Lazy analysis**: Defer analysis until character needs to reference image
4. **Progressive streaming**: Return partial observations during analysis

---

## Future Enhancements

### Short-term (Phase 5+)

**Re-analysis Feature:**
- UI button: "Re-analyze with current model"
- Compare results from different models
- Track analysis history

**Performance Metrics:**
- Add vision stats to `/system/status` endpoint
- Track cache hit rate, average processing time
- VRAM usage monitoring

**Error Recovery:**
- Graceful degradation if vision model unavailable
- Retry logic with exponential backoff
- User-friendly error messages

### Long-term (Phase 6+)

**Video Support:**
- Frame extraction from videos
- Multi-frame analysis with temporal reasoning
- Video memory type

**Advanced Vision:**
- OCR extraction for documents
- Face recognition (with user consent)
- Spatial reasoning improvements
- Character-specific vision prompts

**Cloud APIs:**
- OpenAI GPT-4V integration
- Anthropic Claude with vision
- Fallback chain: local → cloud

---

## Philosophy & Design Principles

### Vision as Perception, Not Interpretation

**Core Principle**: The vision model observes, the character interprets.

This separation ensures:
- Characters maintain unique personalities
- Vision capability is shared infrastructure
- Observations are objective and auditable
- Easy to upgrade vision models without retraining characters

### Memory as Truth

Images create memories automatically because:
- Users expect characters to remember images they've shared
- "Remember that mountain photo?" should work naturally
- Visual context enhances conversation continuity
- Memories enable future multi-turn image discussions

### Simplicity Over Perfection

**Intent Detection Simplification**: Original plan had complex trigger phrase detection for Discord. Implementation simplified to "bot invocation = analyze images" because:
- Users expect bots to see images when explicitly invoked
- False negatives (missing images) worse than false positives
- Simpler code = fewer bugs, easier maintenance
- Can always add granular control later if needed

### Backend Agnosticism

Vision system detects backend automatically from LLM config:
- Users don't configure vision backend separately
- Switching Ollama ↔ LM Studio is seamless
- Backend-specific code isolated in VisionService
- Easy to add new backends (e.g., cloud APIs) later

---

## Success Metrics

The vision system is considered successful when:

1. **Adoption**: Users regularly share images with characters
2. **Accuracy**: Vision observations are >90% accurate for common scenarios
3. **Performance**: <15 seconds total latency for single image
4. **Reliability**: <1% failure rate for supported formats
5. **Natural Integration**: Users forget vision is a separate feature
6. **Memory Recall**: Visual memories retrieved naturally in conversations

**Current Status (January 2026)**:
- ✅ Phase 1: Web UI image upload (complete)
- ✅ Phase 1.5: LM Studio support (complete)
- ✅ Phase 3: Discord bridge integration (code complete, testing pending)
- ⏳ Phase 4: Re-analysis and advanced features (partial)
- ⏳ Phase 5: Documentation and polish (in progress)

---

## Related Systems

- **[Memory Intelligence System](MEMORY_INTELLIGENCE_SYSTEM.md)**: Visual memories integrate with memory types and temporal weighting
- **[Semantic Intent Detection](SEMANTIC_INTENT_DETECTION.md)**: Original vision intent plan (simplified in implementation)
- **[ComfyUI Workflow System](COMFYUI_WORKFLOW_SYSTEM.md)**: VRAM coordination between vision and image generation
- **[Prompt Assembly & Token Management](PROMPT_ASSEMBLY_TOKEN_MANAGEMENT.md)**: Vision context injection into character prompts

---

**Living Document**: This design reflects the system as of January 2026. As vision capabilities evolve (video support, advanced features), this document will be updated to reflect new architectural decisions and learned insights.
