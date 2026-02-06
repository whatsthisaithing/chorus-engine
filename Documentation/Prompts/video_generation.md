# Video Generation Prompt Specification

**Version**: 1.0  
**Date**: January 7, 2026  
**Component**: `VideoPromptService`  
**File Location**: `chorus_engine/services/video_prompt_service.py`

---

## Overview

The video generation prompt instructs the LLM to create detailed, motion-focused video descriptions for video generation workflows (e.g., AnimateDiff, CogVideoX) based on user requests and conversation context. Unlike image prompts which emphasize static visual composition, video prompts focus on dynamic action, movement, camera motion, and temporal progression.

**Key Distinction from Image Generation**:
- **Image**: Static visual composition, lighting, framing, atmosphere
- **Video**: Dynamic action, movement, camera motion, temporal flow, what HAPPENS

## Purpose

Video prompt generation serves multiple functions:

1. **Motion Translation**: Convert natural language to motion-focused technical prompts
2. **Context Integration**: Extract action/movement details from conversation history
3. **Dynamic Enhancement**: Add camera movement, pacing, and temporal specifications
4. **Character Animation**: Ensure character movements and expressions are vivid
5. **Action Focus**: Prevent static descriptions, enforce dynamic content

## Architecture

### Generation Flow

```
1. User sends message with video request
   ↓
2. Semantic intent detection (embedding similarity)
   ↓
3. If detected: prepare prompt preview
   ↓
4. LLM generates detailed motion-focused prompt
   ↓
5. User confirms (or edits prompt)
   ↓
6. Inject into ComfyUI video workflow
   ↓
7. Video generates (longer timeout: 600s vs 300s for images)
   ↓
8. Video displays inline with thumbnail extraction
```

### Where It's Used

- **Service**: `VideoPromptService.generate_video_prompt()`
- **Orchestrator**: `VideoGenerationOrchestrator.generate_video()`
- **API Endpoint**: `POST /threads/{thread_id}/generate-video`
- **Model**: Uses character's preferred LLM model
- **Temperature**: 0.3 (lower than images for motion consistency)

### Processing Mode

**Synchronous** (user waits for prompt preview):
- Detection: semantic intent detection (embedding similarity)
- Prompt generation: 2-5 seconds (LLM call)
- User confirms before video generation
- Can edit prompt before submission
- Video generation: 30-120 seconds (longer than images)

---

## Request Detection

### Semantic Intent Detection

**Method**: `SemanticIntentDetector` (embedding-based)  
**Location**: `chorus_engine/services/semantic_intent_detection.py`  
**Integration**: API sets `semantic_has_video` and only calls the video orchestrator when the `send_video` intent is detected (`chorus_engine/api/app.py`).

**Behavior**:
- Uses prototype embeddings + cosine similarity (no LLM call)
- Applies per-intent thresholds with an ambiguity margin
- Supports sentence-level detection for long messages (hybrid mode)

---

## The Video Generation Prompt

### Complete System Prompt Template

**Location**: `VideoPromptService._build_system_prompt()` - Lines 48-123

```
You are a video generation prompt engineer. Your job is to create detailed, 
motion-focused prompts for video generation based on conversation context.

Character: {character.name}
Character Description: {character.role}

Video Generation Settings:
- Default Style: {default_style}
- Character Appearance: {self_description}
- Default Negative Prompt: {negative_prompt}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt.
It will be added automatically when needed.

Your task is to generate a MOTION-FOCUSED, dynamic video prompt based on 
the user's request.

CONTEXT USAGE:
You will receive recent conversation context. Use it intelligently to 
enhance your video prompt:
- EXTRACT motion-related details mentioned in the context (actions, movements, 
  scenes with activity)
- UNDERSTAND references ("that moment", "what you described", "from the story", 
  "your favorite")
- FOCUS on describable motion and action (ignore conversational mechanics and 
  meta-discussion)
- SYNTHESIZE details from multiple messages if the user references earlier 
  discussion
- If the user says "show me X from [earlier topic]", pull ALL relevant 
  motion/action details from context
- When a story or description was shared, extract specific dynamic elements 
  (actions, movements, changes, progression)

CRITICAL - CHARACTER DEPICTION RULES:
When the user requests a video involving the character (you/yourself):
- ALWAYS depict the character at their CURRENT age and appearance (use Character 
  Appearance description above)
- Even if the request references a past memory/story, show the character NOW 
  unless explicitly asked for a historical depiction
- Example: "show me that moment from your childhood story" → Show current 
  character in a scene inspired by the story, NOT as a child
- Example: "video of you walking on the beach" → Show current character 
  appearance walking on beach
- Only depict the character at a different age/appearance if explicitly 
  requested: "show yourself as a child in that scene"
- When in doubt, default to current character appearance

CRITICAL RULES:
1. Focus on MOTION, DYNAMIC ACTION, and TEMPORAL PROGRESSION
2. Describe what HAPPENS in the video, not just what it looks like
3. Include camera movement if relevant (pan, zoom, tracking shot, crane shot, 
   dolly zoom)
4. Specify timing/pacing when appropriate (slow motion, quick cuts, smooth 
   transition)
5. Maximum 150 words - be concise yet vivid
6. Use present tense and active verbs (flows, moves, transforms, swirls, 
   drifts, cascades)
7. DO NOT include dialogue or text that would render as on-screen captions
8. DO NOT use quotation marks around the prompt itself

GOOD VIDEO PROMPTS (emphasize action):
- A cat leaps gracefully through the air, paws extended, landing softly on a 
  windowsill as sunlight streams through
- Ocean waves crash against rocky cliffs in slow motion, water spraying upward 
  and catching golden hour light
- Camera slowly orbits around a steaming cup of coffee, revealing swirling 
  cream patterns forming and dissolving
- Leaves tumble and spiral through autumn air, dancing in wind currents, 
  casting moving shadows on the ground

BAD VIDEO PROMPTS (too static):
- A beautiful landscape with mountains and trees (no action!)
- A portrait of a person smiling (no movement!)
- A still shot of a building (explicitly static!)

Extract the essence of what should MOVE and CHANGE in the scene, then describe 
that motion vividly.

Return ONLY valid JSON in this format:
{
  "prompt": "detailed motion-focused video description (100-300 words) with 
             action verbs, camera movement, and temporal progression",
  "negative_prompt": "things to avoid in the video (static shots, jerky motion, 
                      poor lighting, etc.)",
  "reasoning": "brief 1-2 sentence explanation of the video concept"
}

Remember: Motion and action are key! Describe what MOVES and CHANGES in the scene.
```

### Key Differences from Image Prompts

| Aspect | Image Prompt | Video Prompt |
|--------|--------------|--------------|
| **Focus** | Static composition, lighting, atmosphere | Dynamic action, movement, temporal flow |
| **Length** | 100-300 words | Max 150 words (prompt rule; schema text still says 100-300) |
| **Verbs** | Descriptive (is, has, shows) | Action (moves, flows, transforms, drifts) |
| **Camera** | Optional framing guidance | Essential motion guidance (pan, track, orbit) |
| **Temperature** | 0.3 | 0.3 (same - need consistency) |
| **Timeout** | 300 seconds | 600 seconds (video takes longer) |
| **Examples** | Good: "sunset over ocean" | Bad: "sunset over ocean" (too static!) |
| | Bad: "waves crashing" (too vague) | Good: "waves crash in slow motion, spray catches light" |

---

## Scene Capture (Video Version)

### Third-Person Observer Perspective

**Method**: `VideoPromptService.generate_scene_capture_prompt()`  
**Trigger**: User clicks 🎥 "Capture Scene" button  
**Perspective**: Always third-person observer capturing current motion

### Scene Capture System Prompt

**Location**: `VideoPromptService._build_scene_capture_system_prompt()` - Lines 375-469

**Key Differences from Normal Video Generation**:

| Feature | Normal Generation | Scene Capture |
|---------|------------------|---------------|
| **Trigger** | User request detected | Manual button click |
| **Perspective** | Variable | Always third-person observer |
| **Context Window** | Last 3 messages | Last 10 messages |
| **Focus** | User's specific request | Current scene state with motion |
| **Character** | Current age (unless specified) | Current age, external view with action |
| **Participants** | May exclude user | Includes all participants in motion |
| **Temperature** | 0.3 | 0.3 (same for consistency) |

### Scene Capture Philosophy

**Goal**: Capture the current moment as a motion-focused third-person observer would see it.

**Example Transformations**:

1. **Context**: "I lean back and sigh, watching the clouds drift by..."
   - **Video Prompt**: "{character.name} leans back in their seat, chest rising with a deep sigh, gaze following clouds drifting slowly across the sky, peaceful contemplation. Camera slowly pushes in on their face, then pulls back to reveal the scene. Wind gently moves their hair."

2. **Context**: "I stand up excitedly, gesturing towards the window..."
   - **Video Prompt**: "{character.name} rises quickly from seated position, arms gesturing animatedly toward the window, excited energy in their movements, dynamic transition from still to active. Camera tracks the upward motion."

3. **Context**: "We're discussing art techniques..."
   - **Video Prompt**: "Close shot of {character.name} speaking, hands moving expressively as they explain concepts, facial expressions shifting with enthusiasm, subtle head tilts and nods during conversation. Gentle camera drift following hand gestures."

---

## Context Weighting and Message History

### Recent Message Priority

**Critical Rule**: Last 2-3 messages define CURRENT SCENE STATE

**Context Window**:
- **Normal Generation**: Last 3 messages (focused on specific request)
- **Scene Capture**: Last 10 messages (broader scene context)

**Message Weighting Logic**:
```
Messages 1-7: Background context (setting, history)
Messages 8-10: Current action/state (what's happening NOW)
```

If recent messages contradict earlier ones, RECENT messages are authoritative.

### Context Extraction Examples

**Scenario 1: Action Reference**
```
User: "Remember when you described dancing in the rain?"
Assistant: "Oh yes! I was twirling with arms outstretched..."
User: "Make a video of that moment"
```

**Extracted Motion**:
- Twirling motion (spinning)
- Arms outstretched (gesture)
- Rain environment (water droplets, splashing)
- Joyful energy (enthusiastic movement)

**Generated Prompt**: "{character.name} spins gracefully in falling rain, arms extended outward, water droplets catching light as she twirls. Her movements are fluid and joyful, hair and clothing flowing with each rotation. Camera slowly orbits around her, capturing water splashing underfoot and rain streaking past in slow motion. Golden hour lighting filters through the rain clouds."

**Scenario 2: Static Conversation to Subtle Motion**
```
User: "What do you think about philosophy?"
Assistant: "I find it fascinating. There's something about..."
User: "Video of us talking about this"
```

**Extracted Motion** (since conversation is static, focus on subtle movements):
- Speaking (lips moving, expressions changing)
- Gestures (hands moving to emphasize points)
- Head movements (nods, tilts)
- Environment motion (background activity, lighting shifts)

**Generated Prompt**: "Close shot of {character.name} engaged in animated conversation, hands gesturing thoughtfully as she speaks. Her expression shifts between contemplative and enthusiastic, eyes bright with interest. Camera slowly drifts between medium shot and close-up, capturing the intimate discussion. Soft lighting shifts as clouds move past a nearby window. Subtle ambient motion in background (books on shelf, plants swaying gently)."

---

## Validation and Quality Control

### Prompt Validation

**Method**: `VideoPromptService.validate_prompt()`  
**Checks**:
1. Length: 10-500 characters
2. Motion keywords present
3. No obvious static descriptions

**Motion Keywords** (must have at least one):
```python
motion_keywords = [
    'move', 'flow', 'drift', 'swirl', 'rotate', 'spin', 'cascade',
    'ripple', 'wave', 'pulse', 'transform', 'shift', 'glide', 'leap',
    'tumble', 'dance', 'sway', 'orbit', 'pan', 'zoom', 'track', 'fly'
]
```

**Warning Signs** (logged but not failed):
- No motion keywords → "May be too static"
- Too short (< 50 words) → "Might lack detail"
- Too long (> 300 words) → "Might be too complex"

### JSON Response Parsing

**Method**: `VideoPromptService._parse_llm_response()`  
**Handles**:
- Standard JSON format
- Markdown code blocks (```json)
- Malformed JSON (fallback extraction)
- Missing fields (provides defaults)

**Fallback Hierarchy**:
1. Try standard JSON parse
2. Extract from markdown code blocks
3. Regex extract "prompt" field
4. Strip markdown and use raw text
5. Clean and truncate to 500 chars

---

## Integration Points

### API Endpoints

**Normal Generation**:
```
POST /threads/{thread_id}/generate-video
Body: {
  "prompt": "optional pre-generated prompt",
  "custom_instruction": "optional user guidance",
  "workflow_id": "workflow database ID",
  "seed": optional_seed
}
```

**Scene Capture**:
```
POST /threads/{thread_id}/capture-video-scene
Body: {
  "workflow_id": "workflow database ID"
}
```

### Service Dependencies

```
VideoPromptService
  ├─ LLMClient (prompt generation)
  └─ CharacterConfig (character context)

VideoGenerationOrchestrator
  ├─ VideoPromptService (prompt generation)
  ├─ ComfyUIClient (workflow submission)
  ├─ VideoStorageService (file storage)
  ├─ VideoRepository (database records)
  └─ WorkflowManager (prompt injection)
```

---

### Research Areas

1. **Motion Prompt Tuning**: Further refinement of motion keywords
2. **Scene Complexity**: Handle multi-character dynamic scenes
3. **Temporal Coherence**: Improve frame-to-frame consistency
4. **Performance Optimization**: Reduce generation time
5. **Quality Metrics**: Automated motion quality assessment

---

## Troubleshooting

### Debug Logging

**Key Log Points**:
- `[VIDEO PROMPT SERVICE] Calling LLM` → Prompt generation start
- `[VIDEO PROMPT SERVICE] LLM returned` → Prompt generated
- `[VIDEO ORCHESTRATOR] Generated prompt: ...` → Final prompt
- `[VIDEO ORCHESTRATOR] ComfyUI job ID: ...` → Submission
- `Waiting for generation (timeout: 600s)` → Generation in progress

---

## Version History

### 1.0 (January 7, 2026)
- Initial specification based on implemented VideoPromptService
- Motion-focused prompt engineering
- Scene capture support
- Character age/appearance rules
- Context weighting logic
- Validation and quality control

---

## See Also

- [Image Generation Prompts](image_generation.md) - Static image prompt specification
- [Scene Capture Prompts](scene_capture.md) - Third-person observer perspective
- [Workflow Guide](../WORKFLOW_GUIDE.md) - ComfyUI workflow management
- [ComfyUI Integration](../Design/COMFYUI_WORKFLOW_SYSTEM.md) - Architecture overview
