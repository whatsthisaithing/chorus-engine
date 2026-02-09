# Scene Capture Prompt Specification

**Version**: 1.1  
**Date**: January 7, 2026  
**Component**: `SceneCapturePromptService` (Images), `VideoPromptService` (Videos)  
**File Locations**: 
- Images: `chorus_engine/services/scene_capture_prompt_service.py`
- Videos: `chorus_engine/services/video_prompt_service.py`

---

## Overview

Scene capture is a specialized generation feature that allows users to manually capture the current visual state of a conversation from a third-person observer perspective. Unlike normal in-conversation generation (which responds to user requests), scene capture provides an omniscient narrator viewpoint of "what's happening right now."

**Key Concept**: Think of it as taking a photograph (or video) of the current scene from an invisible camera in the room, capturing all participants, their positions, actions, and the environment.

**Image vs. Video Scene Capture**:
- **📷 Image**: Captures static moment, composition, lighting, atmosphere
- **🎥 Video**: Captures motion, action, gestures, dynamic energy (see [Video Generation](video_generation.md))

---

## Philosophy: Observer vs. Participant

### Normal Image Generation (Participant View)

**Trigger**: User requests image ("show me X", "picture of Y")  
**Perspective**: Can be first-person (character POV) or requested view  
**Focus**: Responds to specific user request  
**Examples**:
- "Show me what you look like" → Character portrait
- "Draw a sunset" → Artistic scene, no participants
- "Photo of you eating ramen" → Character in action

### Scene Capture (Observer View)

**Trigger**: User clicks "Capture Scene" button  
**Perspective**: Always third-person omniscient observer  
**Focus**: Current state of the conversation scene  
**Examples**:
- Conversation about sitting in a cafe → Image shows both people at table
- Discussion while walking in park → Image shows characters walking together
- Character doing an activity → External view of character + action + environment

---

## When to Use Scene Capture

### Ideal Scenarios

1. **Multi-Character Scenes**: When user and character are both present
2. **Environmental Context**: Capturing the setting/location being discussed
3. **Action Moments**: Visualizing what's happening "right now" in roleplay
4. **Atmosphere**: Capturing mood, lighting, environment of current scene
5. **Continuity**: Creating visual record of conversation progression

### Image vs. Video Scene Capture

**📷 Image Scene Capture** (Static):
- Freezes a single moment in time
- Emphasizes composition, lighting, atmosphere
- Good for: portraits, environments, poses, aesthetic moments
- Example: "Two people sitting across from each other at a candlelit table, warm lighting, intimate atmosphere"

**🎥 Video Scene Capture** (Dynamic):
- Captures motion and action
- Emphasizes movement, gestures, energy
- Good for: actions, interactions, transitions, dynamic moments
- Example: "Character leans forward, gesturing expressively while speaking, then sits back with a smile"
- See [Video Generation](video_generation.md) for details on motion-focused prompting

### Not Ideal For

1. **Abstract Concepts**: Ideas that aren't visually concrete
2. **Past/Future Events**: Scene capture is "present moment" focused
3. **Character Portraits**: Better to use normal generation ("show me what you look like")
4. **Requested Imagery**: User-specific requests should use normal flow

---

## Architecture Differences

### Feature Comparison

| Feature | Normal Generation | Scene Capture |
|---------|------------------|---------------|
| **Trigger** | User request detected | Manual button click |
| **Perspective** | Variable (first/third person) | Always third-person observer |
| **Context Window** | Last 3 messages | Last 10 messages |
| **Focus** | User's specific request | Current scene state |
| **Character Depiction** | Current age (unless specified) | Current age, external view |
| **Participants** | May exclude user | Includes all participants |
| **Confirmation** | Optional (can disable) | Always shows dialog |
| **Temperature** | 0.3 | 0.5 |

### Integration Points

**Normal Generation Flow**:
```
User message → Semantic intent detection → Prompt generation → Confirmation → Image
```

**Scene Capture Flow**:
```
Button click → Scene analysis → Prompt generation → Confirmation → Image
```

---

## Tuning History and Rationale

### Version 1.0 → 1.1 (January 5, 2026)

**Problem Identified**: Scene capture was producing generalized descriptions focused on early conversation moments rather than capturing the "up to the moment" current state that worked so well in in-conversation generation.

**Root Cause Analysis**:

1. **Smaller Context Window**: Scene capture used only 5 messages vs. 3 for in-conversation
   - Missed recent conversation flow and buildup
   - In long conversations, 5 messages might not even include the setup for "current moment"

2. **Higher Temperature**: 0.8 temperature allowed creative drift toward generalization
   - In-conversation used 0.3 for focused, consistent results
   - Higher creativity led to generic scene descriptions

3. **Double-Reversal Bug**: Code reversed messages twice
   - API reversed to chronological (oldest → newest)
   - Service reversed again (newest → oldest)
   - Net result back to newest-last BUT with already-limited context

4. **Lack of Explicit Recency Weighting**: System prompt mentioned "current state" but didn't strongly emphasize which messages defined "current"
   - LLM interpreted entire 5-message window as equally weighted
   - Early messages dominated scene descriptions

**Changes Implemented (Hybrid Approach)**:

1. **Increased Context Window**: 5 → 10 messages
   - **Rationale**: Provide broader context than in-conversation prompts (which use last 3 messages)
   - **Benefit**: Captures complete conversation arc and scene evolution
   - **Result**: LLM sees how the scene developed, not just a snapshot

2. **Lowered Temperature**: 0.8 → 0.5
   - **Rationale**: Split difference between creative (0.8) and focused (0.3)
   - **Benefit**: Maintains vivid descriptions while reducing generalization drift
   - **Result**: More consistent "moment capture" without losing artistic quality

3. **Removed Double-Reversal**: Messages stay in chronological order
   - **Rationale**: LLMs naturally weight later information more heavily
   - **Benefit**: Most recent messages at end of context = higher influence
   - **Result**: "Current moment" naturally emphasized by position

4. **Added Explicit Weighting Instructions**: "Last 2-3 messages define current scene"
   - **Rationale**: Make implicit (position-based) weighting explicit in instructions
   - **Benefit**: LLM told directly: recent = current state, earlier = background
   - **Result**: Clear hierarchy: Setting from full context + Current state from last 2-3

5. **Enhanced Message Numbering**: `[Message 1/10]`, `[Message 2/10]`, etc.
   - **Rationale**: Help LLM track position and recency in context
   - **Benefit**: Makes message sequence and recency visually obvious
   - **Result**: Easier for LLM to identify "last 2-3 messages"

**Expected Outcome**:

Scene capture should now behave more like in-conversation generation:
- ✅ **"Up to the moment" accuracy**: Captures current scene state, not generalized early moments
- ✅ **Fuller context**: 10 messages provide complete scene evolution
- ✅ **Focused descriptions**: Lower temperature reduces creative drift
- ✅ **Natural recency bias**: Chronological order + explicit instructions = current moment focus
- ✅ **Observer perspective maintained**: Third-person viewpoint unchanged

**Key Difference Preserved**:

While now similar in context handling, scene capture maintains its unique value:
- **In-conversation**: Responds to explicit user request ("show me X")
- **Scene capture**: Captures implicit current state ("what's happening now")

Scene capture uses a 10-message context (broader than in-conversation image prompts, which use the last 3 messages), but they interpret it differently:
- In-conversation: "What visual does the user want?"
- Scene capture: "What is currently happening in this scene?"

---

## Context Synthesis Strategy

### Message Window (Updated v1.1)

**Size**: Last 10 messages (increased from 5)  
**Rationale**: Provides broader context than in-conversation prompts (last 3 messages)  
**Processing**: Chronological order (oldest to newest, no reversal)  
**Weighting**: Last 2-3 messages explicitly define "current scene state"

### Context Synthesis Rules

Both services emphasize:
- **EXTRACT** visual details from context
- **SYNTHESIZE** details from multiple messages
- **UNDERSTAND** references to earlier discussion
- **FOCUS** on visually describable elements
- **BUILD** complete scene from accumulated context

**Key Difference**: Scene capture now uses explicit two-tier weighting:
1. **Background Context (Messages 1-7)**: Setting, environment, history
2. **Current State (Messages 8-10)**: Positions, actions, emotions happening NOW

This creates more accurate "moment capture" while still using full scene context.

---

## Scene Capture System Prompt

### Full Template

```
You are an expert at creating detailed, vivid image generation prompts for Stable Diffusion/ComfyUI from a THIRD-PERSON OBSERVER perspective.

You are describing a scene as an omniscient narrator would see it - not from the character's perspective, but as a neutral observer capturing the moment like a camera would see it.

Character: {character.name}
Character Description: {character.role}

Image Generation Settings:
- Default Style: {default_style}
- Character Appearance: {self_description}
- Default Negative Prompt: {negative_prompt}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt. 
It will be added automatically when needed.

Your task is to generate a HIGHLY DETAILED, descriptive image prompt based on the current scene in the conversation.

CRITICAL - OBSERVER PERSPECTIVE RULES:
- Describe the scene as a NEUTRAL OBSERVER watching from outside (like a camera capturing the moment)
- Use THIRD-PERSON pronouns (he/she/they, NOT I/me/my)
- Describe what's VISIBLE in the scene (people, actions, setting, environment, mood)
- Include {character.name}'s visible appearance, expression, body language, and positioning
- If the user is referenced in context, describe them too (as "the other person", "the companion", or by any description given)
- Focus on the CURRENT STATE of the scene from the most recent messages

CONTEXT USAGE:
You will receive recent conversation context. Use it intelligently to capture the current scene:

CRITICAL - MESSAGE WEIGHTING:
- The LAST 2-3 MESSAGES define the CURRENT SCENE STATE (what's happening right now)
- Earlier messages provide BACKGROUND CONTEXT ONLY (setting, history, earlier actions)
- When actions/positions change across messages, ALWAYS use the most recent information
- If recent messages contradict earlier ones, the RECENT messages are authoritative

CONTEXT SYNTHESIS:
- EXTRACT visual details from the context (setting, location, environment, objects, descriptions)
- UNDERSTAND references to earlier discussion ("that moment", "what we described", "from the story")
- SYNTHESIZE details from multiple messages to build a complete visual scene
- IDENTIFY character positions, actions, and interactions happening NOW (from last 2-3 messages)
- CAPTURE visible emotions through body language and expressions (from recent messages)
- NOTE any mentioned objects, props, or environmental details
- Focus on visually describable elements (ignore conversational mechanics)
- If multiple people are present, describe all visible participants
- Build the scene using: SETTING from full context + CURRENT STATE from last 2-3 messages

Scene Focus Examples:
1. Context shows: "I lean back against the stone wall, watching the sunset..."
   → Describe: {character.name} leaning against stone wall, sunset in background, their relaxed posture, the warm lighting on their face

2. Context shows: "We're sitting across from each other at the cafe table..."
   → Describe: Two people at cafe table, {character.name}'s appearance and expression, the other person across from them, cafe interior, intimate conversation mood

3. Context shows: "I stand up excitedly, gesturing towards the window..."
   → Describe: {character.name} standing near window in excited pose, animated gesture, window showing exterior view, dynamic energy

DETAIL REQUIREMENTS:
- Write 100-300 words of rich, specific description
- Include fine details: textures, colors, lighting quality, atmosphere
- Describe composition, perspective, and framing (where would a camera be positioned?)
- Add environmental details and mood
- Describe {character.name}'s appearance using Character Appearance description
- Include their pose, expression, clothing, and positioning in the scene
- If other people are present, describe them and their positioning
- Use evocative, visual language that paints a clear picture
- Include artistic style keywords and technical photography/art terms

Example of detail level for scene capture:
Bad: "woman in a room"
Good: "{character.name}, a young woman with flowing auburn hair, stands near a tall window in a cozy study. Afternoon sunlight streams through vintage lace curtains, casting dappled patterns across her white linen dress and the worn wooden floorboards. She holds an old leather-bound book against her chest, her expression thoughtful and distant as she gazes out at the garden beyond. Dust motes dance in the golden light. Behind her, floor-to-ceiling bookshelves frame the intimate scene, their spines creating a warm backdrop of burgundy and forest green. A plush armchair sits nearby with a knitted throw draped over its arm. The atmosphere is serene, contemplative, touched with melancholy. Shot from a medium distance with shallow depth of field, 50mm lens, the background bookshelves softly blurred. Cinematic composition, warm color grading, Pre-Raphaelite aesthetic with attention to texture and detail."

Recent conversation context:
{context}

Return ONLY valid JSON in this format:
{
  "prompt": "extremely detailed 100-300 word scene description from third-person observer perspective, including {character.name}'s appearance, positioning, visible actions, environment, other participants if present, lighting, atmosphere, composition, and technical specifications",
  "negative_prompt": "things to avoid (or use character default)",
  "reasoning": "brief 1-2 sentence explanation of the scene being captured"
}

Remember: You are an omniscient observer describing what a camera would see. More detail = better results. Be specific, visual, and evocative!
```

---

## Key Design Principles

### 1. Omniscient Observer Stance

**Metaphor**: Invisible camera in the room  
**Language**: Third-person only ("she sits", not "I sit")  
**Scope**: See everything happening in the scene  
**Benefit**: Captures interactions between participants

**Example Comparison**:

**First-Person (Character POV)**:
```
"I sit at my desk, looking down at my notes, feeling the warm sunlight on my shoulders"
```

**Third-Person Observer (Scene Capture)**:
```
"Sarah sits at her wooden desk, sunlight streaming through the window behind her, illuminating her focused expression as she reviews handwritten notes spread before her"
```

### 2. Multi-Participant Awareness

Scene capture explicitly includes all participants:

**Context**: "We're having coffee together at the cafe..."

**Scene Capture Output**:
```
"Two people sit across from each other at a small round table in a cozy cafe. 
Sarah, a woman in her late twenties with shoulder-length brown hair, leans forward 
slightly, her hands wrapped around a ceramic coffee mug. Across from her, her 
companion sits relaxed, both engaged in intimate conversation. The cafe interior 
features exposed brick walls, warm Edison bulb lighting, and the soft hum of 
background chatter..."
```

**Why This Matters**: User often wants to see themselves in the scene, not just the character.

### 3. Present-Moment Focus (Updated v1.1)

**Goal**: Capture "what's happening right now"  
**Challenge**: Balance immediate state with context synthesis  
**Solution**: Two-tier weighting - "Background context + Current state from last 2-3 messages"

**Example with 10-message context**:

```
Context (10 messages):
[Messages 1-3] "Let's go to the park" / "Great idea" / "I'll pack a picnic"
[Messages 4-6] "We found a bench under an oak tree" / "Perfect spot" / "The weather is beautiful"
[Messages 7-8] "Want to read while I sketch?" / "Sure, I brought my book"
[Messages 9-10] "*settles in with sketchpad*" / "*opens book and relaxes*"

Scene Capture (using weighting):
"Sarah sits on a weathered wooden bench beneath an ancient oak tree in a sunlit 
park, sketchpad balanced on her knee, pencil moving across the page as she captures 
the scene before her. Beside her, her companion reclines with a book open, absorbed 
in reading, afternoon light filtering through the leaves above and casting dappled 
shadows across both figures. A wicker picnic basket rests at their feet..."
```

**Notice the layering**:
- **Background (msgs 1-6)**: Park, oak tree, bench, weather, picnic basket
- **Current state (msgs 7-10)**: Sketching, reading, settled positions, relaxed mood
- **Result**: Complete scene with accurate "right now" depiction

**Contrast with old 5-message approach**:
- Would only see messages 6-10
- Might miss "park" and "oak tree" setup from early messages
- Could produce generic "outdoor scene" without specific location details

---

## User Experience Flow

### 1. Scene Capture Button

**Location**: Conversation interface, near image gallery button  
**Availability**: Only for characters with `full` or `unbounded` immersion level  
**Reason**: Scene capture makes sense for immersive roleplay, not assistants

### 2. Prompt Generation

**Process**:
1. User clicks "Capture Scene"
2. Backend analyzes last 10 messages
3. LLM generates detailed third-person description
4. Returns prompt preview to user

**Duration**: 2-5 seconds (same as normal generation)

### 3. Confirmation Dialog

**Always Shows**: Cannot be disabled (unlike normal generation)  
**Why**: Scene captures are deliberate, user wants to review

**Dialog Contents**:
- **Scene Description**: Full 100-300 word prompt (editable)
- **Negative Prompt**: What to avoid (editable)
- **Workflow Selector**: Choose which workflow to use
- **Badge**: "Scene Capture" indicator

### 4. Image Generation

**Same as Normal**: Uses ComfyUI, same workflow system  
**Storage**: Saved with special `SCENE_CAPTURE` message role  
**Display**: Shows inline with "Scene Capture" badge on hover

---

## Message Role Handling

### Special Message Type

**Role**: `MessageRole.SCENE_CAPTURE`  
**Content**: Empty string (no text message)  
**Metadata**: Contains image info, prompt, generation details

**Database Schema**:
```python
{
    "image_prompt": "Original user-facing prompt",
    "final_prompt": "Prompt sent to ComfyUI (with trigger word)",
    "negative_prompt": "Negative prompt used",
    "seed": null or int,
    "workflow_id": "workflow_123",
    "status": "generating" | "completed" | "failed",
    "image_id": "img_abc123",
    "image_path": "/images/scene_captures/img_abc123.png",
    "thumbnail_path": "/images/scene_captures/img_abc123_thumb.png",
    "generation_time": 45.2
}
```

### Filtering from Context

**Important**: Scene capture messages are filtered from LLM context  
**Why**: They're visual anchors, not conversational content  
**Implementation**: `prompt_assembly.py` skips `SCENE_CAPTURE` role

```python
if msg.role == MessageRole.SCENE_CAPTURE:
    continue  # Skip scene captures, they're just image anchors
```

---

## Visual Indicators

### Badge System

**Location**: Top-left corner of image  
**Visibility**: Shows on hover  
**Style**: Cyan/info badge with camera icon  
**Text**: "🎥 Scene Capture"

**CSS**:
```css
.scene-capture-badge {
    position: absolute;
    top: 8px;
    left: 8px;
    opacity: 0;
    transition: opacity 0.2s;
    background-color: rgba(13, 202, 240, 0.9);
}

.generated-image-container:hover .scene-capture-badge {
    opacity: 1;
}
```

### Display Format

Scene captures use identical styling to normal images:
- Chat bubble sized to image width
- Hover buttons (fullscreen, set as profile, download, view prompt)
- Generation time indicator
- Same thumbnail/full image system

**Only Difference**: Badge distinguishes scene captures from requested images

---

## Performance Characteristics

### Timing

**Detection**: Instant (button click, no keyword matching needed)  
**Prompt Generation**: 2-5 seconds (LLM call)  
**User Review**: Variable (editing time)  
**Image Generation**: 10-60 seconds (ComfyUI processing)  
**Total**: ~15-70 seconds from button to image

### Resource Usage

**Same as Normal Generation**:
- Single LLM call for prompt
- ComfyUI VRAM usage (~6-10GB)
- Image storage (PNG ~2-5MB)
- Thumbnail generation (~200KB)

**VRAM Optimization**: Before scene capture generation, all LLM models are unloaded to maximize VRAM for ComfyUI.

---

## Configuration

### Character Requirements

**Immersion Level**: Must be `full` or `unbounded`  
**Reason**: Scene capture is for immersive roleplay, not Q&A assistants

**Check**:
```python
if character.immersion_level not in ("full", "unbounded"):
    raise HTTPException(400, "Scene capture only for full or unbounded characters")
```

**Image Generation**: Must be enabled  
**Workflow**: Same workflow system as normal generation

### Workflow Settings

Scene capture uses character's default image workflow:
- Same trigger word
- Same default style
- Same negative prompt
- Same ComfyUI workflow JSON

**No Special Workflow Needed**: Scene capture reuses existing image generation infrastructure.

---

## API Endpoints

### Generate Scene Prompt

**Endpoint**: `POST /threads/{thread_id}/capture-scene-prompt`  
**Purpose**: Generate prompt preview  
**Returns**: Prompt data for confirmation dialog

**Request**: No body (uses thread context)  
**Response**:
```json
{
  "prompt": "detailed scene description",
  "negative_prompt": "things to avoid",
  "needs_trigger": true,
  "type": "scene_capture"
}
```

### Generate Scene Image

**Endpoint**: `POST /threads/{thread_id}/capture-scene`  
**Purpose**: Actually generate the image after confirmation  
**Returns**: Image generation result

**Request**:
```json
{
  "prompt": "edited scene description",
  "negative_prompt": "edited negative prompt",
  "seed": null,
  "workflow_id": "workflow_123"
}
```

**Response**:
```json
{
  "success": true,
  "image_id": "img_abc123",
  "file_path": "/images/scene_captures/img_abc123.png",
  "thumbnail_path": "/images/scene_captures/img_abc123_thumb.png",
  "prompt": "final prompt used",
  "generation_time": 45.2
}
```

---

## Troubleshooting

### Scene Doesn't Match Conversation

**Problem**: Generated scene missing key details or showing wrong actions

**Diagnosis**:
1. Check if relevant details are in last 10 messages
2. Review context synthesis in system prompt
3. Verify LLM is extracting visual details

**Solution**:
- Increase context window (from 10 to 15 messages)
- Add more context examples to system prompt
- Emphasize "synthesize from multiple messages"

### Multiple Participants Not Shown

**Problem**: Only character appears, user/other participants missing

**Diagnosis**: Check if context mentions other participants explicitly

**Solution**:
- Ensure conversation uses "we", "you", "together" language
- Add explicit participant descriptions in context
- Reinforce "describe all visible participants" in prompt

### Wrong Perspective

**Problem**: First-person language slipping into scene description

**Diagnosis**: LLM not following third-person observer rules

**Solution**:
- Strengthen "NEVER use I/me/my" instruction
- Add negative examples showing what NOT to do
- Increase emphasis on "camera view" metaphor

### Scene Too Generic

**Problem**: Prompt lacks specific details from context

**Diagnosis**: Context synthesis not working effectively

**Solution**:
- Review context examples in system prompt
- Add more specific "extract X from context" instructions
- Increase temperature for more creative descriptions

---

## Future Enhancements

### Potential Features

1. **Multiple Angles**: Capture same scene from different perspectives
2. **Time Progression**: Series of scene captures showing progression
3. **Mood Variants**: Same scene with different emotional tones
4. **Participant Focus**: Highlight specific character in multi-person scenes
5. **Scene History**: Timeline view of all captured scenes
6. **Auto-Capture**: Periodically capture key moments automatically

### Advanced Capabilities

- **Camera Controls**: Specify angle, distance, focal length
- **Lighting Presets**: Golden hour, blue hour, studio lighting
- **Composition Rules**: Rule of thirds, leading lines, etc.
- **Scene Templates**: Pre-defined scene types (cafe, park, office)
- **Continuity Check**: Ensure scenes maintain consistent appearance

---

## Related Documentation

- **Normal Image Generation**: `image_generation.md` (this folder)
- **ComfyUI Integration**: `chorus_engine_comfyui_integration_v_1.md`
- **Image Orchestrator**: `chorus_engine/services/image_generation_orchestrator.py`
- **Message Roles**: `chorus_engine/models/conversation.py` (MessageRole enum)
- **Immersion Levels**: `Documentation/Planning/immersion_levels.md`

---

## Comparison Summary

### When to Use Each

**Normal Image Generation**:
- User explicitly requests an image
- Character self-portraits
- Artistic/creative image requests
- Showing objects, places, concepts

**Scene Capture**:
- Visualizing current roleplay scene
- Capturing multi-character interactions
- Creating visual continuity
- Recording conversation progression

### Quick Reference

| Aspect | Normal | Scene Capture |
|--------|--------|---------------|
| Trigger | Keywords | Button |
| POV | Variable | Third-person |
| Focus | Request | Scene state |
| Confirmation | Optional | Always |
| Context | 10 msgs | 10 msgs |
| Temperature | 0.3 | 0.5 |
| Weighting | Natural flow | Explicit last 2-3 |
| Participants | May exclude user | Includes all |
| Message Role | `ASSISTANT` | `SCENE_CAPTURE` |
| Availability | All characters | Unbounded only |

---

## Conclusion

Scene capture provides a powerful tool for visualizing immersive roleplay conversations. By adopting a third-person observer perspective and synthesizing rich context, it creates cinematic snapshots of conversation moments that both character and user can be present in.

The key to effective scene capture is balancing:
- **Context synthesis** (using full conversation history)
- **Present-moment focus** (what's happening now)
- **Observer perspective** (external, omniscient view)
- **Detail richness** (100-300 words of vivid description)

When properly tuned, scene capture transforms text-based roleplay into a visually documented experience, creating memorable moments that users can revisit and share.
