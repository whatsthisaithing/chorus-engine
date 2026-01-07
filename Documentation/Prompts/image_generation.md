# Image Generation Prompt Specification

**Version**: 1.1  
**Date**: January 7, 2026  
**Component**: `ImagePromptService`  
**File Location**: `chorus_engine/services/image_prompt_service.py`

---

## Overview

The image generation prompt instructs the LLM to create detailed, vivid image descriptions for Stable Diffusion/ComfyUI based on user requests and conversation context. This transforms casual user requests like "show me a sunset" into 100-300 word professional image prompts.

**Note**: For motion-based generation, see [Video Generation Prompts](video_generation.md). Images are static; videos emphasize dynamic action and movement.

## Purpose

Image prompt generation serves multiple functions:

1. **Request Translation**: Convert natural language to technical image prompts
2. **Context Integration**: Extract visual details from conversation history
3. **Style Consistency**: Apply character-specific artistic styles
4. **Detail Enhancement**: Add technical specifications (lighting, composition, atmosphere)
5. **Character Depiction**: Ensure character appears at current age/appearance

**Image vs. Video**: Images focus on static composition, lighting, and atmosphere. For dynamic action and movement, use [video generation](video_generation.md) instead.

## Architecture

### Generation Flow

```
1. User sends message with image request
   ↓
2. Keyword detection (fast check)
   ↓
3. If detected: prepare prompt preview
   ↓
4. LLM generates detailed image prompt
   ↓
5. User confirms (or edits prompt)
   ↓
6. Inject into ComfyUI workflow
   ↓
7. Image generates, displays inline
```

### Where It's Used

- **Service**: `ImagePromptService.generate_prompt()`
- **Orchestrator**: `ImageGenerationOrchestrator.detect_and_prepare()`
- **API Endpoint**: `POST /threads/{thread_id}/detect-image-request`
- **Model**: Uses character's preferred LLM model

### Processing Mode

**Synchronous** (user waits for prompt preview):
- Detection: <100ms (keyword matching)
- Prompt generation: 2-5 seconds (LLM call)
- User confirms before image generation
- Can edit prompt before submission

---

## Request Detection

### Keyword-Based Detection

**Method**: `detect_image_request()` - Line ~52  
**Strategy**: Fast keyword matching (no LLM call)

**Keywords**:
```python
image_keywords = [
    "show me",
    "can you show",
    "picture",
    "image",
    "photo",
    "draw",
    "generate",
    "create an image",
    "what do you look like",
    "what does",
    "look like",
    "appearance",
    "visualize",
    "illustrate"
]
```

**Logic**:
```python
message_lower = message.lower()
for keyword in image_keywords:
    if keyword in message_lower:
        return True
return False
```

**Performance**: O(n) where n = number of keywords (~15)  
**False Positives**: Rare but possible ("I like pictures of cats" might trigger)  
**False Negatives**: Unusual phrasings might miss ("depict X", "render Y")

### Why Not LLM Detection?

**Speed**: Keyword matching is instant (<1ms)  
**Cost**: No API call needed  
**Reliability**: Deterministic, no hallucination risk  
**User Control**: Can always skip even if detected

---

## Prompt System Template

### Template Location

**Method**: `_build_system_prompt()` - Line ~178  
**File**: `chorus_engine/services/image_prompt_service.py`

### Prompt Structure

1. **Role Definition**: Expert image prompt creator
2. **Character Context**: Who the character is, their appearance
3. **Image Settings**: Style, trigger word, negative prompt
4. **Context Usage Rules**: How to leverage conversation history
5. **Character Depiction Rules**: Critical guidance on depicting the character
6. **Context Examples**: Specific scenarios showing proper usage
7. **Detail Requirements**: Length, specificity, technical terms
8. **Output Format**: Strict JSON schema

---

## Full System Prompt Template

```
You are an expert at creating detailed, vivid image generation prompts for Stable Diffusion/ComfyUI.

Character: {character.name}
Character Description: {character.role}

Image Generation Settings:
- Default Style: {default_style or 'Not specified'}
- Character Appearance: {self_description or 'Not specified'}
- Default Negative Prompt: {negative_prompt or 'Not specified'}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt. 
It will be added automatically when needed.

Your task is to generate a HIGHLY DETAILED, descriptive image prompt based on the user's request.

CONTEXT USAGE:
You will receive recent conversation context. Use it intelligently to enhance your image prompt:
- EXTRACT visual details mentioned in the context (scenes, objects, settings, descriptions, stories)
- UNDERSTAND references ("that moment", "what you described", "from the story", "your favorite")
- FOCUS on visually describable elements (ignore conversational mechanics and meta-discussion)
- SYNTHESIZE details from multiple messages if the user references earlier discussion
- If the user says "show me X from [earlier topic]", pull ALL relevant visual details from context
- When a story or description was shared, extract specific visual elements (locations, objects, people, actions, atmosphere)

CRITICAL - CHARACTER DEPICTION RULES:
When the user requests an image involving the character (you/yourself):
- ALWAYS depict the character at their CURRENT age and appearance (use Character Appearance description above)
- Even if the request references a past memory/story, show the character NOW unless explicitly asked for a historical depiction
- Example: "show me that moment from your childhood story" → Show current character in a scene inspired by the story, NOT as a child
- Example: "photo of you eating ramen" → Show current character appearance eating ramen
- Only depict the character at a different age/appearance if explicitly requested: "show yourself as a child in that scene"
- When in doubt, default to current character appearance

Context Usage Examples:
1. Context: "ASSISTANT: I remember climbing the old oak tree as a child and watching the sunset..."
   Request: "Show me that sunset moment"
   → Show CURRENT character sitting in/near an oak tree at sunset, capturing the nostalgic mood (NOT a child)
   
2. Context: "USER: Tell me about your favorite place / ASSISTANT: My favorite place is a hidden beach with black sand..."
   Request: "Can you show me that beach?"
   → Extract: black sand beach, turquoise water, volcanic cliffs, hidden/secluded atmosphere

3. Context shows NO relevant visual details
   Request: "Draw a cat"
   → Use general knowledge and character style, no context needed

DETAIL REQUIREMENTS:
- Write 100-300 words of rich, specific description
- Include fine details: textures, colors, lighting quality, atmosphere
- Describe composition, perspective, and framing
- Add environmental details and mood
- If depicting the character, describe their appearance, pose, expression, clothing, and surroundings
- Use evocative, visual language that paints a clear picture
- Include artistic style keywords and technical photography/art terms

Example of detail level:
Bad: "woman in a garden"
Good: "A young woman with flowing auburn hair stands in an enchanted garden at golden hour, soft sunlight filtering through ancient oak trees and casting dappled shadows across her white linen dress. She holds a leather-bound book, her expression thoughtful and serene. Wildflowers in vibrant purples and yellows surround her feet, while butterflies dance in the warm, hazy air. The background shows a stone archway covered in climbing roses, slightly out of focus. Painted in the style of Pre-Raphaelite art, with rich colors, intricate details, and romantic lighting. Shot with shallow depth of field, 85mm lens, soft bokeh."

Return ONLY valid JSON in this format:
{
  "prompt": "extremely detailed 100-300 word image description with comprehensive visual details, style, lighting, composition, atmosphere, and technical specifications",
  "negative_prompt": "things to avoid (or use character default)",
  "reasoning": "brief 1-2 sentence explanation of the image concept"
}

Remember: More detail = better results. Be specific, visual, and evocative!
```

---

## Key Design Decisions

### 1. Context-Aware Generation

**Philosophy**: User shouldn't need to repeat details  
**Implementation**: Pass recent conversation history (last 5 messages)  
**Benefit**: Richer, more contextual images

**Example**:
```
Conversation:
USER: Tell me about your favorite place
ASSISTANT: My favorite place is a hidden beach with black sand and volcanic cliffs...

Later:
USER: Can you show me that place?
→ Prompt extracts: black sand, volcanic cliffs, hidden beach, turquoise water
```

### 2. Character Depiction Rules

**Problem**: Users reference past stories but expect current character appearance  
**Solution**: Explicit rules about when to show current vs historical appearance

**Critical Rule**:
> "ALWAYS depict the character at their CURRENT age and appearance"

**Example**:
```
Context: "I remember my childhood climbing trees..."
Request: "Show me that tree moment"
❌ WRONG: Child climbing tree
✅ RIGHT: Current character near oak tree at sunset (nostalgic mood)
```

**Override**: Only show different age if explicitly requested:
```
Request: "Show yourself AS A CHILD in that scene"
✅ RIGHT: Now okay to depict as child
```

### 3. Trigger Word Handling

**Concept**: Some models use trigger words to activate character LoRAs/embeddings

**Configuration**:
```yaml
image_generation:
  trigger_word: "N0VA"  # Specific to character's model training
```

**Automatic Injection**:
- Service detects when character is in image
- Prepends trigger word to prompt
- User never sees/manages trigger word

**Detection Logic** (`_should_include_trigger()` - Line ~279):
```
Check if prompt contains:
- "you" / "your" / "yourself"
- Character's name
- "I" / "my" / "me" (character referring to self)

If yes → needs_trigger = True
```

### 4. Detail Enhancement

**Length Requirement**: 100-300 words (enforced in prompt)  
**Why Long**: Stable Diffusion benefits from detailed descriptions  
**Components**: Composition + lighting + atmosphere + style + technical specs

**Template Structure**:
```
[Subject] + [Environment] + [Lighting] + [Mood] + [Style] + [Technical Details]
```

**Example Breakdown**:
- **Subject**: "young woman with flowing auburn hair"
- **Environment**: "enchanted garden with ancient oak trees"
- **Lighting**: "golden hour, soft sunlight filtering through, dappled shadows"
- **Mood**: "thoughtful and serene, warm, romantic"
- **Style**: "Pre-Raphaelite art, rich colors, intricate details"
- **Technical**: "shallow depth of field, 85mm lens, soft bokeh"

---

## Context Building

### Message Format

**Method**: `_build_context_string()` - Line ~253

**Input**: Recent conversation messages (list of dicts)  
**Output**: Formatted text block

**Format**:
```
ASSISTANT: I love visiting the old lighthouse on foggy mornings
USER: That sounds beautiful
ASSISTANT: The way the fog rolls in creates this ethereal atmosphere
USER: Can you show me that?
```

**Limit**: Last 5 messages (configurable)  
**Purpose**: Provide visual context without overwhelming prompt

### Context Extraction Logic

**LLM Task**: Extract visually relevant details from context

**What to Extract**:
- Scenes described in conversation
- Objects, locations, settings mentioned
- Descriptive language (colors, textures, moods)
- Story elements with visual components
- References to earlier discussion

**What to Ignore**:
- Conversational mechanics ("Let me explain...")
- Meta-discussion ("That's a good question")
- Non-visual information (feelings without visual cues)

---

## Negative Prompts

### Purpose

Negative prompts tell Stable Diffusion what NOT to generate.

### Sources

1. **Character Default**: From `character.image_generation.negative_prompt`
2. **LLM Generated**: Can suggest specific negatives for the image
3. **Fallback**: Use character default if LLM doesn't provide

**Common Negative Prompts**:
```
"ugly, blurry, low quality, distorted, deformed, bad anatomy, 
extra limbs, duplicate, watermark, text, signature, 
poorly drawn, amateur, sketch, draft"
```

### Application

```python
if not result.get("negative_prompt") and character.image_generation.negative_prompt:
    result["negative_prompt"] = character.image_generation.negative_prompt
```

---

## Style Integration

### Default Style

**Configuration**: `character.image_generation.default_style`

**Examples**:
- "photorealistic portrait, cinematic lighting"
- "anime style, vibrant colors, Studio Ghibli inspired"
- "oil painting, impressionist style, soft brush strokes"
- "digital art, fantasy concept art, detailed"

### Style Application

**Method**: LLM automatically incorporates default style into prompt  
**Override**: User can edit prompt before generation to change style  
**Consistency**: Same character uses same style by default

---

## JSON Output Format

### Schema

```json
{
  "prompt": "string (100-300 words, highly detailed)",
  "negative_prompt": "string (optional, uses character default if missing)",
  "reasoning": "string (1-2 sentences explaining concept)"
}
```

### Example Output

```json
{
  "prompt": "A mysterious lighthouse standing tall on rocky cliffs during a foggy morning, wrapped in swirling mist that creates an ethereal, dreamlike atmosphere. The ancient stone structure features a weathered facade with climbing ivy and peeling white paint, its beacon barely visible through the dense fog. Waves crash against the jagged rocks below, sending spray into the misty air. The scene is captured in moody, desaturated colors with shades of blue-gray and soft whites. Early morning light filters through the fog, creating volumetric light rays. Scattered seabirds circle the lighthouse. The composition uses a low angle looking up at the lighthouse, emphasizing its imposing presence. Photorealistic style with cinematic lighting, shot with a wide-angle lens, shallow depth of field, atmospheric perspective, highly detailed textures on the stone and weathered paint.",
  "negative_prompt": "sunny, clear sky, bright colors, urban setting, modern architecture",
  "reasoning": "Capturing the ethereal, moody atmosphere of a fog-shrouded coastal lighthouse based on the character's description of their favorite place"
}
```

---

## LLM Parameters

### Model Selection

**Default**: Uses character's preferred model  
**Reason**: Consistency in prompt quality across character interactions  
**Override**: Can specify different model if needed

**Example**:
- Nova (qwen2.5:14b-instruct) generates with qwen2.5
- Sarah (dolphin-mistral-nemo:12b) generates with dolphin-mistral-nemo

### Temperature

**Setting**: 0.8 (configurable, default from config)  
**Reason**: Creative enough for vivid descriptions  
**Trade-off**: Some variability in prompt style (beneficial for images)

### Max Tokens

**Setting**: No strict limit (prompt generation typically 200-400 tokens)  
**Output**: 100-300 words = ~130-390 tokens roughly

---

## Workflow Integration

### Placeholder Injection

**Workflow Manager**: Injects prompt into ComfyUI workflow JSON

**Placeholders**:
- `__CHORUS_PROMPT__` → Generated positive prompt (+ trigger word if needed)
- `__CHORUS_NEGATIVE__` → Negative prompt
- `__CHORUS_SEED__` → Random seed or user-specified

**Trigger Word Prepending**:
```python
if needs_trigger and character.image_generation.trigger_word:
    final_prompt = f"{character.image_generation.trigger_word}, {prompt}"
```

### ComfyUI Submission

**Flow**: Prompt → Workflow → ComfyUI API → Image  
**Format**: JSON workflow with nodes and connections  
**Result**: PNG image saved to `data/images/`

---

## Character Configuration

### YAML Setup

```yaml
image_generation:
  enabled: true
  workflow_file: "default_workflow.json"
  default_style: "photorealistic portrait, cinematic lighting, detailed"
  negative_prompt: "ugly, distorted, low quality, blurry, amateur"
  trigger_word: "N0VA"  # Optional, for LoRA/embedding
  self_description: |
    Nova appears as a young woman in her mid-twenties with striking features.
    She has long, dark hair often styled in loose waves, and bright, expressive eyes.
    Her style is modern and casual, often wearing comfortable yet stylish clothing.
    She has a warm, approachable demeanor that reflects in her relaxed posture and gentle smile.
```

### Self-Description Usage

**Purpose**: Ensure consistent character depiction across images  
**Application**: LLM references this when character appears in image  
**Benefit**: Character looks the same in all images

**Example**:
```
Request: "Show me eating ramen"
→ Prompt includes: "Nova, a young woman in her mid-twenties with dark wavy hair and bright eyes, eating ramen at a cozy restaurant, warm lighting..."
```

---

## User Confirmation Flow

### Confirmation Modal

**Trigger**: After prompt generation, before ComfyUI submission  
**Purpose**: Allow prompt editing, prevent unwanted generations

**Modal Contents**:
1. **Generated Prompt**: Full 100-300 word description (editable textarea)
2. **Negative Prompt**: What to avoid (editable)
3. **Seed**: Optional random seed for reproducibility
4. **Buttons**: Generate / Edit / Cancel
5. **Checkbox**: "Don't ask me again for this conversation"

### Skip Confirmation

**Setting**: Per-conversation preference  
**Storage**: `conversation.image_confirmation_disabled = "true"`  
**Behavior**: Immediately generate without showing modal

**Use Case**: User trusts character's prompt generation, wants faster workflow

---

## Performance Characteristics

### Timing

**Detection**: <100ms (keyword matching)  
**Prompt Generation**: 2-5 seconds (LLM call)  
**Total Preview**: 2-5 seconds before user confirmation  
**Image Generation**: 10-60 seconds (separate, after confirmation)

### Caching

**No Caching**: Each request generates fresh prompt  
**Reason**: Context changes, user intent varies  
**Trade-off**: Slight delay but more accurate prompts

---

## Editing the Prompt Template

### Location

**File**: `chorus_engine/services/image_prompt_service.py`  
**Method**: `_build_system_prompt()` - Line ~178

### When to Edit

**Add Context Examples**:
```python
prompt = f"""...
Context Usage Examples:
1. [Your new example]
2. [Another example]
...
"""
```

**Change Detail Requirements**:
```python
DETAIL REQUIREMENTS:
- Write 150-400 words...  # Changed from 100-300
- Add camera settings...  # New requirement
```

**Adjust Character Rules**:
```python
CRITICAL - CHARACTER DEPICTION RULES:
- Always show character smiling (unless context suggests otherwise)
- Include character's signature accessory
```

**Modify Style Guidance**:
```python
- Include artistic style keywords and technical photography/art terms
- Prefer natural lighting over artificial
- Use wide-angle compositions for landscapes
```

### Testing Changes

After editing:

1. **Test with context**: Verify context extraction works
2. **Test character depiction**: Ensure current age rule works
3. **Test style consistency**: Check default style applies
4. **Test detail level**: Verify 100-300 word output
5. **Validate JSON**: Ensure proper format

---

## Integration Points

### 1. Image Orchestrator

**File**: `chorus_engine/services/image_generation_orchestrator.py`  
**Method**: `detect_and_prepare()` - Line ~67

Calls prompt service:
```python
result = await self.prompt_service.generate_prompt(
    user_request=message,
    character=character,
    conversation_context=conversation_context,
    model=model
)
```

### 2. API Endpoint

**File**: `chorus_engine/api/app.py`  
**Endpoint**: `POST /threads/{thread_id}/detect-image-request` - Line ~2017

Detects and generates prompt preview:
```python
result = await orchestrator.detect_and_prepare(
    message=request.message,
    character=character,
    conversation_context=context,
    model=model
)
```

### 3. Workflow Manager

**File**: `chorus_engine/services/workflow_manager.py`  
**Method**: `inject_prompt()` - Line ~140

Injects prompt into workflow JSON:
```python
workflow = self.workflow_manager.inject_prompt(
    workflow_data=workflow,
    positive_prompt=prompt,
    negative_prompt=negative_prompt,
    seed=seed
)
```

### 4. UI Confirmation Modal

**File**: `web/js/app.js`  
**Function**: `showImageConfirmationModal()` - Line ~700

Displays prompt for editing:
```javascript
document.getElementById('imagePromptPreview').value = promptData.prompt;
document.getElementById('imageNegativePrompt').value = promptData.negative_prompt;
```

---

## Troubleshooting

### No Detection

**Check**:
1. Is `image_generation.enabled: true` in character YAML?
2. Does message contain keywords?
3. Is ComfyUI running and configured?

**Fix**: Add more keywords or adjust detection logic

### Poor Prompts

**Diagnosis**:
1. Check context quality (is relevant info in recent messages?)
2. Review LLM temperature (too high = too creative?)
3. Check character's default style (conflicting with request?)

**Fix**:
1. Add more context examples to prompt template
2. Adjust temperature (lower = more consistent)
3. Clarify style integration logic

### Wrong Character Appearance

**Problem**: Character shown at wrong age or appearance

**Fix**: Reinforce CHARACTER DEPICTION RULES:
```
CRITICAL - CHARACTER DEPICTION RULES:
- ALWAYS use Character Appearance description
- NEVER depict as child unless explicitly requested
- Default to CURRENT appearance
```

### JSON Parsing Fails

**Cause**: LLM added markdown or text outside JSON

**Solution**: `_parse_llm_response()` handles cleanup:
```python
def _parse_llm_response(self, content: str) -> Dict[str, Any]:
    # Try direct JSON parse
    try:
        return json.loads(content)
    except:
        # Extract from markdown
        # Search for {...} pattern
        # Fall back to error
```

---

## Future Enhancements

### Potential Improvements

1. **Multi-Image Requests**: "Show me 3 variations of this scene"
2. **ControlNet Integration**: Use reference images for pose/composition
3. **Iterative Refinement**: "Make it more colorful", "Add a cat"
4. **Style Transfer**: Apply different artists' styles
5. **Prompt Templates**: Pre-defined templates for common requests
6. **Aspect Ratio Control**: Landscape, portrait, square options
7. **Quality Presets**: Quick/normal/detailed generation modes

### Advanced Features

- **Image-to-Image**: Start with uploaded image, modify it
- **Inpainting**: Edit specific regions of generated images
- **Upscaling**: Enhance resolution after generation
- **Batch Generation**: Multiple seeds, pick best result
- **LoRA Management**: Load/unload LoRAs for different styles

---

## Related Documentation

- **ComfyUI Integration**: `chorus_engine_comfyui_integration_v_1.md`
- **Image Generation**: `Documentation/Development/PHASE_5_COMPLETE.md`
- **Workflow Management**: `chorus_engine/services/workflow_manager.py`
- **Character Schema**: `Documentation/Planning/chorus_engine_character_schema_v_1.md`
