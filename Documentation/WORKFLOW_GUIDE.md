# Chorus Engine Workflow Guide

This guide explains how to create and manage ComfyUI workflows for use with Chorus Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Types](#workflow-types)
3. [Workflow Structure](#workflow-structure)
4. [Image Workflows](#image-workflows)
5. [TTS Workflows](#tts-workflows)
6. [Video Workflows](#video-workflows)
7. [Testing Workflows](#testing-workflows)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Chorus Engine uses a **workflow-first** approach to generation. Instead of guessing which models or settings to use, you explicitly define ComfyUI workflows that specify:

- Which models to use
- Which LoRAs or other modifiers to apply
- How to process inputs
- What outputs to generate

This eliminates cross-model accidents and gives you precise control over generation.

### Workflow-First Design Philosophy

**Benefits**:
- **Explicit Control**: You define exactly what happens during generation
- **No Guessing**: System doesn't try to pick models or settings for you
- **Model Awareness**: Each workflow is built for specific models
- **Reproducibility**: Same workflow produces consistent results
- **Flexibility**: Easy to experiment with different models and settings

**How It Works**:
1. Create a workflow in ComfyUI
2. Add Chorus Engine placeholder nodes
3. Export the workflow as JSON
4. Upload to Chorus Engine
5. Set as default or use on demand

---

## Workflow Types

Chorus Engine supports three types of workflows:

| Type | Purpose | Placeholders |
|------|---------|--------------|
| **Image** | Generate images from text prompts | `__CHORUS_PROMPT__`, `__CHORUS_NEGATIVE__`, `__CHORUS_SEED__` |
| **TTS** | Generate voice audio from text | `__CHORUS_TEXT__`, `__CHORUS_VOICE_SAMPLE__`, `__CHORUS_VOICE_TRANSCRIPT__` |
| **Video** | Generate short videos (2-4 seconds) | `__CHORUS_PROMPT__`, `__CHORUS_NEGATIVE__`, `__CHORUS_SEED__` |

Each workflow type has specific placeholders that Chorus Engine replaces at generation time.

**Note**: All workflow types use the same placeholder injection system. The system automatically searches for placeholders in any text field within the workflow and replaces them with actual values.

---

## Workflow Structure

### File Organization

Workflows are stored in type-specific folders:
```
workflows/
  <character_id>/
    image/
      my_flux_workflow.json
      my_sdxl_workflow.json
    audio/
      default_tts_workflow.json
      custom_voice_workflow.json
    video/
      (future)
```

### Workflow JSON Format

ComfyUI workflows are saved as JSON files containing:
- Node definitions
- Node connections
- Widget values
- Metadata

**Example Structure**:
```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "__CHORUS_REFERENCE_IMAGE__"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "__CHORUS_PROMPT__"
    }
  }
}
```

### Placeholder Replacement

Chorus Engine scans the workflow JSON for placeholder strings and replaces them before sending to ComfyUI:

1. Load workflow JSON
2. Search for placeholder strings
3. Replace with actual values
4. Send to ComfyUI for execution

**Important**: Placeholders are case-sensitive and must match exactly.

---

## Image Workflows

### Overview

Image workflows generate images from text prompts, optional reference images, and character context.

### Supported Placeholders

| Placeholder | Purpose | Example Value | Required? |
|-------------|---------|---------------|----------|
| `__CHORUS_PROMPT__` | The image generation prompt | "A futuristic cityscape at sunset" | Yes |
| `__CHORUS_NEGATIVE__` | Negative prompt (things to avoid) | "blurry, low quality, distorted" | No |
| `__CHORUS_SEED__` | Seed for reproducibility | 42 | No |

**Note**: These placeholders can appear anywhere in any text field in your workflow. The system will find and replace them automatically.

### Creating an Image Workflow

**Step 1: Design in ComfyUI**

1. Open ComfyUI
2. Load or create your workflow
3. Add the models, LoRAs, and settings you want
4. Test the workflow with sample inputs

**Step 2: Add Placeholder Nodes**

1. Replace your prompt text with `__CHORUS_PROMPT__`
   - In **CLIPTextEncode** node, set text to `__CHORUS_PROMPT__`
   - In **any prompt node**, use `__CHORUS_PROMPT__` as the input

2. (Optional) Add reference image support
   - Add a **LoadImage** node
   - Set image path to `__CHORUS_REFERENCE_IMAGE__`
   - Connect to your workflow as needed

3. (Optional) Add character name support
   - Use `__CHORUS_CHARACTER_NAME__` in prompts or conditioning
   - Example: `"Portrait of __CHORUS_CHARACTER_NAME__, __CHORUS_PROMPT__"`

**Step 3: Configure Output**

1. Add a **SaveImage** node at the end of your workflow
2. Set the output path/filename pattern
3. Ensure output format is compatible (PNG recommended)

**Step 4: Export Workflow**

1. In ComfyUI, click **Comfy** button
2. Choose the **File** menu
3. Choose **Export (API)**
4. Save with a descriptive name (e.g., `flux_dev_highres.json`)

**Step 5: Upload to Chorus Engine**

1. Go to **Workflows** tab in web interface
2. Select character
3. Select **Image** as type
4. Upload the JSON file
5. Optionally set as default

### Image Workflow Best Practices

**Model Selection**:
- Use models appropriate for your use case (FLUX for quality, SDXL for speed)
- Test models in ComfyUI before integrating
- Document which models your workflow requires

**LoRA Usage**:
- Scope LoRAs per workflow (don't assume availability)
- Test LoRA weights carefully
- Consider character-specific LoRAs

**Prompt Engineering**:
- Design workflows that enhance prompts (style, quality tags)
- Consider adding negative prompts
- Test with various prompt types

**Performance**:
- Balance quality vs. generation time
- Consider using lower step counts for faster generation
- Test memory usage with your settings

---

## TTS Workflows

### Overview

TTS (Text-to-Speech) workflows generate voice audio from text using voice cloning techniques. They use a voice sample and transcript to replicate a specific voice.

### Required Placeholders

| Placeholder | Purpose | Example Value |
|-------------|---------|---------------|
| `__CHORUS_TEXT__` | The text to synthesize | "Hello, how are you today?" |
| `__CHORUS_VOICE_SAMPLE__` | Path to voice sample audio | `data/voice_samples/character/sample.wav` |
| `__CHORUS_VOICE_TRANSCRIPT__` | Transcript of voice sample | "This is a sample of my voice." |

### Creating a TTS Workflow

**Step 1: Choose a TTS Model**

Common TTS models for ComfyUI:
- **F5-TTS**: High-quality voice cloning, moderate speed
- **Coqui TTS**: Fast, good quality, easy to use
- **XTTS**: Excellent multi-lingual support
- **StyleTTS2**: Natural prosody, emotional control

Install your chosen model's ComfyUI nodes.

**Step 2: Design in ComfyUI**

1. Open ComfyUI
2. Load your TTS model nodes
3. Add nodes for:
   - Loading voice sample audio
   - Processing text input
   - Voice cloning/synthesis
   - Audio output

**Step 3: Add Placeholder Nodes**

1. **Text Input**:
   - In your TTS node's text input, set value to `__CHORUS_TEXT__`
   - Example: TextInput node with `text = "__CHORUS_TEXT__"`

2. **Voice Sample**:
   - In LoadAudio or similar node, set path to `__CHORUS_VOICE_SAMPLE__`
   - Example: LoadAudio node with `audio = "__CHORUS_VOICE_SAMPLE__"`

3. **Voice Transcript** (if supported by model):
   - Some models use transcript for better cloning
   - Set transcript field to `__CHORUS_VOICE_TRANSCRIPT__`
   - Example: VoiceCloning node with `transcript = "__CHORUS_VOICE_TRANSCRIPT__"`

**Step 4: Configure Output**

1. Add a **SaveAudio** node at the end
2. Set output format (WAV recommended, 16-bit, 24kHz+)
3. Ensure output path is writable

**Step 5: Test the Workflow**

1. Replace placeholders with test values:
   - `__CHORUS_TEXT__` → "This is a test."
   - `__CHORUS_VOICE_SAMPLE__` → path to a real audio file
   - `__CHORUS_VOICE_TRANSCRIPT__` → transcript of that audio
2. Run the workflow in ComfyUI
3. Verify audio output quality
4. Adjust settings as needed

**Step 6: Export Workflow**

1. In ComfyUI, restore placeholders (replace test values back)
2. Click **Save** → **Export as JSON**
3. Save with a descriptive name (e.g., `f5tts_voice_clone.json`)

**Step 7: Upload to Chorus Engine**

1. Go to **Workflows** tab in web interface
2. Select character
3. Select **TTS** as type
4. Upload the JSON file
5. Optionally set as default

### TTS Workflow Best Practices

**Voice Sample Quality**:
- Use clean, high-quality audio (24kHz+ sample rate, 16-bit depth)
- 5-15 seconds duration is ideal
- Minimal background noise
- Clear, natural speech
- Match the character's intended voice style

**Transcript Accuracy**:
- Transcribe exactly what is spoken in the voice sample
- Include punctuation for natural pacing
- Don't add or remove words

**Text Processing**:
- Consider adding preprocessing nodes:
  - Remove markdown formatting
  - Expand abbreviations
  - Normalize punctuation
- Test with various text lengths

**Model Settings**:
- Adjust temperature for variation (lower = more consistent)
- Tune repetition penalty to avoid loops
- Test generation speed vs. quality trade-offs

**Output Format**:
- WAV format recommended (lossless, widely compatible)
- Sample rate: 24kHz or higher
- Bit depth: 16-bit minimum
- Mono or stereo depending on use case

---

## Video Workflows

### Overview

Video workflows generate short videos from motion-focused text prompts. Videos emphasize dynamic action, movement, and transitions rather than static composition.

**Key Differences from Images**:
- **Motion-Focused**: Prompts describe what HAPPENS, not just what things look like
- **Longer Timeout**: 600 seconds (vs 300 for images) due to computational complexity
- **Automatic Thumbnails**: First frame automatically extracted as thumbnail
- **Format Detection**: System auto-detects video format (.mp4, .webm, .gif, etc.)

### Supported Placeholders

| Placeholder | Purpose | Example Value | Required? |
|-------------|---------|---------------|----------|
| `__CHORUS_PROMPT__` | Motion-focused video prompt | "Person walks forward, waves, and smiles" | Yes |
| `__CHORUS_NEGATIVE__` | Negative prompt (things to avoid) | "static, still image, frozen" | No |
| `__CHORUS_SEED__` | Seed for reproducibility | 42 | No |

**Note**: Video prompts are automatically optimized for motion. The system generates prompts that emphasize:
- **Action verbs**: walks, waves, turns, gestures
- **Camera movement**: pans, zooms, follows
- **Timing**: begins, then, finally
- **Energy**: gracefully, energetically, smoothly

See [Video Generation Prompts](Prompts/video_generation.md) for detailed prompt engineering guidance.

### Creating a Video Workflow

**Step 1: Design in ComfyUI**

Open ComfyUI and setup your preferred video generation workflow

**Step 2: Add Placeholder Nodes**

1. Replace your positive prompt text with `__CHORUS_PROMPT__`

2. (Optional) Add negative prompt support
   - Use `__CHORUS_NEGATIVE__` in negative prompt text fields
   - Example: "static, still image, frozen, __CHORUS_NEGATIVE__"

3. (Optional) Add seed control
   - Use `__CHORUS_SEED__` in KSampler or noise seed fields
   - Must be in a numeric field (not text)

**Step 3: Configure Output**

1. Add a **SaveVideo** or **VHSVideoCombine** node (or equivalent)
2. Set output format (webp or MP4 recommended)

**Step 4: Test with Motion Prompts**

1. Replace `__CHORUS_PROMPT__` with a test motion prompt:
   - Good: "cat walks across room, pauses, then sits down"
   - Bad: "portrait of a cat sitting" (too static)

2. Run the workflow in ComfyUI
3. Verify video shows actual motion
4. Adjust settings if needed (frame count, FPS, steps)
5. Restore placeholders before export

**Step 5: Export Workflow**

1. In ComfyUI, click **Save** → **Export (API Format)**
2. Save with descriptive name (e.g., `wan22_motion.json`)

**Step 6: Upload to Chorus Engine**

1. Go to **Workflows** tab in web interface
2. Select character
3. Select **Video** as type
4. Upload the JSON file
5. Optionally set as default

### Video Scene Capture

Video workflows also support scene capture (🎥 button in UI):
- Captures current conversation moment as short video
- Emphasizes gestures, expressions, and movement
- Uses third-person observer perspective
- Prompts automatically focus on motion and action

See [Scene Capture Guide](Prompts/scene_capture.md) for details on video vs. image scene capture.

---

## Testing Workflows

### Testing in ComfyUI

**Before Uploading**:
1. Test the workflow thoroughly in ComfyUI
2. Verify all nodes execute without errors
3. Check output quality meets your standards
4. Test with various inputs (different prompts, lengths, etc.)

**Placeholder Testing**:
1. Replace placeholders with realistic test values
2. Run the workflow
3. Verify output is correct
4. Restore placeholders before export

### Testing in Chorus Engine

**After Uploading**:
1. Upload the workflow to Chorus Engine
2. Set it as the default (or use it explicitly)
3. Generate content using the workflow
4. Check `server_log_fixed.txt` for errors
5. Verify output quality and correctness

**Iteration**:
1. If issues found, return to ComfyUI
2. Adjust workflow settings
3. Re-export and re-upload
4. Test again

### Common Issues During Testing

**Placeholder Not Replaced**:
- Check spelling is exact (case-sensitive)
- Ensure placeholder is in a text field, not a dropdown
- Verify workflow JSON contains the placeholder string

**ComfyUI Execution Fails**:
- Check ComfyUI logs for errors
- Verify all required nodes/models are installed
- Test the workflow manually in ComfyUI first

**Output Not Saved**:
- Verify SaveImage/SaveAudio node is present
- Check output path is writable
- Ensure output format is supported

**Poor Quality Output**:
- Adjust model settings (steps, CFG, temperature, etc.)
- Try different models
- Improve input quality (prompts, voice samples, etc.)

---

## Troubleshooting

### Workflow Upload Issues

**Upload Fails**:
- Check JSON is valid (use a JSON validator)
- Ensure file size is reasonable (< 10 MB typically)
- Verify character exists in system
- Check `server_log_fixed.txt` for error details

**Workflow Not Found**:
- Verify workflow name matches file name (without .json extension)
- Check workflow is in correct folder (`workflows/<character>/<type>/`)
- Ensure workflow type is correct (image/audio/video)

### Generation Issues

**Workflow Fails to Execute**:
- Verify ComfyUI is running and accessible
- Check all required models/nodes are installed in ComfyUI
- Look for errors in ComfyUI console
- Check `server_log_fixed.txt` for Chorus Engine errors

**Placeholders Not Replaced**:
- Confirm placeholders are spelled correctly (case-sensitive)
- Check placeholders are in appropriate node fields
- Verify Chorus Engine is passing values correctly (check logs)

**Wrong Workflow Used**:
- Check character's YAML configuration for default workflow
- Verify workflow name matches configuration
- Ensure workflow is set as default in database

### Quality Issues

**Poor Image Quality**:
- Increase sampling steps
- Adjust CFG scale
- Try a different model or LoRA
- Improve prompt quality

**Poor Audio Quality**:
- Use higher quality voice sample
- Adjust TTS model settings (temperature, repetition penalty)
- Try a different TTS model
- Ensure voice sample transcript is accurate

### Performance Issues

**Slow Generation**:
- Reduce sampling steps (images)
- Use faster models
- Check GPU utilization in ComfyUI
- Consider smaller image sizes or shorter audio

**ComfyUI Out of Memory**:
- Reduce batch sizes
- Use smaller models
- Lower resolution/quality settings
- Restart ComfyUI to clear VRAM

---

## Advanced Topics

### Dynamic Workflows

**Conditional Generation**:
- Use ComfyUI's conditional nodes
- Switch between different model paths
- Adapt workflow based on input

**Multi-Stage Pipelines**:
- Combine multiple models in sequence
- Upscale or refine initial outputs
- Apply post-processing effects

### Workflow Versioning

**Best Practices**:
- Use descriptive workflow names with versions: `flux_dev_v2.json`
- Keep old versions for comparison
- Document changes in workflow descriptions
- Test new versions before setting as default

### Custom Nodes

**Using Custom ComfyUI Nodes**:
- Install required custom nodes in ComfyUI
- Document dependencies in workflow description
- Share installation instructions with users
- Test compatibility across ComfyUI versions

---

## Resources

### ComfyUI Resources

- **ComfyUI GitHub**: https://github.com/comfyanonymous/ComfyUI
- **ComfyUI Manager**: Plugin for installing custom nodes
- **Community Workflows**: ComfyUI forums and Discord

### Model Resources

**Image Models**:
- Hugging Face: https://huggingface.co/models
- Civitai: https://civitai.com/
- FLUX Models: https://github.com/black-forest-labs/flux

**TTS Models**:
- F5-TTS: https://github.com/SWivid/F5-TTS
- Coqui TTS: https://github.com/coqui-ai/TTS
- XTTS: https://huggingface.co/coqui/XTTS-v2

### Chorus Engine Resources

- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **Getting Started**: [GETTING_STARTED.md](../GETTING_STARTED.md)
- **API Documentation**: See `chorus_engine/api/app.py` for endpoint details

---

**Last Updated**: Phase 7 Complete (Video Generation) - January 7, 2026
