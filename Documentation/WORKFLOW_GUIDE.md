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
| **Image** | Generate images from text prompts | `__CHORUS_PROMPT__`, `__CHORUS_REFERENCE_IMAGE__`, `__CHORUS_CHARACTER_NAME__` |
| **TTS** | Generate voice audio from text | `__CHORUS_TEXT__`, `__CHORUS_VOICE_SAMPLE__`, `__CHORUS_VOICE_TRANSCRIPT__` |
| **Video** | Generate videos (future) | TBD |

Each workflow type has specific placeholders that Chorus Engine replaces at generation time.

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

### Required Placeholders

| Placeholder | Purpose | Example Value |
|-------------|---------|---------------|
| `__CHORUS_PROMPT__` | The image generation prompt | "A futuristic cityscape at sunset" |
| `__CHORUS_REFERENCE_IMAGE__` | Optional reference image path | `data/images/character/reference.png` |
| `__CHORUS_CHARACTER_NAME__` | Character name | "Nova" |

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

### Status

Video workflow support is planned for a future phase. This section will be updated when video generation is implemented.

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

**Last Updated**: Phase 6 Complete (TTS Integration)
