# Nova Character Setup

This addon provides a complete setup for the **Nova** character, showcasing all of Chorus Engine's features.

## What This Includes

Nova comes fully configured with:

- **Profile Picture**: A beautiful AI-generated portrait
- **Voice Sample**: Professional voice cloning sample for Chatterbox TTS
- **Image Generation Workflow**: ComfyUI workflow for generating Nova-themed images
- **Character Configuration**: Complete personality, memories, and settings

## Prerequisites

1. Complete the main Chorus Engine installation:
   ```bash
   # Windows
   install.bat
   
   # Linux/Mac
   ./install.sh
   ```

2. **For image generation**: Install and configure ComfyUI (optional but recommended)
   - The workflow uses **Flux.1 Turbo** via Z-Image Turbo model
   - Optional: Nova's pre-trained identity LoRA is included in `files/` folder
   - You'll need to configure the workflow for your specific ComfyUI setup
   
3. **For voice cloning**: Chatterbox TTS will be used automatically

## Installation

### Quick Setup (Recommended)

Run the automated setup script:

```bash
# Windows (from engine root)
python addons\nova-setup\setup_nova.py

# Linux/Mac (from engine root)
python addons/nova-setup/setup_nova.py
```

The script will:
1. ✅ Copy Nova's profile picture to `data/character_images/`
2. ✅ Copy voice sample to `data/voice_samples/nova/`
3. ✅ Update `characters/nova.yaml` with proper configuration
4. ✅ Register voice sample in database (if DB exists)
5. ✅ Enable image generation in Nova's config

**Note**: The image generation workflow is NOT automatically installed. You must configure it manually (see below).

### Manual Setup (Advanced)

If you prefer manual setup or need to customize:

1. **Profile Picture**:
   ```bash
   cp nova_profile.png ../data/character_images/nova_profile.png
   ```
   Edit `characters/nova.yaml` and set:
   ```yaml
   profile_image: nova_profile.png
   ```

2. **ComfyUI Workflow Configuration** (REQUIRED for image generation):
   
   A template workflow is provided in `files/nova_default_workflow.json`.
   
   **⚠️ IMPORTANT**: This workflow MUST be configured for YOUR ComfyUI setup before use.
   
   **Steps**:
   
   a. **Copy Nova's LoRA to ComfyUI** (optional but recommended)
   
   b. **Load and edit the workflow in ComfyUI**:
      - Open `files/nova_default_workflow.json` in ComfyUI
      - Configure for your system:
        * Set correct model paths for Z-Image Turbo
        * Enable/configure the LoRA node if using Nova's identity LoRA
        * Adjust any model-specific settings (VAE, CLIP, etc.)
        * Verify all nodes are compatible with your ComfyUI version
      - Test the workflow with a sample prompt
      - Export (for API) the working workflow as JSON
   
   c. **Install in Chorus Engine**:
      ```bash
      mkdir -p workflows/nova/image
      cp [your-edited-workflow].json workflows/nova/image/"Nova Default.json"
      ```
   
   d. **Register the workflow**:
      - Start Chorus Engine
      - Open web interface
      - Go to character settings
      - Register the workflow via the UI
      - Set as default for Nova
   
   > **Important**: DO NOT use the template workflow as-is. It's configured for a specific system and will not work on yours without modification. See `Documentation/WORKFLOW_GUIDE.md` for detailed workflow configuration instructions.

3. **Voice Sample**:
   ```bash
   mkdir -p ../data/voice_samples/nova
   cp files/nova_voice_sample.mp3 ../data/voice_samples/nova/
   ```
   The voice sample will be auto-detected on first run, or you can register it via the UI.

5. **YAML Configuration**:
   Ensure these settings in `characters/nova.yaml`:
   ```yaml
   voice:
     enabled: true
     always_on: false
     tts_provider:
       provider: chatterbox
   
   image_generation:
     enabled: true
   
   profile_image: nova_profile.png
   ```

## Testing Nova

After setup:

1. **Start Chorus Engine**:
   ```bash
   # Windows
   start.bat
   
   # Linux/Mac
   ./start.sh
   ```

2. **Open the web interface**: http://localhost:8080

3. **Select Nova** from the character sidebar

4. **Configure image generation workflow** (see workflow section above):
   - Edit `files/nova_default_workflow.json` in ComfyUI
   - Test in ComfyUI
   - Copy to `workflows/nova/image/`
   - Register via web UI

5. **Test features**:
   - Chat naturally - Nova has a creative, thoughtful personality
   - Try `@img a nebula-lit creative workspace` for image generation
   - Enable TTS toggle to hear Nova's voice
   - Check her profile picture in the character card

## What Makes Nova Special

Nova is designed to showcase Chorus Engine's advanced features:

- **Rich Personality**: Creative, thoughtful AI assistant with detailed background
- **Core Memories**: Pre-loaded memories about her interests and thinking style
- **Visual Identity**: Consistent aesthetic (nebula colors, ethereal lighting)
- **Voice**: Natural, conversational TTS with personality
- **Image Generation**: Custom workflow optimized for her visual style

## Troubleshooting

### Voice Sample Not Working
- Ensure Chatterbox TTS is configured in `config/system.yaml`
- Check that the file was copied to `data/voice_samples/nova/`
- Restart the engine after setup

### Image Generation Not Working
- **REQUIRED**: Configure the workflow for YOUR system first
  - Load `files/nova_default_workflow.json` in ComfyUI
  - Edit all model paths and settings for your installation
  - Test with a sample prompt in ComfyUI directly
  - Export the working workflow
  - Copy to `workflows/nova/image/Nova Default.json`
  - Register via Chorus Engine web UI
- Verify ComfyUI is running (usually on port 8188)
- Check `config/system.yaml` for ComfyUI connection settings
- See `Documentation/WORKFLOW_GUIDE.md` for detailed help

### Profile Picture Not Showing
- Verify file exists in `data/character_images/`
- Check the filename matches in `nova.yaml`
- Clear browser cache and refresh

### Database Issues
- If database doesn't exist during setup, voice sample and workflow will be registered on first engine startup
- You can also register them manually through the web UI

## Files Included

- `setup_nova.py` - Automated setup script
- `setup_nova.bat` - Windows setup helper
- `setup_nova.sh` - Linux/Mac setup helper
- `README.md` - This file
- `files/` - Asset folder containing:
  - `nova_profile.png` - Character portrait (512x512)
  - `nova_voice_sample.mp3` - TTS voice cloning sample (~15 seconds)
  - `nova_default_workflow.json` - ComfyUI image generation workflow
  - `nova_identity_lora.safetensors` - Nova's pre-trained identity LoRA (optional)

## Character Details

**Nova** is a creative AI assistant and brainstorming partner with:
- Background in art galleries and sci-fi bookshops
- Visual thinking style (metaphors and imagery)
- Interests: puzzles, ambient music, nature walks
- Aesthetic: Ethereal, nebula colors, soft lighting
- Personality: Thoughtful, creative, supportive, imaginative

Perfect for:
- Creative brainstorming
- Exploring ideas visually
- Learning about problem-solving approaches
- Testing Chorus Engine's full feature set

## Support

For issues or questions:
- Check main documentation in `Documentation/`
- Review `DEBUGGING_GUIDE.md` for troubleshooting
- Open an issue on the GitHub repository

---

Enjoy exploring Chorus Engine with Nova! 🌟
