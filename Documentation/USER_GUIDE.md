# Chorus Engine User Guide

This guide covers how to use Chorus Engine's features as an end user.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Model Management](#model-management)
3. [Character Cards](#character-cards)
4. [Text Conversations](#text-conversations)
5. [Image Generation](#image-generation)
6. [Scene Capture & Image Gallery](#scene-capture--image-gallery)
7. [Voice Generation (TTS)](#voice-generation-tts)
8. [Memory Management](#memory-management)
9. [Ambient Activities](#ambient-activities)
10. [Workflows](#workflows)
11. [Troubleshooting](#troubleshooting)
12. [User Identity](#user-identity)
13. [Time Context](#time-context)

---

## Getting Started

See [GETTING_STARTED.md](../GETTING_STARTED.md) for initial setup, including:
- Python environment setup
- LLM provider configuration
- ComfyUI integration
- Database initialization

Once setup is complete, access the web interface at `http://localhost:8000`.

---

## Model Management

### Overview

When using **Ollama** as your LLM provider, Chorus Engine includes an integrated **Model Manager** for easy model discovery, download, and management directly from the web interface.

**Features**:
- **Curated Model Library**: Pre-vetted models tested with Chorus Engine, with performance ratings
- **HuggingFace Import**: Import any GGUF model from HuggingFace with automatic chat template extraction
- **VRAM Estimation**: Automatic GPU memory requirement calculations with fit indicators
- **Database Tracking**: Persistent model inventory that survives restarts
- **Character Integration**: Easy model selection per character in the character editor

### Quick Start

1. **Access Model Manager**: Click gear icon (⚙️) → **Model Management**
2. **Browse Models**: View curated models with ratings and VRAM requirements
3. **Download**: Select quantization and click Download (progress tracked in real-time)
4. **Use in Characters**: Select downloaded models in character editor's LLM settings

### Model Types

**Curated Models**:
- Pre-tested with Chorus Engine
- Performance ratings (conversation, memory, creativity, etc.)
- Multiple quantization options with VRAM estimates
- Categorized: Balanced, Creative, Technical, Advanced

**Custom HuggingFace Models**:
- Import any GGUF model from HuggingFace
- Automatic chat template extraction (no manual Modelfile needed)
- Direct Ollama integration via `hf.co/` format

**See [MODEL_MANAGER.md](MODEL_MANAGER.md) for complete documentation**.

### Model Compatibility Note

Performance varies significantly across different models:

- **Conversation Quality**: Some models are more conversational and character-consistent
- **Memory Extraction**: Accuracy of automatic fact extraction varies by model
- **Image Prompt Generation**: Quality of image descriptions differs
- **Instruction Following**: Some models better adhere to system prompts and rules

**Tested Models**:
- `qwen2.5:14b-instruct` - Generally strong all-around performance
- `dolphin-mistral-nemo:12b` - Good conversational quality, requires defensive filters for memory extraction

**Recommendation**: Start with `qwen2.5:14b-instruct` for balanced performance, then experiment with other models to find what works best for your use case and hardware.

**Provider Selection**:
- **Ollama**: Simpler setup, automatic VRAM management, faster model switching
- **LM Studio**: Better UI for browsing models, manual control, more model formats

⚠️ **CRITICAL FOR LM STUDIO USERS**: When loading a model in LM Studio, you **must configure the context window** to match your Chorus Engine configuration:

1. **In LM Studio**: Set the "Context Length" (n_ctx) when loading the model (e.g., 32768 for 32K context)
   - Default is often only 4096 which is insufficient for rich conversations
2. **In Chorus Engine**: Set `context_window` in `config/system.yaml` or character's `preferred_llm` settings to the **same value**
3. **If they don't match**: You'll get empty responses, context overflow errors, or underutilized context

See [GETTING_STARTED.md](../GETTING_STARTED.md#critical-lm-studio-context-window-configuration) for detailed setup instructions.

### Archivist Model (Conversation Analysis)

Chorus Engine can use a **dedicated archivist model** for conversation analysis (summary + memory extraction). This is configured in `config/system.yaml` via:

```yaml
llm:
  archivist_model: qwen3:4b
```

**Provider notes:**
- **Ollama**: Use the Ollama model name (recommended default `qwen3:4b`)
- **LM Studio**: Use the exact LM Studio model name
- **KoboldCpp**: Leave blank (single-model runtime)

If set, the archivist model overrides character preferences for analysis and runs at temperature **0.0** for consistency.
You can edit this in the **System Settings** UI (Settings → System Settings → LLM Provider Configuration).

---

## Character Cards

### Overview

**Character cards** are portable character configuration files embedded as base64-encoded metadata in PNG images. This feature enables easy sharing, distribution, and importing of characters across platforms.

**Key Features**:
- Import/export characters as PNG files with embedded metadata
- SillyTavern V2 format compatibility for cross-platform character sharing
- Profile image management with focal point selection
- Automatic core memory loading on import

### Exporting a Character Card

Export any character as a shareable PNG card:

1. **Open Character Editor**: Click gear icon (⚙️) → Select character → Click gear icon again
2. **Click "Export as Card"** button in the character editor
3. **Configure Export Options**:
   - Character is pre-selected
   - ✓ Include voice configuration (optional)
   - ✓ Include workflow preferences (optional)
   - Voice sample URL (optional - for referencing external voice samples)
4. **Preview Metadata**: Expand the metadata preview to see what will be exported
5. **Click "Export"**: Downloads a PNG file named `CharacterName.card.png`

**What Gets Exported**:
- Character identity (name, role, system prompt, traits)
- Personality configuration and response style
- Core memories
- Profile image
- Voice settings (if enabled)
- Workflow preferences (if enabled)
- Metadata (creator, version, tags)

**What Doesn't Get Exported** (system-specific):
- Conversation history
- Memory store data (except core memories)
- LLM model preferences
- Absolute file paths

### Importing a Character Card

Import characters from PNG cards (Chorus Engine or SillyTavern format):

1. **Click "Import Character Card"** button on the character selection page
2. **Select PNG File**: 
   - Use the file picker, or
   - Drag and drop a PNG file onto the import area
3. **Preview the Card**:
   - Review character name, role, and description
   - See profile image preview
   - Check format (Chorus Engine or SillyTavern)
   - View immersion level, role type, and capabilities
   - See personality traits and core memories count
4. **Click "Import"** to confirm, or "Cancel" to abort

**Import Process**:
- Character name collisions are handled automatically (appends suffix if needed)
- Profile image is extracted and saved to `data/character_images/`
- Core memories are automatically loaded into the vector store
- Character becomes immediately available for use

**SillyTavern Compatibility**:

Chorus Engine can import SillyTavern V2 character cards with intelligent field mapping:

| SillyTavern Field | Chorus Engine Mapping |
|-------------------|-----------------------|
| `name` | `name` |
| `description` | `role` |
| `personality` | Extracted into `personality_traits` |
| `description` + `personality` | Combined into `system_prompt` |
| `scenario` | Added to `system_prompt` context |
| `first_mes` | Not used (Chorus generates greetings) |
| `mes_example` | Not used (Chorus learns from conversation) |
| `creator` | Preserved in metadata |
| `tags` | Preserved in metadata |

**Intelligent Defaults**:
- `immersion_level` set to `"unbounded"` (SillyTavern characters typically have fewer boundaries)
- `role_type` set to `"companion"` (most common use case)
- Personality traits extracted via regex patterns (20+ common traits detected)
- Description paragraphs converted to core memories

### Profile Image Management

#### Uploading a Profile Picture

1. **Open Character Editor**: Select character and click gear icon
2. **Profile Image Section**: Find the "Profile Image" field
3. **Click Upload Button** (📤 icon next to the profile image field)
4. **Select Image**: Choose PNG, JPG, JPEG, or WEBP file (max 10MB)
5. **Image Processing**: 
   - Image is automatically converted to PNG
   - Saved as `{character_id}.png` in `data/character_images/`
   - Filename is automatically updated in the character configuration

#### Setting Image Focal Point

For non-square images displayed in circular avatars, set a focal point to ensure important parts (like faces) are centered:

1. **Upload or Select Profile Image** first
2. **Click Focal Point Button** (🎯 bullseye icon next to the upload button)
3. **Interactive Modal Opens**:
   - Click anywhere on the character image
   - A gold marker shows your selected focal point
   - Coordinates update in real-time
4. **Click "Save Focus Point"** to apply
5. **Result**: 
   - Focal point coordinates saved to character YAML
   - Sidebar avatar and profile modal update immediately
   - Non-destructive (original image unchanged, CSS handles display)

**Focal Point Tips**:
- Click on the character's face or most important feature
- Default is center (50%, 50%) if not set
- Can be changed anytime without re-uploading the image
- Works with any image aspect ratio

### Character Card Troubleshooting

**Import Fails**:
- Ensure file is a valid PNG image
- Check file size is under 10MB
- Verify card has embedded metadata (not all PNG files are character cards)

**SillyTavern Card Looks Different**:
- Expected - Chorus Engine uses different character structure
- System prompt combines multiple SillyTavern fields
- Core memories created from character description
- Adjust character config after import if needed

**Profile Image Not Showing**:
- Check `data/character_images/` directory exists
- Verify file permissions
- Check browser console for 404 errors
- Reload character list after upload

**Focal Point Not Applied**:
- Ensure you clicked "Save Focus Point" in the modal
- Refresh the page to reload character data
- Check character YAML contains `profile_image_focus` field

---

## Text Conversations

### Starting a Conversation

1. Open the web interface
2. Select a character from the dropdown (top-left)
3. Select or create a conversation thread
4. Type your message in the text box at the bottom
5. Press **Send** or hit `Enter`

### Conversation Features

**Thread Management**:
- **New Thread**: Click the "+" button to start a fresh conversation
- **Switch Threads**: Use the thread dropdown to switch between conversations
- **Delete Thread**: Click the trash icon to delete a thread

**Message Controls**:
- **Edit Messages**: Click the pencil icon to edit your messages
- **Delete Messages**: Click the trash icon to remove messages
- **Regenerate Response**: Click the refresh icon to get a new response

**Privacy Levels**:
Messages can be marked with different privacy levels that affect memory persistence:
- **Public**: Stored in long-term memory (default)
- **Private**: Not stored in long-term memory
- **Whisper**: Hidden from character context

---

## Image Generation

### Using Image Generation

1. Type a message requesting an image
2. The character will respond with text first
3. If image generation is triggered, you'll see the image appear below the message
4. Images are automatically saved to `data/images/<character_id>/`

### Image Workflows

**Default Workflows**:
Each character can have a default image workflow specified in their YAML configuration:
```yaml
image_generation:
  default_workflow: my_flux_workflow
```

**Custom Workflows**:
- Access the **Workflows** tab in the web interface
- Upload new workflows or manage existing ones
- Set a workflow as the default for the character

**Workflow Placeholders**:
Image workflows use placeholders that are replaced at generation time:
- `__CHORUS_PROMPT__`: The image prompt
- `__CHORUS_REFERENCE_IMAGE__`: Optional reference image path
- `__CHORUS_CHARACTER_NAME__`: Character name

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for details on creating custom workflows.

---

## Scene Capture & Image Gallery

### Overview

Scene Capture allows you to manually generate an image of the current conversation scene at any moment. This is different from conversationally-requested images - instead of the character generating an image of what they describe, scene capture creates a **third-person observer view** of the conversation itself.

### Scene Capture Button

**Availability**:
- The scene capture button (📷 camera icon) appears in the chat interface
- Only visible for characters with **full or unbounded immersion level**
- Located next to other chat controls

**Using Scene Capture**:
1. Click the camera button during any conversation
2. The character's LLM analyzes the recent conversation (last 6000 tokens)
3. An **observer-perspective prompt** is automatically generated, describing:
   - The physical setting and environment
   - Character appearance and positioning
   - Current mood and atmosphere
   - Visual details and lighting
4. A confirmation modal appears with the generated prompt
5. **Review and edit** the prompt if desired
6. Click **"Generate"** to create the scene image
7. Generation takes ~18 seconds (same as normal images)
8. The image appears in the chat with a **blue camera badge**

**Observer Perspective**:
Scene captures are generated from a **3rd person cinematic perspective**, as if an invisible observer is witnessing the conversation. The prompts describe:
- "View of [character] sitting at their desk..."
- "The room is dimly lit with..."
- "Their expression shows..."

This differs from conversational images where the character describes what they're imagining or creating.

### Image Gallery

**Opening the Gallery**:
- Click the **Gallery toggle button** at the top of the interface (70px from top)
- The gallery panel slides out showing all images from the current conversation

**Gallery Features**:
- **Thumbnail Grid**: All images displayed as clickable thumbnails
- **Camera Badge**: Blue 📷 icon appears on scene-captured images
- **Chronological Order**: Newest images appear first
- **Image Types**:
  - **Conversational Images**: Generated during conversation when character describes something visual
  - **Scene Captures**: Manually captured observer-perspective scenes with camera badge

**Viewing Images**:
- Click any thumbnail to view full-size in a modal
- Click outside the modal or the X button to close
- Navigate between images using the modal

**Gallery Persistence**:
- Gallery shows images from the current conversation
- Switches when you change conversations
- Images are permanently stored in `data/images/{conversation_id}/`

### Technical Details

**File Storage**:
- Full images: `data/images/{conversation_id}/{image_id}_full.png`
- Thumbnails: `data/images/{conversation_id}/{image_id}_thumb.png`
- Organized by conversation for easy management

**Database Tracking**:
- Scene captures stored with `SCENE_CAPTURE` message role
- Separate from regular conversation messages
- Includes metadata (prompt, generation time, workflow)

**VRAM Management**:
- Scene capture uses the same ComfyUI lock as normal images
- Only one image generates at a time
- Models automatically unload/reload for memory efficiency

### Tips and Best Practices

**When to Use Scene Capture**:
- Capture memorable moments in the conversation
- Create visual records of important scenes
- Experiment with different perspectives on the same conversation
- Build a visual narrative of your interactions

**Editing Prompts**:
- The auto-generated prompt is a suggestion - feel free to edit
- Add specific details you want to see
- Adjust lighting, camera angle, or atmosphere
- Use the same workflow settings as normal images

**Gallery Management**:
- Images persist forever unless manually deleted from filesystem
- No built-in deletion yet (planned for future update)
- Gallery loads all images - performance may slow with 100+ images

**Character Requirements**:
- Scene capture requires **full or unbounded immersion** characters
- These characters have deeper contextual awareness
- Lower immersion characters don't have scene capture enabled

---

## Voice Generation (TTS)

### Overview

Chorus Engine can generate voice audio for character responses using ComfyUI TTS workflows.

### Voice Sample Management

**Uploading Voice Samples**:
1. Go to the **Voice Samples** tab
2. Click **Upload Voice Sample**
3. Select an audio file (WAV, MP3, FLAC, OGG)
4. Provide a transcript of the audio
5. Click **Upload**

**Voice Sample Requirements**:
- **Audio Format**: WAV (recommended), MP3, FLAC, or OGG
- **Duration**: 3-30 seconds ideal
- **Quality**: Clear, minimal background noise
- **Transcript**: Exact transcription of the audio content

**Managing Voice Samples**:
- View all voice samples in the **Voice Samples** tab
- Delete samples with the trash icon
- One voice sample per character (latest upload replaces previous)

### Enabling TTS

**Per-Conversation Toggle**:
1. Start or select a conversation
2. Toggle the **TTS** switch (top-right of message area)
3. When enabled, character responses will generate audio automatically

**Character-Level Default**:
Set TTS to always be enabled by default in the character YAML:
```yaml
tts_generation:
  enabled: true
  always_on: true  # TTS enabled by default for all new conversations
  default_workflow: default_tts_workflow
```

### Using TTS

**Automatic Generation**:
- When TTS is enabled, audio generates automatically after each character response
- Audio appears as a player below the message text
- Generation happens in the background (responses appear immediately)

**Audio Player Controls**:
- **Play/Pause**: Click the play button
- **Seek**: Click or drag the progress bar
- **Volume**: Adjust with the volume slider
- **Download**: Right-click the player and select "Save audio as..."

**Manual Regeneration**:
- Click the **Regenerate Audio** button (🔄 icon) to re-generate audio for a message
- Useful if audio quality is poor or generation failed

### TTS Workflows

**Default Workflow**:
The character's YAML configuration specifies the default TTS workflow:
```yaml
tts_generation:
  enabled: true
  default_workflow: my_custom_tts_workflow
```

**Custom TTS Workflows**:
1. Go to the **Workflows** tab
2. Select **TTS** as the workflow type
3. Upload a ComfyUI TTS workflow
4. Set it as default if desired

**TTS Workflow Placeholders**:
TTS workflows use special placeholders:
- `__CHORUS_TEXT__`: The text to speak
- `__CHORUS_VOICE_SAMPLE__`: Path to the voice sample audio
- `__CHORUS_VOICE_TRANSCRIPT__`: Transcript of the voice sample

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#tts-workflows) for details on creating TTS workflows.

### TTS Best Practices

**Voice Sample Tips**:
- Use clean, high-quality audio
- Record in a quiet environment
- Keep samples between 5-15 seconds
- Use neutral tone and clear pronunciation
- Match the character's intended voice style

**Transcript Accuracy**:
- Transcribe exactly what is spoken
- Include punctuation for natural pacing
- Avoid adding extra words or corrections

**Audio Quality**:
- First generation may take longer (model loading)
- Subsequent generations are faster
- Audio quality depends on the TTS model in your workflow
- Experiment with different workflows for best results

### Troubleshooting TTS

**No Voice Sample Error**:
- Upload a voice sample in the **Voice Samples** tab
- Ensure the sample is assigned to the correct character

**Audio Not Auto-Playing**:
- Browsers block auto-play; click the play button manually
- Check browser auto-play settings

**Workflow Not Found**:
- Verify the workflow exists in `workflows/<character>/audio/`
- Check the workflow name in the character's YAML configuration
- Ensure the workflow has the correct placeholders

**Audio Quality Issues**:
- Try a different TTS model in your ComfyUI workflow
- Ensure voice sample is high quality
- Check ComfyUI logs for errors

**Generation Fails**:
- Check ComfyUI is running and accessible
- Verify the TTS workflow is valid
- Check `server_log_fixed.txt` for error details

---

## Memory Management

### Memory System Overview

Chorus Engine uses a **multi-layered memory system** that automatically learns and remembers facts about you across conversations:

**Memory Types**:
- **Core Memory**: Character's immutable backstory and personality (from configuration)
- **Explicit Memory**: Facts you explicitly tell the character to remember
- **Fact Memory**: Discrete information automatically extracted from your messages
- **Project Memory**: Your ongoing goals, activities, and plans
- **Experience Memory**: Shared moments and collaborations within conversations
- **Story Memory**: Emotionally significant narratives you share from your past
- **Relationship Memory**: Trust developments and vulnerable moments

**Automatic Extraction**:
- Facts are extracted automatically from every message you send
- Conversations are analyzed when completed to extract projects, experiences, stories, and relationships
- The system uses semantic deduplication to prevent storing duplicate information
- All memories are searchable and retrievable based on relevance

### Memory Types Explained

#### Core Memory
Immutable facts about the character's backstory and personality, defined in their configuration file.

**Example**: "Nova is a visual artist specializing in watercolor painting"

**Usage**: Provides consistent character foundation across all conversations

#### Explicit Memory
Facts you explicitly tell the character to remember using commands or the Memory Manager UI.

**Example**: "My anniversary is October 15th"

**Usage**: Important dates, preferences, or information you want guaranteed recall

#### Fact Memory
Discrete information automatically extracted from your messages in real-time.

**Examples**:
- "User's name is Sarah"
- "User prefers tea over coffee"  
- "User works as a data scientist"
- "User lives in Seattle"

**Usage**: Builds your profile automatically without explicit "remember this" commands

#### Project Memory
Ongoing activities, goals, or work you mention throughout conversations.

**Examples**:
- "User is training for a marathon in April"
- "User is building a REST API for their startup"
- "User is learning Spanish for upcoming trip"

**Usage**: Character remembers what you're working on across conversations

#### Experience Memory
Shared moments and collaborations that happened within a specific conversation.

**Examples**:
- "Explored watercolor techniques for capturing fog together"
- "Debugged Docker networking issue collaboratively"
- "Had deep discussion about AI ethics"

**Usage**: Character remembers what you did together in past conversations

#### Story Memory
Emotionally significant narratives you shared from your past.

**Examples**:
- "User shared memory of childhood dog who taught them about unconditional love"
- "User described overcoming fear of public speaking at work conference"
- "User talked about grandmother's influence on their love of art"

**Usage**: Character remembers emotionally important stories you've shared

#### Relationship Memory
Trust developments, vulnerability moments, and relationship milestones.

**Examples**:
- "User felt comfortable sharing anxiety about work performance"
- "User opened up about struggles with self-confidence"
- "Trust deepened through vulnerable conversation about family"

**Usage**: Character tracks relationship depth and emotional trust levels

### Memory Manager

Access the **Memory Manager** by clicking the brain icon in the sidebar.

**Features**:
- **View Memories**: Browse all memories by type using filter tabs
- **Add Memory**: Create explicit memories manually
- **Delete Memory**: Remove incorrect or outdated memories
- **Search**: Semantic search to find relevant memories
- **Categories**: Fact memories show category badges (personal info, preference, experience, etc.)

**Filter Tabs**:
- **All**: Show all memory types
- **Core**: Character backstory only
- **Explicit**: User-created memories
- **Facts**: Automatically extracted discrete information
- **Projects**: Ongoing activities and goals
- **Experiences**: Shared conversation moments
- **Stories**: Emotional narratives from your past
- **Relationship**: Trust and vulnerability developments

### Conversation Analysis

Characters automatically analyze completed conversations to extract thematic memories. You can also trigger manual analysis.

**Manual Analysis**:
1. Open a conversation
2. Click the **Analyze Now** button (lightbulb icon in the header)
3. Review the analysis preview
4. Click **Analyze** to confirm

**Analysis Shows**:
- Memory count by type (facts, projects, experiences, stories, relationships)
- Conversation summary (2-3 sentences)
- Main themes discussed
- Overall tone (supportive, playful, deep, technical, etc.)
- Emotional arc (how emotions evolved during conversation)
- Detailed list of extracted memories with confidence scores

**Analysis History**:
1. Click the **Analysis History** button (clock icon in the header)
2. View chronological list of past analyses
3. Click any analysis to see full details
4. See which analyses were manual vs automatic

**When Analysis Happens Automatically**:
- When conversation reaches 10,000 tokens (substantial conversation)
- When you start a new conversation (previous one analyzed)
- When conversation has 2,500+ tokens and 24 hours of inactivity

### Memory Profiles (Immersion Levels)

Characters have different **memory profiles** based on their immersion level, which controls what types of memories they extract:

**Minimal** (Utilitarian assistants):
- Extracts: Facts, Projects
- Example: Code assistant who remembers your tech stack and what you're building

**Balanced** (Professional collaborators):
- Extracts: Facts, Projects, Experiences
- Example: Work assistant who remembers collaborations and what you explored together

**Full** (Conversational companions):
- Extracts: Facts, Projects, Experiences, Stories
- Example: Creative companion who remembers emotional narratives you've shared

**Unbounded** (Deep roleplay):
- Extracts: All memory types including Relationship tracking
- Example: Companion who tracks trust developments and emotional milestones

**Note**: Memory profiles are configured per character and can be customized in the character YAML configuration.
- **Core memories**: Character-defining facts loaded from YAML

### Viewing Memories

1. Go to the **Memories** tab
2. Select a character from the dropdown
3. View all memories associated with that character
4. Search memories using the search box

### Memory Types

**Core Memories**:
- Defined in the character's YAML file
- Loaded automatically on character initialization
- Cannot be deleted (edit the YAML file instead)

**Extracted Memories**:
- Automatically extracted from conversations
- Stored in the vector database
- Retrieved based on relevance to current conversation

### Managing Memories

**Adding Memories**:
- Memories are extracted automatically during conversations
- No manual addition required (happens in background)

**Deleting Memories**:
- Click the trash icon next to a memory
- Core memories cannot be deleted

**Memory Relevance**:
- The system retrieves relevant memories based on conversation context
- More relevant memories have higher scores
- Adjust `memory.retrieval_limit` in `config/system.yaml` to control how many memories are retrieved

---

## Ambient Activities

### Overview

Ambient activities allow characters to post updates even when you're not actively chatting. This creates a more immersive, living-world experience.

### Enabling Ambient Activities

**Per-Character Configuration**:
Edit the character's YAML file:
```yaml
ambient:
  enabled: true
  base_interval_minutes: 60  # Average time between activities
  randomness: 0.3  # 30% variation
  active_hours:
    start: 8  # 8 AM
    end: 23   # 11 PM
  activity_prompts:
    - "What is {character_name} doing right now?"
    - "Describe {character_name}'s current thoughts or activities."
```

**Global Configuration**:
Edit `config/system.yaml`:
```yaml
ambient:
  enabled: true
  polling_interval_seconds: 300  # Check every 5 minutes
```

### How It Works

1. System checks periodically if a character should post
2. If enough time has passed, generates an ambient activity
3. Activity appears as a new message in the thread
4. You can respond to ambient activities like normal messages

### Activity Types

Activities can include:
- Character thoughts and reflections
- Current actions or hobbies
- Reactions to in-world events
- Updates on ongoing projects

### Controlling Ambient Activity

**Frequency**:
- Adjust `base_interval_minutes` to control how often activities occur
- Higher values = less frequent activities

**Randomness**:
- Adjust `randomness` (0.0 to 1.0) to add variation
- 0.0 = perfectly regular intervals
- 0.5 = ±50% variation

**Active Hours**:
- Set `active_hours` to limit when ambient activities can occur
- Uses 24-hour format (0-23)

**Disabling**:
Set `enabled: false` in character YAML or system config.

---

## Workflows

### Overview

Workflows are ComfyUI workflows that define how images, audio, or video are generated. Chorus Engine uses a workflow-first approach.

### Workflow Types

- **Image**: ComfyUI image generation workflows
- **TTS**: ComfyUI text-to-speech workflows
- **Video**: ComfyUI video generation workflows (future)

### Managing Workflows

**Viewing Workflows**:
1. Go to the **Workflows** tab
2. Select a character from the dropdown
3. View all workflows for that character
4. Filter by type (Image | TTS | Video)

**Uploading Workflows**:
1. Click **Upload Workflow**
2. Select the workflow type
3. Choose a workflow JSON file (exported from ComfyUI)
4. Optionally set as default
5. Click **Upload**

**Setting Default Workflows**:
- Click the **Set as Default** button for a workflow
- The default workflow is used when no specific workflow is requested
- Each workflow type has its own default

**Deleting Workflows**:
- Click the trash icon next to a workflow
- Default workflows cannot be deleted (unset default first)

### Workflow Placeholders

Workflows use placeholders that Chorus Engine replaces at generation time. Different workflow types have different placeholders.

**Image Placeholders**:
- `__CHORUS_PROMPT__`: Image generation prompt
- `__CHORUS_REFERENCE_IMAGE__`: Optional reference image path
- `__CHORUS_CHARACTER_NAME__`: Character name

**TTS Placeholders**:
- `__CHORUS_TEXT__`: Text to synthesize
- `__CHORUS_VOICE_SAMPLE__`: Path to voice sample audio
- `__CHORUS_VOICE_TRANSCRIPT__`: Transcript of voice sample

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for detailed information on creating workflows.

---

## Troubleshooting

### General Issues

**Cannot Connect to API**:
- Ensure the server is running: `python -m chorus_engine.main` or `.\start.ps1`
- Check the server is accessible at `http://localhost:8000`
- Check `server_log_fixed.txt` for errors

**Character Not Loading**:
- Verify the character YAML file exists in the `characters/` directory
- Check YAML syntax is valid
- Look for errors in `server_log_fixed.txt`

**ComfyUI Connection Failed**:
- Ensure ComfyUI is running
- Verify the ComfyUI URL in `config/system.yaml`
- Test ComfyUI is accessible at the configured URL

### LLM / Context Issues

**Empty or No Response from LLM** (LM Studio users):
- **Most Common Cause**: Context window mismatch between LM Studio and Chorus Engine
- **Solution**:
  1. Check what context length you set when loading the model in LM Studio (often defaults to 4096)
  2. Verify `context_window` in `config/system.yaml` or character's `preferred_llm` settings
  3. **Reload the model in LM Studio** with matching context length (recommended: 32768)
  4. Restart Chorus Engine
- See [GETTING_STARTED.md](../GETTING_STARTED.md#critical-lm-studio-context-window-configuration) for detailed instructions

**Context Overflow Errors**:
- LM Studio: Your configured `context_window` is larger than the model's loaded context
- Check server logs for specific error message
- Reload model in LM Studio with larger context window
- Or reduce `context_window` in config to match LM Studio's setting

**Document Analysis Returns Empty Results**:
- May indicate context window is too small for document chunks
- Increase context window in both LM Studio (when loading) and Chorus Engine config
- Recommended minimum: 16K, optimal: 32K for document analysis

### Memory Issues

**Memories Not Being Created**:
- Check `memory.extraction.enabled` is `true` in `config/system.yaml`
- Verify the LLM provider is configured correctly
- Look for extraction errors in `server_log_fixed.txt`

**Irrelevant Memories Retrieved**:
- Adjust `memory.retrieval_limit` in `config/system.yaml`
- Check memory embeddings are being created (vector store should have data)
- Consider deleting irrelevant memories manually

### Image Generation Issues

**Images Not Generating**:
- Check ComfyUI is running and accessible
- Verify the workflow exists and is valid
- Check workflow has correct placeholders
- Look for ComfyUI errors in its console

**Wrong Workflow Used**:
- Check the character's `image_generation.default_workflow` in YAML
- Verify the workflow name matches the file in `workflows/<character>/image/`

### TTS Issues

**Audio Not Generating**:
- Upload a voice sample for the character
- Enable TTS toggle in the conversation
- Check ComfyUI is running
- Verify TTS workflow has correct placeholders

**Poor Audio Quality**:
- Use a higher-quality voice sample
- Try a different TTS model in your ComfyUI workflow
- Ensure voice sample transcript is accurate

### Performance Issues

**Slow Response Times**:
- Check LLM provider latency
- Reduce `memory.retrieval_limit` to retrieve fewer memories
- Disable ambient activities if not needed
- Check ComfyUI performance (GPU usage)

**High Memory Usage**:
- Large vector stores can consume memory
- Consider periodic cleanup of old memories
- Restart the server periodically

### Database Issues

**Database Locked**:
- Only one server instance should run at a time
- Check no other processes are accessing the database
- Restart the server

**Migrations Fail**:
- Check migration scripts in `testing/` directory
- Backup database before running migrations
- Check `server_log_fixed.txt` for error details

---

## Getting Help

- **Documentation**: See `Documentation/` directory for detailed guides
- **Issues**: Check `server_log_fixed.txt` for error messages
- **Community**: [Add your support channels here]
- **Contributing**: See `CONTRIBUTING.md` for development guidelines

---

## User Identity

User Identity is a system configuration setting used to address you consistently across chats. It is not a memory and is not inferred from conversation.

**Fields**:
- Display name: the name used for canonical addressing
- Aliases: optional alternate names

You can update this in **Settings -> User Identity**.

## Time Context

When enabled, the system injects the server-local time into conversation prompts to improve time awareness.

**Defaults**:
- Enabled by default
- Blank timezone uses server local time

You can configure this in **Settings -> System Settings -> Time Context**.

---

**Last Updated**: Phase 6 Complete (TTS Integration)
