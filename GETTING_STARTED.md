# Chorus Engine - Getting Started

Local AI Character Orchestration System - A FastAPI-based system for managing AI character interactions with persistent conversations, memory, and more.

## Quick Start

### Prerequisites

**Choose Your Installation Mode:**

#### Portable/Embedded Mode (Recommended for Users)
- ✅ No Python installation needed (downloaded automatically)
- ✅ Self-contained environment
- ✅ Works on any Windows machine
- **Requirements:**
  - **Visual C++ Build Tools** (for ChromaDB compilation) - See below
  - **Windows Long Paths enabled** - See below
  - **NVIDIA GPU with CUDA support** (recommended for TTS) - See GPU Setup below

#### Developer Mode (For Contributors)
- **Python 3.11 or higher** (3.13 recommended) installed on your system
- **Visual C++ Build Tools** (Windows only, for ChromaDB compilation)
- All other requirements same as Portable mode

**Both Modes Require:**
- **LLM Provider** - Choose one:
  - **Integrated Provider** (built-in, recommended) - Downloads models automatically, GPU-accelerated
  - **[Ollama](https://ollama.ai/)** - External service, good for sharing models across apps
  - **[LM Studio](https://lmstudio.ai/)** - External service with model management UI
  - **[KoboldCpp](https://github.com/LostRuins/koboldcpp)** - Lightweight external service

### Windows-Specific Setup

#### Enable Long Paths (Required)

ChromaDB and transformers require Windows long path support:

**Option 1: Group Policy Editor**
1. Press `Win + R`, type `gpedit.msc`, press Enter
2. Navigate to: `Computer Configuration` → `Administrative Templates` → `System` → `Filesystem`
3. Double-click "Enable Win32 long paths"
4. Select "Enabled", click OK

**Option 2: Registry (Administrator PowerShell)**
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Restart your PowerShell session after enabling.

#### Install Visual C++ Build Tools

Required for compiling ChromaDB's native components:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --silent --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Or download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### GPU Setup (Highly Recommended)

**Why GPU?** 
- **TTS (Chatterbox)**: GPU provides ~10-50x faster voice synthesis (2-3s vs 30-60s per voice line)
- **Integrated LLM**: GPU enables local AI chat at reasonable speeds (vs very slow CPU inference)
- **Overall Experience**: GPU transforms Chorus Engine from "technically works" to "actually usable"

**Requirements:**
- NVIDIA GPU (GTX 1060 6GB or better recommended)
- ~4GB VRAM minimum (8GB+ recommended for integrated LLM)
- **CUDA Toolkit 12.1+** (only for integrated LLM provider)

**Important Notes:**
- **PyTorch**: Bundles its own CUDA runtime - no system CUDA installation needed for TTS
- **llama-cpp-python**: Requires CUDA 12.1 runtime installed system-wide for GPU acceleration
- If you only use external LLM providers (Ollama/LM Studio), no CUDA installation needed at all

**Installation:**

**CUDA 12.1 Runtime** (required ONLY for integrated LLM with GPU):
- Download: https://developer.nvidia.com/cuda-12-1-0-download-archive
- Select: Windows > x86_64 > 11 > exe (network)
- **Custom Installation**: Check only "CUDA Runtime Libraries" (saves 2.5GB)
- This is all you need - PyTorch already includes CUDA for TTS

**Verify Installation:**
```powershell
# Check NVIDIA driver and CUDA version
nvidia-smi

# Check compute capability (5.0+ required for modern features)
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**GPU Compatibility:**
- **RTX 50-series (5090, etc.)**: ✅ Fully supported (PyTorch 2.9 cu130)
- **RTX 40-series (4090, 4080, etc.)**: ✅ Supported
- **RTX 30-series (3090, 3080, etc.)**: ✅ Supported
- **GTX 10/20-series**: ✅ Supported

*Note: PyTorch 2.9 with cu130 supports all modern NVIDIA GPUs from Pascal (GTX 10-series) through Blackwell (RTX 50-series). The CUDA runtime is bundled in the PyTorch wheel. For the integrated LLM, llama-cpp-python needs CUDA 12.1 runtime installed separately.*

*Note: We use PyTorch 2.9 with CUDA 13.0 which supports all modern NVIDIA GPUs from Pascal (GTX 10-series) through Blackwell (RTX 50-series) architectures. The integrated LLM uses llama-cpp-python with CUDA 12.1 wheels for maximum compatibility.*

**CPU-Only Alternative:**
If you don't have an NVIDIA GPU or can't install CUDA 12.1:
- TTS will work but be very slow (30-60s per voice line vs 2-3s with GPU)
- Integrated LLM will work but be very slow (consider using Ollama/LM Studio instead)
- All other features work normally

**CUDA Installation** (optional - only needed for integrated LLM GPU acceleration):
- **For integrated LLM**: Install CUDA 12.1 runtime (see instructions above)
- **For TTS only**: No CUDA installation needed (PyTorch includes it)
- Restart your terminal/PowerShell after CUDA installation

**Note:** The portable installation (`install.bat`) automatically installs CUDA-enabled PyTorch and llama-cpp-python. It will detect if CUDA 12.1 is installed and inform you if it's missing.

### Installation

**Choose your installation method:**

#### Option A: Portable/Embedded Installation (Recommended for Windows Users)

Run the automated installer:

```cmd
install.bat
```

This will:
- Download Python 3.11.7 embedded (~30MB)
- Install PyTorch 2.9 with CUDA 13.0 support
- Install llama-cpp-python with CUDA 12.1 support
- Install all dependencies automatically
- Set up a self-contained environment
- Check for CUDA runtime and provide guidance if missing

**No Python installation required!** Everything is downloaded and configured for you.

**After Installation:**
- If you see CUDA warnings, install the required CUDA versions (see GPU Setup above)
- The server will still start and work (falling back to CPU where needed)
- For best performance, install both CUDA 13.0 and 12.1

---

#### Option B: Developer/Manual Installation

**Prerequisites:** Python 3.11+ must be installed on your system

1. **Clone or download** the repository

2. **Install dependencies:**

   **IMPORTANT:** Install PyTorch with CUDA support FIRST (before requirements.txt):
   
   ```bash
   # For CUDA 13.0 (recommended - supports RTX 50-series and all modern GPUs)
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
   
   # Then install remaining dependencies
   pip install -r requirements.txt
   
   # Finally install llama-cpp-python with CUDA 12.1 support (for integrated LLM)
   pip install llama-cpp-python --prefer-binary --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   
   **Alternative CUDA versions for PyTorch:**
   - CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124` (older but stable)
   - CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121` (compatibility)
   - CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118` (legacy GPUs)
   - CPU only: Just use `pip install torch torchaudio` (very slow for TTS/LLM)

   **Note**: First install may take several minutes:
   - ChromaDB compiles native components (requires C++ build tools)
   - PyTorch downloads (~2GB with CUDA)
   - llama-cpp-python CUDA wheel (~450MB)
   - Sentence transformers model downloads on first use (~80MB)

---

### Post-Installation Setup

3. **Configure your LLM provider:**

The system supports multiple LLM providers. Configure in `config/system.yaml`:

**Option A: Integrated Provider (Recommended - Built-in)**
```yaml
llm:
  provider: integrated  # Built-in llama.cpp, no external service needed
  model: "data/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"  # Path to GGUF model
  n_gpu_layers: -1  # -1 = all layers on GPU (requires CUDA 12.1)
  n_threads: 8      # CPU threads for CPU-only inference
```

Download models via web UI (Settings > Model Management) or manually from HuggingFace.
See `Documentation/INTEGRATED_LLM_GUIDE.md` for detailed setup and model recommendations.

**Option B: Ollama (External Service with Integrated Model Manager)**
```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  model: mistral:7b-instruct
```

Install and start Ollama:
```bash
# Download from https://ollama.ai/
# Start Ollama service
```

**Model Management**: When using Ollama, Chorus Engine provides an **integrated Model Manager** for downloading and managing models through the web interface. This includes:
- Curated model library with performance ratings and VRAM estimates
- Direct HuggingFace GGUF import (automatic chat template extraction)
- Database tracking of downloaded models
- Easy model selection in character editor

See [Documentation/MODEL_MANAGER.md](Documentation/MODEL_MANAGER.md) for complete guide.

**Quick Start with Model Manager**:
1. Start Chorus Engine with Ollama provider configured
2. Click gear icon (⚙️) → **Model Management**
3. Browse curated models or import from HuggingFace
4. Download with one click (automatic VRAM estimation)
5. Select models per character in character editor

**Manual Model Installation** (alternative):
```bash
ollama pull mistral:7b-instruct
```

**Option C: LM Studio (External Service)**
```yaml
llm:
  provider: lmstudio
  base_url: http://localhost:1234
  model: local-model  # Name shown in LM Studio
```

**Option D: KoboldCpp (External Service)**
- Download a model from the model library
- Load the model in LM Studio

**Configure Chorus Engine** - Edit `config/system.yaml`:

⚠️ **IMPORTANT**: The `config/system.yaml` file has **LM Studio as the default** (uncommented). Before starting Chorus Engine:

1. **Open** `config/system.yaml`
2. **Check which LLM provider block is uncommented** (active configuration)
3. **If using Ollama**: Comment out the `lmstudio` block (add `#` at start of each line) and uncomment the `ollama` block
4. **If using LM Studio**: The default configuration is already correct - just verify your settings

Example configuration blocks:
```yaml
# For Ollama (comment out lmstudio block, uncomment this):
# llm:
#   provider: ollama
#   base_url: http://localhost:11434
#   model: qwen2.5:14b-instruct

# For LM Studio (DEFAULT - currently active):
llm:
  provider: lmstudio
  base_url: http://localhost:1234
  model: qwen/qwen2.5-coder-14b
```

4. **Configure HuggingFace (Required for TTS)**

Chatterbox TTS requires HuggingFace authentication to download models:

**Get a HuggingFace Token:**
- Go to https://huggingface.co/settings/tokens
- Create a new token (read access is sufficient)
- Copy the token

**Login with your token:**

*If using standard Python:*
```bash
huggingface-cli login
```

*If using embedded Python (portable installation):*
```powershell
python_embeded\Scripts\huggingface-cli.exe login
```

Paste your token when prompted. This is a one-time setup that enables model downloads.

### Model Selection

Chorus Engine supports models from **both Ollama and LM Studio**, but performance characteristics vary:

**Performance Considerations**:
- **Conversation Quality**: Character voice consistency and naturalness
- **Memory Extraction**: Automatic fact extraction accuracy (with defensive filters)
- **Image Descriptions**: Quality of generated image prompts
- **Instruction Adherence**: Following system prompts and rules

**Tested Models** (as of 2026-01-02):
- **qwen2.5:14b-instruct** - Balanced all-around performance
  - Ollama: `ollama pull qwen2.5:14b`
  - LM Studio: Download "Qwen 2.5 14B Instruct" from model library
- **dolphin-mistral-nemo:12b** - Good conversational quality, uncensored
  - Ollama: `ollama pull CognitiveComputations/dolphin-mistral-nemo:12b`
  - LM Studio: Search for "Dolphin Mistral Nemo" in model library

**Recommendation**: Start with `qwen2.5:14b-instruct` for reliable performance across all features. Experiment with other models based on your hardware capabilities and use case preferences.

**Provider Comparison**:
- **Ollama**: Simpler CLI, automatic model management, faster model switching
- **LM Studio**: Better UI, manual model control, more model format support

### Critical: LM Studio Context Window Configuration

⚠️ **IMPORTANT FOR LM STUDIO USERS**: You must configure the context window in **both places**:

**1. In LM Studio (when loading the model):**
- Open LM Studio's model loading interface
- Find the **"Context Length"** or **"n_ctx"** setting
- Set it to match your desired context window (e.g., **31500** for 32K context)
- **Default is often 4096** which is too small and will cause context overflow errors
- Click "Load Model" with the updated context setting

**2. In Chorus Engine configuration:**

Edit `config/system.yaml` and set the `context_window` value to match:
```yaml
llm:
  provider: lmstudio
  base_url: http://localhost:1234
  model: qwen/qwen2.5-coder-14b
  context_window: 31500  # MUST MATCH LM Studio's loaded context!
```

Or in a character's `preferred_llm` settings (see [CHARACTER_CONFIGURATION.md](Documentation/CHARACTER_CONFIGURATION.md)):
```yaml
preferred_llm:
  temperature: 0.4
  context_window: 31500  # Character-specific override
```

**Why Both?**
- **LM Studio setting**: Controls the actual model's context window in memory
- **Chorus Engine setting**: Used for token budget calculations and context management
- **They must match** or you'll get context overflow errors or underutilized context

**Symptoms of Mismatch:**
- Empty responses from the LLM
- Context overflow errors in logs
- Unexpectedly truncated conversations
- Document analysis returning no results

**Recommended Settings:**
- **8K context** (8192): Minimum for basic conversations
- **16K context** (16384): Good for conversations with some memory
- **32K context** (32768): **Recommended** - Supports rich conversations with documents and extensive memory
- **128K+ context**: For very long conversations or heavy document analysis (requires compatible models)

**Note**: Ollama users don't need to worry about this - Ollama automatically manages context windows based on the model's capabilities.

### Running Chorus Engine

#### Option 1: Use Startup Scripts (Recommended)

**Windows:**
```cmd
start.bat
```

**PowerShell:**
```powershell
.\start.ps1
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

The startup scripts will:
- Check dependencies
- Start the backend server
- Open the web interface in your browser
- Provide graceful shutdown with Ctrl+C

#### Option 2: Manual Start

```bash
python -m chorus_engine.main
```

Then open your browser to `http://localhost:8080`

---

## First-Time Setup: Try Nova! 🌟

After installation, we **highly recommend** running the Nova character setup to see all of Chorus Engine's features in action:

```bash
# Windows
addons\nova-setup\setup_nova.bat

# Linux/Mac
./addons/nova-setup/setup_nova.sh
```

### What the Nova Setup Provides

The automated setup configures Nova with:
- ✨ **Profile Picture**: Beautiful AI-generated portrait
- 🎤 **Voice Sample**: Professional voice cloning for TTS
- 🎨 **Image Generation**: ComfyUI workflow template (requires configuration)

**Perfect for:**
- First-time users wanting to explore all features
- Testing the complete Chorus Engine experience
- Learning how to configure characters

### After Setup

1. Start Chorus Engine (using `start.bat` or `python -m chorus_engine.main`)
2. Open the web interface at `http://localhost:8080`
3. Select **Nova** from the character sidebar
4. Start chatting!
5. Enable the TTS toggle to hear Nova's voice
6. Try image generation by asking her to "take a photo" (after workflow configuration)

> **Note**: Image generation requires ComfyUI setup and workflow configuration. See `addons/nova-setup/README.md` for detailed instructions on configuring the workflow for your system.

---

## Using the Web Interface

The web interface provides a complete chat experience:

1. **Select a Character** - Choose from available characters (Nova, Alex)
2. **Create a Conversation** - Click "New Conversation"
3. **Start Chatting** - Type your messages and get AI responses
4. **Manage Conversations** - View history, edit titles, switch between conversations
5. **Multiple Threads** - Create separate discussion threads within conversations

### Features

- ✅ Persistent conversation history (stored in SQLite)
- ✅ Multiple conversations per character
- ✅ Thread support for organizing discussions
- ✅ Message history with timestamps
- ✅ Beautiful Bootstrap UI with dark sidebar
- ✅ Responsive design (mobile-friendly)

---

## API Endpoints

### Health & Characters

- `GET /health` - Check system health
- `GET /characters` - List all characters
- `GET /characters/{id}` - Get character details

### Conversations

- `POST /conversations` - Create new conversation
- `GET /conversations` - List conversations
- `GET /conversations/{id}` - Get conversation details
- `PATCH /conversations/{id}` - Update title
- `DELETE /conversations/{id}` - Delete conversation

### Threads

- `POST /conversations/{id}/threads` - Create thread
- `GET /conversations/{id}/threads` - List threads
- `GET /threads/{id}` - Get thread details
- `PATCH /threads/{id}` - Update thread title
- `DELETE /threads/{id}` - Delete thread

### Messages

- `GET /threads/{id}/messages` - Get all messages
- `POST /threads/{id}/messages` - Send message and get response

### API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

---

## Try it out (Command Line)

1. **Check health:**
```bash
curl http://localhost:8080/health
```

2. **List characters:**
```bash
curl http://localhost:8080/characters
```

3. **Create a conversation:**
```bash
curl -X POST http://localhost:8080/conversations \
  -H "Content-Type: application/json" \
  -d '{"character_id": "nova", "title": "Creative Brainstorm"}'
```

4. **Send a message:**
```bash
# Replace {thread_id} with actual thread ID from conversation
curl -X POST http://localhost:8080/threads/{thread_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "Help me brainstorm creative project ideas!"}'
```

4. Chat with Alex:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What are some best practices for error handling in Python?\", \"character_id\": \"alex\"}"
```

## Project Structure

```
Code/
├── chorus_engine/           # Main application package
│   ├── api/                # FastAPI routes and app
│   ├── config/             # Configuration models and loader
│   ├── llm/                # LLM client integration
│   ├── models/             # Data models (future: SQLAlchemy)
│   ├── utils/              # Utility functions
│   └── main.py             # Entry point
├── characters/             # Character YAML configurations
│   ├── nova.yaml          # Creative AI assistant
│   └── alex.yaml          # Technical problem solver
├── config/                 # System configuration
│   └── system.yaml        # LLM, memory, ComfyUI settings
├── workflows/              # ComfyUI workflow files (future)
├── data/                   # Database and persistent data
└── requirements.txt        # Python dependencies
```

## Configuration

### System Config (`config/system.yaml`)

Configure LLM backend, memory settings, and API options. Falls back to defaults if not present.

### Character Configs (`characters/*.yaml`)

Each character has:
- **id**: Unique identifier (used in API calls)
- **name**: Display name
- **role**: Character's role/specialty
- **system_prompt**: Core personality and behavior
- **personality_traits**: List of traits
- **emotional_range**: Allowed emotional states
- **visual_identity**: Settings for image generation (future)
- **ambient_activity**: Auto-generated activity settings (future)

## API Endpoints

### `GET /health`
Check system health and LLM availability

### `GET /characters`
List all available characters

### `GET /characters/{character_id}`
Get details for a specific character

### `POST /chat`
Send a message to a character:
```json
{
  "message": "Your message here",
  "character_id": "nova"
}
```

Response:
```json
{
  "response": "Character's response",
  "character_name": "Nova"
}
```

## Development Status

**Phase 1 (Current)**: ✅ Basic functionality
- FastAPI server with health check
- Configuration loading and validation
- Ollama LLM integration
- Simple chat endpoint (no conversation history yet)
- Two example characters (Nova and Alex)

**Phase 2 (Next)**: Conversations, memory, streaming
- Conversation and thread management
- Message history persistence
- Streaming responses (SSE)
- Vector memory with ChromaDB
- Character management and immersion levels
- Automatic memory extraction
- Privacy mode for conversations
- ComfyUI image generation integration

**Future Enhancements**:
- Voice interaction (TTS/STT) - *Deferred to future*
- Ambient activities system
- Long conversation management
- Advanced memory exploration

## Customization

### Creating New Characters

1. Copy an existing character YAML from `characters/`
2. Modify the fields (ensure unique `id`)
3. Restart the server

### Changing LLM Model

Edit `config/system.yaml`:
```yaml
llm:
  model: llama3.1:8b  # or any other Ollama model
```

---

## Image Generation (Phase 5)

Chorus Engine can generate images during conversations using ComfyUI integration.

### Prerequisites

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** installed and running at `http://localhost:8188`
- **At least one working workflow** that generates images
- **Stable Diffusion model** loaded in ComfyUI

### Quick Setup

1. **Configure Chorus Engine** - Edit `config/system.yaml`:
```yaml
comfyui:
  enabled: true
  url: "http://localhost:8188"
  timeout_seconds: 300
  polling_interval_seconds: 2
  max_concurrent_jobs: 2
```

2. **Export Your ComfyUI Workflow**:
   - In ComfyUI, open your working workflow
   - In the positive prompt node, enter: `__CHORUS_PROMPT__`
   - In the negative prompt node (optional), enter: `__CHORUS_NEGATIVE__`
   - Click Settings (gear icon) → Enable "Enable Dev mode Options"
   - Click "Save (API Format)"
   - Save as `workflow.json`

3. **Add Workflow to Chorus Engine**:
```powershell
mkdir workflows\nova
copy path\to\workflow.json workflows\nova\workflow.json
```

4. **Configure Character for Images** - Edit `characters/nova.yaml`:
```yaml
image_generation:
  enabled: true
  workflow_file: workflow.json
  trigger_word: nvai  # Optional: for character-specific LoRAs
  default_style: "digital art, ethereal, cosmic aesthetic"
  negative_prompt: "blurry, low quality, distorted"
  self_description: "A creative AI assistant"
```

### Using Image Generation

Simply ask your character naturally:
- "Show me what you look like"
- "Can you draw a sunset over the ocean?"
- "Generate a picture of a cozy coffee shop"

The system will:
1. Detect your image request
2. Generate a detailed 100-300 word prompt using AI
3. Show a confirmation modal with the prompt
4. Generate the image via ComfyUI
5. Display it inline in the conversation

### Workflow Management

Use the "Manage Workflows" button in the UI to:
- Upload new workflows
- Set a default workflow for each character
- Rename or delete workflows
- View existing workflows

### VRAM Optimization (Optional)

If running both the LLM and ComfyUI on the same GPU:

```yaml
llm:
  unload_during_image_generation: true
```

This unloads the LLM from VRAM before image generation, then reloads it after. Adds a few seconds of reload time but gives ComfyUI full access to your GPU memory.

**Character-Specific Models**: Each character can specify a preferred LLM model and parameters in their config:

```yaml
preferred_llm:
  provider: null              # Optional: specify provider (defaults to Ollama)
  model: "dolphin-mistral-nemo:12b"  # Model name from Ollama
  temperature: 0.9            # 0.0 = focused, 1.0 = creative
  max_tokens: 2048            # Maximum response length
```

Available parameters:
- **model**: The Ollama model name (e.g., `qwen2.5:14b-instruct`, `dolphin-mistral-nemo:12b`)
- **temperature**: Controls randomness (0.0-1.0). Lower = more focused, higher = more creative
- **max_tokens**: Maximum length of generated responses (default: 2048)
- **provider**: Optional provider specification (null defaults to Ollama)

Chorus Engine automatically switches models when you switch characters. The system ensures that all operations (chat, memory extraction, image prompt generation) use the character's preferred model consistently. If you notice unexpected model loading, check `log_ollama_status()` diagnostic output in the console.

### Placeholder System

Chorus Engine replaces these placeholders in your workflows:
- `__CHORUS_PROMPT__` - Generated positive prompt
- `__CHORUS_NEGATIVE__` - Negative prompt (optional)
- `__CHORUS_SEED__` - Random seed for variation (optional)

### Troubleshooting Images

**ComfyUI not connecting:**
```powershell
curl http://localhost:8188/system_stats
```
If no response, start ComfyUI: `python main.py` in the ComfyUI directory.

**"Workflow file not found":**
```powershell
ls workflows\nova\workflow.json
```
Make sure it's in the correct character subfolder.

**"Failed to inject prompts":**
- Re-export from ComfyUI with placeholders in place
- Use API format (not regular save)
- Ensure `__CHORUS_PROMPT__` is in your workflow

For detailed ComfyUI setup, see `Documentation/Development/PHASE_5_COMFYUI_SETUP.md`.

---

## Phase 6: Text-to-Speech (TTS)

Chorus Engine can generate voice audio for character responses using ComfyUI workflows.

### TTS Setup

**Prerequisites:**
- ComfyUI server running (same as image generation)
- TTS nodes installed in ComfyUI (e.g., F5-TTS, E2-TTS, or your preferred TTS model)
- A default TTS workflow exported from ComfyUI

**1. Create TTS Workflow in ComfyUI:**
- Set up your preferred TTS model in ComfyUI
- Add these placeholders to your workflow:
  - `__CHORUS_TEXT__` - Text to speak (required)
  - `__CHORUS_VOICE_SAMPLE__` - Path to voice sample file (optional, for voice cloning)
  - `__CHORUS_VOICE_TRANSCRIPT__` - Transcript of voice sample (optional)
- Export workflow in API format
- Save as `workflows/defaults/default_tts_workflow.json`

**2. Upload Voice Sample (Optional):**
- In the web UI, click character settings
- Upload a 5-30 second voice sample
- Enter the exact transcript of what's spoken
- Check "Set as default" to use for all conversations

**3. Enable TTS for Conversation:**
- In a conversation, find the TTS toggle checkbox
- Check it to enable automatic audio generation
- Character responses will now generate audio automatically

### TTS Features

**Voice Sample Management:**
- Upload multiple voice samples per character
- One default sample per character
- Samples stored in `data/voice_samples/<character>/`

**Per-Conversation Toggle:**
- Enable/disable TTS for each conversation independently
- Setting persists across sessions
- Falls back to character default if not set

**Custom Workflows:**
- Upload custom TTS workflows via Workflow Management modal
- Select "TTS" workflow type when uploading
- Set as default for the character
- Workflows stored in `workflows/<character>/audio/`

**Audio Controls:**
- Regenerate audio with different settings
- Delete audio to free up disk space
- Audio served via `/audio/{filename}` endpoint
- Files stored in `data/audio/`

### TTS Configuration

Add to character YAML (optional):

```yaml
tts_generation:
  enabled: true                    # Allow TTS for this character
  always_on: false                 # Auto-generate for every message
  default_workflow: default_tts_workflow  # Which workflow to use
```

### TTS Placeholder System

Chorus Engine replaces these placeholders in TTS workflows:
- `__CHORUS_TEXT__` - The character's response text (preprocessed from markdown)
- `__CHORUS_VOICE_SAMPLE__` - Absolute path to the default voice sample audio file
- `__CHORUS_VOICE_TRANSCRIPT__` - The transcript text of the voice sample

### Troubleshooting TTS

**"No voice sample uploaded":**
- Upload a voice sample in character settings
- Make sure transcript matches the audio exactly
- Check the sample is set as default

**TTS generates but no audio plays:**
- Check browser console for auto-play blocking (normal behavior)
- Click the play button on the audio player
- Some browsers block auto-play until user interaction

**"Workflow not found":**
- Make sure `default_tts_workflow.json` exists in `workflows/defaults/`
- Or upload a custom TTS workflow via the UI
- Check the workflow is set as default for the character

**Audio quality poor:**
- Use higher quality voice samples (24kHz+, clear audio)
- Ensure transcript is exact (including punctuation)
- Try a different TTS model in ComfyUI

For detailed TTS setup, see `Documentation/Development/PHASE_6_COMPLETE.md`.

---

## Troubleshooting

**"LLM not available"**
- **If using Ollama**:
  - Make sure Ollama is running: `ollama serve`
  - Check the model is pulled: `ollama list`
  - Verify URL is `http://localhost:11434` in `config/system.yaml`
- **If using LM Studio**:
  - Make sure LM Studio's local server is enabled
  - Check a model is loaded in LM Studio
  - Verify URL is `http://localhost:1234` in `config/system.yaml`
  - Verify provider is set to `"lmstudio"` in `config/system.yaml`

**Character not found**
- Check the character YAML file exists in `characters/`
- Verify the `id` field matches the filename (without .yaml)
- Check server logs for validation errors

**Server won't start**
- Check Python version: `python --version` (need 3.11+)
- Verify all dependencies: `pip install -r requirements.txt`
- Check port 8080 isn't already in use

## License

See [LICENSE.md](LICENSE.md)

## Documentation

See the `Documentation/Development/` directory for detailed guides:
- **PHASE_5_COMFYUI_SETUP.md** - Complete ComfyUI integration guide
- **DEVELOPMENT_PHASES.md** - Full development roadmap and phase status
- **PHASE_5_COMPLETE.md** - Phase 5 implementation details and results
- API specifications and technical documentation
- Character schema and configuration
- Memory model and token budget management
- And more...
