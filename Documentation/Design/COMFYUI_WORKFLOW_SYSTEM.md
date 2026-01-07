# ComfyUI Workflow Orchestration System Design

**Phases**: Phase 5 (Image Generation), Phase 6 (Audio Generation), Phase 7 (Video Generation)  
**Created**: December 2025  
**Updated**: January 2026  
**Status**: Image, Audio, and Video Complete

---

## Overview

The ComfyUI Workflow Orchestration System provides a workflow-first approach to multimodal generation (images, audio, video) that treats generation as composable, character-specific workflows rather than black-box API calls. Instead of abstracting ComfyUI away, the system embraces its workflow paradigm—enabling users to design custom generation pipelines with full control over models, LoRAs, samplers, and post-processing while Chorus Engine handles placeholder injection, job management, and file storage.

This design document captures the philosophy, architecture, and design decisions behind integrating ComfyUI as the generation engine for visual and audio content.

---

## Core Philosophy

### The Workflow-First Principle

**Central Insight**: Generation quality depends on specific models, LoRAs, samplers, and post-processing. Don't hide these details—expose them as user-controllable workflows.

**The Problem with Black-Box Generation**:
- ❌ Hidden model selection (user can't control quality)
- ❌ No LoRA support (can't use character-specific styles)
- ❌ Fixed sampling parameters (can't fine-tune)
- ❌ Limited post-processing (no upscaling, no face restoration)
- ❌ Vendor lock-in (tied to specific API)

**The Workflow Solution**:
```
User designs workflow in ComfyUI:
  1. Select base model (SDXL, Flux, etc.)
  2. Add LoRAs for character style
  3. Configure sampler (steps, CFG, scheduler)
  4. Add post-processing (upscaling, face detail)
  5. Export as JSON

Chorus Engine:
  1. Loads character's workflow
  2. Injects prompt text (__CHORUS_PROMPT__)
  3. Submits to ComfyUI
  4. Polls for completion
  5. Stores result with metadata
```

**Why Workflow-First Works**:
- User has complete control over generation pipeline
- Character-specific workflows (Nova uses anime LoRAs, Alex uses photorealistic)
- Supports any ComfyUI node (future-proof)
- No vendor lock-in (runs locally)
- Community can share workflows

---

### The Character-Scoped Workflow Principle

**Central Insight**: Each character should have their own visual/audio identity through custom workflows.

**Implementation**:
```
workflows/
  nova/
    image/
      default.json    # Nova's default image workflow
      portrait.json   # Closeup portrait workflow
      scene.json      # Full scene workflow
    audio/
      default.json    # Nova's TTS workflow (F5-TTS + voice sample)
  alex/
    image/
      default.json    # Alex's image workflow (different models)
    audio/
      default.json    # Alex's TTS workflow
```

**Workflow Database**:
```sql
CREATE TABLE workflows (
    id INTEGER PRIMARY KEY,
    character_name VARCHAR(50),
    workflow_name VARCHAR(100),
    workflow_file_path VARCHAR(500),  -- Path to JSON file
    workflow_type VARCHAR(20),        -- 'image', 'audio', 'video'
    is_default BOOLEAN
);
```

**Why Character-Scoped Works**:
- Nova gets anime-style images with her specific LoRAs
- Alex gets photorealistic technical diagrams
- Each character's audio uses their voice sample
- Easy to add new characters (create folder, add workflows)

---

### The Placeholder-Injection Principle

**Central Insight**: Workflows are templates. Chorus Engine injects dynamic content at generation time.

**Standard Placeholders**:
| Placeholder | Purpose | Example |
|-------------|---------|---------|
| `__CHORUS_PROMPT__` | Positive text prompt | "Portrait of Nova, anime style, vibrant colors" |
| `__CHORUS_NEGATIVE__` | Negative prompt | "low quality, blurry, distorted" |
| `__CHORUS_SEED__` | Random seed for reproducibility | 42 |
| `__CHORUS_TEXT__` | TTS input text | "Hello! How can I help you today?" |
| `__CHORUS_VOICE_SAMPLE__` | Path to voice sample WAV | "/data/voice_samples/nova_sample.wav" |
| `__CHORUS_VOICE_TRANSCRIPT__` | Transcript of voice sample | "This is a sample of my voice." |

**Injection Process**:
```python
def inject_prompt(workflow_data: dict, positive_prompt: str, negative_prompt: str = "", seed: int = -1):
    """Recursively inject placeholders into workflow JSON."""
    for node_id, node in workflow_data.items():
        if "inputs" in node:
            for key, value in node["inputs"].items():
                if value == "__CHORUS_PROMPT__":
                    node["inputs"][key] = positive_prompt
                elif value == "__CHORUS_NEGATIVE__":
                    node["inputs"][key] = negative_prompt
                elif value == "__CHORUS_SEED__" and seed >= 0:
                    node["inputs"][key] = seed
    return workflow_data
```

**Why Placeholder Injection Works**:
- Workflows stay generic (shareable)
- Chorus Engine provides dynamic content
- Same workflow works for many prompts
- User can manually override (set seed, adjust prompt)

---

### The VRAM Coordination Principle

**Central Insight**: LLM and ComfyUI both need VRAM. Coordinate to prevent OOM errors.

**The Problem**:
- LLM: 14B model = ~8GB VRAM
- ComfyUI: SDXL + LoRAs = 6-10GB VRAM
- Total needed: 14-18GB (exceeds most GPUs)

**The Solution** (Optional, configurable):
```python
# Phase 5: Optional VRAM coordination
if config.unload_during_image_generation:
    await llm_client.unload_all_models()  # Free VRAM
    # ComfyUI has maximum VRAM available
    result = await generate_image(...)
    # LLM auto-reloads on next message
```

**Phase 6.3: ComfyUI Lock**:
```python
# Ensure sequential execution (no concurrent jobs)
async with comfyui_lock:
    # Only one generation at a time
    result = await comfyui_client.submit_workflow(...)
```

**Why Coordination Works**:
- Optional unloading (user configures via system.yaml)
- Provider-agnostic (Ollama supports unload, LM Studio auto-manages)
- Lock prevents simultaneous jobs (audio + image conflict)
- Transparent to user (happens automatically)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                Character Configuration (YAML)                │
│  default_image_workflow: "default"                          │
│  default_audio_workflow: "default"                          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  WorkflowManager        │
        │  - Load workflows       │
        │  - Inject placeholders  │
        │  - Validate structure   │
        └────────────┬────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼──────┐ ┌────▼──────┐ ┌────▼─────────┐
│  Image     │ │  Audio    │ │  Video       │
│  Orchestr. │ │  Orchestr.│ │  (Future)    │
└─────┬──────┘ └────┬──────┘ └──────────────┘
      │              │
      └──────────────┼─────────────────────┐
                     │                     │
        ┌────────────▼────────────┐        │
        │    ComfyUIClient        │        │
        │  - submit_workflow()    │        │
        │  - poll_workflow()      │        │
        │  - get_history()        │        │
        │  - health_check()       │        │
        └─────────────────────────┘        │
                     │                     │
        ┌────────────▼────────────┐        │
        │    ComfyUI Server       │        │
        │  (External Process)     │        │
        │  - Runs on localhost    │        │
        │  - Processes workflows  │        │
        │  - Returns images/audio │        │
        └─────────────────────────┘        │
                     │                     │
        ┌────────────▼────────────┐        │
        │  Storage Services       │        │
        │  - ImageStorage         │        │
        │  - AudioStorage         │        │
        │  Organize files by      │        │
        │  conversation           │        │
        └─────────────────────────┘        │
                                            │
        ┌───────────────────────────────────▼────┐
        │      Database (SQLite)                 │
        │  - images table (metadata)             │
        │  - audio_messages table                │
        │  - workflows table                     │
        └────────────────────────────────────────┘
```

### Workflow Types

```python
from enum import Enum

class WorkflowType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"  # Future

# Folder structure
workflows/
  {character_id}/
    image/        # WorkflowType.IMAGE
      *.json
    audio/        # WorkflowType.AUDIO
      *.json
    video/        # WorkflowType.VIDEO (future)
      *.json
```

---

## Key Components

### 1. WorkflowManager Service

**Purpose**: Load, validate, and prepare workflows for submission.

**Key Methods**:
```python
class WorkflowManager:
    def load_workflow_by_type(
        self,
        character_id: str,
        workflow_type: WorkflowType,
        workflow_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load character's workflow of specified type."""
        pass
    
    def inject_prompt(
        self,
        workflow_data: Dict,
        positive_prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict:
        """Inject image generation placeholders."""
        pass
    
    def inject_audio_placeholders(
        self,
        workflow_data: Dict,
        text: str,
        voice_sample_path: Optional[str] = None,
        transcript: Optional[str] = None
    ) -> Dict:
        """Inject audio generation placeholders."""
        pass
    
    def list_workflows_by_type(
        self,
        character_id: str,
        workflow_type: WorkflowType
    ) -> List[str]:
        """List available workflows for character and type."""
        pass
    
    def save_workflow_by_type(
        self,
        character_id: str,
        workflow_type: WorkflowType,
        workflow_name: str,
        workflow_data: Dict
    ) -> None:
        """Save workflow to filesystem and database."""
        pass
```

**Validation**:
- Checks for required placeholders
- Validates JSON structure
- Ensures ComfyUI nodes exist

---

### 2. ComfyUIClient Service

**Purpose**: Async HTTP client for ComfyUI API communication.

**Key Methods**:
```python
class ComfyUIClient:
    async def health_check(self) -> bool:
        """Check if ComfyUI is running."""
        pass
    
    async def submit_workflow(self, workflow_data: Dict) -> str:
        """Submit workflow and return prompt_id."""
        pass
    
    async def poll_workflow(
        self,
        prompt_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0
    ) -> List[str]:
        """Poll until complete, return output file paths."""
        pass
    
    async def get_history(self, prompt_id: str) -> Dict:
        """Get workflow execution history."""
        pass
    
    async def list_models(self) -> Dict:
        """Get available models from ComfyUI."""
        pass
```

**Error Handling**:
- `ComfyUIConnectionError`: Server not reachable
- `ComfyUIJobError`: Job failed during generation
- `ComfyUITimeoutError`: Job exceeded timeout

**Polling Strategy**:
```python
async def poll_workflow(self, prompt_id: str, timeout: float = 300.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        history = await self.get_history(prompt_id)
        if prompt_id in history:
            # Check if complete
            status = history[prompt_id].get("status", {})
            if status.get("completed"):
                return self._extract_output_files(history[prompt_id])
        await asyncio.sleep(1.0)
    raise ComfyUITimeoutError(f"Workflow {prompt_id} exceeded timeout")
```

---

### 3. Image Generation Orchestrator

**Purpose**: Coordinates image generation pipeline end-to-end.

**Generation Flow**:
```python
async def generate_image(
    db: Session,
    conversation_id: str,
    thread_id: str,
    character: CharacterConfig,
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    message_id: Optional[int] = None,
    workflow_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete image generation pipeline:
    1. Load character's workflow
    2. Inject prompts
    3. Optional: Unload LLM (VRAM coordination)
    4. Submit to ComfyUI
    5. Poll for completion
    6. Copy to conversation folder
    7. Generate thumbnail
    8. Save metadata to database
    9. Optional: Reload LLM
    10. Return image URL and metadata
    """
```

**Database Schema**:
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    conversation_id VARCHAR(36),
    thread_id VARCHAR(36),
    character_id VARCHAR(50),
    message_id INTEGER,  -- Optional: image tied to specific message
    prompt TEXT,
    negative_prompt TEXT,
    seed INTEGER,
    workflow_name VARCHAR(100),
    filename VARCHAR(255),
    thumbnail_filename VARCHAR(255),
    width INTEGER,
    height INTEGER,
    generation_time_seconds REAL,
    created_at TIMESTAMP
);
```

---

### 4. Audio Generation Orchestrator

**Purpose**: Coordinates TTS audio generation pipeline.

**Generation Flow**:
```python
async def generate_audio(
    message_id: str,
    text: str,
    character_id: str,
    voice_sample_repository: VoiceSampleRepository,
    audio_repository: AudioRepository,
    workflow_name: Optional[str] = None
) -> AudioResult:
    """
    Complete audio generation pipeline:
    1. Validate text (no markdown, emojis)
    2. Preprocess markdown to plain text
    3. Load voice sample (if available)
    4. Load audio workflow
    5. Inject placeholders (__CHORUS_TEXT__, __CHORUS_VOICE_SAMPLE__)
    6. Acquire ComfyUI lock
    7. Submit to ComfyUI
    8. Poll for completion
    9. Copy to conversation folder
    10. Save metadata to database
    11. Release lock
    12. Return audio URL and metadata
    """
```

**Database Schema**:
```sql
CREATE TABLE audio_messages (
    id INTEGER PRIMARY KEY,
    message_id INTEGER UNIQUE,  -- Links to messages table
    character_id VARCHAR(50),
    workflow_name VARCHAR(100),
    voice_sample_id INTEGER,    -- Links to voice_samples table
    filename VARCHAR(255),
    duration_seconds REAL,
    generation_time_seconds REAL,
    created_at TIMESTAMP
);
```

---

### 5. Storage Services

**ImageStorage**:
```python
class ImageStorageService:
    def save_image(
        self,
        source_path: Path,
        conversation_id: str,
        filename: str
    ) -> tuple[Path, Path]:
        """
        Save image and generate thumbnail.
        
        Returns:
            (full_path, thumbnail_path)
        """
        # Save to data/images/{conversation_id}/
        # Generate 200x200 thumbnail
        # Return both paths
```

**AudioStorage**:
```python
class AudioStorageService:
    def save_audio(
        self,
        source_path: Path,
        conversation_id: str,
        message_id: int
    ) -> Path:
        """
        Save audio file.
        
        Returns:
            Path to saved audio file
        """
        # Save to data/audio/{conversation_id}/
        # Naming: message_{message_id}_{timestamp}.wav
```

---

## Generation Workflows

### Image Generation Workflow

**User Triggers**:
1. Click "Generate Image" button
2. Specify prompt or use LLM-generated description

**Process**:
```
1. Frontend: POST /threads/{thread_id}/images
   Body: { prompt: "...", negative_prompt: "...", seed: 42 }

2. Backend: Load character's default image workflow

3. LLM Prompt Service (optional): Generate detailed prompt
   "Portrait of Nova, anime style, vibrant colors, detailed eyes..."

4. WorkflowManager: Inject prompts into workflow JSON

5. VRAM Coordination (optional):
   - Unload LLM models
   - Free VRAM for ComfyUI

6. ComfyUI Client: Submit workflow
   - POST /prompt with workflow JSON
   - Receive prompt_id

7. Poll for completion:
   - GET /history/{prompt_id} every 1 second
   - Check status.completed = true

8. Download output:
   - GET /view?filename={output_filename}
   - Save to data/images/{conversation_id}/

9. Generate thumbnail (200x200)

10. Save metadata to database:
    - prompt, seed, workflow_name, dimensions, generation_time

11. Reload LLM (if unloaded)

12. Return: { image_url, thumbnail_url, metadata }

13. Frontend: Display image in chat
```

**Typical Latency**:
- Workflow load: ~10ms
- LLM unload: ~500ms (optional)
- ComfyUI generation: 10-60 seconds (model/GPU dependent)
- File copy: ~100ms
- Thumbnail: ~200ms
- Database save: ~10ms
- **Total**: 10-60 seconds (dominated by ComfyUI)

---

### Audio Generation Workflow

**User Triggers**:
1. Automatic: Character responds with text → auto-generate audio
2. Manual: Click "Generate Audio" button on existing message

**Process**:
```
1. Frontend: POST /conversations/{id}/messages/{msg_id}/audio

2. Backend: Load message text from database

3. Audio Preprocessing:
   - Remove markdown formatting (**, __, *, etc.)
   - Remove emojis and special characters
   - Convert to plain English

4. Voice Sample Lookup:
   - Query voice_samples table for character
   - Get WAV file path and transcript

5. WorkflowManager: Load audio workflow
   - workflows/{character_id}/audio/default.json

6. Inject placeholders:
   - __CHORUS_TEXT__: Preprocessed message text
   - __CHORUS_VOICE_SAMPLE__: /data/voice_samples/nova_sample.wav
   - __CHORUS_VOICE_TRANSCRIPT__: "This is a sample..."

7. Acquire ComfyUI lock (sequential execution)

8. ComfyUI Client: Submit workflow
   - POST /prompt with workflow JSON
   - Receive prompt_id

9. Poll for completion:
   - GET /history/{prompt_id} every 1 second
   - Check status.completed = true

10. Download output:
    - GET /view?filename={output_audio_filename}
    - Save to data/audio/{conversation_id}/message_{id}_{timestamp}.wav

11. Save metadata to database:
    - message_id, workflow_name, voice_sample_id, duration, generation_time

12. Release ComfyUI lock

13. Return: { audio_url, duration, metadata }

14. Frontend: Display audio player for message
```

**Typical Latency**:
- Text preprocessing: ~10ms
- Workflow load: ~10ms
- ComfyUI generation: 5-30 seconds (text length dependent)
- File copy: ~50ms
- Database save: ~10ms
- **Total**: 5-30 seconds

---

## Design Decisions & Rationale

### Decision: ComfyUI Instead of Custom Pipeline

**Alternatives Considered**:
1. **Stable Diffusion WebUI API**
   - ❌ Less flexible than ComfyUI nodes
   - ❌ Limited workflow composability

2. **Direct diffusers library** (Python)
   - ❌ Requires maintaining our own pipeline code
   - ❌ No GUI for users to design workflows
   - ❌ Slower iteration (code changes vs. workflow changes)

3. **Cloud APIs** (Replicate, Together, etc.)
   - ❌ API costs
   - ❌ No local control
   - ❌ Privacy concerns

4. **ComfyUI** (chosen) ✅
   - ✅ Node-based workflow system (infinitely flexible)
   - ✅ GUI for workflow design
   - ✅ Runs locally (no API costs, full privacy)
   - ✅ Active community (tons of custom nodes)
   - ✅ Supports image AND audio generation

**Why ComfyUI Works**:
- Users design workflows visually in ComfyUI
- Export JSON and place in workflows/ folder
- Chorus Engine handles automation (prompts, job management, storage)
- Future-proof (any ComfyUI node works)

---

### Decision: Character-Specific Workflows Instead of Global

**Alternatives Considered**:
1. **Single global workflow**
   - ❌ All characters look the same
   - ❌ Can't use character-specific LoRAs

2. **Model selection at runtime**
   - ❌ Doesn't handle LoRAs, samplers, post-processing
   - ❌ Too limited

3. **Character-specific workflows** (chosen) ✅
   - ✅ Each character has distinct visual/audio identity
   - ✅ Nova: anime LoRAs, vibrant colors
   - ✅ Alex: photorealistic, technical diagrams
   - ✅ Full control over entire generation pipeline

**Why Character-Scoped Works**:
- Visual consistency per character
- User can customize per character
- Easy to share (send workflow JSON + character YAML)

---

### Decision: Placeholder Injection Instead of Hardcoded Prompts

**Alternatives Considered**:
1. **Hardcoded prompts in workflow**
   - ❌ Workflow tied to specific prompt
   - ❌ Not reusable

2. **String replacement in JSON** (e.g., `{PROMPT}`)
   - ❌ No standard format
   - ❌ Brittle (breaks if user types `{PROMPT}` in prompt)

3. **Unique placeholder strings** (chosen) ✅
   - ✅ `__CHORUS_PROMPT__` unlikely to collide
   - ✅ Clear naming convention
   - ✅ Extensible (add more placeholders as needed)

**Standard Placeholders**:
- `__CHORUS_PROMPT__`: Positive prompt
- `__CHORUS_NEGATIVE__`: Negative prompt
- `__CHORUS_SEED__`: Random seed
- `__CHORUS_TEXT__`: TTS text
- `__CHORUS_VOICE_SAMPLE__`: Voice sample path
- `__CHORUS_VOICE_TRANSCRIPT__`: Voice transcript

---

### Decision: Optional VRAM Coordination Instead of Always-Unload

**Alternatives Considered**:
1. **Always unload LLM before ComfyUI**
   - ❌ Adds latency even if user has enough VRAM
   - ❌ Annoying for high-VRAM users (RTX 4090 with 24GB)

2. **Never unload** (user manages manually)
   - ❌ OOM errors for low-VRAM users (RTX 3080 with 10GB)
   - ❌ Poor user experience

3. **Optional unloading** (chosen) ✅
   - ✅ User configures via system.yaml
   - ✅ High-VRAM users: disable unloading
   - ✅ Low-VRAM users: enable unloading
   - ✅ Provider-agnostic (works with Ollama, skips gracefully with LM Studio)

**Configuration**:
```yaml
# system.yaml
visual_generation:
  unload_during_image_generation: true  # Default: false
```

---

### Decision: Sequential Execution Lock Instead of Queue

**Alternatives Considered**:
1. **No coordination** (allow concurrent jobs)
   - ❌ Image + audio generation simultaneously = OOM
   - ❌ Unpredictable VRAM usage

2. **Job queue with priority**
   - ❌ Added complexity
   - ❌ Overkill for single-user system

3. **Simple async lock** (chosen) ✅
   - ✅ `async with comfyui_lock:` ensures sequential execution
   - ✅ If image generating, audio waits
   - ✅ If audio generating, image waits
   - ✅ Automatic release via context manager

**Implementation**:
```python
# Phase 6.3: ComfyUI lock
comfyui_lock = asyncio.Lock()

# In image generation endpoint
async with comfyui_lock:
    result = await generate_image(...)

# In audio generation endpoint
async with comfyui_lock:
    result = await generate_audio(...)
```

---

## Known Limitations

### 1. ComfyUI Must Be Running Separately

**Limitation**: User must start ComfyUI manually before using Chorus Engine.

**Why**: ComfyUI is a separate application (Python with custom nodes).

**Workaround**:
- Health check warns if ComfyUI not available
- UI disables generation buttons if health check fails

**Future**: Could auto-start ComfyUI as subprocess.

---

### 2. No Real-Time Progress Updates

**Limitation**: Frontend only knows "started" → "completed". No intermediate progress.

**Why**: ComfyUI doesn't expose per-step progress easily via API.

**Workaround**: Show spinner with estimated time.

**Future**: WebSocket connection to ComfyUI for real-time progress events.

---

### 3. Workflow Validation Is Limited

**Limitation**: System checks for placeholders but can't validate if ComfyUI workflow will actually work.

**Why**: ComfyUI has hundreds of custom nodes, impossible to validate all.

**Workaround**:
- User tests workflow in ComfyUI before exporting
- Error messages from ComfyUI logged for debugging

**Future**: Pre-submit workflow validation via ComfyUI API.

---

### 4. No Batch Generation

**Limitation**: One prompt = one image/audio. Can't generate multiple variations.

**Why**: Simplified initial implementation.

**Workaround**: Generate multiple times with different seeds.

**Future**: Batch generation endpoint:
```
POST /threads/{id}/images/batch
{ prompts: [...], seeds: [...] }
```

---

## Performance Characteristics

**Workflow Load**: O(1), ~10ms (read JSON file)

**Placeholder Injection**: O(n), ~1-5ms (n = number of nodes in workflow)

**ComfyUI Health Check**: O(1), ~10-50ms (HTTP request to localhost)

**Image Generation Latency**:
- SDXL (20 steps, no upscaling): 10-20 seconds
- SDXL (20 steps, 4x upscaling): 30-60 seconds
- Flux (varies significantly by hardware)

**Audio Generation Latency** (F5-TTS):
- Short message (<50 words): 5-10 seconds
- Medium message (50-150 words): 10-20 seconds
- Long message (>150 words): 20-30 seconds

**VRAM Usage**:
- LLM (14B model): ~8GB
- ComfyUI (SDXL + LoRAs): 6-10GB
- With unloading: Sequential (max 10GB at any time)
- Without unloading: Concurrent (18GB needed)

**Storage**:
- Image: 2-8MB per image (SDXL output)
- Thumbnail: 20-50KB per image
- Audio: 500KB-5MB per message (depends on length)

---

## Testing & Validation

### Unit Tests

```python
def test_workflow_load():
    manager = WorkflowManager()
    workflow = manager.load_workflow_by_type("nova", WorkflowType.IMAGE, "default")
    assert workflow is not None
    assert "prompt" in workflow or isinstance(workflow, dict)

def test_placeholder_injection():
    workflow = {"1": {"inputs": {"text": "__CHORUS_PROMPT__"}}}
    result = inject_prompt(workflow, "Test prompt")
    assert result["1"]["inputs"]["text"] == "Test prompt"

def test_audio_placeholder_injection():
    workflow = {
        "1": {"inputs": {"text": "__CHORUS_TEXT__"}},
        "2": {"inputs": {"audio": "__CHORUS_VOICE_SAMPLE__"}}
    }
    result = inject_audio_placeholders(workflow, "Hello", "/path/to/sample.wav")
    assert result["1"]["inputs"]["text"] == "Hello"
    assert result["2"]["inputs"]["audio"] == "/path/to/sample.wav"
```

### Integration Tests

**Requires Running ComfyUI**:
```python
@pytest.mark.integration
async def test_image_generation_end_to_end():
    # Assumes ComfyUI running on localhost:8188
    client = ComfyUIClient()
    assert await client.health_check()
    
    workflow = load_workflow("nova", WorkflowType.IMAGE, "default")
    workflow = inject_prompt(workflow, "Portrait of Nova")
    
    prompt_id = await client.submit_workflow(workflow)
    assert prompt_id
    
    outputs = await client.poll_workflow(prompt_id, timeout=60.0)
    assert len(outputs) > 0
```

---

## Migration Guide

### Adding ComfyUI Generation to Existing Character

**Step 1**: Design workflow in ComfyUI
- Open ComfyUI
- Load/create your workflow
- Add placeholders (e.g., `__CHORUS_PROMPT__` in CLIPTextEncode)
- Test with sample prompt
- Export as JSON (right-click → "Save (API Format)")

**Step 2**: Place workflow in character folder
```
workflows/
  your_character_id/
    image/
      default.json    # Your exported workflow
```

**Step 3**: Update character config
```yaml
# characters/your_character_id.yaml
default_image_workflow: "default"  # References default.json
```

**Step 4**: Test generation
- Start ComfyUI
- Start Chorus Engine
- Send message to character
- Click "Generate Image"
- Verify image appears in chat

---

### Migrating from Phase 5 (Image Only) to Phase 6+ (Image + Audio)

**Changes**:
- Workflows now organized by type: `workflows/{char}/image/` and `workflows/{char}/audio/`
- Old flat structure: `workflows/{char}/workflow.json`
- New structure: `workflows/{char}/image/default.json`

**Migration Script** (automatic on startup):
```python
# If workflows/{char}/workflow.json exists but workflows/{char}/image/ doesn't
# Move workflow.json → image/default.json
```

---

## Future Enhancements

### High Priority

**1. Video Generation Support**
- WorkflowType.VIDEO
- AnimateDiff, SVD (Stable Video Diffusion)
- MP4 storage and streaming

**2. Real-Time Progress Updates**
- WebSocket connection to ComfyUI
- Per-step progress (Step 5/20, Upscaling, etc.)
- Frontend progress bar

**3. Batch Generation**
- Multiple prompts/seeds in one request
- Grid view of variations
- A/B testing workflows

### Medium Priority

**4. Workflow Validation**
- Pre-submit validation via ComfyUI API
- Check for missing models/nodes
- Warn about incompatible settings

**5. Auto-Start ComfyUI**
- Launch ComfyUI as subprocess
- Graceful shutdown on exit
- Health monitoring with auto-restart

**6. Workflow Templates Library**
- Community-shared workflows
- One-click import
- Tagging and search

### Low Priority

**7. Generation History View**
- Gallery of all generated images/audio
- Filter by character, conversation, date
- Regenerate with same seed

**8. Workflow Versioning**
- Track workflow changes over time
- Diff view for changes
- Rollback to previous version

---

## Conclusion

The ComfyUI Workflow Orchestration System represents Chorus Engine's commitment to user control and flexibility in multimodal generation. By embracing ComfyUI's workflow paradigm rather than abstracting it away, the system provides users with complete control over generation pipelines while handling the automation details (placeholder injection, job management, storage, VRAM coordination).

Key achievements:
- **Workflow-First Design**: User-controlled pipelines via ComfyUI JSON workflows
- **Character-Scoped Workflows**: Each character has distinct visual/audio identity
- **Placeholder Injection**: Workflows stay generic, content injected at runtime
- **Type-Based Organization**: Image, audio, video workflows organized by type
- **VRAM Coordination**: Optional LLM unloading and sequential job execution
- **Complete Automation**: Job submission, polling, storage, metadata management

The system has proven successful through:
- Nova: Anime-style images with character-specific LoRAs
- Alex: Photorealistic technical diagrams
- Both characters: Custom TTS workflows with voice samples
- Video: Motion-focused generation with AnimateDiff/CogVideoX
- Community can share workflows (just export JSON)

Future enhancements (real-time progress, batch generation) build naturally on this foundation. The workflow-first approach provides the perfect balance of user control and automated convenience.

**Status**: Production-ready for image, audio, and video generation. Recommended pattern for all multimodal generation needs.

---

## Video Generation Architecture

### Video vs. Image: Key Differences

**Video Generation** (Phase 7, January 2026):
- **Focus**: Motion, dynamic action, temporal progression
- **Prompt Strategy**: Action verbs, camera movement, pacing
- **Timeout**: 600 seconds (vs. 300 for images)
- **File Size**: 5-50MB (vs. 1-5MB for images)
- **Workflows**: Whatever you like (Wan 2.2 tested)
- **Validation**: Motion keyword checking

### VideoGenerationOrchestrator

**Service**: `VideoGenerationOrchestrator`  
**Location**: `chorus_engine/services/video_generation_orchestrator.py`

**Workflow**:
```
1. Generate motion-focused prompt (VideoPromptService)
2. Load video workflow from database
3. Inject prompt via WorkflowManager
4. Submit to ComfyUI
5. Poll for completion (longer timeout)
6. Download result
7. Extract thumbnail from first frame
8. Store video file
9. Create database record
```

**Key Methods**:
- `generate_video()` - Full generation pipeline
- Reuses WorkflowManager for prompt injection (same as images)
- Longer timeout (600s vs 300s)
- Video-specific storage handling

### VideoPromptService

**Service**: `VideoPromptService`  
**Location**: `chorus_engine/services/video_prompt_service.py`

**Motion-Focused Prompting**:
- Emphasizes dynamic action and movement
- Includes camera motion guidance (pan, zoom, track)
- Specifies pacing/timing (slow motion, transitions)
- Validates for motion keywords
- Temperature: 0.3 (same as images for consistency)

**Prompt Template Differences**:

| Aspect | Image Prompt | Video Prompt |
|--------|--------------|--------------|
| Focus | Static composition | Dynamic motion |
| Length | 100-300 words | 100-150 words |
| Verbs | Descriptive (is, has) | Action (moves, flows) |
| Camera | Optional framing | Essential motion |
| Examples | "sunset over ocean" | "waves crash in slow motion" |

**Context Usage**:
- Extracts motion-related details from conversation
- Last 3 messages for normal generation
- Last 10 messages for scene capture
- Prioritizes recent messages for current state

### Video Scene Capture

**Trigger**: 🎥 "Capture Scene" button  
**Perspective**: Third-person observer  
**Purpose**: Capture current conversation moment with motion

**Scene Capture Prompt Strategy**:
- Always third-person perspective
- Focuses on what's actively happening
- Includes all participants in motion
- Describes visible actions, gestures, expressions
- Camera movement to capture scene dynamically

**Example**: Conversation about discussing art → "Close shot of {character.name} speaking, hands moving expressively as they explain concepts, facial expressions shifting with enthusiasm, subtle head tilts and nods during conversation. Gentle camera drift following hand gestures."

### Video Storage

**Service**: `VideoStorageService`  
**Location**: `chorus_engine/services/video_storage.py`

**Storage Structure**:
```
data/videos/
  {conversation_id}/
    {video_id}.mp4 (or .webm, .gif, etc.)
```

**File Handling**:
- Accepts any video format ComfyUI produces
- Extension auto-detected from ComfyUI output
- Atomic file operations (temp → final)
- Database record includes file size, format, duration

### Video Workflows

**Workflow Placeholders**:
- `__CHORUS_PROMPT__` - Motion-focused description
- `__CHORUS_NEGATIVE_PROMPT__` - Things to avoid
- `__CHORUS_SEED__` - Reproducibility (optional)
- Future: `__CHORUS_DURATION__`, `__CHORUS_FPS__`

**Workflow Configuration** (Database):
```yaml
default_style: "cinematic, smooth motion"
self_description: "{character} appearance for consistent depiction"
negative_prompt: "static shots, jerky motion, poor lighting"
trigger_word: "FLUX_TRIGGER" # Character-specific LoRA trigger
```
---

## Workflow Orchestration Comparison

### Image vs. Audio vs. Video

| Feature | Image | Audio | Video |
|---------|-------|-------|-------|
| **Prompt Service** | ImagePromptService | (N/A - direct text) | VideoPromptService |
| **Orchestrator** | ImageGenerationOrchestrator | AudioGenerationOrchestrator | VideoGenerationOrchestrator |
| **Prompt Focus** | Static composition | N/A | Dynamic motion |
| **Timeout** | 300s | 120s | 600s |
| **File Size** | 1-5MB | 0.5-2MB | 5-50MB |
| **Post-Processing** | Thumbnail | Waveform | Thumbnail extraction |
| **Context Window** | 10 messages | N/A | 3-10 messages |
| **Validation** | Visual keywords | N/A | Motion keywords |

### Common Architecture

All three generation types share:
- **WorkflowManager** for placeholder injection
- **ComfyUIClient** for job submission/polling
- **Database-managed workflows** (not filesystem)
- **Confirmation dialogs** before generation
- **User can edit prompt** before submission
- **Inline display** in conversation thread

---

**Document Version**: 1.1  
**Last Updated**: January 7, 2026  
**Author**: System Design Documentation  
**Phases**: Phase 5 (Image), Phase 6 (Audio), Phase 7 (Video)
