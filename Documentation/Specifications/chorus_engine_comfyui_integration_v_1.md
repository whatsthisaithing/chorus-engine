# Chorus Engine – ComfyUI Integration Mechanics (v1)

This document defines how Chorus Engine integrates with ComfyUI for visual generation, including API communication, workflow management, queue handling, and file organization.

ComfyUI integration enables:
- Character-aware image generation
- Workflow-based model/LoRA management
- Asynchronous job processing
- Conversation-associated visual artifacts

---

## Background: Understanding ComfyUI

### What Is ComfyUI?

**ComfyUI** is a node-based visual generation system that executes workflows defined as directed graphs.

**Key concepts**:
- **Workflows**: JSON files defining nodes and connections
- **Nodes**: Processing steps (load model, apply LoRA, generate, save)
- **Queues**: Jobs submitted via API, executed sequentially or in parallel
- **Outputs**: Images saved to ComfyUI's output directory

### Why ComfyUI for Chorus Engine?

**Advantages**:
- **Model agnostic**: Supports FLUX, SDXL, SD1.5, Z-Image, etc.
- **Workflow-first**: Matches our visual specification approach
- **Local execution**: No API costs, full privacy
- **Extensible**: Custom nodes and models
- **Active development**: Rapidly evolving ecosystem

**Trade-offs**:
- External dependency (must be running)
- Requires separate installation
- File management complexity

---

## Architecture Overview

```
Chorus Engine                          ComfyUI
     |                                    |
     | 1. Submit workflow + params        |
     |-------------------------------->   |
     |                                    | 2. Queue job
     |                                    | 3. Execute workflow
     | 4. Poll for status                 |
     |<--------------------------------   |
     |                                    | 5. Complete, save output
     | 6. Retrieve image path             |
     |<--------------------------------   |
     | 7. Move to conversation folder     |
     |                                    |
```

---

## ComfyUI API Endpoints

ComfyUI exposes a REST API (default port: 8188).

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/prompt` | POST | Submit workflow for execution |
| `/queue` | GET | Get current queue status |
| `/history` | GET | Get execution history |
| `/history/{prompt_id}` | GET | Get specific job status and outputs |
| `/view` | GET | Download output image |
| `/object_info` | GET | Get available nodes/models |
| `/system_stats` | GET | Get system status |

### WebSocket Interface

ComfyUI also provides WebSocket for real-time updates:
- Connect to `ws://localhost:8188/ws`
- Receive execution progress events
- Get completion notifications

**v1 decision**: Use HTTP polling (simpler), not WebSocket.

---

## Workflow Loading and Validation

### Workflow File Structure

ComfyUI workflows are JSON files with this structure:

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "flux-dev.safetensors"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "PROMPT_PLACEHOLDER",
      "clip": ["1", 1]
    }
  },
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 30,
      "cfg": 4.5,
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    }
  }
  // ... more nodes
}
```

### Loading Workflows

```python
import json
from pathlib import Path

class WorkflowLoader:
    def __init__(self, workflows_dir: str = "workflows/"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows = {}
        self.load_all_workflows()
    
    def load_all_workflows(self):
        """Load all workflow JSON files from workflows directory."""
        for workflow_file in self.workflows_dir.glob("*.json"):
            workflow_name = workflow_file.stem
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                
                self.workflows[workflow_name] = {
                    'name': workflow_name,
                    'path': str(workflow_file),
                    'data': workflow_data,
                    'nodes': self._extract_node_info(workflow_data)
                }
                
            except Exception as e:
                logger.error(f"Failed to load workflow {workflow_name}: {e}")
    
    def _extract_node_info(self, workflow_data: dict) -> dict:
        """Extract key node information for validation."""
        nodes = {}
        for node_id, node_data in workflow_data.items():
            nodes[node_id] = {
                'type': node_data.get('class_type'),
                'inputs': node_data.get('inputs', {})
            }
        return nodes
    
    def get_workflow(self, name: str) -> dict:
        """Get workflow by name."""
        if name not in self.workflows:
            raise ValueError(f"Workflow '{name}' not found")
        return self.workflows[name]
```

### Workflow Validation

```python
def validate_workflow(workflow_data: dict) -> dict:
    """
    Validate workflow structure and required nodes.
    
    Checks:
    - Has required node types (checkpoint loader, sampler, saver)
    - Has prompt placeholder
    - Has output node
    """
    
    errors = []
    warnings = []
    
    # Extract node types
    node_types = [
        node.get('class_type') 
        for node in workflow_data.values()
    ]
    
    # Check for required node types
    required_types = ['CheckpointLoaderSimple', 'KSampler', 'SaveImage']
    for req_type in required_types:
        if req_type not in node_types:
            errors.append(f"Missing required node type: {req_type}")
    
    # Check for prompt placeholder
    has_prompt_placeholder = False
    for node in workflow_data.values():
        inputs = node.get('inputs', {})
        for key, value in inputs.items():
            if isinstance(value, str) and 'PROMPT_PLACEHOLDER' in value:
                has_prompt_placeholder = True
                break
    
    if not has_prompt_placeholder:
        warnings.append("No PROMPT_PLACEHOLDER found - prompt injection may not work")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
```

---

## Parameter Injection

### Identifying Injectable Parameters

Workflows can define placeholders for dynamic values:

```json
{
  "inputs": {
    "text": "PROMPT_PLACEHOLDER",
    "seed": "SEED_PLACEHOLDER",
    "steps": "STEPS_PLACEHOLDER",
    "cfg": "CFG_PLACEHOLDER"
  }
}
```

### Simplified Parameter Injection

**Only inject the positive prompt** - workflows handle everything else.

```python
def inject_prompt_only(workflow_data: dict, prompt: str) -> dict:
    """
    Inject only the positive prompt into workflow.
    
    Parameters:
        workflow_data: Original workflow JSON
        prompt: Complete positive prompt string generated by LLM
    
    Returns:
        Modified workflow with prompt injected
    """
    
    # Deep copy to avoid modifying original
    workflow = json.loads(json.dumps(workflow_data))
    
    # Recursively find and replace PROMPT_PLACEHOLDER
    def replace_prompt(obj):
        if isinstance(obj, dict):
            return {k: replace_prompt(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_prompt(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace('PROMPT_PLACEHOLDER', prompt)
        else:
            return obj
    
    return replace_prompt(workflow)
```

**Optional parameters** (if workflow supports them):
- User can override seed via API: `{"seed": 42}`
- Workflow must have `SEED_PLACEHOLDER` to use it
- If workflow has no placeholder, parameter is ignored (no error)

**Recommended workflow setup**:
- Positive prompt: `PROMPT_PLACEHOLDER` (required)
- Seed: Either hardcode -1 (random) or use `SEED_PLACEHOLDER` (optional)
- Everything else: Hardcoded in workflow (steps, CFG, size, model, LoRAs, etc.)

---

## Prompt Generation Philosophy

### Simplified Approach

**Key principle**: Workflows are self-contained black boxes. The LLM generates a complete, rich prompt. No mechanical assembly needed.

### LLM-Generated Prompts

The LLM generates the visual prompt using:
1. User's request
2. Character's visual context (from config)
3. Current activity (if relevant to the request)
4. General visual generation best practices

**Character visual context** (in config):
```yaml
visual_identity:
  default_workflow: "nova_portrait"
  prompt_context: |
    When generating images of Nova:
    - Always include "nova-style" trigger word
    - Emphasize soft, painterly aesthetic
    - Warm, expressive mood
    - Creative, artistic settings work well
```

**LLM prompt for image generation**:
```
User request: "send me a selfie of you sketching"

Character context: {character.visual_identity.prompt_context}
Current activity: sketching ideas for a new painting

Generate a detailed image generation prompt that:
- Includes all necessary trigger words
- Describes the scene, mood, and visual style
- Incorporates the current activity if relevant
- Uses comma-separated descriptors
- Focuses on visual details, not narrative

Output only the prompt, no explanation.
```

**LLM output**:
```
nova-style, young woman sketching in cozy art studio, soft painterly style, 
warm natural lighting, casual creative attire, sketchbook and art supplies 
visible, serene focused expression, artistic atmosphere, high quality portrait, 
detailed, expressive
```

**Benefits**:
- LLM handles semantic understanding (what makes sense visually)
- No hardcoded keyword mappings
- Naturally incorporates context
- Flexible to any request
- Character's "voice" extends to visual descriptions

---

## Workflow Self-Containment

### No Dynamic LoRA Injection

**v1 Philosophy**: Workflows are pre-configured with all settings, including LoRAs.

**Why this is simpler**:
- Users configure workflows in ComfyUI (familiar interface)
- No node manipulation by Chorus Engine
- Transparent: what you see in ComfyUI is what executes
- Easier debugging: test workflow directly in ComfyUI
- No version conflicts or node structure assumptions

**Shipped Workflows**:
- `nova_portrait.json` - Has Nova's LoRA pre-loaded at optimal weight
- `alex_portrait.json` - Configured for Alex's neutral style
- `generic_portrait.json` - No character-specific LoRAs

**Custom Workflows**:
Users can:
1. Create/modify workflows in ComfyUI
2. Save the JSON file to `workflows/` directory
3. Reference in character config: `default_workflow: "my_custom_workflow"`
4. Chorus Engine just passes the prompt - workflow handles everything else

**No runtime modification needed**.

---

## Job Submission

### ComfyUI Prompt API

```python
import requests
import uuid

class ComfyUIClient:
    def __init__(self, base_url: str = "http://localhost:8188"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def submit_workflow(
        self,
        workflow_data: dict,
        client_id: str = None
    ) -> dict:
        """
        Submit workflow to ComfyUI for execution.
        
        Returns:
            {
                'prompt_id': 'uuid',
                'number': 123,  # Queue position
                'node_errors': {}
            }
        """
        
        if client_id is None:
            client_id = str(uuid.uuid4())
        
        payload = {
            'prompt': workflow_data,
            'client_id': client_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"ComfyUI submission failed: {e}")
            raise ComfyUIConnectionError(f"Failed to submit workflow: {e}")
    
    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a specific prompt."""
        try:
            response = self.session.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ComfyUIConnectionError(f"Failed to get history: {e}")
    
    def get_queue(self) -> dict:
        """Get current queue status."""
        try:
            response = self.session.get(
                f"{self.base_url}/queue",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ComfyUIConnectionError(f"Failed to get queue: {e}")
    
    def check_connection(self) -> bool:
        """Check if ComfyUI is accessible."""
        try:
            response = self.session.get(
                f"{self.base_url}/system_stats",
                timeout=3
            )
            return response.status_code == 200
        except:
            return False
```

---

## Job Tracking and Polling

### Job Status States

```python
class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### Job Manager

```python
import time
from datetime import datetime, timedelta

class VisualGenerationJobManager:
    def __init__(self, comfy_client: ComfyUIClient, db_connection):
        self.comfy = comfy_client
        self.db = db_connection
        self.active_jobs = {}  # In-memory tracking
    
    def create_job(
        self,
        conversation_id: str,
        thread_id: str,
        workflow_name: str,
        prompt: str,
        parameters: dict,
        message_id: str = None
    ) -> dict:
        """Create a new visual generation job."""
        
        job_id = str(uuid.uuid4())
        
        job = {
            'id': job_id,
            'conversation_id': conversation_id,
            'thread_id': thread_id,
            'message_id': message_id,
            'workflow': workflow_name,
            'prompt': prompt,
            'parameters': parameters,
            'status': JobStatus.QUEUED,
            'created_at': datetime.now().isoformat(),
            'comfy_prompt_id': None,
            'result': None,
            'error': None
        }
        
        # Store in database
        self.db.execute("""
            INSERT INTO visual_generation_jobs
            (id, conversation_id, thread_id, message_id, workflow,
             prompt, parameters, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, conversation_id, thread_id, message_id, workflow_name,
            prompt, json.dumps(parameters), JobStatus.QUEUED, job['created_at']
        ))
        
        # Add to active jobs
        self.active_jobs[job_id] = job
        
        return job
    
    def submit_job(self, job_id: str, workflow_data: dict) -> dict:
        """Submit job to ComfyUI."""
        
        job = self.active_jobs.get(job_id)
        if not job:
            job = self._load_job_from_db(job_id)
        
        try:
            # Submit to ComfyUI
            result = self.comfy.submit_workflow(workflow_data)
            
            # Update job with ComfyUI prompt ID
            job['comfy_prompt_id'] = result['prompt_id']
            job['status'] = JobStatus.PROCESSING
            job['started_at'] = datetime.now().isoformat()
            
            self.db.execute("""
                UPDATE visual_generation_jobs
                SET comfy_prompt_id = ?, status = ?, started_at = ?
                WHERE id = ?
            """, (result['prompt_id'], JobStatus.PROCESSING, 
                  job['started_at'], job_id))
            
            return job
            
        except ComfyUIConnectionError as e:
            # Mark as failed
            job['status'] = JobStatus.FAILED
            job['error'] = {
                'code': 'COMFY_UNAVAILABLE',
                'message': str(e)
            }
            
            self.db.execute("""
                UPDATE visual_generation_jobs
                SET status = ?, error = ?
                WHERE id = ?
            """, (JobStatus.FAILED, json.dumps(job['error']), job_id))
            
            raise
    
    def poll_job(self, job_id: str) -> dict:
        """Poll job status from ComfyUI."""
        
        job = self.active_jobs.get(job_id)
        if not job:
            job = self._load_job_from_db(job_id)
        
        if job['status'] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return job  # Already terminal state
        
        try:
            # Get history from ComfyUI
            history = self.comfy.get_history(job['comfy_prompt_id'])
            
            if job['comfy_prompt_id'] in history:
                # Job has completed
                job_result = history[job['comfy_prompt_id']]
                
                # Extract output images
                outputs = self._extract_outputs(job_result)
                
                if outputs:
                    job['status'] = JobStatus.COMPLETED
                    job['completed_at'] = datetime.now().isoformat()
                    job['result'] = outputs
                    
                    self.db.execute("""
                        UPDATE visual_generation_jobs
                        SET status = ?, completed_at = ?, result = ?
                        WHERE id = ?
                    """, (JobStatus.COMPLETED, job['completed_at'],
                          json.dumps(outputs), job_id))
                else:
                    # No outputs = failed
                    job['status'] = JobStatus.FAILED
                    job['error'] = {
                        'code': 'NO_OUTPUT',
                        'message': 'ComfyUI completed but produced no output'
                    }
            
            return job
            
        except ComfyUIConnectionError as e:
            logger.error(f"Failed to poll job {job_id}: {e}")
            # Don't mark as failed on poll errors - might be temporary
            return job
    
    def _extract_outputs(self, job_result: dict) -> dict:
        """Extract output image paths from ComfyUI job result."""
        
        outputs = job_result.get('outputs', {})
        
        for node_id, node_outputs in outputs.items():
            if 'images' in node_outputs:
                images = node_outputs['images']
                if images:
                    # Get first image (handle multiple outputs later)
                    image_info = images[0]
                    return {
                        'filename': image_info['filename'],
                        'subfolder': image_info.get('subfolder', ''),
                        'type': image_info.get('type', 'output')
                    }
        
        return None
```

---

## Polling Strategy

### Adaptive Polling

Adjust polling frequency based on expected duration.

```python
class AdaptivePoller:
    def __init__(self, job_manager: VisualGenerationJobManager):
        self.job_manager = job_manager
    
    async def poll_until_complete(
        self,
        job_id: str,
        timeout_seconds: int = 300,
        initial_interval: float = 2.0,
        max_interval: float = 10.0
    ) -> dict:
        """
        Poll job until completion or timeout.
        
        Polling strategy:
        - Start with 2s interval
        - Increase to 5s after 30s
        - Increase to 10s after 60s
        """
        
        start_time = time.time()
        interval = initial_interval
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                # Mark as failed
                job = self.job_manager.active_jobs[job_id]
                job['status'] = JobStatus.FAILED
                job['error'] = {
                    'code': 'GENERATION_TIMEOUT',
                    'message': f'Generation exceeded {timeout_seconds}s timeout'
                }
                return job
            
            # Poll status
            job = self.job_manager.poll_job(job_id)
            
            # Check if terminal state
            if job['status'] in [JobStatus.COMPLETED, JobStatus.FAILED, 
                                JobStatus.CANCELLED]:
                return job
            
            # Adaptive interval
            if elapsed > 60:
                interval = max_interval
            elif elapsed > 30:
                interval = 5.0
            
            # Wait before next poll
            await asyncio.sleep(interval)
```

---

## File Management

### ComfyUI Output Handling

ComfyUI saves images to its own output directory (configurable, typically `ComfyUI/output/`).

**Challenge**: Files scattered across ComfyUI outputs, not organized by conversation.

**Solution**: Move (not copy) files to Chorus conversation structure.

### File Organization Strategy

```
data/
└── conversations/
    └── {conversation_id}/
        └── images/
            ├── {job_id}.png
            ├── {job_id}_metadata.json
            └── ...
```

### File Mover Implementation

```python
import shutil
from pathlib import Path

class ComfyUIFileMover:
    def __init__(
        self,
        comfy_output_dir: str = "ComfyUI/output",
        chorus_data_dir: str = "data/conversations"
    ):
        self.comfy_output_dir = Path(comfy_output_dir)
        self.chorus_data_dir = Path(chorus_data_dir)
    
    def move_output_to_conversation(
        self,
        comfy_output: dict,
        conversation_id: str,
        job_id: str
    ) -> dict:
        """
        Move ComfyUI output to Chorus conversation folder.
        
        Args:
            comfy_output: Output info from ComfyUI
                {
                    'filename': 'ComfyUI_00001.png',
                    'subfolder': '',
                    'type': 'output'
                }
            conversation_id: Conversation UUID
            job_id: Job UUID
        
        Returns:
            {
                'original_path': '/path/to/ComfyUI/output/file.png',
                'chorus_path': 'data/conversations/{conv}/images/{job}.png',
                'absolute_path': '/absolute/path/to/file.png'
            }
        """
        
        # Construct source path
        subfolder = comfy_output.get('subfolder', '')
        filename = comfy_output['filename']
        
        if subfolder:
            source_path = self.comfy_output_dir / subfolder / filename
        else:
            source_path = self.comfy_output_dir / filename
        
        # Construct destination path
        conv_images_dir = self.chorus_data_dir / conversation_id / 'images'
        conv_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Use job_id as filename to avoid conflicts
        file_extension = source_path.suffix
        dest_filename = f"{job_id}{file_extension}"
        dest_path = conv_images_dir / dest_filename
        
        # Move file (not copy)
        try:
            shutil.move(str(source_path), str(dest_path))
            
            # Save metadata alongside image
            metadata_path = conv_images_dir / f"{job_id}_metadata.json"
            metadata = {
                'original_filename': filename,
                'comfy_subfolder': subfolder,
                'moved_at': datetime.now().isoformat(),
                'job_id': job_id,
                'conversation_id': conversation_id
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'original_path': str(source_path),
                'chorus_path': str(dest_path.relative_to(self.chorus_data_dir.parent)),
                'absolute_path': str(dest_path.absolute()),
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to move file: {e}")
            raise FileMovementError(f"Could not move ComfyUI output: {e}")
```

### Handling File Permissions

```python
def ensure_file_accessible(file_path: Path) -> bool:
    """Ensure file is readable and movable."""
    
    try:
        # Check if file exists
        if not file_path.exists():
            return False
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            logger.warning(f"File not readable: {file_path}")
            return False
        
        # Check if parent directory is writable (for move operation)
        if not os.access(file_path.parent, os.W_OK):
            logger.warning(f"Cannot write to directory: {file_path.parent}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking file accessibility: {e}")
        return False
```

---

## Complete Generation Flow

### End-to-End Workflow

```python
async def generate_image(
    conversation_id: str,
    thread_id: str,
    user_prompt: str,
    character: dict,
    parameters: dict = None,
    workflow_override: str = None
) -> dict:
    """
    Complete image generation flow.
    
    Steps:
    1. Determine workflow
    2. Assemble visual prompt
    3. Load and configure workflow
    4. Create job
    5. Submit to ComfyUI
    6. Poll until complete
    7. Move file to conversation
    8. Return result (Simplified)

### End-to-End Workflow

```python
async def generate_image(
    conversation_id: str,
    thread_id: str,
    user_request: str,
    character: dict,
    activity: dict = None,
    workflow_override: str = None,
    seed_override: int = None
) -> dict:
    """
    Simplified image generation flow.
    
    Steps:
    1. LLM generates visual prompt
    2. Load workflow
    3. Inject prompt (and optionally seed)
    4. Submit to ComfyUI
    5. Poll until complete
    6. Move file to conversation
    7. Return result
    """
    
    # 1. Determine workflow
    workflow_name = workflow_override or \
                   character.get('visual_identity', {}).get('default_workflow')
    
    if not workflow_name:
        raise ValueError("No workflow specified and character has no default")
    
    # 2. LLM generates the visual prompt
    visual_prompt = await generate_visual_prompt_via_llm(
        user_request=user_request,
        character=character,
        activity=activity
    )
    
    # 3. Load workflow
    workflow_loader = WorkflowLoader()
    workflow = workflow_loader.get_workflow(workflow_name)
    workflow_data = workflow['data']
    
    # 4. Inject prompt (and seed if provided)
    workflow_data = inject_prompt_only(workflow_data, visual_prompt)
    
    if seed_override is not None:
        # Optional: inject seed if workflow supports it
        workflow_data = inject_optional_parameter(
            workflow_data, 
            'SEED_PLACEHOLDER', 
            seed_override
        )
    
    # 5. Create job
    job_manager = VisualGenerationJobManager(comfy_client, db)
    job = job_manager.create_job(
        conversation_id=conversation_id,
        thread_id=thread_id,
        workflow_name=workflow_name,
        prompt=visual_prompt,
        parameters={'seed': seed_override} if seed_override else {}
    )
    
    # 6. Submit to ComfyUI
    try:
        job = job_manager.submit_job(job['id'], workflow_data)
    except ComfyUIConnectionError as e:
        return {
            'job': job,
            'error': {
                'code': 'COMFY_UNAVAILABLE',
                'message': 'ComfyUI is not available',
                'recoverable': True
            }
        }
    
    # 7. Poll until complete (async)
    poller = AdaptivePoller(job_manager)
    job = await poller.poll_until_complete(
        job['id'],
        timeout_seconds=300
    )
    
    # 8. Move file if successful
    if job['status'] == JobStatus.COMPLETED:
        file_mover = ComfyUIFileMover()
        file_info = file_mover.move_output_to_conversation(
            job['result'],
            conversation_id,
            job['id']
        )
        
        # Update job with final paths
        job['result']['chorus_path'] = file_info['chorus_path']
        job['result']['absolute_path'] = file_info['absolute_path']
        
        job_manager.db.execute("""
            UPDATE visual_generation_jobs
            SET result = ?
            WHERE id = ?
        """, (json.dumps(job['result']), job['id']))
    
    return {
        'job': job,
        'success': job['status'] == JobStatus.COMPLETED
    }


async def generate_visual_prompt_via_llm(
    user_request: str,
    character: dict,
    activity: dict = None
) -> str:
    """
    Use LLM to generate a complete visual prompt.
    
    The LLM has context about:
    - User's request
    - Character's visual identity context
    - Current activity (if relevant)
    
    Returns a complete, comma-separated prompt ready for ComfyUI.
    """
    
    visual_context = character.get('visual_identity', {}).get('prompt_context', '')
    activity_description = activity.get('description', '') if activity else ''
    
    system_prompt = """You are a visual prompt generator. Convert requests into 
detailed, comma-separated image generation prompts suitable for Stable Diffusion/FLUX.

Include:
- Any required trigger words or style descriptors
- Visual details (lighting, mood, composition)
- Quality tags (high quality, detailed, etc.)

Output ONLY the prompt, no explanation."""

    user_prompt = f"""User request: {user_request}

Character visual context:
{visual_context}

Current activity: {activity_description if activity_description else 'None'}

Generate the image prompt:"""

    # Call LLM (using character's preferred model)
    response = await llm_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.strip()   'error': {
            'code': 'WORKFLOW_NOT_FOUND',
            'message': f"Workflow '{workflow_name}' not found in workflows/",
            'available_workflows': list(workflow_loader.workflows.keys())
        }
    }
```

**3. Model/LoRA Not Found**
```python
# ComfyUI will return node_errors in submission response
if 'node_errors' in submission_result and submission_result['node_errors']:
    return {
        'error': {
            'code': 'WORKFLOW_NODE_ERROR',
            'message': 'Workflow contains invalid nodes',
            'details': submission_result['node_errors'],
            'suggestion': 'Check that all models and LoRAs exist in ComfyUI'
        }
    }
```

**4. Generation Timeout**
```python
if elapsed > timeout_seconds:
    return {
        'error': {
            'code': 'GENERATION_TIMEOUT',
            'message': f'Generation exceeded {timeout_seconds}s timeout',
            'suggestion': 'Try reducing steps or using a faster model'
        }
    }
```

**5. Output File Not Found**
```python
if not source_path.exists():
    return {
        'error': {
            'code': 'OUTPUT_FILE_NOT_FOUND',
            'message': 'ComfyUI reported completion but output file not found',
            'details': {'expected_path': str(source_path)}
        }
    }
```

### Retry Logic

```python
def submit_with_retry(
    job_id: str,
    workflow_data: dict,
    max_retries: int = 3,
    backoff: float = 2.0
) -> dict:
    """Submit job with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            return job_manager.submit_job(job_id, workflow_data)
        except ComfyUIConnectionError as e:
            if attempt == max_retries - 1:
                raise  # Final attempt failed
            
            wait_time = backoff ** attempt
            logger.warning(f"Submission failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
```

---

## Queue Management

### Concurrent Jobs

ComfyUI can handle multiple jobs, but system resources are limited.

```python
class QueueManager:
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self.queue = []
    
    def can_submit(self) -> bool:
        """Check if we can submit another job."""
        return self.active_count < self.max_concurrent
    
    def submit_or_queue(self, job_id: str, workflow_data: dict):
        """Submit job or add to queue if at capacity."""
        
        if self.can_submit():
            self.active_count += 1
            return job_manager.submit_job(job_id, workflow_data)
        else:
            self.queue.append((job_id, workflow_data))
            return {
                'queued': True,
                'position': len(self.queue)
            }
    
    def job_completed(self, job_id: str):
        """Mark job as complete and submit next from queue."""
        self.active_count -= 1
        
        if self.queue:
            next_job_id, next_workflow = self.queue.pop(0)
            self.submit_or_queue(next_job_id, next_workflow)
```

---

## Configuration

### System Configuration

```yaml
comfyui:
  url: "http://localhost:8188"
  timeout_seconds: 300
  connection_retry_attempts: 3
  connection_retry_backoff: 2.0
  
  polling:
    initial_interval_seconds: 2.0
    max_interval_seconds: 10.0
  
  queue:
    max_concurrent_jobs: 2
    max_queue_size: 10
  
  file_management:
    output_directory: "ComfyUI/output"
    move_files: true  # Move instead of copy
    cleanup_comfy_outputs: true  # Clean up after move
  
  defaults:
    steps: 30
    cfg: 4.5
    width: 1024
    height: 1024
```

---

## Monitoring & Debug

### Debug Information

```json
{
  "job": {...},
  "debug": {
    "workflow_used": "flux_dev_portrait",
    "prompt_components": {
      "user": "A serene portrait",
      "character_triggers": ["nova-style", "soft painterly"],
      "activity": null
    },
    "parameters_injected": {
      "seed": 42,
      "steps": 30,
      "cfg": 4.5
    },
    "loras_applied": [
      {"name": "nova_style", "weight": 0.8}
    ],
    "comfy_submission": {
      "prompt_id": "uuid",
      "queue_position": 1,
      "submitted_at": "2025-12-27T10:00:00Z"
    },
    "timing": {
      "queue_wait_ms": 500,
      "generation_ms": 45000,
      "file_move_ms": 150,
      "total_ms": 45650
    }
  }
}
```

---

## Summary

ComfyUI integration in Chorus Engine:

1. **Loads workflows** from JSON files with validation
2. **Injects parameters** dynamically (prompt, seed, steps, etc.)
3. **Applies character LoRAs** based on visual identity
4. **Assembles visual prompts** from multiple sources
5. **Submits jobs** to ComfyUI via REST API
6. **Polls adaptively** until completion or timeout
7. **Moves files** to conversation-specific folders
8. **Handles errors gracefully** with clear messaging
9. **Manages queue** to prevent resource exhaustion
10. **Provides transparency** via debug information

This system enables character-aware visual generation while maintaining the workflow-first, model-agnostic architecture.

---

## Implementation Checklist

- [ ] Build ComfyUI REST client with connection checking
- [ ] Implement workflow loader and validator
- [ ] Build parameter injection system
- [ ] Create visual prompt assembler
- [ ] Implement LoRA injection
- [ ] Build job manager with database persistence
- [ ] Create adaptive polling system
- [ ] Implement file mover with error handling
- [ ] Add queue management for concurrent jobs
- [ ] Create timeout and retry logic
- [ ] Add debug endpoints for troubleshooting
- [ ] Test with multiple workflows (FLUX, SDXL, etc.)
- [ ] Document workflow creation guide for users

**Next**: Configuration validation, Ambient activity state machine, or other gaps identified.
