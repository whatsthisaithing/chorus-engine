"""
Video generation orchestrator.

Coordinates video generation workflow: prompt creation, ComfyUI submission,
result polling, and storage. Workflow-agnostic - accepts any video format.
"""

import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from chorus_engine.services.comfyui_client import ComfyUIClient
from chorus_engine.services.video_prompt_service import VideoPromptService
from chorus_engine.services.video_storage import VideoStorageService
from chorus_engine.repositories.video_repository import VideoRepository
from chorus_engine.models.conversation import GeneratedVideo

logger = logging.getLogger(__name__)


class VideoGenerationError(Exception):
    """Base exception for video generation errors."""
    pass


class VideoGenerationOrchestrator:
    """
    Orchestrates end-to-end video generation workflow.
    
    Workflow:
    1. Generate motion-focused prompt from conversation
    2. Load and prepare video workflow
    3. Submit to ComfyUI
    4. Poll for completion (longer timeout than images)
    5. Download and store result
    6. Create database record
    
    Workflow-agnostic: Accepts any format ComfyUI produces.
    """
    
    def __init__(
        self,
        comfyui_client: ComfyUIClient,
        video_prompt_service: VideoPromptService,
        video_storage: VideoStorageService,
        video_timeout: int = 600
    ):
        """
        Initialize video generation orchestrator.
        
        Args:
            comfyui_client: ComfyUI client
            video_prompt_service: Prompt generation service
            video_storage: Video storage service
            video_timeout: Timeout for video generation in seconds (default: 600)
        """
        self.comfyui_client = comfyui_client
        self.prompt_service = video_prompt_service
        self.storage = video_storage
        
        # Video-specific timeout (longer than images)
        self.video_timeout = video_timeout
        self.poll_interval = 2  # seconds
        
        # Import workflow manager for prompt injection
        from chorus_engine.services.workflow_manager import WorkflowManager
        from pathlib import Path
        self.workflow_manager = WorkflowManager(workflows_dir=Path("workflows"))
        
        logger.info(
            f"Video generation orchestrator initialized: timeout={self.video_timeout}s"
        )
    
    async def generate_video(
        self,
        video_repository: VideoRepository,
        conversation_id: str,
        thread_id: str,
        messages: List,
        character,
        character_name: str,
        character_id: str,
        workflow_entry,
        prompt: Optional[str] = None,
        custom_instruction: Optional[str] = None,
        trigger_words: Optional[List[str]] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> GeneratedVideo:
        """
        Generate video from conversation context.
        
        Args:
            video_repository: Video database repository
            conversation_id: Conversation ID
            thread_id: Thread ID
            messages: Recent conversation messages
            character_name: Character name
            character_id: Character ID
            workflow_entry: Workflow database entry
            prompt: Optional pre-generated prompt (if provided, skips generation)
            custom_instruction: Optional user prompt guidance (used only if prompt not provided)
            trigger_words: Optional workflow trigger words (used only if prompt not provided)
            negative_prompt: Optional negative prompt
            seed: Optional seed for reproducibility
        
        Returns:
            GeneratedVideo database record
        
        Raises:
            VideoGenerationError: Generation failed
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Use provided prompt or generate new one
            logger.info(f"[VIDEO ORCHESTRATOR] Received prompt parameter: {repr(prompt)[:200] if prompt is not None else 'None'}")
            
            if prompt is None or prompt.strip() == "":
                logger.info("[VIDEO ORCHESTRATOR] No prompt provided, generating new prompt...")
                prompt_info = await self.prompt_service.generate_video_prompt(
                    messages=messages,
                    character=character,
                    character_name=character_name,
                    custom_instruction=custom_instruction,
                    trigger_words=trigger_words,
                    workflow_config=workflow_entry.workflow_config if hasattr(workflow_entry, 'workflow_config') else None
                )
                prompt = prompt_info["prompt"]
                # Use generated negative prompt if not explicitly provided
                if negative_prompt is None:
                    negative_prompt = prompt_info.get("negative_prompt", "")
                logger.info(f"[VIDEO ORCHESTRATOR] Generated prompt: {prompt[:100]}...")
                if negative_prompt:
                    logger.info(f"[VIDEO ORCHESTRATOR] Generated negative prompt: {negative_prompt[:100]}...")
            else:
                logger.info(f"[VIDEO ORCHESTRATOR] Using provided prompt (len={len(prompt)}): {prompt[:100]}...")
                # Use default negative prompt if not provided
                if negative_prompt is None:
                    negative_prompt = ""
            
            # Step 2: Load workflow from file
            logger.info(f"Loading workflow: {workflow_entry.workflow_name}")
            workflow_path = Path(workflow_entry.workflow_file_path)
            if not workflow_path.exists():
                raise VideoGenerationError(f"Workflow file not found: {workflow_path}")
            
            with open(workflow_path, 'r') as f:
                workflow_data = json.load(f)
            
            # Step 3: Inject prompts into workflow using workflow_manager (same as image generation)
            workflow_with_prompts = self.workflow_manager.inject_prompt(
                workflow_data=workflow_data,
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed
            )
            
            # Handle workflow format (may have "prompt" wrapper)
            if "prompt" in workflow_with_prompts:
                workflow_nodes = workflow_with_prompts["prompt"]
            else:
                workflow_nodes = workflow_with_prompts
            
            # Step 4: Submit to ComfyUI
            logger.info("Submitting to ComfyUI...")
            job_id = await self.comfyui_client.submit_workflow(
                workflow_data=workflow_nodes
            )
            logger.info(f"ComfyUI job ID: {job_id}")
            
            # Step 5: Wait for completion (longer timeout for videos)
            logger.info(f"Waiting for generation (timeout: {self.video_timeout}s)...")
            await self.comfyui_client.wait_for_completion(
                job_id=job_id,
                callback=None,
                timeout=self.video_timeout
            )
            
            # Step 6: Get result
            logger.info("Retrieving video...")
            video_data, extension = await self.comfyui_client.get_result(job_id)
            
            if not video_data:
                raise VideoGenerationError("No video data returned from ComfyUI")
            
            # Step 7: Save to temporary file with correct extension
            temp_dir = Path("data/videos/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_video_path = temp_dir / f"{job_id}_video{extension}"
            temp_video_path.write_bytes(video_data)
            logger.info(f"Saved temporary video with extension {extension}")
            
            # Step 8: Store video and extract thumbnail
            logger.info("Storing video...")
            
            # Create video record to get ID
            video_record = video_repository.create_video(
                conversation_id=conversation_id,
                file_path="",  # Will update after saving
                prompt=prompt,
                negative_prompt=negative_prompt,
                workflow_file=workflow_entry.workflow_file_path,
                comfy_prompt_id=job_id
            )
            
            # Save video with ID (no thumbnail needed for videos)
            video_path, thumb_path = await self.storage.save_video(
                video_path=temp_video_path,
                conversation_id=conversation_id,
                video_id=video_record.id,
                create_thumbnail=False  # Videos display inline, no thumbnail needed
            )
            
            # Step 8: Extract video metadata from saved file
            metadata = await self._extract_metadata(video_path)
            
            # Step 9: Update database record
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            video_record.file_path = str(video_path)
            video_record.thumbnail_path = str(thumb_path) if thumb_path else None
            video_record.format = metadata.get('format')
            video_record.duration_seconds = metadata.get('duration')
            video_record.width = metadata.get('width')
            video_record.height = metadata.get('height')
            video_record.generation_time_seconds = generation_time
            
            video_repository.db.commit()
            
            logger.info(
                f"Video generation complete: {video_record.id} "
                f"({generation_time:.1f}s, {metadata.get('format', 'unknown')})"
            )
            
            # Cleanup temp file
            await asyncio.to_thread(temp_video_path.unlink, missing_ok=True)
            
            return video_record
        
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise VideoGenerationError(f"Video generation failed: {e}")
    
    async def generate_scene_capture(
        self,
        video_repository: VideoRepository,
        conversation_id: str,
        thread_id: str,
        messages: List,
        character,
        character_name: str,
        character_id: str,
        workflow_entry,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> GeneratedVideo:
        """
        Generate video for scene capture (🎥 button).
        
        Args:
            video_repository: Video database repository
            conversation_id: Conversation ID
            thread_id: Thread ID
            messages: Recent messages
            character: Character configuration object
            character_name: Character name
            character_id: Character ID
            workflow_entry: Workflow database entry
            prompt: Optional user-provided prompt (if None, will generate)
            negative_prompt: Optional negative prompt
            seed: Optional seed for reproducibility
        
        Returns:
            GeneratedVideo record
        """
        try:
            # Use provided prompt or generate scene-specific prompt
            if prompt is None:
                prompt_info = await self.prompt_service.generate_scene_capture_prompt(
                    messages=messages,
                    character=character,
                    workflow_config=workflow_entry.workflow_config if hasattr(workflow_entry, 'workflow_config') else None
                )
                prompt = prompt_info["prompt"]
                # Use generated negative prompt if not explicitly provided
                if negative_prompt is None:
                    negative_prompt = prompt_info.get("negative_prompt", "")
            
            # Use standard generation flow with the prompt
            return await self.generate_video(
                video_repository=video_repository,
                conversation_id=conversation_id,
                thread_id=thread_id,
                messages=messages,
                character=character,
                character_name=character_name,
                character_id=character_id,
                workflow_entry=workflow_entry,
                prompt=prompt,  # Pass the prompt directly
                custom_instruction=None,
                trigger_words=None,
                negative_prompt=negative_prompt,
                seed=seed
            )
        
        except Exception as e:
            logger.error(f"Scene capture failed: {e}")
            raise VideoGenerationError(f"Scene capture failed: {e}")
    
    async def _load_workflow(self, workflow_file: str) -> Dict[str, Any]:
        """Load workflow JSON file."""
        try:
            workflow_path = Path(workflow_file)
            if not workflow_path.is_absolute():
                workflow_path = Path("workflows") / workflow_file
            
            with open(workflow_path, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            raise VideoGenerationError(f"Failed to load workflow: {e}")
    
    async def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract video metadata (format, duration, dimensions).
        
        Uses ffprobe if available.
        """
        metadata = {
            'format': video_path.suffix.lstrip('.').lower(),
            'duration': None,
            'width': None,
            'height': None
        }
        
        try:
            # Try ffprobe
            import subprocess
            
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract duration
                if 'format' in data and 'duration' in data['format']:
                    metadata['duration'] = float(data['format']['duration'])
                
                # Extract dimensions from video stream
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        metadata['width'] = stream.get('width')
                        metadata['height'] = stream.get('height')
                        break
        
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
