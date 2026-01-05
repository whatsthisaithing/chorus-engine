"""
Audio Generation Orchestrator
Coordinates TTS audio generation using ComfyUI workflows.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from chorus_engine.services.comfyui_client import ComfyUIClient
from chorus_engine.services.workflow_manager import WorkflowManager, WorkflowType
from chorus_engine.services.audio_preprocessing import AudioPreprocessingService
from chorus_engine.services.audio_storage import AudioStorageService
from chorus_engine.repositories.voice_sample_repository import VoiceSampleRepository
from chorus_engine.repositories.audio_repository import AudioRepository
from chorus_engine.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class AudioGenerationError(Exception):
    """Base exception for audio generation errors."""
    pass


class AudioResult:
    """
    Result of audio generation operation.
    
    Attributes:
        success: Whether generation succeeded
        audio_filename: Filename of generated audio (if successful)
        duration: Generation duration in seconds
        error_message: Error description (if failed)
        preprocessed_text: The plain text sent to TTS
    """
    
    def __init__(
        self,
        success: bool,
        audio_filename: Optional[str] = None,
        duration: Optional[float] = None,
        error_message: Optional[str] = None,
        preprocessed_text: Optional[str] = None
    ):
        self.success = success
        self.audio_filename = audio_filename
        self.duration = duration
        self.error_message = error_message
        self.preprocessed_text = preprocessed_text


class AudioGenerationOrchestrator:
    """
    Orchestrates text-to-speech audio generation using ComfyUI.
    
    Coordinates:
    - Text preprocessing (markdown to plain English)
    - Voice sample loading
    - Workflow preparation and placeholder injection
    - ComfyUI job submission and polling
    - Audio file storage
    - Database record creation
    - VRAM coordination (unload LLM if needed)
    """
    
    def __init__(
        self,
        comfyui_client: Optional[ComfyUIClient] = None,
        workflow_manager: Optional[WorkflowManager] = None,
        preprocessing_service: Optional[AudioPreprocessingService] = None,
        storage_service: Optional[AudioStorageService] = None
    ):
        """
        Initialize the audio generation orchestrator.
        
        Args:
            comfyui_client: ComfyUI client (creates default if None)
            workflow_manager: Workflow manager (creates default if None)
            preprocessing_service: Text preprocessing service (creates default if None)
            storage_service: Audio storage service (creates default if None)
        """
        self.comfyui_client = comfyui_client or ComfyUIClient()
        self.workflow_manager = workflow_manager or WorkflowManager()
        self.preprocessing = preprocessing_service or AudioPreprocessingService()
        self.storage = storage_service or AudioStorageService()
        
        logger.info("AudioGenerationOrchestrator initialized")
    
    async def generate_audio(
        self,
        message_id: str,
        text: str,
        character_id: str,
        voice_sample_repository: VoiceSampleRepository,
        audio_repository: AudioRepository,
        workflow_name: Optional[str] = None
    ) -> AudioResult:
        """
        Generate TTS audio for a message.
        
        Full workflow:
        1. Validate text is suitable for TTS
        2. Preprocess markdown to plain English
        3. Load voice sample (if available)
        4. Load and prepare workflow
        5. Submit to ComfyUI and wait for completion
        6. Save audio file and database record
        
        Args:
            message_id: ID of the message to generate audio for
            text: The message text (markdown format)
            character_id: Character ID
            voice_sample_repository: Repository for voice sample access
            audio_repository: Repository for audio record access
            workflow_name: Specific workflow to use (uses default if None)
        
        Returns:
            AudioResult with success status and audio filename or error
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate text
            is_valid, error_msg = self.preprocessing.validate_text_for_tts(text)
            if not is_valid:
                logger.warning(f"Text validation failed for message {message_id}: {error_msg}")
                return AudioResult(
                    success=False,
                    error_message=f"Text not suitable for TTS: {error_msg}"
                )
            
            # Step 2: Preprocess text
            logger.info(f"Preprocessing text for message {message_id}")
            preprocessed_text = self.preprocessing.preprocess_for_tts(text)
            
            if not preprocessed_text:
                logger.warning(f"No text remaining after preprocessing for message {message_id}")
                return AudioResult(
                    success=False,
                    error_message="No text remaining after preprocessing"
                )
            
            logger.debug(f"Preprocessed text ({len(preprocessed_text)} chars): {preprocessed_text[:100]}...")
            
            # Step 3: Load voice sample (if available)
            voice_sample = voice_sample_repository.get_default_for_character(character_id)
            
            voice_sample_path = None
            voice_transcript = None
            voice_sample_id = None
            
            if voice_sample:
                # Get ABSOLUTE path to voice sample file (ComfyUI requires absolute paths)
                voice_samples_dir = Path("data/voice_samples") / character_id
                voice_sample_file = voice_samples_dir / voice_sample.filename
                
                if voice_sample_file.exists():
                    # Convert to absolute path for ComfyUI
                    voice_sample_path = str(voice_sample_file.absolute())
                    voice_transcript = voice_sample.transcript
                    voice_sample_id = voice_sample.id
                    logger.info(f"Using voice sample: {voice_sample.filename} (absolute path: {voice_sample_path})")
                else:
                    logger.warning(f"Voice sample file not found: {voice_sample_file}")
            else:
                logger.info(f"No voice sample found for character {character_id}, using empty placeholders")
            
            # Step 4: Load and prepare workflow
            logger.info(f"Loading audio workflow for character {character_id}")
            
            try:
                workflow_data = self.workflow_manager.load_workflow_by_type(
                    character_id=character_id,
                    workflow_type=WorkflowType.AUDIO,
                    workflow_name=workflow_name
                )
            except Exception as e:
                logger.error(f"Failed to load audio workflow: {e}")
                return AudioResult(
                    success=False,
                    error_message=f"Failed to load workflow: {str(e)}"
                )
            
            # Inject audio placeholders
            workflow_data = self.workflow_manager.inject_audio_placeholders(
                workflow_data=workflow_data,
                text=preprocessed_text,
                voice_sample_path=voice_sample_path,
                transcript=voice_transcript
            )
            
            logger.debug(f"Workflow prepared with placeholders injected")
            
            # Step 5: Submit to ComfyUI
            logger.info(f"Submitting audio generation to ComfyUI")
            
            try:
                prompt_id = await self.comfyui_client.submit_workflow(workflow_data)
                logger.info(f"Audio generation submitted: prompt_id={prompt_id}")
            except Exception as e:
                logger.error(f"Failed to submit workflow to ComfyUI: {e}")
                return AudioResult(
                    success=False,
                    error_message=f"Failed to submit to ComfyUI: {str(e)}"
                )
            
            # Step 6: Poll for completion
            logger.info(f"Waiting for audio generation to complete...")
            
            try:
                # Wait for completion and get the audio data from ComfyUI
                await self.comfyui_client.wait_for_completion(prompt_id)
                
                # Retrieve audio file data via ComfyUI API (not filesystem)
                audio_data = await self._get_audio_result(prompt_id)
                
                if not audio_data:
                    raise AudioGenerationError("No audio data retrieved from ComfyUI")
                
                logger.info(f"Audio generation complete, retrieved {len(audio_data)} bytes")
            except Exception as e:
                logger.error(f"Audio generation failed: {e}")
                return AudioResult(
                    success=False,
                    error_message=f"Generation failed: {str(e)}"
                )
            
            # Step 7: Save audio file
            logger.info(f"Saving audio file for message {message_id}")
            
            try:
                audio_filename = self.storage.save_audio(
                    audio_data=audio_data,
                    message_id=message_id
                )
                logger.info(f"Audio saved: {audio_filename}")
            except Exception as e:
                logger.error(f"Failed to save audio file: {e}")
                return AudioResult(
                    success=False,
                    error_message=f"Failed to save audio: {str(e)}"
                )
            
            # Step 8: Create database record
            generation_duration = time.time() - start_time
            
            try:
                audio_repository.create(
                    message_id=message_id,
                    audio_filename=audio_filename,
                    workflow_name=workflow_name or "workflow",
                    generation_duration=generation_duration,
                    text_preprocessed=preprocessed_text,
                    voice_sample_id=voice_sample_id
                )
                logger.info(f"Audio record created for message {message_id}")
            except Exception as e:
                logger.error(f"Failed to create audio record: {e}")
                # Audio file exists but no DB record - log warning but return success
                logger.warning(f"Audio generated successfully but database record creation failed")
            
            return AudioResult(
                success=True,
                audio_filename=audio_filename,
                duration=generation_duration,
                preprocessed_text=preprocessed_text
            )
        
        except Exception as e:
            logger.error(f"Unexpected error during audio generation: {e}", exc_info=True)
            return AudioResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    async def _get_audio_result(self, prompt_id: str) -> Optional[bytes]:
        """
        Retrieve generated audio data from ComfyUI.
        
        Similar to image retrieval, downloads audio via ComfyUI's /view endpoint
        instead of accessing filesystem directly.
        
        Args:
            prompt_id: ComfyUI prompt ID
        
        Returns:
            Audio file bytes or None if not found
        """
        try:
            # Get job history to find output files
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.comfyui_client.base_url}/history/{prompt_id}")
                response.raise_for_status()
                
                history = response.json()
                
                if prompt_id not in history:
                    logger.error(f"Prompt {prompt_id} not found in history")
                    return None
                
                job_data = history[prompt_id]
                
                if "outputs" not in job_data:
                    logger.error(f"Prompt {prompt_id} has no outputs")
                    return None
                
                outputs = job_data["outputs"]
                
                # Look for audio output node
                audio_info = None
                for node_id, node_output in outputs.items():
                    if "audio" in node_output:
                        audio_list = node_output["audio"]
                        if audio_list and len(audio_list) > 0:
                            audio_info = audio_list[0]
                            break
                
                if not audio_info:
                    logger.error(f"No audio found in prompt {prompt_id} outputs")
                    return None
                
                # Download the audio file
                filename = audio_info["filename"]
                subfolder = audio_info.get("subfolder", "")
                audio_type = audio_info.get("type", "output")
                
                params = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": audio_type
                }
                
                logger.info(f"Downloading audio: {filename} from ComfyUI")
                
                response = await client.get(
                    f"{self.comfyui_client.base_url}/view",
                    params=params
                )
                response.raise_for_status()
                
                logger.info(f"Successfully retrieved audio for prompt {prompt_id}")
                return response.content
                
        except Exception as e:
            logger.error(f"Failed to retrieve audio from ComfyUI: {e}")
            return None
