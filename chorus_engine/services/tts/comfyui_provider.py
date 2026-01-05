"""
ComfyUI TTS Provider

Workflow-based TTS using ComfyUI server.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .base_provider import BaseTTSProvider, TTSRequest, TTSResult
from ..audio_preprocessing import AudioPreprocessingService
from ..workflow_manager import WorkflowManager, WorkflowType

logger = logging.getLogger(__name__)


class ComfyUITTSProvider(BaseTTSProvider):
    """TTS provider using ComfyUI workflows."""
    
    def __init__(
        self,
        comfyui_client,
        comfyui_lock,
        workflow_manager: WorkflowManager,
        audio_storage
    ):
        """
        Initialize ComfyUI TTS provider.
        
        Args:
            comfyui_client: ComfyUI client for workflow submission
            comfyui_lock: Async lock for ComfyUI operations
            workflow_manager: Workflow manager for loading/injecting workflows
            audio_storage: Audio storage service for saving files
        """
        self.comfyui_client = comfyui_client
        self.comfyui_lock = comfyui_lock
        self.workflow_manager = workflow_manager
        self.audio_storage = audio_storage
        self.preprocessor = AudioPreprocessingService()
    
    @property
    def provider_name(self) -> str:
        return "comfyui"
    
    async def generate_audio(self, request: TTSRequest) -> TTSResult:
        """
        Generate audio using ComfyUI workflow.
        
        This wraps the existing AudioGenerationOrchestrator logic,
        refactored to fit the provider interface.
        """
        start_time = time.time()
        
        # Get workflow name from provider config or use default
        workflow_name = None
        if request.provider_config:
            workflow_name = request.provider_config.get('workflow_name')
        
        if not workflow_name:
            workflow_name = "default_tts_workflow"
        
        try:
            # Load workflow
            logger.info(f"[ComfyUI] Loading workflow '{workflow_name}' for character {request.character_id}")
            
            try:
                workflow_data = self.workflow_manager.load_workflow_by_type(
                    character_id=request.character_id,
                    workflow_type=WorkflowType.AUDIO,
                    workflow_name=workflow_name
                )
            except Exception as e:
                logger.error(f"[ComfyUI] Failed to load workflow: {e}")
                return TTSResult(
                    success=False,
                    error_message=f"Failed to load workflow '{workflow_name}': {str(e)}",
                    provider_name=self.provider_name
                )
            
            # Inject placeholders
            logger.debug(f"[ComfyUI] Injecting audio placeholders")
            workflow_data = self.workflow_manager.inject_audio_placeholders(
                workflow_data=workflow_data,
                text=request.text,
                voice_sample_path=request.voice_sample_path,
                transcript=request.voice_transcript
            )
            
            # Submit to ComfyUI (with lock to prevent concurrent jobs)
            async with self.comfyui_lock:
                logger.info(f"[ComfyUI] Submitting workflow to ComfyUI")
                
                try:
                    prompt_id = await self.comfyui_client.submit_workflow(workflow_data)
                    logger.info(f"[ComfyUI] Workflow submitted: prompt_id={prompt_id}")
                except Exception as e:
                    logger.error(f"[ComfyUI] Failed to submit workflow: {e}")
                    return TTSResult(
                        success=False,
                        error_message=f"Failed to submit to ComfyUI: {str(e)}",
                        provider_name=self.provider_name
                    )
                
                # Wait for completion
                logger.info(f"[ComfyUI] Waiting for generation to complete...")
                
                try:
                    await self.comfyui_client.wait_for_completion(prompt_id)
                    
                    # Retrieve audio data
                    audio_data = await self._get_audio_result(prompt_id)
                    
                    if not audio_data:
                        raise Exception("No audio data retrieved from ComfyUI")
                    
                    logger.info(f"[ComfyUI] Audio generation complete, retrieved {len(audio_data)} bytes")
                    
                except Exception as e:
                    logger.error(f"[ComfyUI] Generation failed: {e}")
                    return TTSResult(
                        success=False,
                        error_message=f"Generation failed: {str(e)}",
                        provider_name=self.provider_name
                    )
            
            # Save audio file
            logger.info(f"[ComfyUI] Saving audio file for message {request.message_id}")
            
            try:
                audio_filename = self.audio_storage.save_audio(
                    audio_data=audio_data,
                    message_id=request.message_id
                )
                
                audio_path = Path("data/audio") / audio_filename
                
                generation_duration = time.time() - start_time
                
                logger.info(f"[ComfyUI] Audio saved: {audio_filename} (duration: {generation_duration:.2f}s)")
                
                return TTSResult(
                    success=True,
                    audio_filename=audio_filename,
                    audio_path=audio_path,
                    generation_duration=generation_duration,
                    provider_name=self.provider_name,
                    metadata={
                        'workflow_name': workflow_name,
                        'prompt_id': prompt_id,
                        'voice_cloned': bool(request.voice_sample_path)
                    }
                )
                
            except Exception as e:
                logger.error(f"[ComfyUI] Failed to save audio file: {e}")
                return TTSResult(
                    success=False,
                    error_message=f"Failed to save audio: {str(e)}",
                    provider_name=self.provider_name
                )
        
        except Exception as e:
            logger.error(f"[ComfyUI] Unexpected error during TTS generation: {e}")
            import traceback
            traceback.print_exc()
            return TTSResult(
                success=False,
                error_message=str(e),
                provider_name=self.provider_name
            )
    
    async def _get_audio_result(self, prompt_id: str) -> Optional[bytes]:
        """
        Retrieve audio data from ComfyUI after generation.
        
        Args:
            prompt_id: The ComfyUI prompt ID
        
        Returns:
            Audio file data as bytes, or None if not found
        """
        try:
            # Get job history to find output files
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.comfyui_client.base_url}/history/{prompt_id}")
                response.raise_for_status()
                
                history = response.json()
                
                if prompt_id not in history:
                    logger.error(f"No history found for prompt {prompt_id}")
                    return None
                
                outputs = history[prompt_id].get('outputs', {})
                
                # Find the audio output (typically from SaveAudio node)
                for node_id, node_output in outputs.items():
                    if 'audio' in node_output:
                        audio_files = node_output['audio']
                        
                        if audio_files and len(audio_files) > 0:
                            audio_info = audio_files[0]
                            filename = audio_info.get('filename')
                            subfolder = audio_info.get('subfolder', '')
                            file_type = audio_info.get('type', 'output')
                            
                            # Download the audio file via ComfyUI's view endpoint
                            params = {
                                "filename": filename,
                                "subfolder": subfolder,
                                "type": file_type
                            }
                            
                            logger.debug(f"Downloading audio: {filename} from subfolder: {subfolder}")
                            
                            download_response = await client.get(
                                f"{self.comfyui_client.base_url}/view",
                                params=params
                            )
                            download_response.raise_for_status()
                            
                            logger.info(f"Successfully retrieved audio for prompt {prompt_id}")
                            return download_response.content
                
                logger.error(f"No audio output found in ComfyUI results for prompt {prompt_id}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve audio result from ComfyUI: {e}")
            return None
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Check if ComfyUI is accessible."""
        try:
            # ComfyUI client has a health check method
            if hasattr(self.comfyui_client, 'is_connected'):
                if not self.comfyui_client.is_connected():
                    return False, "ComfyUI not connected"
            
            return True, None
            
        except Exception as e:
            return False, f"ComfyUI not accessible: {str(e)}"
    
    def get_estimated_duration(self, text: str) -> float:
        """
        Estimate ComfyUI generation time.
        
        ComfyUI is async, typically 5-30 seconds depending on model.
        """
        # Estimate audio duration
        audio_duration = self.preprocessor.estimate_duration(text)
        
        # Add overhead for ComfyUI processing (typically 10-20 seconds)
        return audio_duration + 15.0
    
    def is_available(self) -> bool:
        """Check if ComfyUI provider is available."""
        is_valid, _ = self.validate_config()
        return is_valid
