"""
Image generation orchestrator - coordinates all Phase 5 services.

Manages the complete image generation flow:
- Request detection
- Prompt generation
- Workflow loading
- ComfyUI submission
- Image storage
- Database recording
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from chorus_engine.config.models import SystemConfig, CharacterConfig
from chorus_engine.services.comfyui_client import ComfyUIClient, ComfyUIError
from chorus_engine.services.workflow_manager import WorkflowManager, WorkflowError
from chorus_engine.services.image_prompt_service import ImagePromptService
from chorus_engine.services.image_storage import ImageStorageService
from chorus_engine.repositories.image_repository import ImageRepository
from chorus_engine.repositories.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)


class ImageGenerationOrchestrator:
    """
    Orchestrates complete image generation workflow.
    
    Coordinates:
    - Prompt generation
    - Workflow management
    - ComfyUI communication
    - Image storage
    - Database updates
    """
    
    def __init__(
        self,
        system_config: SystemConfig,
        comfyui_client: ComfyUIClient,
        workflow_manager: WorkflowManager,
        prompt_service: ImagePromptService,
        storage_service: ImageStorageService
    ):
        """
        Initialize orchestrator.
        
        Args:
            system_config: System configuration
            comfyui_client: ComfyUI client
            workflow_manager: Workflow manager
            prompt_service: Prompt generation service
            storage_service: Image storage service
        """
        self.system_config = system_config
        self.comfyui_client = comfyui_client
        self.workflow_manager = workflow_manager
        self.prompt_service = prompt_service
        self.storage_service = storage_service
        
        logger.info("Image generation orchestrator initialized")
    
    async def detect_and_prepare(
        self,
        message: str,
        character: CharacterConfig,
        conversation_context: Optional[list] = None,
        model: Optional[str] = None,  # Add model parameter to pass through
        db_session: Optional[Session] = None  # Add db_session for workflow config fetch
    ) -> Optional[Dict[str, Any]]:
        """
        Detect image request and prepare prompt preview.
        
        Args:
            message: User's message
            character: Character config
            conversation_context: Recent conversation messages
            model: Optional model override (uses character's preferred model if not specified)
            db_session: Optional database session for fetching workflow config
        
        Returns:
            Dict with prompt preview if request detected, None otherwise
        """
        # Check if enabled
        if not self.system_config.comfyui.enabled:
            logger.debug("ComfyUI disabled in config")
            return None
        
        if not character.image_generation.enabled:
            logger.debug(f"Image generation disabled for character {character.id}")
            return None
        
        # NOTE: Image intent detection already happened at API level (KeywordIntentDetector)
        # This method only gets called when detected_intents.generate_image == True
        # So we can skip redundant detection and proceed directly to prompt generation
        
        logger.info(f"Image request detected for character {character.id}")
        logger.info(f"[ORCHESTRATOR] Generating image prompt with model: {model}")
        
        # Fetch workflow config from database if session provided
        workflow_config = None
        if db_session:
            try:
                workflow_repo = WorkflowRepository(db_session)
                workflow_entry = workflow_repo.get_default_for_character_and_type(
                    character_name=character.id,
                    workflow_type="image"
                )
                if workflow_entry:
                    # Build config dict from workflow fields
                    workflow_config = {
                        "trigger_word": workflow_entry.trigger_word,
                        "default_style": workflow_entry.default_style,
                        "negative_prompt": workflow_entry.negative_prompt,
                        "self_description": workflow_entry.self_description
                    }
                    logger.debug(f"Fetched workflow config for character {character.id}")
            except Exception as e:
                logger.warning(f"Failed to fetch workflow config: {e}")
        
        # Generate prompt preview
        try:
            prompt_data = await self.prompt_service.generate_prompt(
                user_request=message,
                character=character,
                conversation_context=conversation_context,
                model=model,  # Pass through model parameter
                workflow_config=workflow_config  # Pass workflow config
            )
            
            # Prepare final prompt
            final_prompt = self.prompt_service.prepare_final_prompt(
                base_prompt=prompt_data["prompt"],
                character=character,
                include_trigger=prompt_data["needs_trigger"],
                apply_character_style=True,
                workflow_config=workflow_config
            )
            
            # Get negative prompt from workflow config or prompt data
            neg_prompt = prompt_data.get("negative_prompt")
            if not neg_prompt and workflow_config:
                neg_prompt = workflow_config.get("negative_prompt")
            
            result = {
                "detected": True,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "needs_trigger": prompt_data["needs_trigger"],
                "reasoning": prompt_data.get("reasoning", "")
            }
            
            # Log to image_prompts.jsonl if db_session available
            if db_session:
                try:
                    from datetime import datetime
                    import json
                    from pathlib import Path
                    
                    # Get conversation_id from thread if possible
                    from chorus_engine.repositories import ThreadRepository
                    thread_repo = ThreadRepository(db_session)
                    # Note: We don't have thread_id here, would need to pass it
                    # For now, we'll log at the API level instead
                    
                except Exception as log_error:
                    logger.warning(f"Failed to log in-conversation image request: {log_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to prepare image prompt: {e}", exc_info=True)
            return None
    
    async def generate_image(
        self,
        db: Session,
        conversation_id: str,
        thread_id: str,
        character: CharacterConfig,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        message_id: Optional[int] = None,
        progress_callback=None,
        workflow_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image end-to-end.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            thread_id: Thread ID
            character: Character config
            prompt: Image prompt
            negative_prompt: Optional negative prompt
            seed: Optional seed
            message_id: Optional associated message ID
            progress_callback: Optional async callback(status_dict)
            workflow_name: Optional specific workflow name to use (if None, uses default)
        
        Returns:
            Dict with image_id, file_path, generation_time
        
        Raises:
            Various exceptions from services
        """
        start_time = time.time()
        
        try:
            # 1. Check ComfyUI health
            is_healthy = await self.comfyui_client.health_check()
            if not is_healthy:
                raise ComfyUIError("ComfyUI server is not responding")
            
            # 2. Fetch workflow config from database
            workflow_repo = WorkflowRepository(db)
            
            if workflow_name:
                # Use specified workflow
                workflow_entry = workflow_repo.get_by_name(
                    character_name=character.id,
                    workflow_name=workflow_name
                )
                if not workflow_entry:
                    raise WorkflowError(f"Workflow '{workflow_name}' not found for character {character.id}")
                # Verify it's an image workflow
                if workflow_entry.workflow_type != "image":
                    raise WorkflowError(f"Workflow '{workflow_name}' is not an image workflow (type: {workflow_entry.workflow_type})")
            else:
                # Use default workflow
                workflow_entry = workflow_repo.get_default_for_character_and_type(
                    character_name=character.id,
                    workflow_type="image"
                )
                if not workflow_entry:
                    raise WorkflowError(f"No default image workflow found for character {character.id}")
            
            # 3. Load character's workflow (Phase 6: uses type-based subfolders)
            from chorus_engine.services.workflow_manager import WorkflowType
            workflow = self.workflow_manager.load_workflow_by_type(
                character_id=character.id,
                workflow_type=WorkflowType.IMAGE,
                workflow_name=workflow_entry.workflow_name
            )
            
            # 4. Inject prompts
            workflow_with_prompt = self.workflow_manager.inject_prompt(
                workflow_data=workflow,
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed
            )
            
            # 5. Submit to ComfyUI
            # ComfyUI API expects just the nodes (either from "prompt" key or directly)
            if "prompt" in workflow_with_prompt:
                workflow_nodes = workflow_with_prompt["prompt"]
            else:
                workflow_nodes = workflow_with_prompt
            
            job_id = await self.comfyui_client.submit_workflow(
                workflow_data=workflow_nodes
            )
            
            logger.info(f"Submitted image generation job: {job_id}")
            
            # 5. Wait for completion
            await self.comfyui_client.wait_for_completion(
                job_id=job_id,
                callback=progress_callback
            )
            
            # 6. Retrieve image
            image_data = await self.comfyui_client.get_result(job_id)
            
            # 7. Save to disk
            image_repo = ImageRepository(db)
            
            # Create temporary ID for file naming
            temp_id = int(time.time() * 1000) % 1000000
            
            full_path, thumb_path = await self.storage_service.save_image(
                image_data=image_data,
                conversation_id=conversation_id,
                image_id=temp_id,
                create_thumbnail=True
            )
            
            # Get image dimensions
            image_info = await self.storage_service.get_image_info(full_path)
            
            # 8. Save to database
            generation_time = time.time() - start_time
            
            image_record = image_repo.create(
                conversation_id=conversation_id,
                thread_id=thread_id,
                character_id=character.id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                workflow_file=workflow_entry.workflow_file_path,
                file_path=str(full_path),
                thumbnail_path=str(thumb_path) if thumb_path else None,
                width=image_info["width"],
                height=image_info["height"],
                seed=seed,
                message_id=message_id,
                generation_time=generation_time
            )
            
            logger.info(f"Image generated successfully: ID {image_record.id}, {generation_time:.1f}s")
            
            return {
                "image_id": image_record.id,
                "file_path": str(full_path),
                "thumbnail_path": str(thumb_path) if thumb_path else None,
                "width": image_info["width"],
                "height": image_info["height"],
                "generation_time": generation_time,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def regenerate_image(
        self,
        db: Session,
        image_id: int,
        new_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Regenerate an existing image with a different seed.
        
        Args:
            db: Database session
            image_id: ID of image to regenerate
            new_seed: Optional new seed (None for random)
        
        Returns:
            Dict with new image data
        """
        repo = ImageRepository(db)
        original = repo.get_by_id(image_id)
        
        if not original:
            raise ValueError(f"Image {image_id} not found")
        
        # Load character config (would need character loader service)
        # For now, this is a simplified version
        # In real implementation, would load from character_id
        
        logger.info(f"Regenerating image {image_id} with new seed")
        
        # Use original parameters but new seed
        return await self.generate_image(
            db=db,
            conversation_id=original.conversation_id,
            thread_id=original.thread_id,
            character=None,  # Would load from character_id
            prompt=original.prompt,
            negative_prompt=original.negative_prompt,
            seed=new_seed,
            message_id=original.message_id
        )
