"""
Image attachment handling service for vision system.

Handles saving uploaded images, calling vision service for analysis,
and storing results in the database.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.models import ImageAttachment, Message, Memory, MemoryType
from chorus_engine.services.vision_service import VisionService, VisionServiceError
import uuid

logger = logging.getLogger(__name__)


class ImageAttachmentService:
    """
    Service for handling image attachments with vision analysis.
    
    Responsibilities:
    - Save uploaded images to disk
    - Call vision service for analysis
    - Store results in database
    - Link attachments to messages
    """
    
    def __init__(
        self,
        vision_service: VisionService,
        base_storage_path: Path = Path("data/images")
    ):
        """
        Initialize image attachment service.
        
        Args:
            vision_service: VisionService instance for analysis
            base_storage_path: Root directory for image storage
        """
        self.vision_service = vision_service
        self.base_storage_path = Path(base_storage_path)
        self.base_storage_path.mkdir(parents=True, exist_ok=True)
    
    async def save_and_analyze_image(
        self,
        db: Session,
        message_id: str,
        conversation_id: str,
        character_id: str,
        image_file: bytes,
        original_filename: str,
        mime_type: str,
        source: str = "web",
        user_description: Optional[str] = None,
        message_content: Optional[str] = None
    ) -> ImageAttachment:
        """
        Save uploaded image and analyze with vision service.
        
        Args:
            db: Database session
            message_id: ID of the message this image is attached to
            conversation_id: ID of the conversation
            character_id: ID of the character
            image_file: Binary image data
            original_filename: Original filename from upload
            mime_type: MIME type of the image
            source: Source platform (web, discord, slack)
            user_description: Optional user-provided description
            message_content: Optional message text for intent detection
            
        Returns:
            ImageAttachment record with vision analysis results
            
        Raises:
            Exception: If saving or analysis fails
        """
        # Generate unique ID
        import uuid
        attachment_id = str(uuid.uuid4())
        
        # Create conversation directory
        conv_dir = self.base_storage_path / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        extension = Path(original_filename).suffix or ".jpg"
        
        # Save original image
        original_path = conv_dir / f"{attachment_id}_original{extension}"
        with open(original_path, 'wb') as f:
            f.write(image_file)
        
        logger.info(f"Saved image to {original_path} ({len(image_file)} bytes)")
        
        # Get image dimensions
        try:
            from PIL import Image
            img = Image.open(original_path)
            width, height = img.size
        except Exception as e:
            logger.warning(f"Could not read image dimensions: {e}")
            width, height = None, None
        
        # Create database record
        attachment = ImageAttachment(
            id=attachment_id,
            message_id=message_id,
            conversation_id=conversation_id,
            character_id=character_id,
            original_path=str(original_path),
            original_filename=original_filename,
            file_size=len(image_file),
            mime_type=mime_type,
            width=width,
            height=height,
            uploaded_at=datetime.utcnow(),
            source=source,
            description=user_description
        )
        
        db.add(attachment)
        db.flush()  # Get the ID but don't commit yet
        
        # Determine if we should analyze the image
        should_analyze = self.vision_service.should_analyze_image(
            message_content or "",
            source=source
        )
        
        if not should_analyze:
            attachment.vision_skipped = "true"
            attachment.vision_skip_reason = "intent_not_detected"
            logger.info(f"Skipping vision analysis for {attachment_id} (intent not detected)")
            db.commit()
            return attachment
        
        # Analyze image with vision service
        try:
            logger.info(f"Analyzing image {attachment_id} with vision service...")
            result = await self.vision_service.analyze_image(
                image_path=original_path,
                context=user_description or message_content,
                character_id=character_id
            )
            
            # Update attachment with vision results
            attachment.vision_processed = "true"
            attachment.vision_model = result.model
            attachment.vision_backend = result.backend
            attachment.vision_processed_at = datetime.utcnow()
            attachment.vision_processing_time_ms = result.processing_time_ms
            attachment.vision_observation = result.observation
            attachment.vision_confidence = result.confidence
            attachment.vision_tags = json.dumps(result.tags)
            
            # Store processed image path if it exists
            if result.processing_time_ms > 0:  # Vision service created processed version
                processed_path = original_path.parent / f"processed_{original_path.name}"
                if processed_path.exists():
                    attachment.processed_path = str(processed_path)
            
            logger.info(
                f"Vision analysis complete for {attachment_id}: "
                f"confidence={result.confidence:.2f}, time={result.processing_time_ms}ms"
            )
            
            # Create visual memory if auto_create is enabled
            memory_config = self.vision_service.config.get("memory", {})
            if memory_config.get("auto_create", True):
                min_confidence = memory_config.get("min_confidence", 0.6)
                if result.confidence >= min_confidence:
                    try:
                        await self._create_visual_memory(
                            db=db,
                            attachment=attachment,
                            result=result,
                            message_id=message_id,
                            conversation_id=conversation_id,
                            character_id=character_id
                        )
                    except Exception as e:
                        logger.error(f"Failed to create visual memory for {attachment_id}: {e}", exc_info=True)
                        # Don't fail the whole operation if memory creation fails
            
        except VisionServiceError as e:
            logger.error(f"Vision analysis failed for {attachment_id}: {e}")
            attachment.vision_skipped = "true"
            attachment.vision_skip_reason = f"analysis_failed: {str(e)[:80]}"
            
        except Exception as e:
            logger.error(f"Unexpected error during vision analysis for {attachment_id}: {e}", exc_info=True)
            attachment.vision_skipped = "true"
            attachment.vision_skip_reason = f"unexpected_error: {str(e)[:80]}"
        
        db.commit()
        return attachment
    
    def get_vision_context(self, attachment: ImageAttachment) -> Optional[str]:
        """
        Get formatted vision context for prompt injection.
        
        Args:
            attachment: ImageAttachment with vision analysis
            
        Returns:
            Formatted vision context string, or None if not analyzed
        """
        if attachment.vision_processed != "true" or not attachment.vision_observation:
            return None
        
        # Parse structured data if available
        try:
            data = json.loads(attachment.vision_observation)
            
            # Build concise context
            parts = []
            if data.get("main_subject"):
                parts.append(f"Main subject: {data['main_subject']}")
            if data.get("objects"):
                parts.append(f"Objects: {', '.join(data['objects'][:5])}")
            if data.get("people") and isinstance(data["people"], dict):
                count = data["people"].get("count", 0)
                if count > 0:
                    parts.append(f"People: {count}")
            if data.get("text_content"):
                parts.append(f"Text: {data['text_content'][:100]}")
            if data.get("mood"):
                parts.append(f"Mood: {data['mood']}")
            
            return " | ".join(parts) if parts else attachment.vision_observation[:200]
            
        except json.JSONDecodeError:
            # Fallback: use raw observation (truncated)
            return attachment.vision_observation[:200]
    
    async def _create_visual_memory(
        self,
        db: Session,
        attachment: ImageAttachment,
        result,  # VisionAnalysisResult
        message_id: str,
        conversation_id: str,
        character_id: str
    ) -> Optional[Memory]:
        """
        Create a visual memory for the analyzed image.
        
        Args:
            db: Database session
            attachment: ImageAttachment record
            result: VisionAnalysisResult from analysis
            message_id: Source message ID
            conversation_id: Source conversation ID
            character_id: Character ID
            
        Returns:
            Created Memory or None if creation failed
        """
        memory_config = self.vision_service.config.get("memory", {})
        
        # Build memory content from vision observation
        try:
            data = json.loads(result.observation)
            
            # Create descriptive memory content
            parts = []
            
            if data.get("main_subject"):
                parts.append(f"User showed me an image of {data['main_subject']}")
            else:
                parts.append("User showed me an image")
            
            # Add key details
            details = []
            if data.get("objects"):
                details.append(f"containing {', '.join(data['objects'][:3])}")
            if data.get("people") and isinstance(data["people"], dict):
                count = data["people"].get("count", 0)
                if count > 0:
                    details.append(f"with {count} {'person' if count == 1 else 'people'}")
            if data.get("mood"):
                details.append(f"conveying a {data['mood']} mood")
            
            if details:
                parts.append(". The image " + ", ".join(details))
            
            memory_content = "".join(parts) + "."
            
        except json.JSONDecodeError:
            # Fallback: use simple format
            memory_content = f"User showed me an image. {result.observation[:200]}"
        
        # Create memory record
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            conversation_id=conversation_id,
            character_id=character_id,
            memory_type=MemoryType.EXPLICIT,  # Visual memories are explicit (user action)
            content=memory_content,
            priority=memory_config.get("default_priority", 70),
            category=memory_config.get("category", "visual"),
            confidence=result.confidence,
            status="auto_approved",  # Auto-approve visual memories
            source_messages=[message_id],
            source="web",  # Will be updated based on actual source
            meta_data={
                "image_attachment_id": attachment.id,
                "vision_model": result.model,
                "vision_backend": result.backend,
                "vision_tags": result.tags,
                "original_filename": attachment.original_filename
            },
            created_at=datetime.utcnow()
        )
        
        db.add(memory)
        db.flush()  # Get the ID
        
        logger.info(
            f"Created visual memory {memory_id} for image {attachment.id} "
            f"(confidence={result.confidence:.2f}, priority={memory.priority})"
        )
        
        return memory
