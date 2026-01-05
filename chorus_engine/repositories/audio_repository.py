"""
Audio Repository
Manages audio message records for TTS-generated audio.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import AudioMessage

logger = logging.getLogger(__name__)


class AudioRepository:
    """
    Repository for audio message database operations.
    
    Handles CRUD operations for TTS-generated audio linked to messages.
    """
    
    def __init__(self, db: Session):
        """
        Initialize repository with database session.
        
        Args:
            db: SQLAlchemy session
        """
        self.db = db
    
    def create(
        self,
        message_id: str,
        audio_filename: str,
        workflow_name: Optional[str] = None,
        generation_duration: Optional[float] = None,
        text_preprocessed: Optional[str] = None,
        voice_sample_id: Optional[int] = None
    ) -> AudioMessage:
        """
        Create a new audio message record.
        
        Args:
            message_id: ID of the message this audio belongs to
            audio_filename: Filename of the audio file
            workflow_name: Name of the workflow used
            generation_duration: How long generation took (seconds)
            text_preprocessed: The preprocessed plain text sent to TTS
            voice_sample_id: ID of the voice sample used (if any)
        
        Returns:
            The created AudioMessage object
        """
        audio_message = AudioMessage(
            message_id=message_id,
            audio_filename=audio_filename,
            workflow_name=workflow_name,
            generation_duration=generation_duration,
            text_preprocessed=text_preprocessed,
            voice_sample_id=voice_sample_id,
            created_at=datetime.now()
        )
        
        self.db.add(audio_message)
        self.db.commit()
        self.db.refresh(audio_message)
        
        logger.info(f"Created audio record for message {message_id}: {audio_filename}")
        return audio_message
    
    def get_by_message_id(self, message_id: str) -> Optional[AudioMessage]:
        """
        Get audio record for a specific message.
        
        Args:
            message_id: The message ID
        
        Returns:
            AudioMessage object or None if no audio exists for this message
        """
        return self.db.query(AudioMessage).filter(
            AudioMessage.message_id == message_id
        ).first()
    
    def delete_by_message_id(self, message_id: str) -> bool:
        """
        Delete audio record for a message.
        
        Note: This only deletes the database record. The audio file
        must be deleted separately using AudioStorageService.
        
        Args:
            message_id: The message ID
        
        Returns:
            True if deleted, False if no audio existed
        """
        audio = self.get_by_message_id(message_id)
        
        if not audio:
            logger.warning(f"No audio record found for message {message_id}")
            return False
        
        self.db.delete(audio)
        self.db.commit()
        
        logger.info(f"Deleted audio record for message {message_id}")
        return True
    
    def update_generation_metadata(
        self,
        message_id: int,
        generation_duration: Optional[float] = None,
        workflow_name: Optional[str] = None
    ) -> Optional[AudioMessage]:
        """
        Update metadata for an audio record.
        
        Args:
            message_id: The message ID
            generation_duration: Generation duration to set
            workflow_name: Workflow name to set
        
        Returns:
            Updated AudioMessage or None if not found
        """
        audio = self.get_by_message_id(message_id)
        
        if not audio:
            logger.warning(f"No audio record found for message {message_id}")
            return None
        
        if generation_duration is not None:
            audio.generation_duration = generation_duration
        if workflow_name is not None:
            audio.workflow_name = workflow_name
        
        self.db.commit()
        self.db.refresh(audio)
        
        logger.info(f"Updated audio metadata for message {message_id}")
        return audio
