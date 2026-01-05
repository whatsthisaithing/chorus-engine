"""
Voice Sample Repository
Manages voice samples for TTS voice cloning.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import VoiceSample

logger = logging.getLogger(__name__)


class VoiceSampleRepository:
    """
    Repository for voice sample database operations.
    
    Handles CRUD operations for voice samples used in TTS generation.
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
        character_id: str,
        filename: str,
        transcript: str,
        is_default: bool = False
    ) -> VoiceSample:
        """
        Create a new voice sample record.
        
        Args:
            character_id: ID of the character this sample belongs to
            filename: Filename of the audio file
            transcript: Exact transcript of the audio
            is_default: Whether this is the default sample for the character
        
        Returns:
            The created VoiceSample object
        """
        # If setting as default, unset any existing default for this character
        if is_default:
            self._unset_default_for_character(character_id)
        
        voice_sample = VoiceSample(
            character_id=character_id,
            filename=filename,
            transcript=transcript,
            is_default=1 if is_default else 0,
            uploaded_at=datetime.now()
        )
        
        self.db.add(voice_sample)
        self.db.commit()
        self.db.refresh(voice_sample)
        
        logger.info(f"Created voice sample: {filename} for character {character_id}")
        return voice_sample
    
    def get_by_id(self, sample_id: int) -> Optional[VoiceSample]:
        """
        Get a voice sample by ID.
        
        Args:
            sample_id: The voice sample ID
        
        Returns:
            VoiceSample object or None if not found
        """
        return self.db.query(VoiceSample).filter(VoiceSample.id == sample_id).first()
    
    def get_default_for_character(self, character_id: str) -> Optional[VoiceSample]:
        """
        Get the default voice sample for a character.
        
        Args:
            character_id: The character ID
        
        Returns:
            VoiceSample object or None if no default is set
        """
        return self.db.query(VoiceSample).filter(
            VoiceSample.character_id == character_id,
            VoiceSample.is_default == 1
        ).first()
    
    def get_all_for_character(self, character_id: str) -> List[VoiceSample]:
        """
        Get all voice samples for a character.
        
        Args:
            character_id: The character ID
        
        Returns:
            List of VoiceSample objects (may be empty)
        """
        return self.db.query(VoiceSample).filter(
            VoiceSample.character_id == character_id
        ).order_by(
            VoiceSample.is_default.desc(),
            VoiceSample.uploaded_at.desc()
        ).all()
    
    def set_default(self, sample_id: int) -> VoiceSample:
        """
        Set a voice sample as the default for its character.
        
        Automatically unsets any other default for the same character.
        
        Args:
            sample_id: The voice sample ID to set as default
        
        Returns:
            The updated VoiceSample object
        
        Raises:
            ValueError: If sample_id doesn't exist
        """
        # Get the sample to find its character_id
        sample = self.get_by_id(sample_id)
        
        if not sample:
            raise ValueError(f"Voice sample not found: {sample_id}")
        
        # Unset any existing default for this character
        self._unset_default_for_character(sample.character_id)
        
        # Set this sample as default
        sample.is_default = 1
        
        self.db.commit()
        self.db.refresh(sample)
        
        logger.info(f"Set voice sample {sample_id} as default for character {sample.character_id}")
        return sample
    
    def update_transcript(self, sample_id: int, transcript: str) -> VoiceSample:
        """
        Update the transcript for a voice sample.
        
        Args:
            sample_id: The voice sample ID
            transcript: The new transcript text
        
        Returns:
            The updated VoiceSample object
        
        Raises:
            ValueError: If sample_id doesn't exist
        """
        sample = self.get_by_id(sample_id)
        
        if not sample:
            raise ValueError(f"Voice sample not found: {sample_id}")
        
        sample.transcript = transcript
        
        self.db.commit()
        self.db.refresh(sample)
        
        logger.info(f"Updated transcript for voice sample {sample_id}")
        return sample
    
    def delete(self, sample_id: int) -> bool:
        """
        Delete a voice sample.
        
        Args:
            sample_id: The voice sample ID to delete
        
        Returns:
            True if deleted, False if sample didn't exist
        """
        sample = self.get_by_id(sample_id)
        
        if not sample:
            logger.warning(f"Voice sample {sample_id} not found for deletion")
            return False
        
        self.db.delete(sample)
        self.db.commit()
        
        logger.info(f"Deleted voice sample {sample_id}")
        return True
    
    def _unset_default_for_character(self, character_id: str) -> None:
        """
        Internal method to unset default flag for all samples of a character.
        
        Args:
            character_id: The character ID
        """
        self.db.query(VoiceSample).filter(
            VoiceSample.character_id == character_id,
            VoiceSample.is_default == 1
        ).update({'is_default': 0})
        # Note: Commit happens in the calling method
