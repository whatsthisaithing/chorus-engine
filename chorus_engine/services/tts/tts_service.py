"""
Unified TTS Service

Facade that routes requests to appropriate TTS provider.
"""

import logging
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from .provider_factory import TTSProviderFactory
from .base_provider import TTSRequest, TTSResult
from ..audio_preprocessing import AudioPreprocessingService
from ..structured_response import parse_structured_response, to_plain_text
from ...config.models import CharacterConfig
from ...repositories.voice_sample_repository import VoiceSampleRepository
from ...repositories.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)


class TTSService:
    """
    Unified TTS service.
    
    Routes TTS requests to appropriate provider based on character config.
    """
    
    def __init__(self, db: Session, system_config=None):
        """
        Initialize TTS service.
        
        Args:
            db: Database session
            system_config: System configuration (optional)
        """
        self.db = db
        self.system_config = system_config
        self.preprocessor = AudioPreprocessingService()
        self.voice_sample_repo = VoiceSampleRepository(db)
        self.workflow_repo = WorkflowRepository(db)
    
    async def generate_audio(
        self,
        text: str,
        character: CharacterConfig,
        message_id: str
    ) -> TTSResult:
        """
        Generate audio using character's configured TTS provider.
        
        Args:
            text: Text to speak (will be preprocessed)
            character: Character configuration
            message_id: Message ID for file naming
        
        Returns:
            TTSResult with audio filename and metadata
        """
        # Step 1: Extract structured text for TTS if applicable
        text_for_tts = text or ""
        include_physical = False
        if self.system_config and hasattr(self.system_config, "tts"):
            include_physical = getattr(self.system_config.tts, "include_physicalaction", False)
        
        if "<assistant_response>" in text_for_tts:
            parsed = parse_structured_response(text_for_tts)
            extracted = to_plain_text(parsed.segments, include_physicalaction=include_physical)
            if extracted:
                text_for_tts = extracted
        
        # Step 2: Validate text
        is_valid, reason = self.preprocessor.validate_text_for_tts(text_for_tts)
        if not is_valid:
            logger.warning(f"[TTS] Text validation failed for message {message_id}: {reason}")
            return TTSResult(
                success=False,
                error_message=f"Text not suitable for TTS: {reason}"
            )
        
        # Step 3: Preprocess text
        logger.info(f"[TTS] Preprocessing text for message {message_id}")
        preprocessed_text = self.preprocessor.preprocess_for_tts(text_for_tts)
        
        if not preprocessed_text:
            logger.warning(f"[TTS] No text remaining after preprocessing for message {message_id}")
            return TTSResult(
                success=False,
                error_message="No text remaining after preprocessing"
            )
        
        logger.debug(f"[TTS] Preprocessed text ({len(preprocessed_text)} chars)")
        
        # Step 3: Get voice sample (if applicable)
        voice_sample = self.voice_sample_repo.get_default_for_character(character.id)
        voice_sample_path = None
        voice_transcript = None
        
        if voice_sample:
            voice_samples_dir = Path("data/voice_samples") / character.id
            voice_sample_file = voice_samples_dir / voice_sample.filename
            if voice_sample_file.exists():
                voice_sample_path = str(voice_sample_file.absolute())
                voice_transcript = voice_sample.transcript
                logger.info(f"[TTS] Using voice sample: {voice_sample.filename}")
            else:
                logger.warning(f"[TTS] Voice sample file not found: {voice_sample_file}")
        
        # Step 4: Determine provider
        # Check system config for default provider, fallback to chatterbox
        provider_name = "chatterbox"  # fallback default
        if self.system_config and hasattr(self.system_config, 'tts'):
            provider_name = getattr(self.system_config.tts, 'default_provider', 'chatterbox')
        
        provider_config = {}
        
        # Character-specific provider overrides system default
        if character.voice and hasattr(character.voice, 'tts_provider'):
            provider_name = character.voice.tts_provider.provider
            
            # Get provider-specific config
            if provider_name == "comfyui":
                # Check database for default workflow first
                db_workflow = self.workflow_repo.get_default_for_character_and_type(
                    character.id,
                    'audio'
                )
                
                if db_workflow:
                    # Database takes precedence
                    provider_config = {"workflow_name": db_workflow.workflow_name}
                    logger.info(f"[TTS] Using database default workflow: {db_workflow.workflow_name}")
                else:
                    # Fallback to character YAML config
                    provider_config = character.voice.tts_provider.comfyui or {}
                    if provider_config.get('workflow_name'):
                        logger.info(f"[TTS] Using YAML config workflow: {provider_config['workflow_name']}")
                    else:
                        provider_config = {"workflow_name": "default_tts_workflow"}
                        logger.info(f"[TTS] Using system default workflow: default_tts_workflow")
                        
            elif provider_name == "chatterbox":
                provider_config = character.voice.tts_provider.chatterbox or {}
        elif character.voice and hasattr(character.voice, 'tts_engine'):
            # Legacy config support
            logger.warning("[TTS] Using legacy tts_engine field, please migrate to tts_provider")
            provider_name = "comfyui"
            
            # Check database for default workflow
            db_workflow = self.workflow_repo.get_default_for_character_and_type(
                character.id,
                'audio'
            )
            provider_config = {"workflow_name": db_workflow.workflow_name if db_workflow else "default_tts_workflow"}
        
        logger.info(f"[TTS] Using provider: {provider_name}")
        
        # Step 5: Get provider
        provider = TTSProviderFactory.get_provider(provider_name)
        
        if not provider:
            available = TTSProviderFactory.list_providers()
            return TTSResult(
                success=False,
                error_message=f"TTS provider '{provider_name}' not available. Available providers: {available}"
            )
        
        if not provider.is_available():
            return TTSResult(
                success=False,
                error_message=f"TTS provider '{provider_name}' is not available (check configuration/model loading)"
            )
        
        # Step 6: Build request
        request = TTSRequest(
            text=preprocessed_text,
            character_id=character.id,
            message_id=message_id,
            voice_sample_path=voice_sample_path,
            voice_transcript=voice_transcript,
            provider_config=provider_config
        )
        
        # Step 7: Generate audio
        logger.info(f"[TTS] Generating audio for message {message_id} using {provider_name}")
        result = await provider.generate_audio(request)
        
        if result.success:
            logger.info(f"[TTS] Audio generated successfully: {result.audio_filename}")
        else:
            logger.error(f"[TTS] Audio generation failed: {result.error_message}")
        
        return result
