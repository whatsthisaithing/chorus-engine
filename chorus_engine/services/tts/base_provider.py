"""
Base TTS Provider Interface

All TTS providers must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class TTSRequest:
    """TTS generation request."""
    text: str                          # Preprocessed text to speak
    character_id: str                  # Character ID
    message_id: str                    # Message ID for file naming
    voice_sample_path: Optional[str] = None   # Path to voice sample (if applicable)
    voice_transcript: Optional[str] = None    # Transcript of voice sample
    provider_config: Optional[Dict[str, Any]] = None  # Provider-specific settings


@dataclass
class TTSResult:
    """TTS generation result."""
    success: bool
    audio_filename: Optional[str] = None      # Filename of generated audio
    audio_path: Optional[Path] = None         # Full path to audio file
    generation_duration: Optional[float] = None  # Time taken (seconds)
    error_message: Optional[str] = None
    provider_name: Optional[str] = None       # Which provider generated this
    metadata: Optional[Dict[str, Any]] = None # Provider-specific metadata


class BaseTTSProvider(ABC):
    """
    Base class for all TTS providers.
    
    Providers must implement:
    - generate_audio() - Core generation method
    - validate_config() - Check if provider is properly configured
    - get_estimated_duration() - Estimate generation time
    - is_available() - Check if provider can accept requests
    
    Optional methods for resource management:
    - unload_model() - Free VRAM for heavy tasks
    - reload_model() - Reload model after heavy tasks
    - is_model_loaded() - Check if model is in memory
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g., 'comfyui', 'chatterbox')."""
        pass
    
    @abstractmethod
    async def generate_audio(self, request: TTSRequest) -> TTSResult:
        """
        Generate audio from text.
        
        Args:
            request: TTSRequest with text, character_id, voice sample, etc.
        
        Returns:
            TTSResult with audio filename and metadata
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate provider configuration.
        
        Returns:
            (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_estimated_duration(self, text: str) -> float:
        """
        Estimate audio generation time in seconds.
        
        Args:
            text: Text to be spoken
        
        Returns:
            Estimated duration in seconds
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is currently available.
        
        Returns:
            True if provider can accept requests
        """
        pass
    
    def unload_model(self) -> None:
        """
        Unload model from memory/VRAM (optional).
        
        Called when heavy tasks (image/video generation) need VRAM.
        Providers with large models should implement this.
        """
        pass
    
    def reload_model(self) -> None:
        """
        Reload model into memory/VRAM (optional).
        
        Called after heavy tasks complete.
        Only needed if unload_model() was implemented.
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is currently loaded in memory.
        
        Returns:
            True if model is ready for generation (default: True)
        """
        return True
