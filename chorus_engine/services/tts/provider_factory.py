"""
TTS Provider Factory

Creates and manages TTS provider instances.
"""

import logging
from typing import Dict, Optional

from .base_provider import BaseTTSProvider

logger = logging.getLogger(__name__)


class TTSProviderFactory:
    """Factory for creating and managing TTS providers."""
    
    _providers: Dict[str, BaseTTSProvider] = {}
    
    @classmethod
    def register_provider(cls, provider: BaseTTSProvider):
        """Register a TTS provider."""
        cls._providers[provider.provider_name] = provider
        logger.info(f"Registered TTS provider: {provider.provider_name}")
    
    @classmethod
    def get_provider(cls, provider_name: str) -> Optional[BaseTTSProvider]:
        """
        Get a TTS provider by name.
        
        Args:
            provider_name: Provider identifier ('comfyui', 'chatterbox', etc.)
        
        Returns:
            Provider instance or None if not found
        """
        return cls._providers.get(provider_name)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, BaseTTSProvider]:
        """Get all available TTS providers."""
        return {
            name: provider 
            for name, provider in cls._providers.items()
            if provider.is_available()
        }
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def initialize_providers(
        cls,
        comfyui_client=None,
        comfyui_lock=None,
        workflow_manager=None,
        audio_storage=None,
        system_config=None
    ):
        """
        Initialize all TTS providers with dependencies.
        
        Called during app startup.
        
        Args:
            comfyui_client: ComfyUI client for workflow-based TTS
            comfyui_lock: Async lock for ComfyUI operations
            workflow_manager: Workflow manager for loading/injecting workflows
            audio_storage: Audio storage service for saving files
            system_config: System configuration
        """
        # Initialize ComfyUI provider
        if comfyui_client and workflow_manager:
            try:
                from .comfyui_provider import ComfyUITTSProvider
                
                comfyui_provider = ComfyUITTSProvider(
                    comfyui_client=comfyui_client,
                    comfyui_lock=comfyui_lock,
                    workflow_manager=workflow_manager,
                    audio_storage=audio_storage
                )
                cls.register_provider(comfyui_provider)
            except ImportError as e:
                logger.warning(f"ComfyUI TTS provider not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize ComfyUI provider: {e}")
        
        # Initialize Chatterbox provider
        try:
            from .chatterbox_provider import ChatterboxTTSProvider
            
            chatterbox_provider = ChatterboxTTSProvider(
                audio_storage=audio_storage,
                system_config=system_config
            )
            cls.register_provider(chatterbox_provider)
        except ImportError:
            logger.warning("Chatterbox TTS not available (package not installed)")
        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox provider: {e}")
        
        # Log summary
        available = cls.get_available_providers()
        if available:
            logger.info(f"Initialized TTS providers: {list(available.keys())}")
        else:
            logger.warning("No TTS providers available!")
