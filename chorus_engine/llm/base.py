"""Base abstract class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator
import httpx
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """LLM response model."""
    content: str
    model: str
    finish_reason: Optional[str] = None


class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM provider clients.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods. Optional model management methods have
    default no-op implementations.
    """
    
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
        context_window: int = 8192,
    ):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL for the LLM provider
            model: Default model identifier
            timeout: Request timeout in seconds
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens to generate
            context_window: Model's context window size (for validation/warnings)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.client = httpx.AsyncClient(timeout=timeout)
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM provider is available and responding.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a non-streaming completion.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Override default model
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_with_history(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a non-streaming completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Override default model
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens as they are generated.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Content chunks as they are generated
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def stream_with_history(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Override default model
            
        Yields:
            Content chunks as they are generated
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    # Optional model management methods (providers can override if supported)
    
    async def get_loaded_models(self) -> list[str]:
        """
        Get list of currently loaded models in memory.
        
        Default implementation returns empty list. Override if provider
        supports querying loaded models.
        
        Returns:
            List of model identifiers currently loaded in memory
        """
        return []
    
    async def unload_model(self) -> bool:
        """
        Unload the current model from memory.
        
        Default implementation returns False. Override if provider
        supports manual model unloading.
        
        Returns:
            True if model was unloaded, False if not supported
        """
        return False
    
    async def unload_all_models(self) -> None:
        """
        Unload all models from memory.
        
        Default implementation is a no-op. Override if provider
        supports unloading all models at once.
        """
        pass
    
    async def reload_model(self) -> bool:
        """
        Reload the current model into memory.
        
        Default implementation returns False. Override if provider
        supports manual model loading.
        
        Returns:
            True if model was loaded, False if not supported
        """
        return False
    
    async def reload_models_after_generation(
        self,
        character_model: str,
        intent_model: str = "qwen2.5:3b-instruct",
    ) -> None:
        """
        Reload models after external generation (e.g., ComfyUI).
        
        Default implementation is a no-op. Override if provider
        requires explicit model reloading after VRAM was freed.
        
        Args:
            character_model: Main character model to reload
            intent_model: Intent detection model to reload
        """
        pass
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """
        Ensure a specific model is loaded and ready.
        
        Default implementation returns True (assumes models auto-load).
        Override if provider requires explicit model loading.
        
        Args:
            model: Model identifier to load
            
        Returns:
            True if model is loaded, False if loading failed
        """
        return True
    
    async def close(self) -> None:
        """Close the HTTP client. Can be overridden if needed."""
        await self.client.aclose()
