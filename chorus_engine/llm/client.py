"""LLM client factory and backward compatibility exports."""

from typing import TYPE_CHECKING
from .base import BaseLLMClient, LLMError, LLMResponse
from .ollama import OllamaLLMClient

# Import LM Studio only when available (Phase 2)
try:
    from .lmstudio import LMStudioLLMClient
    HAS_LMSTUDIO = True
except ImportError:
    HAS_LMSTUDIO = False

# Import KoboldCpp client
try:
    from .koboldcpp import KoboldCppLLMClient
    HAS_KOBOLDCPP = True
except ImportError:
    HAS_KOBOLDCPP = False

if TYPE_CHECKING:
    from chorus_engine.config.models import LLMConfig


def create_llm_client(config: "LLMConfig") -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client based on provider.
    
    Args:
        config: LLM configuration with provider type and settings
        
    Returns:
        Provider-specific LLM client instance
        
    Raises:
        ValueError: If provider is unknown
        NotImplementedError: If provider is not yet implemented
    """
    provider = config.provider.lower()
    
    if provider == "ollama":
        return OllamaLLMClient(
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout_seconds,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            context_window=config.context_window,
        )
    
    elif provider == "lmstudio":
        if not HAS_LMSTUDIO:
            raise NotImplementedError(
                "LM Studio client not yet implemented. "
                "This will be available in Phase 2 of the refactor."
            )
        return LMStudioLLMClient(
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout_seconds,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            context_window=config.context_window,
        )
    
    elif provider == "koboldcpp":
        if not HAS_KOBOLDCPP:
            raise NotImplementedError(
                "KoboldCpp client not available. "
                "Please ensure koboldcpp.py is present in the llm directory."
            )
        return KoboldCppLLMClient(
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout_seconds,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            context_window=config.context_window,
        )
    
    elif provider in ["openai-compatible", "llamacpp"]:
        raise NotImplementedError(
            f"Provider '{provider}' is not yet implemented. "
            f"Currently supported: ollama, lmstudio, koboldcpp"
        )
    
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: ollama, lmstudio, koboldcpp"
        )


# Backward compatibility: LLMClient class wrapper
class LLMClient:
    """
    DEPRECATED: Use create_llm_client() instead.
    
    This wrapper maintains backward compatibility for existing code.
    Creates an Ollama client with the provided parameters.
    """
    
    def __new__(
        cls,
        base_url: str = "http://localhost:11434",
        model: str = "mistral:7b-instruct",
        timeout: float = 120.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Create an Ollama client for backward compatibility."""
        return OllamaLLMClient(
            base_url=base_url,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Re-export all necessary classes for backward compatibility
__all__ = [
    "BaseLLMClient",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "OllamaLLMClient",
    "create_llm_client",
]
