"""Base abstract class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, AsyncIterator, List, Dict, Any
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """LLM response model."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    tool_calls: Optional[list[dict]] = None
    raw_message: Optional[dict] = None


@dataclass
class LLMStreamEvent:
    """Structured streaming event emitted by provider clients."""

    content_delta: str = ""
    provider_tool_calls_delta: Optional[List[Dict[str, Any]]] = None
    provider_raw_event: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


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
        capture_raw_http_debug: bool = False,
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
        self.capture_raw_http_debug = bool(capture_raw_http_debug)
        self.client = httpx.AsyncClient(timeout=timeout)

    def _capture_raw_http_exchange(
        self,
        *,
        provider: str,
        endpoint_path: str,
        payload: dict,
        request_body_bytes: bytes,
        response: httpx.Response,
    ) -> None:
        if not self.capture_raw_http_debug:
            return
        try:
            from .request_debug_context import get_request_debug_context

            ctx = get_request_debug_context()
            conversation_id = str(ctx.get("conversation_id") or "unknown").strip() or "unknown"
            chat_type = str(ctx.get("chat_type") or "normal").strip().lower() or "normal"
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            endpoint_slug = endpoint_path.strip("/").replace("/", "_")
            filename = f"{conversation_id}_{chat_type}_{timestamp}_{endpoint_slug}.json"
            out_dir = Path("data/debug/requests")
            out_dir.mkdir(parents=True, exist_ok=True)

            response_body_bytes = bytes(response.content or b"")
            request_text = request_body_bytes.decode("utf-8", errors="replace")
            response_text = response_body_bytes.decode("utf-8", errors="replace")

            doc = {
                "captured_at_utc": datetime.utcnow().isoformat() + "Z",
                "conversation_id": conversation_id,
                "chat_type": chat_type,
                "context": ctx,
                "provider": provider,
                "base_url": self.base_url,
                "endpoint_path": endpoint_path,
                "model": str(payload.get("model") or self.model),
                "status_code": response.status_code,
                "request": {
                    "content_type": "application/json",
                    "body_text": request_text,
                    "body_bytes_b64": base64.b64encode(request_body_bytes).decode("ascii"),
                },
                "response": {
                    "content_type": response.headers.get("content-type"),
                    "body_text": response_text,
                    "body_bytes_b64": base64.b64encode(response_body_bytes).decode("ascii"),
                },
            }

            (out_dir / filename).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to capture raw %s HTTP exchange: %s", str(provider or "llm"), exc)
    
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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
        response_format: Optional[dict] = None,
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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
        response_format: Optional[dict] = None,
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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
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
    
    async def stream_with_history(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens with conversation history.
        Compatibility wrapper over `stream_with_history_events`.
        """
        async for event in self.stream_with_history_events(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        ):
            chunk = str(getattr(event, "content_delta", "") or "")
            if chunk:
                yield chunk

    async def stream_with_history_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """
        Stream structured events with conversation history.
        Providers should override for richer tool/raw event support.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stream_with_history_events()"
        )

    async def generate_vision(
        self,
        *,
        prompt: str,
        image_base64_list: List[str],
        image_mime_type: str = "image/jpeg",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a multimodal completion (text + image)."""
        raise LLMError(f"Vision generation not supported by {self.__class__.__name__}")
    
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
