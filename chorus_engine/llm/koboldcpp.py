"""KoboldCpp LLM client implementation (OpenAI-compatible)."""

import json
import logging
from typing import Optional, AsyncIterator
from .base import BaseLLMClient, LLMResponse, LLMError
from .text_normalization import normalize_mojibake
import httpx

logger = logging.getLogger(__name__)


class KoboldCppLLMClient(BaseLLMClient):
    """
    Client for interacting with KoboldCpp via OpenAI-compatible API.
    
    KoboldCpp is a lightweight inference engine for GGUF models with:
    - Single model loading (user starts: koboldcpp.exe --model model.gguf --port 5001)
    - OpenAI-compatible API endpoints
    - Excellent CPU support with partial GPU offloading
    - No dynamic model management (model loaded at startup)
    
    Note: Unlike LM Studio/Ollama, KoboldCpp loads ONE model at startup.
    To switch models, user must restart KoboldCpp with different --model flag.
    """
    
    async def health_check(self) -> bool:
        """Check if KoboldCpp is available."""
        try:
            # Try v1/models endpoint (OpenAI-compatible)
            response = await self.client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                return True
            
            # Fallback: try root endpoint
            response = await self.client.get(f"{self.base_url}/api/v1/model")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"KoboldCpp health check failed: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI-compatible chat completions endpoint.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Ignored (KoboldCpp uses the model loaded at startup)
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        try:
            # Build messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,  # KoboldCpp ignores this, but required by API spec
                "messages": messages,
                "stream": False,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            logger.debug(f"KoboldCpp request: messages={len(messages)}, temp={payload['temperature']}, max_tokens={payload['max_tokens']}")
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()

            if response.encoding is None or response.encoding.lower() != "utf-8":
                response.encoding = "utf-8"
            
            data = response.json()
            
            # OpenAI-compatible response format
            choice = data.get("choices", [{}])[0]
            content = normalize_mojibake(choice.get("message", {}).get("content", ""))
            used_model = data.get("model", self.model)
            finish_reason = choice.get("finish_reason")
            if not content.strip():
                logger.warning(
                    f"[KOBOLDCPP] Empty response content: model={used_model}, "
                    f"finish_reason={finish_reason}, choices={len(data.get('choices', []))}, "
                    f"max_tokens={payload.get('max_tokens')}, temperature={payload.get('temperature')}"
                )
            
            return LLMResponse(
                content=content,
                model=used_model,
                finish_reason=finish_reason,
                usage=data.get("usage")
            )
        
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text
                logger.error(f"KoboldCpp error response: {error_body}")
                
                # Check for context overflow
                if "context" in error_body.lower() and ("overflow" in error_body.lower() or "exceed" in error_body.lower()):
                    logger.error(
                        f"Context overflow detected! Your configured context_window is {self.context_window} tokens, "
                        f"but KoboldCpp was started with a smaller --contextsize. "
                        f"Please restart KoboldCpp with: --contextsize {self.context_window} or larger."
                    )
            except:
                pass
            raise LLMError(f"HTTP error during LLM generation: {e}")
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during LLM generation: {e}")
        except Exception as e:
            raise LLMError(f"Failed to generate LLM response: {e}")
    
    async def generate_with_history(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate with conversation history using chat completions endpoint.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Ignored (KoboldCpp uses the model loaded at startup)
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        try:
            payload = {
                "model": self.model,  # Required by API spec, but KoboldCpp uses loaded model
                "messages": messages,
                "stream": False,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            logger.debug(f"KoboldCpp request: messages={len(messages)}, temp={payload['temperature']}, max_tokens={payload['max_tokens']}")
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()

            if response.encoding is None or response.encoding.lower() != "utf-8":
                response.encoding = "utf-8"
            
            data = response.json()
            
            choice = data.get("choices", [{}])[0]
            content = normalize_mojibake(choice.get("message", {}).get("content", ""))
            used_model = data.get("model", self.model)
            finish_reason = choice.get("finish_reason")
            if not content.strip():
                logger.warning(
                    f"[KOBOLDCPP] Empty response content (history): model={used_model}, "
                    f"finish_reason={finish_reason}, choices={len(data.get('choices', []))}, "
                    f"max_tokens={payload.get('max_tokens')}, temperature={payload.get('temperature')}"
                )
            
            return LLMResponse(
                content=content,
                model=used_model,
                finish_reason=finish_reason,
                usage=data.get("usage")
            )
        
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text
                logger.error(f"KoboldCpp error response: {error_body}")
            except:
                pass
            raise LLMError(f"HTTP error during LLM generation: {e}")
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during LLM generation: {e}")
        except Exception as e:
            raise LLMError(f"Failed to generate LLM response: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion token by token.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Ignored (KoboldCpp uses the model loaded at startup)
            
        Yields:
            Content chunks as they arrive
            
        Raises:
            LLMError: If streaming fails
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()

                if response.encoding is None or response.encoding.lower() != "utf-8":
                    response.encoding = "utf-8"
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = normalize_mojibake(delta.get("content", ""))
                            
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        
        except httpx.HTTPStatusError as e:
            logger.error(f"KoboldCpp streaming error: {e}")
            raise LLMError(f"HTTP error during streaming: {e}")
        except httpx.HTTPError as e:
            logger.error(f"KoboldCpp streaming error: {e}")
            raise LLMError(f"HTTP error during streaming: {e}")
        except Exception as e:
            logger.error(f"KoboldCpp streaming error: {e}")
            raise LLMError(f"Failed to stream response: {e}")
    
    async def stream_with_history(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Ignored (KoboldCpp uses the model loaded at startup)
            
        Yields:
            Content chunks as they arrive
            
        Raises:
            LLMError: If streaming fails
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()

                if response.encoding is None or response.encoding.lower() != "utf-8":
                    response.encoding = "utf-8"
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = normalize_mojibake(delta.get("content", ""))
                            
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        
        except httpx.HTTPStatusError as e:
            logger.error(f"KoboldCpp streaming error: {e}")
            raise LLMError(f"HTTP error during streaming: {e}")
        except httpx.HTTPError as e:
            logger.error(f"KoboldCpp streaming error: {e}")
            raise LLMError(f"HTTP error during streaming: {e}")
        except Exception as e:
            logger.error(f"KoboldCpp streaming error: {e}")
            raise LLMError(f"Failed to stream response: {e}")
    
    # Model management methods (KoboldCpp-specific behavior)
    
    async def get_loaded_models(self) -> list[str]:
        """
        Get list of loaded models.
        
        Note: KoboldCpp only loads ONE model at startup, so this always
        returns a single-item list with the configured model name.
        
        Returns:
            List with single model identifier
        """
        try:
            # Try to query model info
            response = await self.client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return [m.get("id", self.model) for m in models]
        except:
            pass
        
        # Fallback: return configured model
        return [self.model]
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """
        Ensure a model is loaded and ready.
        
        Note: KoboldCpp cannot dynamically load models. The model is loaded
        at startup with --model flag. This method only checks if KoboldCpp
        is responding (health check).
        
        Args:
            model: Model identifier (ignored, cannot switch models)
            
        Returns:
            True if KoboldCpp is available, False otherwise
        """
        is_healthy = await self.health_check()
        
        if is_healthy and model != self.model:
            logger.warning(
                f"Character requested model '{model}' but KoboldCpp is running with '{self.model}'. "
                f"To switch models, restart KoboldCpp with: --model {model}"
            )
        
        return is_healthy
    
    async def unload_model(self, model: str) -> bool:
        """
        Unload a model from memory.
        
        Note: KoboldCpp cannot dynamically unload models. To unload, the user
        must stop the KoboldCpp process. This method is a no-op.
        
        Args:
            model: Model identifier (ignored)
            
        Returns:
            False (cannot unload without stopping process)
        """
        logger.warning(
            "KoboldCpp does not support dynamic model unloading. "
            "To free VRAM, stop the KoboldCpp process."
        )
        return False
