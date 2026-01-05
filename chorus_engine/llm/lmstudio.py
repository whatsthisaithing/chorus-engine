"""LM Studio LLM client implementation (OpenAI-compatible)."""

import json
import logging
from typing import Optional, AsyncIterator
from .base import BaseLLMClient, LLMResponse, LLMError
import httpx

logger = logging.getLogger(__name__)


class LMStudioLLMClient(BaseLLMClient):
    """
    Client for interacting with LM Studio via OpenAI-compatible API.
    
    LM Studio uses OpenAI-compatible endpoints and supports:
    - JIT (Just-In-Time) model loading: Models auto-load on first request
    - TTL-based auto-eviction: Models unload after idle time
    - Multiple model instances: Same model can be loaded multiple times
    - Enhanced stats: tokens/second, time to first token
    """
    
    async def health_check(self) -> bool:
        """Check if LM Studio is available."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except Exception as e:
            print(f"LM Studio health check failed: {e}", flush=True)
            logger.warning(f"LM Studio health check failed: {e}")
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
            max_tokens: Override default max tokens (OpenAI standard parameter)
            model: Override default model
            
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
                "model": model if model is not None else self.model,
                "messages": messages,
                "stream": False,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            logger.debug(f"LM Studio request: model={payload['model']}, messages={len(messages)}, temp={payload['temperature']}, max_tokens={payload['max_tokens']}")
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            # OpenAI-compatible response format
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            used_model = data.get("model", model if model is not None else self.model)
            finish_reason = choice.get("finish_reason")
            
            return LLMResponse(
                content=content,
                model=used_model,
                finish_reason=finish_reason
            )
        
        except httpx.HTTPStatusError as e:
            # Log the error response body for debugging
            error_body = ""
            try:
                error_body = e.response.text
                logger.error(f"LM Studio error response: {error_body}")
                
                # Check for context overflow
                if "context" in error_body.lower() and "overflow" in error_body.lower():
                    # Extract context info from error if possible
                    logger.error(
                        f"Context overflow detected! Your configured context_window is {self.context_window} tokens, "
                        f"but LM Studio reports the model is loaded with a smaller context. "
                        f"Please reload the model in LM Studio with a larger context window (e.g., 8192, 16384, or 32768)."
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
        Generate a completion with full conversation history.
        
        Args:
            messages: Full message history array with roles and content
                     Format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Override default model
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        try:
            payload = {
                "model": model if model is not None else self.model,
                "messages": messages,
                "stream": False,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            # Debug logging to help diagnose verbosity issues
            logger.info(f"LM Studio generation request:")
            logger.info(f"  Model: {payload['model']}")
            logger.info(f"  Temperature: {payload['temperature']}")
            logger.info(f"  Max tokens: {payload['max_tokens']}")
            logger.info(f"  Message count: {len(messages)}")
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            # OpenAI-compatible response format
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            used_model = data.get("model", model if model is not None else self.model)
            finish_reason = choice.get("finish_reason")
            
            logger.info(f"LM Studio response: {len(content)} chars, finish_reason={finish_reason}")
            
            return LLMResponse(
                content=content,
                model=used_model,
                finish_reason=finish_reason
            )
            
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
    ) -> AsyncIterator[str]:
        """
        Stream a completion using OpenAI-compatible SSE format.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Chunks of generated text
            
        Raises:
            LLMError: If streaming fails
        """
        try:
            # Build messages array
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
                json=payload
            ) as response:
                response.raise_for_status()
                
                # OpenAI SSE format: "data: {json}\n\n"
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during LLM streaming: {e}")
        except Exception as e:
            raise LLMError(f"Failed to stream LLM response: {e}")
    
    async def stream_with_history(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion with full conversation history.
        
        Args:
            messages: Full message history array with roles and content
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Override default model
            
        Yields:
            Chunks of generated text
            
        Raises:
            LLMError: If streaming fails
        """
        try:
            payload = {
                "model": model if model is not None else self.model,
                "messages": messages,
                "stream": True,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                # OpenAI SSE format: "data: {json}\n\n"
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during LLM streaming: {e}")
        except Exception as e:
            raise LLMError(f"Failed to stream LLM response: {e}")
    
    # LM Studio model management (optional, uses native REST API)
    
    async def get_loaded_models(self) -> list[str]:
        """
        Get list of currently loaded models in LM Studio.
        
        Uses native REST API v0 to query model state.
        
        Returns:
            List of model identifiers currently loaded in memory
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v0/models")
            response.raise_for_status()
            data = response.json()
            
            # Filter for loaded models only
            models = []
            if "data" in data:
                for model_info in data["data"]:
                    if model_info.get("state") == "loaded":
                        models.append(model_info["id"])
            
            return models
            
        except Exception as e:
            logger.warning(f"Failed to get loaded models from LM Studio: {e}")
            return []
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """
        Ensure a specific model is loaded.
        
        LM Studio auto-loads models on first request (JIT loading),
        so we just need to make a minimal request with that model.
        
        Args:
            model: Model identifier to ensure is loaded
            
        Returns:
            True if model is loaded or loads successfully
        """
        try:
            # Check if already loaded
            loaded = await self.get_loaded_models()
            if model in loaded:
                logger.debug(f"Model {model} already loaded")
                return True
            
            # Make a minimal request to trigger JIT loading
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                "max_tokens": 1,
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            logger.info(f"Model {model} loaded via JIT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure model {model} loaded: {e}")
            return False
    
    async def unload_model(self, model: str) -> bool:
        """
        Unload a specific model using LM Studio CLI.
        
        LM Studio doesn't have an HTTP API endpoint for unloading, but the CLI
        supports `lms unload <model>`. We'll use subprocess to call it.
        
        Args:
            model: Model identifier to unload
            
        Returns:
            True if unload succeeded, False otherwise
        """
        try:
            import subprocess
            
            # Try to unload via CLI
            result = subprocess.run(
                ['lms', 'unload', model],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully unloaded model {model} via CLI")
                return True
            else:
                logger.warning(f"Failed to unload model {model}: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning("LM Studio CLI (lms) not found in PATH. Cannot unload models.")
            return False
        except Exception as e:
            logger.error(f"Error unloading model {model}: {e}")
            return False
    
    async def unload_all_models(self) -> None:
        """
        Unload all models for ComfyUI generation using LM Studio CLI.
        
        LM Studio doesn't have an HTTP API endpoint for unloading, but the CLI
        supports `lms unload --all`. We'll use subprocess to call it.
        
        Falls back to logging a warning if CLI is not available.
        """
        logger.info("Unloading all models for ComfyUI generation")
        
        try:
            # Get currently loaded models
            loaded = await self.get_loaded_models()
            
            if not loaded:
                logger.info("No models currently loaded in LM Studio")
                return
            
            logger.info(f"Found {len(loaded)} loaded models: {loaded}")
            
            # Try to unload all via CLI
            import subprocess
            
            logger.info("Calling: lms unload --all")
            result = subprocess.run(
                ['lms', 'unload', '--all'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            logger.info(f"lms unload return code: {result.returncode}")
            if result.stdout:
                logger.info(f"lms stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"lms stderr: {result.stderr}")
            
            if result.returncode == 0:
                logger.info("Successfully unloaded all models via CLI")
                
                # Verify unload
                import asyncio
                await asyncio.sleep(1)  # Give LM Studio time to unload
                
                loaded_after = await self.get_loaded_models()
                if loaded_after:
                    logger.warning(
                        f"Models still loaded after unload attempt: {loaded_after}"
                    )
                else:
                    logger.info("All models unloaded successfully, VRAM freed for ComfyUI")
            else:
                logger.error(f"Failed to unload models via CLI (exit code {result.returncode})")
                logger.info("Models will remain in VRAM until TTL expires")
                
        except FileNotFoundError:
            logger.warning(
                "LM Studio CLI (lms) not found in PATH. Cannot unload models. "
                "Models will remain in VRAM until TTL expires. "
                "To enable unloading, ensure 'lms' command is available in PATH."
            )
        except Exception as e:
            logger.error(f"Error during model unload: {e}", exc_info=True)
    
    async def reload_models_after_generation(
        self,
        character_model: str,
        intent_model: str = "gemma2:9b"
    ) -> None:
        """
        Reload models after ComfyUI generation.
        
        For LM Studio, this is mostly a no-op since models auto-load on first request.
        We just ensure the character model is ready for the next chat request.
        
        Args:
            character_model: Character's preferred model
            intent_model: Intent detection model (not used with LM Studio - keyword detection only)
        """
        logger.info(f"LM Studio: Ensuring {character_model} ready after ComfyUI generation")
        
        try:
            # Just ensure character model is loaded (intent detection uses keywords with LM Studio)
            await self.ensure_model_loaded(character_model)
            logger.info(f"Model {character_model} ready")
            
        except Exception as e:
            logger.error(f"Error reloading model {character_model}: {e}", exc_info=True)
