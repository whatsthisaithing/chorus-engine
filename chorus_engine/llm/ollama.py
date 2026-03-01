"""Ollama LLM client implementation."""

import json
from typing import List

from .text_normalization import normalize_mojibake
import logging
from typing import Optional, AsyncIterator
from .base import BaseLLMClient, LLMResponse, LLMError, LLMStreamEvent
import httpx

logger = logging.getLogger(__name__)


class OllamaLLMClient(BaseLLMClient):
    """Client for interacting with Ollama API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
        context_window: int = 8192,
        use_legacy_chat_api: bool = False,
        capture_raw_http_debug: bool = False,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            capture_raw_http_debug=capture_raw_http_debug,
        )
        self.use_legacy_chat_api = bool(use_legacy_chat_api)

    async def _post_with_optional_raw_capture(self, endpoint_path: str, payload: dict) -> httpx.Response:
        request_body_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        response = await self.client.post(
            f"{self.base_url}{endpoint_path}",
            content=request_body_bytes,
            headers={"Content-Type": "application/json"},
        )
        if self.capture_raw_http_debug:
            self._capture_raw_http_exchange(
                provider="ollama",
                endpoint_path=endpoint_path,
                payload=payload,
                request_body_bytes=request_body_bytes,
                response=response,
            )
        return response

    @staticmethod
    def _apply_openai_sampling_fields(
        payload: dict,
        *,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> None:
        if top_p is not None:
            payload["top_p"] = top_p
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty

    @staticmethod
    def _parse_openai_choice(data: dict) -> tuple[dict, dict]:
        choice = (data.get("choices") or [{}])[0] or {}
        message = choice.get("message", {}) or {}
        return choice, message
    
    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                return True
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            # Print to stderr so it appears in server log
            print(f"Ollama health check failed: {e}", flush=True)
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
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
        Generate a completion from the LLM using chat endpoint.
        
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
        try:
            # Build messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if self.use_legacy_chat_api:
                if any(x is not None for x in (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty)):
                    logger.info("[OLLAMA] Advanced sampling controls ignored for legacy /api/chat transport.")
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": False,
                }
                if temperature is not None or max_tokens is not None:
                    payload["options"] = {}
                    payload["options"]["temperature"] = self.temperature if temperature is None else temperature
                    if max_tokens is not None:
                        payload["options"]["num_predict"] = max_tokens
                response = await self._post_with_optional_raw_capture("/api/chat", payload)
                response.raise_for_status()
                if response.encoding is None or response.encoding.lower() != "utf-8":
                    response.encoding = "utf-8"
                response_text = response.text
                data = response.json()
                message = data.get("message", {}) or {}
                content = normalize_mojibake(message.get("content", ""))
                finish_reason = data.get("done_reason")
                tool_calls = None
                raw_message = message if isinstance(message, dict) else None
            else:
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                }
                self._apply_openai_sampling_fields(
                    payload,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                if tools:
                    payload["tools"] = tools
                    if tool_choice is not None:
                        payload["tool_choice"] = tool_choice
                if isinstance(response_format, dict):
                    payload["response_format"] = response_format
                response = await self._post_with_optional_raw_capture("/v1/chat/completions", payload)
                response.raise_for_status()
                if response.encoding is None or response.encoding.lower() != "utf-8":
                    response.encoding = "utf-8"
                response_text = response.text
                data = response.json()
                choice, message = self._parse_openai_choice(data)
                content = normalize_mojibake(message.get("content", ""))
                finish_reason = choice.get("finish_reason")
                tool_calls = message.get("tool_calls")
                raw_message = message if isinstance(message, dict) else None
            if not content.strip():
                logger.warning(
                    f"[OLLAMA] Empty response raw preview: "
                    f"len={len(response_text)}, "
                    f"keys={list(data.keys())}, "
                    f"preview={response_text[:500]!r}"
                )
                logger.warning(
                    f"[OLLAMA] Empty response content: "
                    f"model={data.get('model', model if model is not None else self.model)}, "
                    f"done_reason={finish_reason}, "
                    f"prompt_tokens={data.get('prompt_eval_count', 0) or data.get('usage', {}).get('prompt_tokens', 0)}, "
                    f"output_tokens={data.get('eval_count', 0) or data.get('usage', {}).get('completion_tokens', 0)}"
                )
            
            # Log Ollama timing metrics for debugging
            load_dur = data.get("load_duration", 0) / 1e9  # nanoseconds to seconds
            prompt_eval_dur = data.get("prompt_eval_duration", 0) / 1e9
            eval_dur = data.get("eval_duration", 0) / 1e9
            total_dur = data.get("total_duration", 0) / 1e9
            prompt_tokens = data.get("prompt_eval_count", 0) or data.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = data.get("eval_count", 0) or data.get("usage", {}).get("completion_tokens", 0)
            
            if load_dur > 1.0 or total_dur > 30:  # Log if model loaded or took >30s
                logger.info(
                    f"[OLLAMA] Request completed: total={total_dur:.1f}s, "
                    f"load={load_dur:.1f}s, prompt_eval={prompt_eval_dur:.1f}s ({prompt_tokens} tokens), "
                    f"gen={eval_dur:.1f}s ({output_tokens} tokens)"
                )
            
            # Use the model from response to confirm what Ollama actually used
            used_model = data.get("model", model if model is not None else self.model)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
                "load_duration_s": load_dur,
                "prompt_eval_duration_s": prompt_eval_dur,
                "eval_duration_s": eval_dur,
                "total_duration_s": total_dur
            }
            
            return LLMResponse(
                content=content,
                model=used_model,
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=tool_calls,
                raw_message=raw_message,
            )
        
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during LLM generation: {e}")
        except Exception as e:
            raise LLMError(f"Failed to generate LLM response: {e}")

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
        try:
            if not image_base64_list:
                raise LLMError("No images provided for vision generation")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if self.use_legacy_chat_api:
                if any(x is not None for x in (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty)):
                    logger.info("[OLLAMA] Advanced sampling controls ignored for legacy /api/chat transport.")
                messages.append({"role": "user", "content": prompt, "images": image_base64_list})
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature if temperature is None else temperature,
                        "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                    },
                }
                endpoint = f"{self.base_url}/api/chat"
            else:
                content_parts = [{"type": "text", "text": prompt}]
                for image_base64 in image_base64_list:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                        }
                    )
                messages.append({"role": "user", "content": content_parts})
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                }
                self._apply_openai_sampling_fields(
                    payload,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                endpoint = f"{self.base_url}/v1/chat/completions"
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            if self.use_legacy_chat_api:
                message = data.get("message", {}) or {}
                content = normalize_mojibake(message.get("content", ""))
                finish_reason = data.get("done_reason")
            else:
                choice, message = self._parse_openai_choice(data)
                content = normalize_mojibake(message.get("content", ""))
                finish_reason = choice.get("finish_reason")
            return LLMResponse(
                content=content,
                model=data.get("model", model if model is not None else self.model),
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0) or data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("eval_count", 0) or data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": (
                        (data.get("prompt_eval_count", 0) or data.get("usage", {}).get("prompt_tokens", 0))
                        + (data.get("eval_count", 0) or data.get("usage", {}).get("completion_tokens", 0))
                    ),
                },
                tool_calls=(message.get("tool_calls") if isinstance(message, dict) else None),
                raw_message=(message if isinstance(message, dict) else None),
            )
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during vision generation: {e}")
        except Exception as e:
            raise LLMError(f"Failed to generate vision response: {e}")
    
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
            if self.use_legacy_chat_api:
                if any(x is not None for x in (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty)):
                    logger.info("[OLLAMA] Advanced sampling controls ignored for legacy /api/chat transport.")
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": False,
                }
                payload["options"] = {
                    "temperature": self.temperature if temperature is None else temperature,
                    "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                }
                logger.debug(
                    "Ollama request: model=%s temp=%s max_tokens=%s messages=%s",
                    payload["model"],
                    payload["options"]["temperature"],
                    payload["options"]["num_predict"],
                    len(messages),
                )
                response = await self._post_with_optional_raw_capture("/api/chat", payload)
            else:
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                }
                self._apply_openai_sampling_fields(
                    payload,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                if tools:
                    payload["tools"] = tools
                    if tool_choice is not None:
                        payload["tool_choice"] = tool_choice
                if isinstance(response_format, dict):
                    payload["response_format"] = response_format
                logger.debug(
                    "Ollama(OpenAI) request: model=%s temp=%s max_tokens=%s messages=%s tools=%s",
                    payload["model"],
                    payload["temperature"],
                    payload["max_tokens"],
                    len(messages),
                    len(tools or []),
                )
                response = await self._post_with_optional_raw_capture("/v1/chat/completions", payload)
            response.raise_for_status()

            if response.encoding is None or response.encoding.lower() != "utf-8":
                response.encoding = "utf-8"

            response_text = response.text
            data = response.json()
            if self.use_legacy_chat_api:
                message = data.get("message", {}) or {}
                content = normalize_mojibake(message.get("content", ""))
                finish_reason = data.get("done_reason")
                tool_calls = None
                raw_message = message if isinstance(message, dict) else None
            else:
                choice, message = self._parse_openai_choice(data)
                content = normalize_mojibake(message.get("content", ""))
                finish_reason = choice.get("finish_reason")
                tool_calls = message.get("tool_calls")
                raw_message = message if isinstance(message, dict) else None
            if not content.strip():
                logger.warning(
                    f"[OLLAMA] Empty response raw preview (history): "
                    f"len={len(response_text)}, "
                    f"keys={list(data.keys())}, "
                    f"preview={response_text[:500]!r}"
                )
                logger.warning(
                    f"[OLLAMA] Empty response content (history): "
                    f"model={data.get('model', model if model is not None else self.model)}, "
                    f"done_reason={finish_reason}, "
                    f"prompt_tokens={data.get('prompt_eval_count', 0) or data.get('usage', {}).get('prompt_tokens', 0)}, "
                    f"output_tokens={data.get('eval_count', 0) or data.get('usage', {}).get('completion_tokens', 0)}"
                )
            
            # Use the model from response to confirm what Ollama actually used
            used_model = data.get("model", model if model is not None else self.model)
            
            logger.debug(f"Ollama response: {len(content)} chars, reason={data.get('done_reason')}")
            
            return LLMResponse(
                content=content,
                model=used_model,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0) or data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("eval_count", 0) or data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": (
                        (data.get("prompt_eval_count", 0) or data.get("usage", {}).get("prompt_tokens", 0))
                        + (data.get("eval_count", 0) or data.get("usage", {}).get("completion_tokens", 0))
                    ),
                    "load_duration_s": data.get("load_duration", 0) / 1e9,
                    "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
                    "eval_duration_s": data.get("eval_duration", 0) / 1e9,
                    "total_duration_s": data.get("total_duration", 0) / 1e9
                },
                tool_calls=tool_calls,
                raw_message=raw_message,
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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion from the LLM using chat endpoint.
        
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
            
            if self.use_legacy_chat_api:
                if any(x is not None for x in (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty)):
                    logger.info("[OLLAMA] Advanced sampling controls ignored for legacy /api/chat transport.")
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }
                if temperature is not None or max_tokens is not None:
                    payload["options"] = {}
                    payload["options"]["temperature"] = self.temperature if temperature is None else temperature
                    if max_tokens is not None:
                        payload["options"]["num_predict"] = max_tokens
                endpoint = f"{self.base_url}/api/chat"
            else:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                }
                self._apply_openai_sampling_fields(
                    payload,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                endpoint = f"{self.base_url}/v1/chat/completions"

            async with self.client.stream(
                "POST",
                endpoint,
                json=payload
            ) as response:
                response.raise_for_status()

                if response.encoding is None or response.encoding.lower() != "utf-8":
                    response.encoding = "utf-8"
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        if self.use_legacy_chat_api:
                            data = json.loads(line)
                            if "message" in data:
                                content = normalize_mojibake(data["message"].get("content", ""))
                                if content:
                                    yield content
                        else:
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                content = normalize_mojibake(delta.get("content", ""))
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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
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
            if self.use_legacy_chat_api:
                if any(x is not None for x in (top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty)):
                    logger.info("[OLLAMA] Advanced sampling controls ignored for legacy /api/chat transport.")
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": True,
                }
                if temperature is not None or max_tokens is not None:
                    payload["options"] = {}
                    payload["options"]["temperature"] = self.temperature if temperature is None else temperature
                    payload["options"]["num_predict"] = self.max_tokens if max_tokens is None else max_tokens
                endpoint = f"{self.base_url}/api/chat"
                logger.debug(
                    "Ollama stream: model=%s temp=%s messages=%s",
                    payload["model"],
                    payload.get("options", {}).get("temperature", self.temperature),
                    len(messages),
                )
            else:
                payload = {
                    "model": model if model is not None else self.model,
                    "messages": messages,
                    "stream": True,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                }
                self._apply_openai_sampling_fields(
                    payload,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                endpoint = f"{self.base_url}/v1/chat/completions"
                logger.debug(
                    "Ollama(OpenAI) stream: model=%s temp=%s messages=%s",
                    payload["model"],
                    payload["temperature"],
                    len(messages),
                )
            
            async with self.client.stream(
                "POST",
                endpoint,
                json=payload
            ) as response:
                response.raise_for_status()

                if response.encoding is None or response.encoding.lower() != "utf-8":
                    response.encoding = "utf-8"
                
                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        if self.use_legacy_chat_api:
                            data = json.loads(line)
                            if "error" in data:
                                logger.error(f"Ollama error: {data['error']}")
                            if "message" in data:
                                content = normalize_mojibake(data["message"].get("content", ""))
                                if content:
                                    chunk_count += 1
                                    yield content
                            if data.get("done") and chunk_count == 0:
                                logger.warning(f"Ollama returned zero content, reason: {data.get('done_reason', 'unknown')}")
                        else:
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                content = normalize_mojibake(delta.get("content", ""))
                                if content:
                                    chunk_count += 1
                                    yield content
                    except json.JSONDecodeError:
                        continue
                
                logger.debug(f"Ollama stream completed: {chunk_count} chunks")
                            
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error during LLM streaming: {e}")
        except Exception as e:
            raise LLMError(f"Failed to stream LLM response: {e}")

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
    ) -> AsyncIterator[LLMStreamEvent]:
        async for chunk in self.stream_with_history(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            model=model,
        ):
            yield LLMStreamEvent(content_delta=str(chunk or ""))
    
    # Ollama-specific model management methods (override base implementations)
    
    async def get_loaded_models(self) -> list[str]:
        """
        Get list of currently loaded models in Ollama.
        
        Returns:
            List of model names currently in VRAM
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            data = response.json()
            
            # Extract model names from running models
            models = []
            if "models" in data:
                for model_info in data["models"]:
                    if "name" in model_info:
                        models.append(model_info["name"])
            
            return models
            
        except Exception as e:
            logger.warning(f"Failed to get loaded models: {e}")
            return []
    
    async def unload_model(self, model: str = None) -> bool:
        """
        Unload a specific model from VRAM (Ollama-specific keep_alive=0).
        
        Args:
            model: Model identifier to unload (defaults to self.model if not specified)
        
        Returns:
            True if successfully unloaded, False otherwise
        """
        try:
            # Use specified model or fall back to self.model
            model_to_unload = model or self.model
            
            # For Ollama, send a request with keep_alive=0 to unload
            payload = {
                "model": model_to_unload,
                "keep_alive": 0
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Unloaded model {model_to_unload} from VRAM")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")
            return False
    
    async def reload_model(self) -> bool:
        """
        Preload the model back into VRAM (Ollama-specific).
        Sends an empty generation request to trigger model loading.
        
        Returns:
            True if successfully reloaded, False otherwise
        """
        try:
            # For Ollama, send a minimal request to load the model
            payload = {
                "model": self.model,
                "prompt": "",
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Reloaded model {self.model} into VRAM")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to reload model: {e}")
            return False
    
    async def unload_all_models(self) -> None:
        """
        Unload ALL currently loaded models.
        
        Used before ComfyUI generation to free maximum VRAM.
        Verifies unload with Ollama status check.
        """
        logger.info("Unloading all models for ComfyUI generation")
        
        try:
            # Get currently loaded models
            loaded = await self.get_loaded_models()
            
            if not loaded:
                logger.info("No models currently loaded")
                return
            
            logger.info(f"Found {len(loaded)} loaded models: {loaded}")
            
            # Unload each model
            for model_name in loaded:
                try:
                    payload = {
                        "model": model_name,
                        "keep_alive": 0
                    }
                    
                    response = await self.client.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    )
                    response.raise_for_status()
                    logger.info(f"Unloaded model: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to unload {model_name}: {e}")
            
            # Verify unload
            import asyncio
            await asyncio.sleep(1)  # Give Ollama time to unload
            
            loaded_after = await self.get_loaded_models()
            if loaded_after:
                logger.warning(
                    f"Models still loaded after unload attempt",
                    extra={"models": loaded_after}
                )
            else:
                logger.info("All models unloaded successfully, VRAM freed for ComfyUI")
                
        except Exception as e:
            logger.error(f"Error during model unload: {e}", exc_info=True)
    
    async def reload_models_after_generation(
        self,
        character_model: str,
        intent_model: str = "qwen2.5:3b-instruct"
    ) -> None:
        """
        Reload intent and character models after ComfyUI generation.
        
        Loads intent model first (faster, needed for next message),
        then character model (slower, needed for response).
        
        Args:
            character_model: Character's preferred model
            intent_model: Intent detection model (default: qwen2.5:3b-instruct)
        """
        import time
        
        logger.info("Reloading models after ComfyUI generation")
        
        try:
            # Load intent model first (2-3 seconds, needed immediately)
            start = time.time()
            payload = {
                "model": intent_model,
                "prompt": "",
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            intent_time = time.time() - start
            logger.info(f"Intent model ({intent_model}) loaded in {intent_time:.2f}s")
            
            # Load character model second (3-5 seconds, needed for response)
            start = time.time()
            payload = {
                "model": character_model,
                "prompt": "",
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            character_time = time.time() - start
            logger.info(f"Character model ({character_model}) loaded in {character_time:.2f}s")
            
            total_time = intent_time + character_time
            logger.info(f"All models reloaded in {total_time:.2f}s, ready for conversation")
            
        except Exception as e:
            logger.error(f"Error during model reload: {e}", exc_info=True)
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """
        Ensure a specific model is loaded into VRAM.
        
        Args:
            model: Model name to ensure is loaded
            
        Returns:
            True if model is loaded, False otherwise
        """
        try:
            # Check if already loaded
            loaded = await self.get_loaded_models()
            if model in loaded:
                logger.debug(f"Model {model} already loaded")
                return True
            
            # Load the model
            payload = {
                "model": model,
                "prompt": "",
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            logger.info(f"Model {model} loaded into VRAM")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure model {model} loaded: {e}")
            return False
