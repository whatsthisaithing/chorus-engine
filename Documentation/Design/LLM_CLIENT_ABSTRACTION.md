# LLM Client Abstraction & Multi-Provider Support Design

**Phase**: Mid-Development Refactor (January 2, 2026)  
**Created**: January 4, 2026  
**Status**: Implemented & Tested

---

## Overview

The LLM Client Abstraction provides a provider-agnostic interface for interacting with different LLM backends (Ollama, LM Studio, OpenAI-compatible services). Rather than hardcoding for a single backend, this system allows users to switch providers without code changes, enabling flexibility in model deployment while maintaining consistent behavior.

This design document captures the architecture, provider implementations, design decisions, and migration strategy for multi-provider LLM support.

---

## Core Philosophy

### The Provider-Agnostic Interface Principle

**Central Insight**: LLM providers have different APIs, but similar capabilities. Abstract the differences behind a common interface.

**Common Operations**:
- Health check (is service available?)
- Text generation (given prompt, return response)
- Streaming generation (token-by-token)
- Chat with history (messages array)
- Model management (load/unload)

**Provider Differences**:
| Feature | Ollama | LM Studio | OpenAI |
|---------|--------|-----------|--------|
| Endpoint | `/api/chat` | `/v1/chat/completions` | `/v1/chat/completions` |
| Streaming | Custom format | SSE (Server-Sent Events) | SSE |
| Model loading | Manual (`/api/generate`) | JIT (Just-In-Time) | N/A (managed) |
| Max tokens param | `options.num_predict` | `max_tokens` | `max_tokens` |
| Keep-alive | `keep_alive: 0` to unload | TTL-based auto-eviction | N/A |

**Solution**: Base interface with provider-specific implementations.

---

### The Factory Pattern Principle

**Central Insight**: Create the right client based on configuration, not manual instantiation.

**Implementation**:
```python
# OLD: Hardcoded client
from chorus_engine.llm.client import LLMClient
client = LLMClient(base_url="http://localhost:11434", ...)

# NEW: Factory-based creation
from chorus_engine.llm import create_llm_client
client = create_llm_client(config)
# Returns: OllamaLLMClient or LMStudioLLMClient or OpenAIClient
```

**Why Factory Works**:
- Configuration drives behavior
- Easy to add new providers
- Backward compatible (old code still works)
- Type hints work correctly

---

### The Fail-Fast Principle

**Central Insight**: Unknown providers should error immediately, not at runtime.

**Implementation**:
```python
def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    if config.provider == "ollama":
        return OllamaLLMClient(...)
    elif config.provider == "lmstudio":
        return LMStudioLLMClient(...)
    elif config.provider in ["openai-compatible", "llamacpp"]:
        raise NotImplementedError(f"Provider '{config.provider}' not yet implemented")
    else:
        raise ValueError(f"Unknown provider: '{config.provider}'")
```

**Why Fail-Fast Works**:
- Configuration errors caught at startup
- Clear error messages guide users
- No silent fallbacks that hide problems
- Forces fixing root cause

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Configuration (YAML)                  │
│  provider: ollama | lmstudio | openai-compatible            │
│  base_url: http://localhost:11434                           │
│  model: qwen2.5:14b-instruct                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  create_llm_client()    │
        │  Factory Function       │
        └────────────┬────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼──────┐ ┌────▼──────┐ ┌────▼─────────┐
│  Ollama    │ │ LM Studio │ │ OpenAI-Compat│
│  Client    │ │ Client    │ │ (Future)     │
└─────┬──────┘ └────┬──────┘ └──────────────┘
      │              │
      └──────────────┼─────────────────────┐
                     │                     │
        ┌────────────▼────────────┐        │
        │    BaseLLMClient        │        │
        │    (Abstract Interface) │        │
        │                         │        │
        │  - health_check()       │        │
        │  - generate()           │        │
        │  - stream_generate()    │        │
        │  - generate_with_history│        │
        │  - stream_with_history()│        │
        │  - list_models()        │        │
        │  - ensure_model_loaded()│        │
        │  - unload_model()       │        │
        └─────────────────────────┘        │
                     │                     │
                     │                     │
        ┌────────────▼────────────┐        │
        │   LLMResponse           │        │
        │  - content: str         │        │
        │  - model: str           │        │
        │  - finish_reason: str   │        │
        └─────────────────────────┘        │
                                            │
        ┌───────────────────────────────────▼────┐
        │        Application Code                │
        │  Uses BaseLLMClient interface only     │
        │  No knowledge of specific providers    │
        └────────────────────────────────────────┘
```

### Base Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standard response from any LLM provider."""
    content: str
    model: str
    finish_reason: Optional[str] = None

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
        context_window: int = 8192,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.client = httpx.AsyncClient(timeout=timeout)
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""
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
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate with conversation history."""
        pass
    
    # Optional methods (default implementations)
    async def stream_generate(self, ...) -> AsyncIterator[str]:
        """Stream generation token by token."""
        raise NotImplementedError("Streaming not supported by this provider")
    
    async def list_models(self) -> List[str]:
        """List available models."""
        return [self.model]  # Default: only configured model
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """Ensure model is loaded (provider-specific)."""
        return True  # Default: assume always loaded
    
    async def unload_model(self, model: str) -> bool:
        """Unload model (provider-specific)."""
        return True  # Default: no-op
```

---

## Provider Implementations

### 1. Ollama Provider

**Characteristics**:
- Local model hosting
- Manual model management (load/unload)
- `/api/chat` for chat completions
- `/api/generate` for model control
- `keep_alive: 0` to unload model
- Custom streaming format

**Implementation Highlights**:
```python
class OllamaLLMClient(BaseLLMClient):
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    async def generate_with_history(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,  # Ollama param name
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        data = response.json()
        
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", model or self.model),
            finish_reason=data.get("done_reason")
        )
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """Load model explicitly (Ollama-specific)."""
        payload = {
            "model": model,
            "prompt": "",  # Empty prompt = just load
            "stream": False
        }
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        return response.status_code == 200
    
    async def unload_model(self, model: str) -> bool:
        """Unload model from VRAM (Ollama-specific)."""
        payload = {
            "model": model,
            "keep_alive": 0  # 0 = unload immediately
        }
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        return response.status_code == 200
```

**Strengths**:
- Easy to run locally
- Good VRAM management
- Simple API
- Active development

**Limitations**:
- Requires Ollama installed
- Manual model management
- Limited to local use

---

### 2. LM Studio Provider

**Characteristics**:
- OpenAI-compatible API
- Just-In-Time (JIT) model loading
- TTL-based auto-eviction
- `/v1/chat/completions` endpoint
- SSE (Server-Sent Events) streaming
- Enhanced stats (tokens/sec, TTFT)

**Implementation Highlights**:
```python
class LMStudioLLMClient(BaseLLMClient):
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except:
            return False
    
    async def generate_with_history(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,  # Standard param name
        }
        
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        data = response.json()
        
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model"),
            finish_reason=choice.get("finish_reason")
        )
    
    async def stream_with_history(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream using SSE format."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
```

**JIT Model Loading**:
- Models auto-load on first request
- No explicit loading needed
- TTL-based auto-unload (configurable)

**Strengths**:
- OpenAI-compatible (easy migration)
- Automatic model management
- Good UI for model browsing
- Enhanced performance stats

**Limitations**:
- Less VRAM control than Ollama
- Newer (less battle-tested)
- Requires LM Studio installed

---

### 3. Future: OpenAI-Compatible Provider

**Target Services**:
- OpenAI API (cloud)
- LocalAI (self-hosted)
- Text Generation WebUI (self-hosted)
- Any OpenAI-compatible endpoint

**Implementation** (planned):
```python
class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, ..., api_key: Optional[str] = None):
        super().__init__(...)
        self.api_key = api_key
        if api_key:
            self.client.headers["Authorization"] = f"Bearer {api_key}"
    
    # Similar to LM Studio (uses /v1/ endpoints)
    # Adds authentication support
```

---

## Design Decisions & Rationale

### Decision: Abstract Base Class vs. Protocol

**Alternatives Considered**:
1. **Protocol (structural typing)**
   - ❌ No shared initialization
   - ❌ Code duplication
   - ❌ Harder to enforce contracts

2. **Concrete base with hooks**
   - ❌ Providers tightly coupled
   - ❌ Difficult to override behavior

3. **Abstract Base Class (chosen)**
   - ✅ Shared initialization logic
   - ✅ Clear contract enforcement
   - ✅ Optional method defaults
   - ✅ Type checking support

**Why ABC Works**:
- Common initialization (base_url, timeout, client)
- Required methods enforced (@abstractmethod)
- Optional methods have defaults (streaming, model mgmt)
- Clear inheritance hierarchy

---

### Decision: Factory Function vs. Dependency Injection

**Alternatives Considered**:
1. **Manual instantiation**
   ```python
   if config.provider == "ollama":
       client = OllamaLLMClient(...)
   ```
   - ❌ Duplicated everywhere
   - ❌ Hard to maintain

2. **Dependency Injection Container**
   - ❌ Overkill for single service
   - ❌ Added complexity
   - ❌ Harder to understand

3. **Factory Function (chosen)**
   ```python
   client = create_llm_client(config)
   ```
   - ✅ Single point of creation
   - ✅ Configuration-driven
   - ✅ Easy to test
   - ✅ Simple to understand

**Why Factory Works**:
- Centralizes provider selection logic
- Easy to add new providers
- Configuration validation in one place
- Backward compatible wrapper available

---

### Decision: Unified Response Type

**Alternatives Considered**:
1. **Provider-specific responses**
   - ❌ Application code knows providers
   - ❌ Hard to switch providers
   - ❌ Type complexity

2. **Dict responses** (untyped)
   - ❌ No type checking
   - ❌ Unclear structure
   - ❌ Runtime errors

3. **Unified LLMResponse (chosen)**
   - ✅ Type-safe
   - ✅ Provider-agnostic
   - ✅ Clear contract
   - ✅ Easy to extend

**Why Unified Response Works**:
- Application code provider-agnostic
- Type hints work correctly
- Easy to add fields (backward compatible)
- Clear documentation

---

### Decision: Optional vs. Required Methods

**Required** (abstract):
- `health_check()` - Every provider must check availability
- `generate()` - Basic completion
- `generate_with_history()` - Chat-style completion

**Optional** (default implementations):
- `stream_generate()` - Not all providers support streaming
- `list_models()` - Some providers can't enumerate
- `ensure_model_loaded()` - Provider-specific
- `unload_model()` - Provider-specific

**Why This Split Works**:
- Core functionality always available
- Advanced features opt-in
- Default implementations prevent NotImplementedError spam
- Providers can override as needed

---

## Known Limitations

### 1. No Automatic Provider Detection
**Limitation**: User must specify provider in config.

**Why**: Different providers can run on same port, no reliable detection.

**Workaround**: Configuration-driven selection.

**Future**: Could attempt probing endpoints for detection.

---

### 2. No Fallback Chain
**Limitation**: If primary provider fails, no automatic fallback.

**Why**: Adds complexity, unclear semantics (which models are equivalent?).

**Workaround**: User switches provider manually.

**Future**: Could add provider list with automatic failover.

---

### 3. Streaming Format Differences
**Limitation**: Ollama and LM Studio have different streaming formats.

**Why**: Different upstream implementations.

**Impact**: Streaming code provider-specific.

**Mitigation**: Abstracted in provider implementations.

---

### 4. Model Name Inconsistencies
**Limitation**: Same model has different names across providers.

**Why**: No standard naming convention.

**Example**: `qwen2.5:14b-instruct` (Ollama) vs. `qwen2.5-14b-instruct-gguf` (LM Studio).

**Workaround**: User configures model name per provider.

---

## Performance Characteristics

**Factory Creation**: O(1), instantaneous

**Health Check**:
- Ollama: Single GET request (~10-50ms)
- LM Studio: Single GET request (~10-50ms)

**Generation**:
- Provider-dependent
- Network latency: ~5-20ms (localhost)
- Model inference: 100ms-10s (model/hardware dependent)

**Streaming**:
- First token latency: Provider-dependent
- Token rate: Provider-dependent (typically 10-50 tokens/sec local)

**Memory Overhead**:
- Base client: ~1KB
- httpx client: ~100KB
- Total per provider: ~100KB

---

## Testing & Validation

### Unit Tests
```python
def test_factory_creates_ollama():
    config = LLMConfig(provider="ollama", base_url="http://localhost:11434")
    client = create_llm_client(config)
    assert isinstance(client, OllamaLLMClient)

def test_factory_creates_lmstudio():
    config = LLMConfig(provider="lmstudio", base_url="http://localhost:1234")
    client = create_llm_client(config)
    assert isinstance(client, LMStudioLLMClient)

def test_factory_rejects_unknown():
    config = LLMConfig(provider="unknown")
    with pytest.raises(ValueError):
        create_llm_client(config)
```

### Integration Tests
- Health check against real Ollama instance
- Generation with real LM Studio instance
- Streaming with both providers
- Error handling (service down, context overflow)

### Backward Compatibility Tests
```python
def test_backward_compatible_import():
    from chorus_engine.llm.client import LLMClient
    client = LLMClient(base_url="http://localhost:11434", model="qwen2.5:14b")
    assert isinstance(client, OllamaLLMClient)
```

---

## Migration Guide

### From Hardcoded Ollama to Abstract Client

**Before** (hardcoded):
```python
from chorus_engine.llm.client import LLMClient

client = LLMClient(
    base_url="http://localhost:11434",
    model="qwen2.5:14b-instruct"
)
```

**After** (provider-agnostic):
```python
from chorus_engine.llm import create_llm_client
from chorus_engine.config import load_config

config = load_config()
client = create_llm_client(config.llm)
```

### Switching Providers

**Ollama Configuration**:
```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  model: qwen2.5:14b-instruct
  temperature: 0.7
  max_response_tokens: 2048
```

**LM Studio Configuration**:
```yaml
llm:
  provider: lmstudio
  base_url: http://localhost:1234
  model: lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
  temperature: 0.7
  max_response_tokens: 2048
```

**No Code Changes Required**: Just update config and restart.

---

## Future Enhancements

### High Priority

**1. OpenAI-Compatible Provider**
- Support cloud OpenAI API
- Support LocalAI (self-hosted)
- Support Text Generation WebUI
- API key authentication

**2. Provider Health Monitoring**
- Periodic health checks
- Automatic retry on failure
- Status dashboard

**3. Request Retry Logic**
- Exponential backoff
- Configurable retry count
- Timeout handling

### Medium Priority

**4. Provider Fallback Chain**
- Primary + secondary providers
- Automatic failover
- Load balancing (future)

**5. Model Registry**
- Map canonical names to provider-specific names
- `"llama3-8b"` → `"qwen2.5:14b-instruct"` (Ollama)
- `"llama3-8b"` → `"llama-3-8b-gguf"` (LM Studio)

**6. Performance Metrics**
- Track request latency
- Token rate monitoring
- Error rate tracking
- Provider comparison dashboard

### Low Priority

**7. Multi-Provider Routing**
- Different models on different providers
- Route by capability (streaming vs. non-streaming)
- Cost-based routing (cloud providers)

**8. Provider-Specific Optimizations**
- Ollama: Better VRAM management
- LM Studio: TTL tuning
- OpenAI: Batch requests

---

## Conclusion

The LLM Client Abstraction provides Chorus Engine with flexibility in LLM deployment without sacrificing code quality or user experience. By abstracting provider differences behind a clean interface, users can switch between Ollama, LM Studio, and future providers with a simple configuration change.

Key achievements:
- **Provider Independence**: Application code doesn't know about providers
- **Easy Switching**: Change one config line, no code changes
- **Type Safety**: Strong typing throughout
- **Backward Compatible**: Existing code works unchanged
- **Extensible**: New providers add easily

The system has proven successful through:
- Seamless Ollama → LM Studio migration
- Zero application code changes needed
- Consistent behavior across providers
- Easy debugging and testing

Future enhancements (OpenAI support, fallback chains, monitoring) build naturally on this foundation. The abstract base class approach provides the right balance of structure and flexibility for supporting multiple LLM backends.

**Status**: Production-ready, tested with multiple providers, recommended pattern for LLM integration.

---

**Document Version**: 1.0  
**Last Updated**: January 4, 2026  
**Author**: System Design Documentation  
**Phase**: Mid-Development Refactor (January 2, 2026)
