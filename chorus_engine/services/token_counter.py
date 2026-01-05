"""
Token Counter Service

Accurate token counting using the model's actual tokenizer.

This is essential for:
- Memory budget management
- Context window management
- Prompt assembly
- API usage tracking

Uses the transformers library to get exact token counts for the
specific model being used (e.g., Qwen2.5).
"""

import logging
from typing import List, Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Counts tokens accurately using the model's tokenizer.
    
    Provides both exact counting (using model tokenizer) and fast estimation
    (character-based) for performance-critical paths.
    """
    
    # Class-level tokenizer cache (shared across instances)
    _tokenizer_cache: Dict[str, any] = {}
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        """
        Initialize token counter.
        
        Args:
            model_name: Hugging Face model name for tokenizer
                       (defaults to Qwen2.5-14B-Instruct)
        """
        self.model_name = model_name
        self._tokenizer = None
        self._tokenizer_available = False
        
        # Try to load tokenizer (lazy loading)
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer from transformers library."""
        # Check class cache first
        if self.model_name in TokenCounter._tokenizer_cache:
            self._tokenizer = TokenCounter._tokenizer_cache[self.model_name]
            self._tokenizer_available = True
            logger.debug(f"Using cached tokenizer for {self.model_name}")
            return
        
        # Skip tokenizer loading for Ollama model names (e.g., "qwen2.5:14b-instruct")
        # These are local model names, not HuggingFace model IDs
        if ':' in self.model_name or '/' not in self.model_name:
            logger.debug(f"Skipping tokenizer load for local model '{self.model_name}', using estimation")
            self._tokenizer_available = False
            return
        
        try:
            from transformers import AutoTokenizer
            
            logger.info(f"Loading tokenizer for {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True  # Required for some models
            )
            
            # Cache at class level (shared)
            TokenCounter._tokenizer_cache[self.model_name] = tokenizer
            self._tokenizer = tokenizer
            self._tokenizer_available = True
            
            logger.info(f"✓ Tokenizer loaded: {self.model_name}")
            
        except ImportError:
            logger.warning(
                "transformers library not available. "
                "Token counting will use estimation (4 chars/token). "
                "Install with: pip install transformers"
            )
            self._tokenizer_available = False
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer for {self.model_name}: {e}. "
                "Falling back to estimation."
            )
            self._tokenizer_available = False
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using model's tokenizer.
        
        Falls back to estimation if tokenizer unavailable.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self._tokenizer_available and self._tokenizer:
            try:
                # Use actual tokenizer
                tokens = self._tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer error, falling back to estimation: {e}")
                return self._estimate_tokens(text)
        else:
            # Fall back to estimation
            return self._estimate_tokens(text)
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token counts
        """
        if not texts:
            return []
        
        if self._tokenizer_available and self._tokenizer:
            try:
                # Batch encoding is more efficient
                encodings = self._tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_length=True
                )
                return encodings['length']
            except Exception as e:
                logger.warning(f"Batch tokenizer error, falling back to estimation: {e}")
                return [self._estimate_tokens(t) for t in texts]
        else:
            return [self._estimate_tokens(t) for t in texts]
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.
        
        Accounts for message formatting overhead (role labels, separators, etc.).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Total token count including formatting
        """
        if not messages:
            return 0
        
        # Format messages as they would appear in prompt
        formatted_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Typical format: "<|role|>\ncontent<|end|>"
            # This varies by model, but we'll use a reasonable approximation
            formatted = f"<|{role}|>\n{content}<|end|>\n"
            formatted_parts.append(formatted)
        
        full_text = "".join(formatted_parts)
        return self.count_tokens(full_text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens using character count.
        
        Uses rough heuristic: ~4 characters per token (English average).
        This is less accurate but faster and doesn't require tokenizer.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Common heuristic: 1 token ≈ 4 characters for English
        # This varies by language and model, but it's a reasonable baseline
        return max(1, len(text) // 4)
    
    def fits_in_context(
        self,
        texts: List[str],
        context_window: int,
        reserve_tokens: int = 0
    ) -> tuple[bool, int]:
        """
        Check if texts fit within context window.
        
        Args:
            texts: List of text strings to check
            context_window: Maximum context size
            reserve_tokens: Tokens to reserve (e.g., for response)
            
        Returns:
            (fits: bool, total_tokens: int)
        """
        total = sum(self.count_tokens(t) for t in texts)
        available = context_window - reserve_tokens
        
        return (total <= available, total)
    
    def truncate_to_budget(
        self,
        texts: List[str],
        token_budget: int,
        keep_first: bool = True
    ) -> List[str]:
        """
        Truncate list of texts to fit within token budget.
        
        Args:
            texts: List of texts
            token_budget: Maximum tokens
            keep_first: If True, keeps first items; if False, keeps last items
            
        Returns:
            Truncated list that fits budget
        """
        if not texts:
            return []
        
        result = []
        total_tokens = 0
        
        # Process in order (first to last or last to first)
        items = texts if keep_first else reversed(texts)
        
        for text in items:
            text_tokens = self.count_tokens(text)
            
            if total_tokens + text_tokens <= token_budget:
                result.append(text)
                total_tokens += text_tokens
            else:
                # Budget exhausted
                break
        
        # Restore original order if we were processing in reverse
        if not keep_first:
            result.reverse()
        
        return result
    
    @property
    def is_exact(self) -> bool:
        """Check if this counter uses exact tokenization."""
        return self._tokenizer_available
    
    @property
    def method(self) -> str:
        """Get counting method being used."""
        return "exact" if self._tokenizer_available else "estimation"


# Global singleton for common model
_default_counter: Optional[TokenCounter] = None


def get_token_counter(model_name: Optional[str] = None) -> TokenCounter:
    """
    Get or create a token counter.
    
    Uses singleton pattern for default model to avoid multiple tokenizer loads.
    
    Args:
        model_name: Optional specific model name
        
    Returns:
        TokenCounter instance
    """
    global _default_counter
    
    if model_name is None:
        # Use default singleton
        if _default_counter is None:
            _default_counter = TokenCounter()
        return _default_counter
    else:
        # Create new instance for specific model
        return TokenCounter(model_name)
