"""Embedding service for converting text to vectors."""

from typing import List, Optional
import logging
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""
    
    # Class-level cache for the model (shared across instances)
    _model_cache = {}
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Default: all-MiniLM-L6-v2 (384 dims, fast, good quality)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self._content_cache = {}  # Instance-level content cache
        
        # Load or get cached model
        if model_name not in self._model_cache:
            logger.info(f"Loading embedding model: {model_name}")
            # Force CPU for embedding model to leave GPU VRAM for TTS/LLM
            self._model_cache[model_name] = SentenceTransformer(model_name, device='cpu')
            logger.info(f"Model loaded: {model_name} (device: CPU)")
        
        self.model = self._model_cache[model_name]
        
        # Get embedding dimensions
        self.dimensions = self.model.get_sentence_embedding_dimension()
        logger.debug(f"Embedding dimensions: {self.dimensions}")
    
    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings for this text
            
        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        if use_cache and text in self._content_cache:
            logger.debug(f"Using cached embedding for text (len={len(text)})")
            return self._content_cache[text]
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()
            
            # Cache if enabled
            if use_cache:
                self._content_cache[text] = embedding
            
            logger.debug(f"Generated embedding for text (len={len(text)}, dims={len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_batch(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to check/update cache
            batch_size: Batch size for encoding (larger = faster but more memory)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Separate cached and uncached texts
        embeddings = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        if use_cache:
            for i, text in enumerate(texts):
                if text in self._content_cache:
                    embeddings[i] = self._content_cache[text]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                logger.debug(f"Generating {len(uncached_texts)} embeddings in batch")
                
                uncached_embeddings = self.model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=len(uncached_texts) > 10,
                    batch_size=batch_size
                ).tolist()
                
                # Place embeddings and update cache
                for idx, embedding in zip(uncached_indices, uncached_embeddings):
                    embeddings[idx] = embedding
                    if use_cache:
                        self._content_cache[texts[idx]] = embedding
                
                logger.debug(f"Batch embedding complete")
                
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise
        
        return embeddings
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings from this model.
        
        Returns:
            Number of dimensions in embedding vectors
        """
        return self.dimensions
    
    def clear_cache(self) -> None:
        """Clear the instance-level content cache."""
        cache_size = len(self._content_cache)
        self._content_cache.clear()
        logger.debug(f"Cleared {cache_size} cached embeddings")
    
    def cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._content_cache)
    
    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear the class-level model cache. Use when switching models."""
        cls._model_cache.clear()
        logger.info("Model cache cleared")
