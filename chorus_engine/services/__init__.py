"""Services package."""

from .embedding_service import EmbeddingService

# Phase 10: Integrated LLM provider services
from .vram_estimator import VRAMEstimator
from .model_manager import ModelManager
from .model_library import ModelLibrary

__all__ = [
    'EmbeddingService',
    # Phase 10
    'VRAMEstimator',
    'ModelManager',
    'ModelLibrary',
]
