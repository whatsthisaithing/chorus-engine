"""
TTS Provider System

Provider-agnostic TTS architecture supporting multiple backends.
"""

from .base_provider import BaseTTSProvider, TTSRequest, TTSResult
from .provider_factory import TTSProviderFactory

__all__ = [
    'BaseTTSProvider',
    'TTSRequest',
    'TTSResult',
    'TTSProviderFactory',
]
