"""
Character Card System
====================

Portable character configurations embedded in PNG images with base64-encoded metadata.

Supports:
- Chorus Engine native format (YAML-based)
- SillyTavern V2 format import (JSON-based)
- Generic profile image upload
"""

from .card_exporter import CharacterCardExporter
from .card_importer import CharacterCardImporter
from .format_detector import CardFormat, FormatDetector
from .metadata_handler import PNGMetadataHandler
from .macro_processor import MacroProcessor, process_character_card_macros

__all__ = [
    'CharacterCardExporter',
    'CharacterCardImporter',
    'CardFormat',
    'FormatDetector',
    'PNGMetadataHandler',
    'MacroProcessor',
    'process_character_card_macros',
]
