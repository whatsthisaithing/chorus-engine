"""
Card Format Detector
===================

Detects character card format from PNG metadata.
"""

import json
import yaml
import logging
from enum import Enum
from typing import Tuple, Optional, Dict, Any

from .metadata_handler import PNGMetadataHandler

logger = logging.getLogger(__name__)


class CardFormat(Enum):
    """Supported character card formats."""
    CHORUS_V1 = "chorus_card_v1"
    SILLYTAVERN_V2 = "chara_card_v2"
    SILLYTAVERN_V3 = "chara_card_v3"
    UNKNOWN = "unknown"


class FormatDetector:
    """Detect character card format from PNG metadata."""
    
    # Metadata keywords to check
    CHORUS_KEYWORD = "chorus_card"
    SILLYTAVERN_KEYWORD = "chara"
    SILLYTAVERN_V3_KEYWORD = "ccv3"
    
    @classmethod
    def detect(cls, png_data: bytes) -> Tuple[CardFormat, Optional[Dict[str, Any]]]:
        """
        Detect character card format and parse metadata.
        
        Args:
            png_data: PNG file data as bytes
            
        Returns:
            Tuple of (CardFormat, parsed_data_dict)
            parsed_data_dict is None if no valid card data found
        """
        # Check for Chorus Engine format first
        chorus_data = PNGMetadataHandler.read_text_chunk(png_data, cls.CHORUS_KEYWORD)
        if chorus_data:
            parsed = cls._parse_chorus(chorus_data)
            if parsed:
                return (CardFormat.CHORUS_V1, parsed)
        
        # Check for SillyTavern V3 format
        st_v3_data = PNGMetadataHandler.read_text_chunk(png_data, cls.SILLYTAVERN_V3_KEYWORD)
        if st_v3_data:
            parsed = cls._parse_sillytavern(st_v3_data)
            if parsed and parsed.get("spec") == "chara_card_v3":
                return (CardFormat.SILLYTAVERN_V3, parsed)
        
        # Check for SillyTavern V2 format
        st_v2_data = PNGMetadataHandler.read_text_chunk(png_data, cls.SILLYTAVERN_KEYWORD)
        if st_v2_data:
            parsed = cls._parse_sillytavern(st_v2_data)
            if parsed:
                # Could be V2 or V3 (V3 also uses 'chara' keyword)
                spec = parsed.get("spec", "")
                if spec == "chara_card_v3":
                    return (CardFormat.SILLYTAVERN_V3, parsed)
                elif spec == "chara_card_v2" or "data" in parsed:
                    return (CardFormat.SILLYTAVERN_V2, parsed)
        
        logger.warning("No valid character card metadata found in PNG")
        return (CardFormat.UNKNOWN, None)
    
    @staticmethod
    def _parse_chorus(data: str) -> Optional[Dict[str, Any]]:
        """
        Parse Chorus Engine card data (YAML format).
        
        Args:
            data: Decoded card data string
            
        Returns:
            Parsed dict or None if invalid
        """
        try:
            parsed = yaml.safe_load(data)
            
            # Validate basic structure
            if not isinstance(parsed, dict):
                logger.warning("Chorus card data is not a dictionary")
                return None
            
            if "spec" not in parsed or "data" not in parsed:
                logger.warning("Chorus card missing required fields (spec, data)")
                return None
            
            if not parsed["spec"].startswith("chorus_card_"):
                logger.warning(f"Unrecognized Chorus card spec: {parsed['spec']}")
                return None
            
            logger.info(f"Successfully parsed Chorus card: {parsed.get('spec')} v{parsed.get('spec_version')}")
            return parsed
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse Chorus card YAML: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing Chorus card: {e}")
            return None
    
    @staticmethod
    def _parse_sillytavern(data: str) -> Optional[Dict[str, Any]]:
        """
        Parse SillyTavern card data (JSON format).
        
        Args:
            data: Decoded card data string
            
        Returns:
            Parsed dict or None if invalid
        """
        try:
            parsed = json.loads(data)
            
            # Validate basic structure
            if not isinstance(parsed, dict):
                logger.warning("SillyTavern card data is not a dictionary")
                return None
            
            # V2 format should have 'spec' and 'data' fields
            # V1 format (old) might not have 'spec' but should have 'name'
            if "spec" in parsed:
                if not parsed["spec"].startswith("chara_card_"):
                    logger.warning(f"Unrecognized SillyTavern card spec: {parsed['spec']}")
                    return None
                logger.info(f"Successfully parsed SillyTavern card: {parsed.get('spec')} v{parsed.get('spec_version')}")
            elif "name" in parsed or "data" in parsed:
                # Likely V1 or malformed V2, try to work with it
                logger.info("Parsed SillyTavern card (possibly V1 format)")
            else:
                logger.warning("SillyTavern card missing required fields")
                return None
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SillyTavern card JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing SillyTavern card: {e}")
            return None
    
    @classmethod
    def get_format_name(cls, format: CardFormat) -> str:
        """Get human-readable format name."""
        names = {
            CardFormat.CHORUS_V1: "Chorus Engine V1",
            CardFormat.SILLYTAVERN_V2: "SillyTavern V2",
            CardFormat.SILLYTAVERN_V3: "SillyTavern V3",
            CardFormat.UNKNOWN: "Unknown Format"
        }
        return names.get(format, "Unknown")
