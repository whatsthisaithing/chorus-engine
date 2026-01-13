"""
Character Card Importer
======================

Import character cards from PNG files (Chorus Engine and SillyTavern formats).
"""

import yaml
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from .metadata_handler import PNGMetadataHandler
from .format_detector import CardFormat, FormatDetector
from .sillytavern_adapter import SillyTavernAdapter
from .models import CardImportResult

logger = logging.getLogger(__name__)


class CharacterCardImporter:
    """Import character cards from PNG files."""
    
    def __init__(self, characters_dir: str, images_dir: str):
        """
        Initialize importer.
        
        Args:
            characters_dir: Path to characters directory
            images_dir: Path to character images directory
        """
        self.characters_dir = Path(characters_dir)
        self.images_dir = Path(images_dir)
        
        # Ensure directories exist
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def import_card(self, png_data: bytes) -> CardImportResult:
        """
        Import character from PNG card.
        
        Args:
            png_data: PNG file data as bytes
            
        Returns:
            CardImportResult with character data, profile image, format, and warnings
            
        Raises:
            ValueError: If PNG contains no valid card data
        """
        logger.info("Importing character card from PNG")
        
        # Detect format
        card_format, card_data = FormatDetector.detect(png_data)
        
        if card_format == CardFormat.UNKNOWN or not card_data:
            raise ValueError("No valid character card data found in PNG")
        
        format_name = FormatDetector.get_format_name(card_format)
        logger.info(f"Detected format: {format_name}")
        
        # Convert to Chorus format
        character_data, warnings = self._convert_to_chorus_format(card_format, card_data)
        
        # Validate required fields
        if "name" not in character_data or not character_data["name"]:
            raise ValueError("Character card missing required field: name")
        
        # Return result (don't save yet - preview first)
        result = CardImportResult(
            character_data=character_data,
            profile_image=png_data,
            format=format_name,
            warnings=warnings
        )
        
        logger.info(f"Successfully imported character card: {character_data['name']}")
        return result
    
    def save_character(
        self,
        character_data: Dict[str, Any],
        profile_image: bytes,
        custom_name: str = None
    ) -> str:
        """
        Save imported character to files.
        
        Args:
            character_data: Character configuration dict
            profile_image: Profile image PNG data
            custom_name: Optional custom character name (overrides card name)
            
        Returns:
            Character filename (without extension)
            
        Raises:
            ValueError: If character name invalid
        """
        # Use custom name if provided, otherwise use card name
        character_name = custom_name or character_data.get("name", "Unnamed")
        
        # Note: Macros are NOT processed here - they're stored in YAML as-is
        # and processed at character load time. This allows:
        # - Name changes to automatically update {{char}} references
        # - Users to edit/add macros in YAML files
        # - Flexibility in macro handling as system evolves
        
        # Sanitize filename
        safe_name = self._sanitize_filename(character_name)
        
        # Handle name collisions
        final_name = self._resolve_name_collision(safe_name)
        
        # Update character name to match filename
        character_data["name"] = character_name
        
        # Set profile image filename
        character_data["profile_image"] = f"{final_name}.png"
        
        # Save character YAML
        char_path = self.characters_dir / f"{final_name}.yaml"
        with open(char_path, 'w', encoding='utf-8') as f:
            yaml.dump(character_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved character YAML: {char_path}")
        
        # Save profile image
        image_path = self.images_dir / f"{final_name}.png"
        PNGMetadataHandler.save_image(profile_image, str(image_path))
        
        logger.info(f"Saved profile image: {image_path}")
        
        return final_name
    
    def _convert_to_chorus_format(
        self,
        card_format: CardFormat,
        card_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Convert card data to Chorus Engine format.
        
        Returns:
            Tuple of (character_data_dict, warnings_list)
        """
        warnings = []
        
        if card_format == CardFormat.CHORUS_V1:
            # Extract data section
            character_data = card_data.get("data", {})
            
            # Check for future version
            spec_version = card_data.get("spec_version", "1.0")
            if spec_version != "1.0":
                warnings.append(f"Card is from future version {spec_version}, some features may not be supported")
            
            return character_data, warnings
        
        elif card_format in (CardFormat.SILLYTAVERN_V2, CardFormat.SILLYTAVERN_V3):
            # Convert using adapter
            character_data = SillyTavernAdapter.from_sillytavern_v2(card_data)
            
            if card_format == CardFormat.SILLYTAVERN_V3:
                warnings.append("SillyTavern V3 format detected - some advanced features may not be fully supported")
            
            # Check for character book (lorebook) - not supported yet
            if card_data.get("data", {}).get("character_book"):
                warnings.append("Character has a lorebook/character book - this feature is not yet supported in Chorus Engine")
            
            return character_data, warnings
        
        else:
            raise ValueError(f"Unsupported card format: {card_format}")
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize string for use as filename."""
        # Remove or replace invalid characters
        safe = re.sub(r'[<>:"/\\|?*]', '', name)
        # Replace spaces and dots with underscores
        safe = safe.replace(' ', '_').replace('.', '_')
        # Remove leading/trailing underscores and spaces
        safe = safe.strip('_ ')
        # Limit length
        if len(safe) > 50:
            safe = safe[:50]
        # Ensure not empty
        if not safe:
            safe = "character"
        return safe.lower()
    
    def _resolve_name_collision(self, base_name: str) -> str:
        """
        Resolve filename collisions by appending suffix.
        
        Args:
            base_name: Base filename (without extension)
            
        Returns:
            Unique filename that doesn't exist
        """
        # Check if base name is available
        if not (self.characters_dir / f"{base_name}.yaml").exists():
            return base_name
        
        # Try appending numbers
        counter = 1
        while True:
            candidate = f"{base_name}_{counter}"
            if not (self.characters_dir / f"{candidate}.yaml").exists():
                logger.info(f"Resolved name collision: {base_name} -> {candidate}")
                return candidate
            counter += 1
            
            # Safety limit
            if counter > 1000:
                raise ValueError(f"Unable to resolve name collision for '{base_name}' after 1000 attempts")
