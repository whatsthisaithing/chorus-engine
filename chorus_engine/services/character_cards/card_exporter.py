"""
Character Card Exporter
======================

Export Chorus Engine characters as PNG character cards.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .metadata_handler import PNGMetadataHandler
from .models import ChorusCharacterCard, CharacterCardData, VoiceConfig, WorkflowPreferences

logger = logging.getLogger(__name__)


class CharacterCardExporter:
    """Export Chorus Engine characters to PNG character cards."""
    
    def __init__(self, characters_dir: str, images_dir: str, default_avatar_path: str):
        """
        Initialize exporter.
        
        Args:
            characters_dir: Path to characters directory
            images_dir: Path to character images directory
            default_avatar_path: Path to default avatar image
        """
        self.characters_dir = Path(characters_dir)
        self.images_dir = Path(images_dir)
        self.default_avatar_path = Path(default_avatar_path)
    
    def export(
        self,
        character_name: str,
        include_voice: bool = True,
        include_workflows: bool = True,
        voice_sample_url: Optional[str] = None
    ) -> bytes:
        """
        Export character as PNG card.
        
        Args:
            character_name: Name of character to export
            include_voice: Include voice configuration
            include_workflows: Include workflow preferences
            voice_sample_url: Optional URL for voice sample download
            
        Returns:
            PNG file data with embedded metadata
            
        Raises:
            FileNotFoundError: If character file not found
            ValueError: If character data invalid
        """
        logger.info(f"Exporting character card for '{character_name}'")
        
        # Load character YAML
        char_path = self.characters_dir / f"{character_name}.yaml"
        if not char_path.exists():
            raise FileNotFoundError(f"Character file not found: {char_path}")
        
        with open(char_path, 'r', encoding='utf-8') as f:
            character_data = yaml.safe_load(f)
        
        # Build card metadata
        card_data = self._build_card_data(
            character_data,
            include_voice,
            include_workflows,
            voice_sample_url
        )
        
        # Create card structure
        card = ChorusCharacterCard(
            spec="chorus_card_v1",
            spec_version="1.0",
            data=card_data
        )
        
        # Serialize to YAML (use mode='json' to get plain values, not Python objects)
        card_dict = card.model_dump(mode='json')
        card_yaml = yaml.dump(card_dict, default_flow_style=False, allow_unicode=True)
        
        # Load profile image
        profile_image = self._get_profile_image(character_name, character_data)
        
        # Embed metadata in PNG
        card_png = PNGMetadataHandler.write_text_chunk(
            profile_image,
            "chorus_card",
            card_yaml
        )
        
        logger.info(f"Successfully exported character card for '{character_name}'")
        return card_png
    
    def _build_card_data(
        self,
        character_data: Dict[str, Any],
        include_voice: bool,
        include_workflows: bool,
        voice_sample_url: Optional[str]
    ) -> CharacterCardData:
        """Build card data from character configuration."""
        
        # Core character data
        card_data = CharacterCardData(
            name=character_data.get("name", "Unnamed Character"),
            role=character_data.get("role", ""),
            role_type=character_data.get("role_type"),
            description=character_data.get("description", ""),
            personality=character_data.get("personality", ""),
            scenario=character_data.get("scenario", ""),
            greeting=character_data.get("greeting", ""),
            example_messages=character_data.get("example_messages", ""),
            creator=character_data.get("creator", ""),
            tags=character_data.get("tags", []),
            created_date=datetime.now().isoformat()
        )
        
        # Note: 'id' is intentionally not exported (it's filename-based and system-specific)
        # On import, the user can choose a new name/id
        
        # System prompt configuration
        card_data.custom_system_prompt = character_data.get("custom_system_prompt", False)
        if character_data.get("system_prompt"):
            card_data.system_prompt = character_data["system_prompt"]
        
        # Immersion settings
        if "immersion_level" in character_data:
            card_data.immersion_level = character_data["immersion_level"]
        if "immersion_settings" in character_data:
            card_data.immersion_settings = character_data["immersion_settings"]
        
        # Core memories (full capture)
        if "core_memories" in character_data:
            card_data.core_memories = character_data["core_memories"]
        
        # Personality traits (simple list)
        if "personality_traits" in character_data:
            card_data.personality_traits = character_data["personality_traits"]
        
        # Emotional range
        if "emotional_range" in character_data:
            card_data.emotional_range = character_data["emotional_range"]
        
        # Memory configuration
        if "memory" in character_data:
            card_data.memory = character_data["memory"]
        if "memory_profile" in character_data:
            card_data.memory_profile = character_data["memory_profile"]
        
        # Feature toggles
        if "image_generation" in character_data:
            card_data.image_generation = character_data["image_generation"]
        if "video_generation" in character_data:
            card_data.video_generation = character_data["video_generation"]
        if "document_analysis" in character_data:
            card_data.document_analysis = character_data["document_analysis"]
        if "code_execution" in character_data:
            card_data.code_execution = character_data["code_execution"]
        
        # Response style
        response_style = character_data.get("response_style")
        if response_style:
            from .models import ResponseStyle
            card_data.response_style = ResponseStyle(**response_style)
        
        # Traits (structured Big Five + lists)
        traits = character_data.get("traits")
        if traits:
            from .models import CharacterTraits
            card_data.traits = CharacterTraits(**traits)
        
        # Temperature (suggested, not enforced)
        if "temperature" in character_data:
            card_data.temperature = character_data["temperature"]
        
        # Voice configuration (comprehensive)
        if include_voice and "voice" in character_data:
            voice_data = character_data["voice"].copy()
            
            # Remove system-specific paths but keep configuration
            if "voice_sample_path" in voice_data:
                del voice_data["voice_sample_path"]
            
            # Add voice sample URL if provided
            if voice_sample_url:
                voice_data["voice_sample_url"] = voice_sample_url
            
            card_data.voice = VoiceConfig(**voice_data)
        
        # Workflow preferences
        if include_workflows:
            workflows = {}
            
            # Only include workflow names, not full paths
            if "default_image_workflow" in character_data:
                workflow_path = character_data["default_image_workflow"]
                workflows["default_image_workflow"] = Path(workflow_path).stem if workflow_path else None
            
            if "default_video_workflow" in character_data:
                workflow_path = character_data["default_video_workflow"]
                workflows["default_video_workflow"] = Path(workflow_path).stem if workflow_path else None
            
            if "auto_generate_images" in character_data:
                workflows["auto_generate_images"] = character_data["auto_generate_images"]
            
            if "auto_generate_videos" in character_data:
                workflows["auto_generate_videos"] = character_data["auto_generate_videos"]
            
            if workflows:
                card_data.workflows = WorkflowPreferences(**workflows)
        
        # Preferred LLM settings - store in extensions as a note
        if "preferred_llm" in character_data:
            if not card_data.extensions:
                card_data.extensions = {}
            card_data.extensions["preferred_llm_note"] = character_data["preferred_llm"]
        
        # Store any other extensions
        if "extensions" in character_data:
            if not card_data.extensions:
                card_data.extensions = {}
            card_data.extensions.update(character_data["extensions"])
        
        return card_data
    
    def _get_profile_image(self, character_name: str, character_data: Dict[str, Any]) -> bytes:
        """Get character profile image or default avatar."""
        
        # Try to find character image
        # Check for various image formats
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            image_path = self.images_dir / f"{character_name}{ext}"
            if image_path.exists():
                logger.debug(f"Using profile image: {image_path}")
                return PNGMetadataHandler.extract_image(str(image_path))
        
        # Check if image path specified in character data
        if "profile_image" in character_data:
            profile_image = character_data["profile_image"]
            # Handle relative paths
            if not Path(profile_image).is_absolute():
                image_path = self.images_dir / profile_image
            else:
                image_path = Path(profile_image)
                
            if image_path.exists() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                logger.debug(f"Using profile image from config: {image_path}")
                return PNGMetadataHandler.extract_image(str(image_path))
        
        # Fall back to default avatar (if it's a PNG)
        logger.debug(f"No profile image found for '{character_name}', using default avatar")
        if self.default_avatar_path.exists():
            # Check if default avatar is a supported format
            if self.default_avatar_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                return PNGMetadataHandler.extract_image(str(self.default_avatar_path))
            else:
                # Default avatar is SVG or other unsupported format - create a blank PNG
                logger.warning(f"Default avatar is {self.default_avatar_path.suffix}, creating blank PNG")
                return self._create_blank_png()
        else:
            # No default avatar at all - create a blank PNG
            logger.warning(f"Default avatar not found: {self.default_avatar_path}, creating blank PNG")
            return self._create_blank_png()
    
    def _create_blank_png(self) -> bytes:
        """Create a simple blank PNG as fallback."""
        from PIL import Image
        from io import BytesIO
        
        # Create a 512x512 gray image
        img = Image.new('RGB', (512, 512), color=(128, 128, 128))
        output = BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
