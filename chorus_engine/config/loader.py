"""Configuration loader with validation and error handling."""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import ValidationError

from .models import SystemConfig, CharacterConfig

logger = logging.getLogger(__name__)

# Immutable default characters (cannot be modified or deleted via API)
IMMUTABLE_CHARACTERS = {"nova", "alex"}


class ConfigLoadError(Exception):
    """Base exception for configuration loading errors."""
    pass


class ConfigValidationError(ConfigLoadError):
    """Configuration validation failed."""
    
    def __init__(self, errors: list[dict], file_path: Path):
        self.errors = errors
        self.file_path = file_path
        super().__init__(self._format_errors())
    
    def _format_errors(self) -> str:
        """Format validation errors for user display."""
        lines = [f"Configuration validation failed for {self.file_path}:\n"]
        for error in self.errors:
            loc = " → ".join(str(l) for l in error['loc'])
            msg = error['msg']
            lines.append(f"  • {loc}: {msg}")
        return "\n".join(lines)


class ConfigLoader:
    """Loads and validates configuration files."""
    
    def __init__(self, config_dir: Path = Path(".")):
        self.config_dir = Path(config_dir)
    
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data
        except FileNotFoundError:
            raise ConfigLoadError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load {file_path}: {e}")
    
    def load_system_config(self, file_path: Optional[Path] = None) -> SystemConfig:
        """
        Load system configuration.
        
        Falls back to defaults if file not found.
        """
        if file_path is None:
            file_path = self.config_dir / "config" / "system.yaml"
        
        try:
            if not file_path.exists():
                logger.info(f"System config not found at {file_path}, using defaults")
                return SystemConfig()
            
            data = self.load_yaml(file_path)
            config = SystemConfig(**data)
            logger.info(f"Loaded system config from {file_path}")
            return config
            
        except ValidationError as e:
            raise ConfigValidationError(e.errors(), file_path)
    
    def load_character(self, character_id: str) -> CharacterConfig:
        """
        Load character configuration by ID.
        
        Raises ConfigLoadError if character not found or invalid.
        """
        file_path = self.config_dir / "characters" / f"{character_id}.yaml"
        
        try:
            data = self.load_yaml(file_path)
            
            # Ensure ID matches filename
            if 'id' not in data:
                data['id'] = character_id
            elif data['id'] != character_id:
                raise ConfigLoadError(
                    f"Character ID mismatch: filename is '{character_id}' but "
                    f"config has id '{data['id']}'"
                )
            
            config = CharacterConfig(**data)
            logger.info(f"Loaded character '{config.name}' from {file_path}")
            return config
            
        except ValidationError as e:
            raise ConfigValidationError(e.errors(), file_path)
    
    def load_all_characters(self) -> Dict[str, CharacterConfig]:
        """
        Load all character configurations.
        
        Returns dict of {character_id: config}.
        Logs warnings for invalid characters but continues loading others.
        """
        characters = {}
        characters_dir = self.config_dir / "characters"
        
        if not characters_dir.exists():
            logger.warning(f"Characters directory not found: {characters_dir}")
            return characters
        
        for file_path in characters_dir.glob("*.yaml"):
            character_id = file_path.stem
            
            # Skip template file
            if character_id == "template":
                logger.debug("Skipping template.yaml (not a real character)")
                continue
            
            try:
                config = self.load_character(character_id)
                characters[character_id] = config
            except ConfigLoadError as e:
                logger.error(f"Failed to load character '{character_id}': {e}")
                # Continue loading other characters
        
        logger.info(f"Loaded {len(characters)} character(s)")
        return characters
    
    def validate_character(self, data: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate character data without loading.
        
        Returns (is_valid, errors).
        Useful for API endpoints that create/update characters.
        """
        try:
            CharacterConfig(**data)
            return True, []
        except ValidationError as e:
            errors = [f"{' → '.join(str(l) for l in err['loc'])}: {err['msg']}" 
                     for err in e.errors()]
            return False, errors
    
    def save_character(self, character: CharacterConfig) -> Path:
        """
        Save character configuration to YAML file.
        
        Args:
            character: Character configuration to save
            
        Returns:
            Path to saved file
            
        Raises:
            ConfigLoadError: If save fails
        """
        characters_dir = self.config_dir / "characters"
        characters_dir.mkdir(exist_ok=True)
        
        file_path = characters_dir / f"{character.id}.yaml"
        
        try:
            # Convert to dict, excluding computed fields
            data = character.model_dump(
                exclude_none=True,
                exclude={"created_at", "updated_at"},
                mode='json'
            )
            
            # Write YAML with clean formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2
                )
            
            logger.info(f"Saved character '{character.name}' to {file_path}")
            return file_path
            
        except Exception as e:
            raise ConfigLoadError(f"Failed to save character to {file_path}: {e}")
    
    def delete_character(self, character_id: str) -> None:
        """
        Delete character configuration file.
        
        Args:
            character_id: ID of character to delete
            
        Raises:
            ConfigLoadError: If character is immutable or deletion fails
        """
        if character_id in IMMUTABLE_CHARACTERS:
            raise ConfigLoadError(
                f"Cannot delete immutable character '{character_id}'"
            )
        
        file_path = self.config_dir / "characters" / f"{character_id}.yaml"
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted character '{character_id}'")
            else:
                raise ConfigLoadError(f"Character file not found: {file_path}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to delete character '{character_id}': {e}")
    
    def is_immutable(self, character_id: str) -> bool:
        """Check if character is an immutable default."""
        return character_id in IMMUTABLE_CHARACTERS
