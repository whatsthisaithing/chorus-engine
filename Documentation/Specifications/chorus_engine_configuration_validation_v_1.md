# Chorus Engine – Configuration Validation & Loading (v1)

This document defines how Chorus Engine validates and loads configuration files using Pydantic models.

Goals:
- Type-safe configuration with runtime validation
- Clear, actionable error messages
- Graceful fallback for missing optional fields
- Cross-platform file handling
- Optional hot-reload support
- Forward-compatible (ignore unknown fields)

---

## Architecture Overview

```
Config Files (YAML/JSON)
    ↓
Load & Parse (PyYAML)
    ↓
Pydantic Validation
    ↓
Type-safe Config Objects
    ↓
Application Usage
```

---

## Pydantic Models

### System Configuration

**File**: `config/system.yaml`

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Literal
from pathlib import Path

class LLMConfig(BaseModel):
    """LLM backend configuration."""
    provider: Literal["ollama", "llamacpp", "openai-compatible"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b-instruct"
    context_window: int = Field(default=8192, gt=0, le=128000)
    max_response_tokens: int = Field(default=2048, gt=0, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=120, gt=0)
    
    @field_validator('base_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')

class MemoryConfig(BaseModel):
    """Memory system configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store: Literal["chroma"] = "chroma"
    implicit_enabled: bool = False
    ephemeral_ttl_hours: int = Field(default=24, gt=0)
    similarity_thresholds: dict = Field(default_factory=lambda: {
        'explicit_minimum': 0.70,
        'implicit_minimum': 0.75,
        'search_api_minimum': 0.65
    })
    default_budget_tokens: int = Field(default=1000, gt=0, le=4000)

class ComfyUIConfig(BaseModel):
    """ComfyUI integration configuration."""
    enabled: bool = True
    url: str = "http://localhost:8188"
    timeout_seconds: int = Field(default=300, gt=0)
    polling_interval_seconds: float = Field(default=2.0, gt=0)
    max_concurrent_jobs: int = Field(default=2, gt=0, le=10)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('url must start with http:// or https://')
        return v.rstrip('/')

class PathsConfig(BaseModel):
    """File path configuration."""
    characters: Path = Path("characters")
    workflows: Path = Path("workflows")
    data: Path = Path("data")
    
    @field_validator('characters', 'workflows', 'data')
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Ensure paths are relative or absolute, cross-platform."""
        # Convert to Path if string, normalize separators
        return Path(v)

class SystemConfig(BaseModel):
    """Top-level system configuration."""
    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    debug: bool = False
    api_host: str = "localhost"
    api_port: int = Field(default=8080, gt=0, le=65535)
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Ensure directories exist
        for path_name in ['characters', 'workflows', 'data']:
            path = getattr(self.paths, path_name)
            path.mkdir(parents=True, exist_ok=True)
```

---

### Character Configuration

**File**: `characters/{character_id}.yaml`

```python
from datetime import datetime
from typing import Optional, Literal

class VoiceConfig(BaseModel):
    """Voice configuration for TTS."""
    tts_engine: str = "xtts_v2"
    voice_sample: Optional[Path] = None
    speaking_style: dict = Field(default_factory=lambda: {
        'pace': 'medium',
        'tone': 'neutral',
        'expressiveness': 'medium'
    })

class EmotionalRange(BaseModel):
    """Emotional constraints for character."""
    baseline: str = "neutral"
    allowed: list[str] = Field(default_factory=lambda: ["neutral", "positive"])

class MemoryPreferences(BaseModel):
    """Character-specific memory settings."""
    scope: Literal["global", "character", "thread"] = "character"
    vector_store: str = "chroma_default"

class AmbientActivityConfig(BaseModel):
    """Ambient activity settings."""
    enabled: bool = False
    default_behavior: Literal["propose_if_idle", "never", "always"] = "propose_if_idle"
    style_guidelines: Optional[str] = None

class VisualIdentityConfig(BaseModel):
    """Visual generation settings."""
    default_workflow: Optional[str] = None
    prompt_context: Optional[str] = None
    
    @field_validator('default_workflow')
    @classmethod
    def validate_workflow_exists(cls, v: Optional[str], info) -> Optional[str]:
        """Check workflow file exists (warning only)."""
        if v is None:
            return v
        
        # Get workflows path from context if available
        workflow_path = Path("workflows") / f"{v}.json"
        if not workflow_path.exists():
            # Log warning but don't fail - allow user to add workflow later
            import logging
            logging.warning(f"Workflow file not found: {workflow_path}")
        
        return v

class PreferredLLMConfig(BaseModel):
    """Character's preferred LLM settings."""
    provider: Optional[str] = None
    model: Optional[str] = None

class CharacterConfig(BaseModel):
    """Complete character configuration."""
    model_config = ConfigDict(extra='ignore')
    
    # Required fields
    id: str = Field(min_length=1, max_length=50, pattern=r'^[a-z0-9_]+$')
    name: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)
    system_prompt: str = Field(min_length=10)
    
    # Optional fields with defaults
    personality_traits: list[str] = Field(default_factory=list)
    preferred_llm: PreferredLLMConfig = Field(default_factory=PreferredLLMConfig)
    voice: Optional[VoiceConfig] = None
    emotional_range: EmotionalRange = Field(default_factory=EmotionalRange)
    memory: MemoryPreferences = Field(default_factory=MemoryPreferences)
    ambient_activity: AmbientActivityConfig = Field(default_factory=AmbientActivityConfig)
    visual_identity: Optional[VisualIdentityConfig] = None
    
    # Metadata (auto-populated)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID is safe for filesystem and URLs."""
        if not v.replace('_', '').isalnum():
            raise ValueError('Character ID must contain only lowercase letters, numbers, and underscores')
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        """Ensure system prompt is substantial."""
        if len(v.strip()) < 10:
            raise ValueError('system_prompt must be at least 10 characters')
        return v.strip()
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Set timestamps if not provided
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
```

---

## Configuration Loader

```python
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

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
```

---

## Error Handling & User Feedback

### Error Categories

**1. File Not Found**
```python
# System config missing → Use defaults (info log)
# Character missing → Error, cannot proceed
# Workflow missing → Warning, allow generation to fail later
```

**2. Invalid YAML Syntax**
```python
# Any config → Clear error with line number if possible
# User-friendly message pointing to YAML validator
```

**3. Validation Errors**
```python
# Show field path and constraint violation
# Provide suggestion for fix when possible
# List all errors, not just first
```

**4. Runtime Validation**
```python
# Missing workflow file → Warning on load, error on use
# Missing voice sample → Warning, fallback to default
# Invalid LLM URL → Error on startup, suggest fix
```

### Example Error Messages

```python
class ErrorFormatter:
    """Format errors for user display."""
    
    @staticmethod
    def format_validation_error(e: ConfigValidationError) -> str:
        """Format Pydantic validation error."""
        output = [
            f"❌ Configuration Error: {e.file_path.name}",
            "",
            "The following issues were found:",
            ""
        ]
        
        for error in e.errors:
            field = " → ".join(str(l) for l in error['loc'])
            msg = error['msg']
            
            # Add user-friendly explanation
            suggestion = ErrorFormatter._get_suggestion(error)
            
            output.append(f"  Field: {field}")
            output.append(f"  Issue: {msg}")
            if suggestion:
                output.append(f"  Fix: {suggestion}")
            output.append("")
        
        output.append("Please fix these issues and restart.")
        return "\n".join(output)
    
    @staticmethod
    def _get_suggestion(error: dict) -> Optional[str]:
        """Get user-friendly suggestion for common errors."""
        error_type = error.get('type', '')
        field = error['loc'][-1] if error['loc'] else ''
        
        suggestions = {
            'value_error.missing': f"Add the required '{field}' field",
            'type_error.integer': f"'{field}' must be a number",
            'value_error.url': "URL must start with http:// or https://",
            'value_error.range': "Value is outside allowed range",
        }
        
        for key, suggestion in suggestions.items():
            if error_type.startswith(key):
                return suggestion
        
        return None
```

---

## Configuration Migration

### Version Tracking

```python
class ConfigVersion(BaseModel):
    """Track config version for migrations."""
    version: str = "1.0.0"
    migrated_from: Optional[str] = None
    migrated_at: Optional[datetime] = None

class VersionedSystemConfig(SystemConfig):
    """System config with version tracking."""
    config_version: ConfigVersion = Field(default_factory=ConfigVersion)
```

### Migration Strategy (v1)

**Simple approach for v1**:
- Ignore unknown fields (forward compatible)
- Provide defaults for new fields (backward compatible)
- No automatic migration needed yet

**Future versions**:
- Detect old config version
- Apply migration transformations
- Save upgraded config
- Keep backup of original

```python
def migrate_config(old_data: dict, from_version: str, to_version: str) -> dict:
    """
    Migrate config from old version to new.
    
    v1: No migrations needed yet.
    Future: Add migration logic here.
    """
    if from_version == to_version:
        return old_data
    
    # Future: Add migration logic
    # if from_version == "1.0.0" and to_version == "2.0.0":
    #     old_data = migrate_1_0_to_2_0(old_data)
    
    return old_data
```

---

## Hot Reload (Optional)

### File Watching

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigWatcher(FileSystemEventHandler):
    """Watch for config file changes."""
    
    def __init__(self, loader: ConfigLoader, on_reload_callback):
        self.loader = loader
        self.on_reload = on_reload_callback
    
    def on_modified(self, event):
        """Handle file modification."""
        if event.src_path.endswith('.yaml'):
            file_path = Path(event.src_path)
            
            # Determine config type and reload
            if file_path.name == 'system.yaml':
                self._reload_system_config(file_path)
            elif file_path.parent.name == 'characters':
                self._reload_character(file_path)
    
    def _reload_system_config(self, file_path: Path):
        """Reload system configuration."""
        try:
            config = self.loader.load_system_config(file_path)
            self.on_reload('system', config)
            logger.info("System config reloaded")
        except ConfigLoadError as e:
            logger.error(f"Failed to reload system config: {e}")
    
    def _reload_character(self, file_path: Path):
        """Reload character configuration."""
        character_id = file_path.stem
        try:
            config = self.loader.load_character(character_id)
            self.on_reload('character', character_id, config)
            logger.info(f"Character '{character_id}' reloaded")
        except ConfigLoadError as e:
            logger.error(f"Failed to reload character '{character_id}': {e}")

def start_config_watcher(config_dir: Path, loader: ConfigLoader, callback):
    """Start watching config directory for changes."""
    event_handler = ConfigWatcher(loader, callback)
    observer = Observer()
    observer.schedule(event_handler, str(config_dir), recursive=True)
    observer.start()
    return observer
```

**Usage**:
```python
def on_config_reload(config_type: str, *args):
    """Handle config reload."""
    if config_type == 'system':
        # Update system config
        pass
    elif config_type == 'character':
        character_id, config = args
        # Update character in memory
        pass

# Optional: Enable hot reload
if system_config.debug:
    watcher = start_config_watcher(Path("."), loader, on_config_reload)
```

---

## Startup Validation Sequence

```python
async def validate_startup_config():
    """
    Comprehensive startup validation.
    
    Validates all configs and dependencies before starting server.
    """
    errors = []
    warnings = []
    
    loader = ConfigLoader()
    
    # 1. Load and validate system config
    try:
        system_config = loader.load_system_config()
        logger.info("✓ System config loaded")
    except ConfigValidationError as e:
        logger.error(ErrorFormatter.format_validation_error(e))
        raise
    
    # 2. Check LLM connectivity
    try:
        llm_connected = await check_llm_connection(system_config.llm)
        if llm_connected:
            logger.info("✓ LLM connection verified")
        else:
            warnings.append("⚠ LLM not available (will retry at runtime)")
    except Exception as e:
        warnings.append(f"⚠ Could not connect to LLM: {e}")
    
    # 3. Check ComfyUI (if enabled)
    if system_config.comfyui.enabled:
        try:
            comfy_connected = await check_comfyui_connection(system_config.comfyui)
            if comfy_connected:
                logger.info("✓ ComfyUI connection verified")
            else:
                warnings.append("⚠ ComfyUI not available (image generation disabled)")
        except Exception as e:
            warnings.append(f"⚠ Could not connect to ComfyUI: {e}")
    
    # 4. Load and validate all characters
    try:
        characters = loader.load_all_characters()
        if len(characters) == 0:
            errors.append("❌ No valid characters found")
        else:
            logger.info(f"✓ Loaded {len(characters)} character(s)")
            
            # Validate each character's workflow
            for char_id, char_config in characters.items():
                if char_config.visual_identity and char_config.visual_identity.default_workflow:
                    workflow_path = system_config.paths.workflows / f"{char_config.visual_identity.default_workflow}.json"
                    if not workflow_path.exists():
                        warnings.append(
                            f"⚠ Character '{char_id}' references missing workflow: "
                            f"{char_config.visual_identity.default_workflow}"
                        )
    except Exception as e:
        errors.append(f"❌ Failed to load characters: {e}")
    
    # 5. Check database
    try:
        db_path = system_config.paths.data / "chorus.db"
        await initialize_database(db_path)
        logger.info("✓ Database initialized")
    except Exception as e:
        errors.append(f"❌ Database initialization failed: {e}")
    
    # 6. Summary
    if errors:
        logger.error("\n".join(errors))
        raise RuntimeError("Startup validation failed. Fix errors and restart.")
    
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info("✓ Startup validation complete")
    return system_config, characters
```

---

## Summary

**Configuration validation strategy**:

1. **Pydantic models** enforce type safety and validation rules
2. **ConfigLoader** handles file I/O and error formatting
3. **Graceful defaults** for system config (can run without file)
4. **Strict validation** for characters (must be valid)
5. **Clear error messages** guide users to fixes
6. **Forward compatible** by ignoring unknown fields
7. **Optional hot reload** for development convenience
8. **Comprehensive startup validation** catches issues early

**Key benefits**:
- Type hints everywhere (IDE support)
- Runtime validation prevents bad data
- User-friendly error messages
- Cross-platform (Path handling)
- Testable (mock configs easily)
- Maintainable (centralized validation logic)

This foundation ensures configuration is always valid and provides a great developer and user experience.

---

## Implementation Checklist

- [ ] Define all Pydantic models
- [ ] Implement ConfigLoader class
- [ ] Create error formatting utilities
- [ ] Add startup validation sequence
- [ ] Write config validation tests
- [ ] Document config file formats for users
- [ ] (Optional) Implement hot reload
- [ ] (Optional) Add config migration framework

**Ready for Phase 1 implementation.**
