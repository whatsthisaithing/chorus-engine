"""Pydantic models for configuration validation."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class LLMConfig(BaseModel):
    """LLM backend configuration."""
    
    provider: Literal["ollama", "lmstudio", "koboldcpp", "openai-compatible"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b-instruct"
    context_window: int = Field(default=8192, gt=0, le=128000)
    max_response_tokens: int = Field(default=2048, gt=0, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=120, gt=0)
    unload_during_image_generation: bool = Field(default=False, description="Unload model from VRAM during image generation to free memory")
    
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
    video_timeout_seconds: int = Field(default=600, gt=0)  # Longer timeout for video generation
    polling_interval_seconds: float = Field(default=2.0, gt=0)
    max_concurrent_jobs: int = Field(default=2, gt=0, le=10)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('url must start with http:// or https://')
        return v.rstrip('/')


class IntentDetectionConfig(BaseModel):
    """Intent detection system configuration (Phase 7)."""
    
    enabled: bool = True
    model: str = "gemma2:9b"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    keep_loaded: bool = True
    fallback_to_keywords: bool = True
    thresholds: dict = Field(default_factory=lambda: {
        'image': 0.7,
        'video': 0.7,
        'memory': 0.8,
        'ambient': 0.6
    })


class PathsConfig(BaseModel):
    """File path configuration."""
    
    characters: Path = Path("characters")
    workflows: Path = Path("workflows")
    data: Path = Path("data")
    
    @field_validator('characters', 'workflows', 'data')
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Ensure paths are cross-platform."""
        return Path(v)


class UIConfig(BaseModel):
    """UI configuration."""
    
    color_scheme: str = Field(
        default="stage-night",
        description="Color scheme/theme name"
    )


class UIConfig(BaseModel):
    """UI configuration."""
    
    color_scheme: str = Field(
        default="stage-night",
        description="Color scheme/theme name"
    )


class SystemDocumentAnalysisConfig(BaseModel):
    """System-level document analysis configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable document analysis system globally"
    )
    default_max_chunks: int = Field(
        default=3,
        gt=0,
        le=100,
        description="Default maximum chunks to retrieve per query"
    )
    max_chunks_cap: int = Field(
        default=25,
        gt=0,
        le=100,
        description="Absolute maximum chunks (safety limit)"
    )
    chunk_token_estimate: int = Field(
        default=512,
        gt=0,
        description="Average tokens per chunk for budget calculations"
    )
    document_budget_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Default portion of context window for documents"
    )


class SystemConfig(BaseModel):
    """Top-level system configuration."""
    
    model_config = ConfigDict(extra='ignore')
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    intent_detection: IntentDetectionConfig = Field(default_factory=IntentDetectionConfig)
    document_analysis: SystemDocumentAnalysisConfig = Field(default_factory=SystemDocumentAnalysisConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    debug: bool = False
    api_host: str = "localhost"
    api_port: int = Field(default=8080, gt=0, le=65535)
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Ensure directories exist
        for path_name in ['characters', 'workflows', 'data']:
            path = getattr(self.paths, path_name)
            path.mkdir(parents=True, exist_ok=True)


# Character configuration models

class TTSProviderConfig(BaseModel):
    """TTS provider-specific configuration."""
    
    # Provider selection
    provider: Literal["comfyui", "chatterbox"] = "comfyui"
    
    # ComfyUI-specific settings
    comfyui: Optional[dict] = Field(
        default_factory=lambda: {"workflow_name": "default_tts_workflow"},
        description="ComfyUI workflow settings"
    )
    
    # Chatterbox-specific settings
    chatterbox: Optional[dict] = Field(
        default_factory=lambda: {
            "temperature": 0.8,
            "use_voice_cloning": True,
            "chunk_threshold": 200  # Characters per chunk (~13-15s audio)
        },
        description="Chatterbox model settings with chunking support"
    )


class VoiceConfig(BaseModel):
    """Voice and TTS configuration."""
    
    enabled: bool = Field(default=False, description="Enable TTS for this character")
    always_on: bool = Field(default=False, description="Auto-generate TTS for every message")
    
    # TTS provider configuration
    tts_provider: TTSProviderConfig = Field(
        default_factory=TTSProviderConfig,
        description="TTS provider settings"
    )
    
    # Voice sample (used by voice cloning providers)
    voice_sample: Optional[Path] = Field(default=None, description="Default voice sample file")
    
    # Deprecated fields (kept for backward compatibility)
    tts_engine: Optional[str] = Field(default=None, description="DEPRECATED: Use tts_provider.provider")
    speaking_style: Optional[dict] = Field(default=None, description="DEPRECATED: Provider-specific setting")


class EmotionalRange(BaseModel):
    """Emotional constraints for character."""
    
    baseline: str = "neutral"
    allowed: list[str] = Field(default_factory=lambda: ["neutral", "positive"])


class ImmersionSettings(BaseModel):
    """Fine-grained immersion control."""
    
    allow_preferences: bool = True
    allow_opinions: bool = True
    allow_experiences: bool = False
    allow_physical_metaphor: bool = False
    allow_physical_sensation: bool = False
    disclaimer_behavior: Literal["never", "only_when_asked", "always"] = "only_when_asked"


class CoreMemory(BaseModel):
    """Immutable character backstory element."""
    
    content: str = Field(min_length=10)
    tags: list[str] = Field(default_factory=list)
    embedding_priority: Literal["low", "medium", "high"] = "medium"


class MemoryPreferences(BaseModel):
    """Character-specific memory settings."""
    
    scope: Literal["global", "character", "thread"] = "character"
    vector_store: str = "chroma_default"


class MemoryProfile(BaseModel):
    """
    Memory extraction profile for Phase 8.
    
    Controls which memory types are extracted based on immersion level.
    """
    
    # Always extracted for all levels
    extract_facts: bool = Field(default=True, description="Basic factual information")
    extract_projects: bool = Field(default=True, description="Goals, plans, and ongoing projects")
    
    # Extracted based on immersion level
    extract_experiences: Optional[bool] = Field(default=None, description="Shared experiences and activities")
    extract_stories: Optional[bool] = Field(default=None, description="Narratives and anecdotes")
    extract_relationship: Optional[bool] = Field(default=None, description="Relationship dynamics and emotional bonds")
    
    # Emotional analysis (all levels can have emotional weight)
    track_emotional_weight: bool = Field(default=True, description="Track emotional significance of memories")
    track_participants: bool = Field(default=True, description="Track who was involved in memories")
    
    def model_post_init(self, __context) -> None:
        """Apply defaults based on None values."""
        # If not explicitly set, these remain None and will use immersion level defaults
        pass


class AmbientActivityConfig(BaseModel):
    """Ambient activity settings."""
    
    enabled: bool = False
    default_behavior: Literal["propose_if_idle", "never", "always"] = "propose_if_idle"
    style_guidelines: Optional[str] = None


class VisualIdentityConfig(BaseModel):
    """Visual generation settings."""
    
    default_workflow: Optional[str] = None
    prompt_context: Optional[str] = None


class ImageGenerationConfig(BaseModel):
    """Image generation configuration for characters (Phase 5).
    
    Note: Workflow settings (workflow_file, trigger_word, default_style, etc.) 
    are now managed in the database via the workflow management system.
    Only the enabled flag is stored in character configuration.
    """
    
    enabled: bool = False


class VideoGenerationConfig(BaseModel):
    """Video generation configuration for characters.
    
    Note: Workflow settings are managed in the database via the workflow management system.
    Only the enabled flag is stored in character configuration.
    """
    
    enabled: bool = False


class DocumentAnalysisConfig(BaseModel):
    """Document analysis configuration for characters (Phase 1-7)."""
    
    enabled: bool = Field(
        default=False,
        description="Enable document ingestion, semantic search, and Q&A with citations"
    )
    max_documents: Optional[int] = Field(
        default=None,
        description="Maximum documents this character can access (None = unlimited)"
    )
    allowed_document_types: Optional[list[str]] = Field(
        default=None,
        description="Restrict document types (e.g., ['policy', 'guideline']). None = all types allowed"
    )
    max_chunks: Optional[int] = Field(
        default=None,
        description="Maximum chunks to retrieve per query (overrides system default)"
    )
    document_budget_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Portion of context window for documents (overrides system default)"
    )


class CodeExecutionConfig(BaseModel):
    """Code execution configuration for characters (Phase 2-7)."""
    
    enabled: bool = Field(
        default=False,
        description="Enable Python code generation and execution for data analysis"
    )
    max_execution_time: int = Field(
        default=30,
        description="Maximum execution time in seconds",
        gt=0,
        le=300
    )
    allowed_libraries: Optional[list[str]] = Field(
        default=None,
        description="Whitelist of allowed Python libraries. None = default whitelist (pandas, numpy, etc.)"
    )

class UIPreferences(BaseModel):
    """UI preferences for character."""
    
    color_scheme: Optional[str] = Field(
        default=None,
        description="Color scheme override for this character (null = use system default)"
    )

class PreferredLLMConfig(BaseModel):
    """Character's preferred LLM settings."""
    
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=8192)
    context_window: Optional[int] = Field(default=None, gt=0, description="Override context window for different model capabilities")


class CharacterConfig(BaseModel):
    """Complete character configuration."""
    
    model_config = ConfigDict(extra='ignore')
    
    # Required fields
    id: str = Field(min_length=1, max_length=50, pattern=r'^[a-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)
    system_prompt: str = Field(min_length=10)
    
    # Character type/role classification (Phase 1-7: Document Analysis)
    role_type: Literal["companion", "assistant", "analyst", "other"] = Field(
        default="assistant",
        description="Character archetype: companion (casual chat), assistant (task help), analyst (data/documents), other (custom)"
    )
    
    # Power user mode: Skip immersion guidance and use raw system prompt
    custom_system_prompt: bool = Field(
        default=False,
        description="If true, uses system_prompt exactly as written without adding immersion guidance. "
                    "WARNING: Power user feature - may break memory extraction, image generation if prompt doesn't follow conventions."
    )
    
    # Immersion configuration (Phase 3)
    immersion_level: Literal["minimal", "balanced", "full", "unbounded"] = "balanced"
    immersion_settings: ImmersionSettings = Field(default_factory=ImmersionSettings)
    
    # Memory profile (Phase 8)
    memory_profile: MemoryProfile = Field(default_factory=MemoryProfile)
    
    # Core memories (Phase 3 - pre-loaded backstory)
    core_memories: list[CoreMemory] = Field(default_factory=list)
    
    # Optional fields with defaults
    personality_traits: list[str] = Field(default_factory=list)
    preferred_llm: PreferredLLMConfig = Field(default_factory=PreferredLLMConfig)
    voice: Optional[VoiceConfig] = None
    emotional_range: EmotionalRange = Field(default_factory=EmotionalRange)
    memory: MemoryPreferences = Field(default_factory=MemoryPreferences)
    ambient_activity: AmbientActivityConfig = Field(default_factory=AmbientActivityConfig)
    visual_identity: Optional[VisualIdentityConfig] = None
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)
    video_generation: VideoGenerationConfig = Field(default_factory=VideoGenerationConfig)
    
    # Document analysis & code execution (Phase 1-7)
    document_analysis: DocumentAnalysisConfig = Field(default_factory=DocumentAnalysisConfig)
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
    
    # UI preferences
    ui_preferences: UIPreferences = Field(default_factory=UIPreferences)
    
    # Profile customization
    profile_image: Optional[str] = Field(default=None, description="Filename of profile image in data/character_images/")
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID is safe for filesystem and URLs."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Character ID must contain only lowercase letters, numbers, underscores, and hyphens')
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        """Ensure system prompt is substantial."""
        if len(v.strip()) < 10:
            raise ValueError('system_prompt must be at least 10 characters')
        return v.strip()
    
    @field_validator('immersion_level')
    @classmethod
    def apply_immersion_presets(cls, v: str, info) -> str:
        """Apply preset immersion settings based on level if not explicitly configured."""
        # This validator runs before immersion_settings, so we'll handle presets in model_post_init
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Set timestamps if not provided
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        
        # Apply immersion level presets if using defaults
        # (Only if user didn't explicitly configure immersion_settings)
        self._apply_immersion_presets()
    
    def _apply_immersion_presets(self) -> None:
        """Apply immersion level presets."""
        presets = {
            "minimal": ImmersionSettings(
                allow_preferences=False,
                allow_opinions=False,
                allow_experiences=False,
                allow_physical_metaphor=False,
                allow_physical_sensation=False,
                disclaimer_behavior="always"
            ),
            "balanced": ImmersionSettings(
                allow_preferences=True,
                allow_opinions=True,
                allow_experiences=False,
                allow_physical_metaphor=False,
                allow_physical_sensation=False,
                disclaimer_behavior="only_when_asked"
            ),
            "full": ImmersionSettings(
                allow_preferences=True,
                allow_opinions=True,
                allow_experiences=True,
                allow_physical_metaphor=True,
                allow_physical_sensation=False,
                disclaimer_behavior="never"
            ),
            "unbounded": ImmersionSettings(
                allow_preferences=True,
                allow_opinions=True,
                allow_experiences=True,
                allow_physical_metaphor=True,
                allow_physical_sensation=True,
                disclaimer_behavior="never"
            ),
        }
        
        # Only apply preset if user is using default ImmersionSettings
        if self.immersion_level in presets:
            # Check if using default settings (all default values)
            default = ImmersionSettings()
            if self.immersion_settings == default:
                self.immersion_settings = presets[self.immersion_level]
    
    @property
    def profile_image_url(self) -> str:
        """Get URL path for character's profile image."""
        if self.profile_image:
            return f"/character_images/{self.profile_image}"
        return "/character_images/default.svg"


class IntentDetectionConfig(BaseModel):
    """Configuration for intent detection system (Phase 7)."""
    
    enabled: bool = True
    model: str = "qwen2.5:3b-instruct"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Very low for consistent classification")
    keep_loaded: bool = Field(default=True, description="Keep intent model loaded for fast detection")
    fallback_to_keywords: bool = Field(default=True, description="Use keyword detection if intent model unavailable")
    
    # Intent-specific confidence thresholds
    thresholds: dict = Field(default_factory=lambda: {
        "image": 0.7,
        "video": 0.7,
        "memory": 0.8,
        "ambient": 0.6,
    })


class VRAMManagementConfig(BaseModel):
    """Configuration for VRAM management (Phase 7)."""
    
    unload_during_comfy: bool = Field(
        default=True,
        description="Unload all models during ComfyUI generation to free maximum VRAM"
    )
    always_reload: list[str] = Field(
        default_factory=lambda: ["intent_model", "character_model"],
        description="Models to reload after generation"
    )
    verify_unload: bool = Field(default=True, description="Verify models unloaded successfully")
    verify_reload: bool = Field(default=True, description="Verify models reloaded successfully")
