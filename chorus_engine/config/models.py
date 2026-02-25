"""Pydantic models for configuration validation."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, List
from pydantic import AliasChoices, BaseModel, Field, field_validator, ConfigDict


class ProviderCapabilityConfig(BaseModel):
    """Per-provider transport capability overrides."""

    supports_native_tools: bool = True
    supports_response_format_json_schema: bool = True
    supports_sentinel_retry: bool = True


class LLMConfig(BaseModel):
    """LLM backend configuration."""
    
    provider: Literal["ollama", "lmstudio", "koboldcpp", "openai-compatible", "integrated"] = "integrated"
    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b-instruct"
    archivist_model: Optional[str] = Field(
        default=None,
        description="Optional dedicated model for conversation analysis (summary + archivist)."
    )
    analysis_max_tokens_summary: int = Field(
        default=4096,
        gt=0,
        le=8192,
        description="Max tokens for conversation summary analysis."
    )
    analysis_max_tokens_memories: int = Field(
        default=4096,
        gt=0,
        le=8192,
        description="Max tokens for conversation memory extraction analysis."
    )
    analysis_min_tokens_summary: int = Field(
        default=500,
        ge=0,
        le=100000,
        description="Minimum tokens required for summary analysis."
    )
    analysis_min_tokens_memories: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="Minimum tokens required for memory extraction analysis."
    )
    context_window: int = Field(default=8192, gt=0, le=128000)
    max_response_tokens: int = Field(default=2048, gt=0, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling cutoff (0.0-1.0)")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling cutoff (integer >= 0)")
    repeat_penalty: Optional[float] = Field(default=None, description="Repeat penalty (> 0)")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty (-2.0 to 2.0)")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty (-2.0 to 2.0)")
    timeout_seconds: int = Field(default=120, gt=0)
    unload_during_image_generation: bool = Field(default=False, description="Unload model from VRAM during image generation to free memory")
    ollama_legacy_chat_api_enabled: bool = Field(
        default=False,
        description="Compat-only: use Ollama legacy /api/chat transport instead of OpenAI-compatible /v1/chat/completions.",
    )
    ollama_capture_raw_http_debug: bool = Field(
        default=False,
        description=(
            "Debug-only: capture raw request/response payloads for Ollama calls "
            "to data/debug/requests/*.json"
        ),
    )
    provider_capabilities: Dict[str, ProviderCapabilityConfig] = Field(
        default_factory=lambda: {
            "ollama": ProviderCapabilityConfig(
                supports_native_tools=True,
                supports_response_format_json_schema=True,
                supports_sentinel_retry=True,
            ),
            "lmstudio": ProviderCapabilityConfig(
                supports_native_tools=True,
                supports_response_format_json_schema=True,
                supports_sentinel_retry=True,
            ),
            "koboldcpp": ProviderCapabilityConfig(
                supports_native_tools=False,
                supports_response_format_json_schema=False,
                supports_sentinel_retry=True,
            ),
        },
        description="Provider transport capability overrides keyed by engine name.",
    )
    
    # Integrated provider specific fields
    n_gpu_layers: Optional[int] = Field(default=-1, description="GPU layers for integrated provider (-1=all, 0=CPU only)")
    n_threads: Optional[int] = Field(default=8, gt=0, description="CPU threads for integrated provider")
    
    @field_validator('base_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')

    @field_validator("top_p", mode="before")
    @classmethod
    def clamp_top_p(cls, value):
        if value is None or value == "":
            return None
        numeric = float(value)
        return min(1.0, max(0.0, numeric))

    @field_validator("top_k", mode="before")
    @classmethod
    def clamp_top_k(cls, value):
        if value is None or value == "":
            return None
        numeric = int(float(value))
        return max(0, numeric)

    @field_validator("repeat_penalty", mode="before")
    @classmethod
    def clamp_repeat_penalty(cls, value):
        if value is None or value == "":
            return None
        numeric = float(value)
        return numeric if numeric > 0.0 else 0.01

    @field_validator("presence_penalty", "frequency_penalty", mode="before")
    @classmethod
    def clamp_penalties(cls, value):
        if value is None or value == "":
            return None
        numeric = float(value)
        return min(2.0, max(-2.0, numeric))


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


class MediaToolingConfig(BaseModel):
    """Media tooling settings for in-conversation tool payloads and offers."""

    enabled: bool = True
    offers_enabled: bool = True
    image_offers_enabled: bool = True
    video_offers_enabled: bool = True
    explicit_min_confidence_image: float = Field(default=0.5, ge=0.0, le=1.0)
    explicit_min_confidence_video: float = Field(default=0.45, ge=0.0, le=1.0)
    offer_min_confidence_image: float = Field(default=0.5, ge=0.0, le=1.0)
    offer_min_confidence_video: float = Field(default=0.45, ge=0.0, le=1.0)
    offer_cooldown_minutes: int = Field(default=30, ge=0, le=1440)
    offer_min_turn_gap: int = Field(default=8, ge=0, le=200)
    max_offers_per_conversation_per_media: int = Field(default=2, ge=0, le=20)
    disable_offers_for_sources: List[str] = Field(default_factory=lambda: ["discord"])


class IntentDetectionConfig(BaseModel):
    """Intent detection system configuration (Phase 7)."""
    
    enabled: bool = False
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


class UserIdentityConfig(BaseModel):
    """System-level canonical user identity."""
    
    display_name: str = ""
    aliases: List[str] = Field(default_factory=list)
    
    @field_validator('display_name')
    @classmethod
    def normalize_display_name(cls, v: str) -> str:
        if v is None:
            return ""
        # Trim and collapse repeated spaces
        return " ".join(v.strip().split())
    
    @field_validator('aliases')
    @classmethod
    def normalize_aliases(cls, v: List[str]) -> List[str]:
        if not v:
            return []
        seen = set()
        normalized = []
        for alias in v:
            if alias is None:
                continue
            cleaned = " ".join(str(alias).strip().split())
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
        return normalized


class TimeContextConfig(BaseModel):
    """Server-local time injection configuration."""
    
    enabled: bool = True
    timezone: Optional[str] = None  # IANA tz; None/blank uses server local time
    format: Literal["iso"] = "iso"


class UserIdentityOverrideConfig(BaseModel):
    """Character-level user identity override settings."""
    
    mode: Literal["canonical", "masked", "role"] = "canonical"
    role_name: Optional[str] = None
    role_aliases: List[str] = Field(default_factory=list)
    
    @field_validator('role_name')
    @classmethod
    def validate_role_name(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return v
        return " ".join(v.strip().split())
    
    def model_post_init(self, __context) -> None:
        if self.mode == "role" and not self.role_name:
            raise ValueError("role_name is required when user_identity.mode is 'role'")


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


class SystemTTSConfig(BaseModel):
    """System-level TTS configuration."""
    
    default_provider: Literal["comfyui", "chatterbox"] = Field(
        default="chatterbox",
        description="Default TTS provider when character doesn't specify one"
    )
    include_physicalaction: bool = Field(
        default=False,
        description="Include physicalaction segments in TTS output"
    )


class VisionConfig(BaseModel):
    """Vision system configuration."""
    
    enabled: bool = Field(default=False, description="Enable vision system")
    max_response_tokens: int = Field(
        default=2048,
        gt=0,
        le=8192,
        description="Maximum response tokens for vision model output."
    )
    model: dict = Field(
        default_factory=lambda: {
            "name": "qwen2-vl:7b",
            "load_timeout_seconds": 60
        },
        description="Vision model configuration"
    )
    processing: dict = Field(
        default_factory=lambda: {
            "max_retries": 2,
            "timeout_seconds": 30,
            "resize_target": 1024,
            "supported_formats": ["jpg", "jpeg", "png", "webp", "gif"],
            "max_file_size_mb": 10
        },
        description="Image processing settings"
    )
    output: dict = Field(
        default_factory=lambda: {
            "format": "structured",
            "include_confidence": True
        },
        description="Output format configuration"
    )
    memory: dict = Field(
        default_factory=lambda: {
            "auto_create": True,
            "category": "visual",
            "default_priority": 70,
            "min_confidence": 0.6
        },
        description="Visual memory creation settings"
    )
    intent: dict = Field(
        default_factory=lambda: {
            "bridge_always_analyze": False,
            "bridge_never_analyze": False,
            "web_ui_always_analyze": True,
            "use_semantic_detection": False,
            "confidence_threshold": 0.45,
            "trigger_phrases": [
                "what do you see",
                "look at",
                "what's in",
                "describe",
                "check this"
            ]
        },
        description="Intent detection settings"
    )
    cache: dict = Field(
        default_factory=lambda: {
            "enabled": True,
            "allow_reanalysis": True
        },
        description="Cache settings"
    )


class ConversationContextConfig(BaseModel):
    """Configuration for conversation context retrieval from past conversations."""
    
    model_config = ConfigDict(extra='ignore')
    
    enabled: bool = Field(
        default=True,
        description="Enable automatic retrieval of relevant past conversation summaries"
    )
    passive_threshold: float = Field(
        default=0.75,
        ge=0.0, le=1.0,
        description="Similarity threshold for passive context retrieval (higher = stricter)"
    )
    triggered_threshold: float = Field(
        default=0.55,
        ge=0.0, le=1.0,
        description="Similarity threshold when user explicitly references past conversations"
    )
    max_summaries: int = Field(
        default=2,
        ge=1, le=5,
        description="Maximum number of past conversation summaries to include"
    )
    token_budget_ratio: float = Field(
        default=0.05,
        ge=0.0, le=0.2,
        description="Portion of context window allocated for conversation summaries"
    )


class StartupConfig(BaseModel):
    """Startup synchronization configuration."""
    
    model_config = ConfigDict(extra='ignore')
    
    sync_summary_vectors: bool = Field(
        default=True,
        description="Sync missing conversation summary vectors on startup (self-healing)"
    )
    sync_memory_vectors: bool = Field(
        default=True,
        description="Sync missing memory vectors on startup (self-healing)"
    )
    sync_moment_pin_vectors: bool = Field(
        default=True,
        description="Sync missing moment pin vectors on startup (self-healing)"
    )
    auto_repair_unhealthy_vectors: bool = Field(
        default=True,
        description="Rebuild unhealthy vector collections detected during startup health checks"
    )


class HeartbeatBackupsConfig(BaseModel):
    """Configuration for heartbeat-driven scheduled character backups."""

    model_config = ConfigDict(extra='ignore')

    enabled: bool = Field(
        default=True,
        description="Enable scheduled heartbeat backups"
    )
    interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Target backup cadence in hours"
    )
    destination_dir: Optional[Path] = Field(
        default=None,
        description="Absolute directory where scheduled backup archives are written"
    )
    retention_days: int = Field(
        default=14,
        ge=1,
        le=365,
        description="Delete scheduled backups older than this many days"
    )
    max_backups_per_cycle: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum characters to process per heartbeat cycle"
    )
    skip_if_unchanged: bool = Field(
        default=True,
        description="Skip scheduled backups when data fingerprint has not changed"
    )
    verify_archive: bool = Field(
        default=True,
        description="Validate ZIP archive integrity after writing"
    )

    @field_validator("destination_dir")
    @classmethod
    def validate_destination_dir(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        if not v.is_absolute():
            raise ValueError("heartbeat.backups.destination_dir must be an absolute path")
        return v


class HeartbeatConfig(BaseModel):
    """Configuration for the heartbeat background processing system."""
    
    model_config = ConfigDict(extra='ignore')
    
    enabled: bool = Field(
        default=True,
        description="Enable heartbeat background processing system"
    )
    interval_seconds: float = Field(
        default=60.0,
        ge=10.0, le=600.0,
        description="Seconds between heartbeat checks when idle"
    )
    idle_threshold_minutes: float = Field(
        default=5.0,
        ge=1.0, le=60.0,
        description="Minutes of inactivity before system is considered idle"
    )
    resume_grace_seconds: float = Field(
        default=2.0,
        ge=0.5, le=10.0,
        description="Grace period after activity before interrupting background tasks"
    )
    
    # Conversation summary analysis settings
    analysis_summary_stale_hours: float = Field(
        default=24.0,
        ge=1.0,
        description="Hours since last activity before summary analysis is eligible"
    )
    analysis_summary_min_messages: int = Field(
        default=10,
        ge=4,
        description="Minimum messages before summary analysis is eligible"
    )
    analysis_summary_batch_size: int = Field(
        default=3,
        ge=1, le=10,
        description="Maximum summary analyses per heartbeat cycle"
    )

    # Conversation memories analysis settings
    analysis_memories_stale_hours: float = Field(
        default=24.0,
        ge=1.0,
        description="Hours since last activity before memory analysis is eligible"
    )
    analysis_memories_min_messages: int = Field(
        default=10,
        ge=4,
        description="Minimum messages before memory analysis is eligible"
    )
    analysis_memories_batch_size: int = Field(
        default=3,
        ge=1, le=10,
        description="Maximum memory analyses per heartbeat cycle"
    )

    # Legacy fields (deprecated; use analysis_summary_* and analysis_memories_*)
    analysis_stale_hours: float = Field(
        default=24.0,
        ge=1.0,
        description="Deprecated: use analysis_summary_stale_hours"
    )
    analysis_min_messages: int = Field(
        default=10,
        ge=4,
        description="Deprecated: use analysis_summary_min_messages"
    )
    analysis_batch_size: int = Field(
        default=3,
        ge=1, le=10,
        description="Deprecated: use analysis_summary_batch_size"
    )
    
    # GPU utilization check (NVIDIA only)
    gpu_check_enabled: bool = Field(
        default=False,
        description="Enable GPU utilization check before background tasks (NVIDIA only)"
    )
    gpu_max_utilization_percent: int = Field(
        default=15,
        ge=5, le=95,
        description="Skip background tasks if GPU utilization exceeds this percentage"
    )
    backups: HeartbeatBackupsConfig = Field(default_factory=HeartbeatBackupsConfig)


class GeneralChatSegmentationConfig(BaseModel):
    """Config for general-chat episodic segmentation."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True, description="Enable general-chat segmentation lifecycle")
    idle_soft_minutes: int = Field(default=30, ge=1, le=1440, description="Soft idle threshold for possible split")
    idle_hard_minutes: int = Field(default=120, ge=1, le=1440, description="Hard idle threshold for definite split")
    sleep_break_minutes: int = Field(default=360, ge=1, le=10080, description="Sleep heuristic threshold for definite split")
    density_window_messages: int = Field(default=8, ge=1, le=100, description="Trailing messages to inspect for activity density")
    density_low_turn_count_max: int = Field(default=4, ge=1, le=100, description="Low-density max trailing turn count")
    density_low_avg_chars_max: int = Field(default=180, ge=1, le=2000, description="Low-density max trailing average characters")
    pending_question_tail_messages: int = Field(default=2, ge=0, le=20, description="Trailing assistant messages scanned for unresolved questions")
    resume_recent_useful_segments: int = Field(default=3, ge=1, le=20, description="Max recent useful segments considered for recap source")
    resume_max_age_hours: int = Field(default=72, ge=1, le=720, description="Max age of segment recap candidate")
    summary_max_tokens: int = Field(default=1200, ge=128, le=8192, description="Max tokens for segment summary generation")
    summary_model_override: Optional[str] = Field(default=None, description="Optional model override for segment summary generation")


class ENSConfig(BaseModel):
    """ENS feature flags for incremental rollout."""

    enabled: bool = Field(default=False, description="Enable ENS routing")
    slice1_chat_ownership: bool = Field(default=False, description="Route non-stream chat through ENS core actions")
    nonstream_intake_only: bool = Field(
        default=True,
        description="When true, non-stream /messages remains on legacy execution and ENS records intake decisions only",
    )
    streaming_intake_only: bool = Field(default=True, description="Stream route uses ENS intake only")
    slice1_compat_postprocessing_enabled: bool = Field(default=False, description="Temporary legacy post-processing after ENS core actions")
    slice2_tool_parsing_ownership: bool = Field(default=False, description="ENS owns non-stream tool payload parsing")
    slice2_tool_dispatch_ownership: bool = Field(default=False, description="ENS owns tool-originated media dispatch")
    slice2_scene_capture_ownership: bool = Field(default=False, description="ENS owns scene-capture preview and confirm flows")
    slice2_scene_capture_legacy_confirm_without_tool_call: bool = Field(
        default=False,
        description="Dev-only compat: allow scene-capture confirm without tool_call_id while scene-capture ownership is enabled",
    )
    slice25_media_gating_ownership: bool = Field(
        default=False,
        description="ENS owns media gating snapshot + tool payload adjudication actions",
    )
    slice3_continuity_writes_ownership: bool = Field(
        default=False,
        description="ENS owns continuity-affecting writes (summaries, memories, pins, continuity state)",
    )
    slice4_config_ownership: bool = Field(
        default=False,
        description="ENS owns config/control-plane writes (system, character, conversation toggles, workflows, core-memory sync)",
    )
    slice6_surface_routing_ownership: bool = Field(
        default=False,
        description="ENS owns normalized surface routing via canonical binding resolution",
    )
    slice65_egress_outbox_ownership: bool = Field(
        default=False,
        description="ENS owns normalized surface egress outbox intent persistence",
    )
    slice7_unified_llm_invocation: bool = Field(
        default=False,
        description="ENS owns unified provider-agnostic LLM invocation path",
    )
    slice75_llm_control_plane_ownership: bool = Field(
        default=False,
        description="ENS owns LLM control-plane operations (health/load/unload/reload/switch)",
    )
    v3_scheduler_enabled: bool = Field(
        default=False,
        description="ENS v3: route runnable work through scheduler/queue boundary",
    )
    v3_arbitration_enabled: bool = Field(
        default=False,
        description="ENS v3: enable arbitration and deterministic scheduler selection rules",
    )
    v3_loop_sessions_enabled: bool = Field(
        default=False,
        description="ENS v3: enable loop sessions as signal producers",
    )
    v3_structured_control_enabled: bool = Field(
        default=False,
        description="ENS v3: accept loop control only from structured control channel",
    )
    v3_assistant_result_enabled: bool = Field(
        default=False,
        description="ENS v3: enable canonical AssistantResult normalization path",
    )
    v3_context_compression_enabled: bool = Field(
        default=False,
        description="ENS v3: enable deterministic context compression hooks",
    )
    loop_memory_compress_every_n_steps: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="ENS v3 compression: trigger fold compression every N loop steps",
    )
    loop_memory_keep_last_k_steps: int = Field(
        default=6,
        ge=1,
        le=500,
        description="ENS v3 compression: keep last K raw loop steps uncompressed",
    )
    v3_sentinel_fallback_enabled: bool = Field(
        default=False,
        description="ENS v3: keep sentinel payload parsing fallback enabled",
    )
    native_tool_transport_enabled: bool = Field(
        default=False,
        description="ENS: enable provider-native tool transport for supported engines.",
    )
    native_tool_transport_force_sentinel: bool = Field(
        default=False,
        description="ENS: force sentinel-only parsing path even when native transport is available.",
    )
    native_tool_transport_sentinel_fallback_enabled: bool = Field(
        default=True,
        description="ENS: allow sentinel fallback when native transport attempt yields no control/tools.",
    )
    native_tool_transport_narrative_v11_split_enabled: bool = Field(
        default=False,
        description="ENS: enable narrative.v1 split loop-step flow (Stage A beat + Stage B control evaluation).",
    )
    native_tool_transport_debug_override_mode: str = Field(
        default="off",
        description=(
            "ENS debug-only native transport override mode: "
            "off|chat_control_only|loop_image_only|both."
        ),
    )
    loop_step_max_passes_per_step: int = Field(
        default=4,
        ge=1,
        le=16,
        description="ENS loop-step orchestration: maximum passes allowed per step execution.",
    )
    loop_step_allow_single_tool_loopback: bool = Field(
        default=True,
        description="ENS loop-step orchestration: allow at most one tool loopback per pass.",
    )
    loop_step_pass_trace_enabled: bool = Field(
        default=True,
        description="ENS loop-step orchestration: include pass_trace diagnostics in loop outputs/events.",
    )

    @field_validator("native_tool_transport_debug_override_mode", mode="before")
    @classmethod
    def _normalize_native_tool_transport_debug_override_mode(cls, value):
        # YAML 1.1 may coerce unquoted 'off' into boolean False.
        if isinstance(value, bool):
            value = "both" if value else "off"
        normalized = str(value or "off").strip().lower()
        allowed = {"off", "chat_control_only", "loop_image_only", "both"}
        if normalized not in allowed:
            raise ValueError(
                "native_tool_transport_debug_override_mode must be one of: "
                "off, chat_control_only, loop_image_only, both"
            )
        return normalized
    scheduler_sync_ticks_per_ingress: int = Field(
        default=1,
        ge=0,
        le=10,
        validation_alias=AliasChoices("scheduler_sync_ticks_per_ingress", "v3_sync_ticks_per_ingress"),
        description="ENS scheduler: maximum synchronous scheduler ticks per ingress",
    )
    scheduler_drain_interval_ms: int = Field(
        default=250,
        ge=250,
        le=500,
        validation_alias=AliasChoices("scheduler_drain_interval_ms", "v3_background_drain_interval_ms"),
        description="ENS scheduler: periodic scheduler drain interval in milliseconds",
    )
    scheduler_max_ticks_per_drain: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias=AliasChoices("scheduler_max_ticks_per_drain", "v3_max_ticks_per_drain"),
        description="ENS scheduler: maximum ticks per background drain cycle",
    )
    scheduler_max_wall_ms_per_drain: int = Field(
        default=50,
        ge=1,
        le=5000,
        validation_alias=AliasChoices("scheduler_max_wall_ms_per_drain", "v3_max_wall_ms_per_drain"),
        description="ENS scheduler: maximum wall-clock milliseconds per background drain cycle",
    )
    scheduler_running_ttl_seconds: int = Field(
        default=180,
        ge=30,
        le=3600,
        description="ENS scheduler: running-state TTL before stale claim recovery",
    )
    scheduler_recovery_check_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="ENS scheduler: interval for stale running recovery sweep",
    )
    scheduler_attention_lock_seconds: int = Field(
        default=120,
        ge=0,
        le=3600,
        description="ENS scheduler: deconfliction attention-lock TTL (weighting only; no reopen gating)",
    )
    scheduler_surface_rate_cap_per_minute: int = Field(
        default=0,
        ge=0,
        le=10000,
        description="ENS scheduler: max completed signals per surface per rolling minute (0 disables)",
    )
    scheduler_assistant_rate_cap_per_minute: int = Field(
        default=0,
        ge=0,
        le=10000,
        description="ENS scheduler: max completed signals per assistant per rolling minute (0 disables)",
    )
    scheduler_surface_cooldown_ms: int = Field(
        default=0,
        ge=0,
        le=60000,
        description="ENS scheduler: minimum gap between completions on same surface in milliseconds (0 disables)",
    )
    debug_capture_full_prompt: bool = Field(
        default=False,
        description="Debug-only: include full assembled system prompt and message payload in ENS conversation logs",
    )


class SystemConfig(BaseModel):
    """Top-level system configuration."""
    
    model_config = ConfigDict(extra='ignore')
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    media_tooling: MediaToolingConfig = Field(default_factory=MediaToolingConfig)
    tts: SystemTTSConfig = Field(default_factory=SystemTTSConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    intent_detection: IntentDetectionConfig = Field(default_factory=IntentDetectionConfig)
    document_analysis: SystemDocumentAnalysisConfig = Field(default_factory=SystemDocumentAnalysisConfig)
    conversation_context: ConversationContextConfig = Field(default_factory=ConversationContextConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    general_chat_segmentation: GeneralChatSegmentationConfig = Field(default_factory=GeneralChatSegmentationConfig)
    ens: ENSConfig = Field(default_factory=ENSConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    startup: StartupConfig = Field(default_factory=StartupConfig)
    user_identity: UserIdentityConfig = Field(default_factory=UserIdentityConfig)
    time_context: TimeContextConfig = Field(default_factory=TimeContextConfig)
    debug: bool = False
    debug_ui: bool = False
    admin_token: Optional[str] = Field(
        default=None,
        description="Optional admin token for privileged endpoints when debug_ui is disabled",
    )
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


class ProactiveOfferMediaOverrideConfig(BaseModel):
    """Per-media proactive offer override fields (null = inherit system default)."""

    enabled: Optional[bool] = None
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ProactiveOffersConfig(BaseModel):
    """Per-character proactive media offer overrides."""

    image: ProactiveOfferMediaOverrideConfig = Field(default_factory=ProactiveOfferMediaOverrideConfig)
    video: ProactiveOfferMediaOverrideConfig = Field(default_factory=ProactiveOfferMediaOverrideConfig)
    offer_cooldown_minutes: Optional[int] = Field(default=None, ge=0, le=1440)
    offer_min_turn_gap: Optional[int] = Field(default=None, ge=0, le=200)
    max_offers_per_conversation_per_media: Optional[int] = Field(default=None, ge=0, le=20)


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


class ContinuityPreferencesConfig(BaseModel):
    """Continuity bootstrap preferences stored per character."""

    default_mode: Literal["ask", "use", "fresh"] = Field(
        default="ask",
        description="Default continuity mode when preference is remembered"
    )


class CharacterBackupConfig(BaseModel):
    """Per-character scheduled backup preferences."""

    enabled: bool = Field(default=False, description="Enable scheduled backups for this character")
    schedule: Literal["daily"] = Field(default="daily", description="Scheduled backup cadence")
    local_time: str = Field(default="03:00", description="24h local time for scheduled backup trigger")
    destination_override: Optional[Path] = Field(
        default=None,
        description="Optional absolute destination directory for this character"
    )
    include_workflows: bool = Field(default=True, description="Include workflows in scheduled backups")
    notes_template: Optional[str] = Field(default=None, description="Optional notes template written into backup manifest")

    @field_validator("local_time")
    @classmethod
    def validate_local_time(cls, v: str) -> str:
        if len(v) != 5 or v[2] != ":":
            raise ValueError("backup.local_time must be in HH:MM format")
        hour_text, minute_text = v.split(":")
        if not (hour_text.isdigit() and minute_text.isdigit()):
            raise ValueError("backup.local_time must be numeric HH:MM")
        hour = int(hour_text)
        minute = int(minute_text)
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError("backup.local_time must be a valid 24h time")
        return v

    @field_validator("destination_override")
    @classmethod
    def validate_destination_override(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        if not v.is_absolute():
            raise ValueError("backup.destination_override must be an absolute path")
        return v


class CharacterFeaturesConfig(BaseModel):
    """Feature flags for character-scoped optional capabilities."""

    interactive_narrative: bool = Field(
        default=False,
        description="Enable interactive narrative loop controls for this character",
    )

class PreferredLLMConfig(BaseModel):
    """Character's preferred LLM settings."""
    
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=8192)
    context_window: Optional[int] = Field(default=None, gt=0, description="Override context window for different model capabilities")
    top_p: Optional[float] = Field(default=None, description="Override top_p (0.0-1.0)")
    top_k: Optional[int] = Field(default=None, description="Override top_k (integer >= 0)")
    repeat_penalty: Optional[float] = Field(default=None, description="Override repeat_penalty (> 0)")
    presence_penalty: Optional[float] = Field(default=None, description="Override presence_penalty (-2.0 to 2.0)")
    frequency_penalty: Optional[float] = Field(default=None, description="Override frequency_penalty (-2.0 to 2.0)")

    @field_validator("top_p", mode="before")
    @classmethod
    def clamp_top_p(cls, value):
        if value is None or value == "":
            return None
        numeric = float(value)
        return min(1.0, max(0.0, numeric))

    @field_validator("top_k", mode="before")
    @classmethod
    def clamp_top_k(cls, value):
        if value is None or value == "":
            return None
        numeric = int(float(value))
        return max(0, numeric)

    @field_validator("repeat_penalty", mode="before")
    @classmethod
    def clamp_repeat_penalty(cls, value):
        if value is None or value == "":
            return None
        numeric = float(value)
        return numeric if numeric > 0.0 else 0.01

    @field_validator("presence_penalty", "frequency_penalty", mode="before")
    @classmethod
    def clamp_penalties(cls, value):
        if value is None or value == "":
            return None
        numeric = float(value)
        return min(2.0, max(-2.0, numeric))


class CharacterConfig(BaseModel):
    """Complete character configuration."""
    
    model_config = ConfigDict(extra='ignore')
    
    # Required fields
    id: str = Field(min_length=1, max_length=50, pattern=r'^[a-z0-9_-]+$')
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    system_prompt: str = Field(min_length=10)
    
    # Character type/role classification (Phase 1-7: Document Analysis)
    role_type: Literal["companion", "assistant", "analyst", "chatbot", "roleplayer", "other"] = Field(
        default="assistant",
        description="Character archetype: companion (casual chat), assistant (task help), analyst (data/documents), chatbot (group chat participant), roleplayer (in-scene roleplay focus), other (custom)"
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
    
    # Structured response configuration (Phase 11)
    response_template: Optional[Literal["A", "B", "C", "D"]] = None
    expressiveness: Optional[Literal["minimal", "balanced", "rich"]] = None
    
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
    proactive_offers: ProactiveOffersConfig = Field(default_factory=ProactiveOffersConfig)
    
    # Document analysis & code execution (Phase 1-7)
    document_analysis: DocumentAnalysisConfig = Field(default_factory=DocumentAnalysisConfig)
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
    
    # UI preferences
    ui_preferences: UIPreferences = Field(default_factory=UIPreferences)

    # Continuity preferences
    continuity_preferences: ContinuityPreferencesConfig = Field(
        default_factory=ContinuityPreferencesConfig
    )
    backup: CharacterBackupConfig = Field(default_factory=CharacterBackupConfig)
    
    # User identity override (conversation runtime only)
    user_identity: UserIdentityOverrideConfig = Field(default_factory=UserIdentityOverrideConfig)
    
    # Profile customization
    profile_image: Optional[str] = Field(default=None, description="Filename of profile image in data/character_images/")
    profile_image_focus: Optional[Dict[str, float]] = Field(default=None, description="Focal point for profile image cropping (x and y percentages)")
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    features: CharacterFeaturesConfig = Field(default_factory=CharacterFeaturesConfig)
    
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
