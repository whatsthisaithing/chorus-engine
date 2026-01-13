"""
Character Card Data Models
=========================

Pydantic models for character card schemas (Chorus Engine and SillyTavern formats).
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


# ===========================
# Chorus Engine Card Format
# ===========================

class ChorusCardSpec(str, Enum):
    """Chorus Engine card specification versions."""
    V1 = "chorus_card_v1"


class ResponseStyle(BaseModel):
    """Character response style configuration."""
    perspective: Optional[str] = None  # first_person, second_person, third_person
    format: Optional[str] = None  # chat, prose, roleplay
    max_response_length: Optional[int] = None


class CharacterTraits(BaseModel):
    """Character personality traits."""
    openness: Optional[float] = None
    conscientiousness: Optional[float] = None
    extraversion: Optional[float] = None
    agreeableness: Optional[float] = None
    neuroticism: Optional[float] = None
    
    # Additional trait fields
    quirks: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)


class VoiceConfig(BaseModel):
    """TTS/Voice configuration."""
    enabled: Optional[bool] = None
    always_on: Optional[bool] = None
    provider: Optional[str] = None
    voice_name: Optional[str] = None
    voice_id: Optional[str] = None
    voice_sample_url: Optional[str] = None
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    tts_provider: Optional[Dict[str, Any]] = None


class WorkflowPreferences(BaseModel):
    """Workflow configuration preferences."""
    default_image_workflow: Optional[str] = None
    default_video_workflow: Optional[str] = None
    auto_generate_images: Optional[bool] = False
    auto_generate_videos: Optional[bool] = False


class CharacterCardData(BaseModel):
    """Character card data payload."""
    
    # Character Identity
    name: str
    role: str = ""
    role_type: Optional[str] = None  # assistant, companion, analyst, creative, specialist
    description: str = ""
    personality: str = ""
    scenario: str = ""
    greeting: str = ""
    example_messages: str = ""
    
    # System Prompt Configuration
    custom_system_prompt: Optional[bool] = False
    system_prompt: Optional[str] = None
    
    # Immersion Settings
    immersion_level: Optional[str] = None  # none, basic, moderate, full
    immersion_settings: Optional[Dict[str, Any]] = None
    
    # Core Memories
    core_memories: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Personality
    personality_traits: List[str] = Field(default_factory=list)
    traits: Optional[CharacterTraits] = None
    emotional_range: Optional[Dict[str, Any]] = None
    
    # Behavior Configuration  
    response_style: Optional[ResponseStyle] = None
    
    # Memory Configuration
    memory: Optional[Dict[str, Any]] = None
    memory_profile: Optional[Dict[str, Any]] = None
    
    # Feature Toggles
    image_generation: Optional[Dict[str, Any]] = None
    video_generation: Optional[Dict[str, Any]] = None
    document_analysis: Optional[Dict[str, Any]] = None
    code_execution: Optional[Dict[str, Any]] = None
    
    # Generation Parameters
    temperature: Optional[float] = None
    
    # Optional Configurations
    voice: Optional[VoiceConfig] = None
    workflows: Optional[WorkflowPreferences] = None
    
    # Metadata
    creator: str = ""
    character_version: str = "1.0"
    tags: List[str] = Field(default_factory=list)
    created_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Extensions for future features (stores preferred_llm, etc.)
    extensions: Dict[str, Any] = Field(default_factory=dict)


class ChorusCharacterCard(BaseModel):
    """Complete Chorus Engine character card structure."""
    spec: ChorusCardSpec = ChorusCardSpec.V1
    spec_version: str = "1.0"
    data: CharacterCardData


# ===========================
# SillyTavern Card Format
# ===========================

class SillyTavernSpec(str, Enum):
    """SillyTavern card specification versions."""
    V2 = "chara_card_v2"
    V3 = "chara_card_v3"


class CharacterBookEntry(BaseModel):
    """World info / lorebook entry."""
    keys: List[str] = Field(default_factory=list)
    content: str = ""
    extensions: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    insertion_order: int = 100
    case_sensitive: Optional[bool] = None
    name: Optional[str] = None
    priority: Optional[int] = None
    id: Optional[int] = None
    comment: Optional[str] = None
    selective: Optional[bool] = None
    secondary_keys: Optional[List[str]] = None
    constant: Optional[bool] = None
    position: Optional[str] = None


class CharacterBook(BaseModel):
    """Character lorebook / world info."""
    name: Optional[str] = None
    description: Optional[str] = None
    scan_depth: Optional[int] = None
    token_budget: Optional[int] = None
    recursive_scanning: Optional[bool] = None
    extensions: Dict[str, Any] = Field(default_factory=dict)
    entries: List[CharacterBookEntry] = Field(default_factory=list)


class SillyTavernCardData(BaseModel):
    """SillyTavern V2 card data structure."""
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    
    creator_notes: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: List[str] = Field(default_factory=list)
    character_book: Optional[CharacterBook] = None
    
    tags: List[str] = Field(default_factory=list)
    creator: str = ""
    character_version: str = ""
    extensions: Dict[str, Any] = Field(default_factory=dict)


class SillyTavernCard(BaseModel):
    """Complete SillyTavern V2 character card structure."""
    spec: SillyTavernSpec = SillyTavernSpec.V2
    spec_version: str = "2.0"
    data: SillyTavernCardData


# ===========================
# Import/Export DTOs
# ===========================

class CardImportResult(BaseModel):
    """Result of character card import operation."""
    character_data: Dict[str, Any]  # Parsed character config
    profile_image: bytes  # Extracted image data
    format: str  # Detected format
    warnings: List[str] = Field(default_factory=list)  # Any issues encountered


class CardExportOptions(BaseModel):
    """Options for character card export."""
    character_name: str
    include_voice: bool = True
    include_workflows: bool = True
    voice_sample_url: Optional[str] = None
