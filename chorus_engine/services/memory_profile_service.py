"""
Memory Profile Service for Phase 8.

Controls memory extraction based on character immersion level and custom profiles.
"""

import logging
from typing import Dict, List, Optional
from chorus_engine.config.models import CharacterConfig, MemoryProfile
from chorus_engine.models.conversation import MemoryType

logger = logging.getLogger(__name__)


# Immersion level defaults for memory extraction
IMMERSION_DEFAULTS: Dict[str, Dict[str, bool]] = {
    "minimal": {
        # Utilitarian: facts and projects only, no stories
        "extract_facts": True,
        "extract_projects": True,
        "extract_experiences": False,
        "extract_stories": False,
        "extract_relationship": False,
    },
    "balanced": {
        # Professional roleplay: facts, projects, some experiences
        "extract_facts": True,
        "extract_projects": True,
        "extract_experiences": True,
        "extract_stories": False,
        "extract_relationship": False,
    },
    "full": {
        # Full roleplay: everything except deep relationship analysis
        "extract_facts": True,
        "extract_projects": True,
        "extract_experiences": True,
        "extract_stories": True,
        "extract_relationship": False,
    },
    "unbounded": {
        # Maximum immersion: all memory types including relationship dynamics
        "extract_facts": True,
        "extract_projects": True,
        "extract_experiences": True,
        "extract_stories": True,
        "extract_relationship": True,
    },
}


class MemoryProfileService:
    """
    Service for determining which memory types to extract based on character profile.
    
    Features:
    - Applies immersion level defaults
    - Supports per-character overrides
    - Provides type filtering for extraction and retrieval
    """
    
    def __init__(self):
        """Initialize memory profile service."""
        logger.info("Memory profile service initialized")
    
    def get_extraction_profile(self, character: CharacterConfig) -> Dict[str, bool]:
        """
        Get the complete extraction profile for a character.
        
        Merges immersion level defaults with character-specific overrides.
        
        Args:
            character: Character configuration
            
        Returns:
            Dictionary mapping extraction types to enabled/disabled
        """
        # Start with immersion level defaults
        defaults = IMMERSION_DEFAULTS.get(
            character.immersion_level,
            IMMERSION_DEFAULTS["balanced"]
        )
        
        # Apply character-specific overrides
        profile = character.memory_profile
        
        result = {
            "extract_facts": profile.extract_facts if profile.extract_facts is not None else defaults["extract_facts"],
            "extract_projects": profile.extract_projects if profile.extract_projects is not None else defaults["extract_projects"],
            "extract_experiences": profile.extract_experiences if profile.extract_experiences is not None else defaults["extract_experiences"],
            "extract_stories": profile.extract_stories if profile.extract_stories is not None else defaults["extract_stories"],
            "extract_relationship": profile.extract_relationship if profile.extract_relationship is not None else defaults["extract_relationship"],
            "track_emotional_weight": profile.track_emotional_weight,
            "track_participants": profile.track_participants,
        }
        
        logger.debug(
            f"Extraction profile for {character.id} ({character.immersion_level}): "
            f"facts={result['extract_facts']}, projects={result['extract_projects']}, "
            f"experiences={result['extract_experiences']}, stories={result['extract_stories']}, "
            f"relationship={result['extract_relationship']}"
        )
        
        return result
    
    def should_extract_type(
        self,
        memory_type: MemoryType,
        character: CharacterConfig
    ) -> bool:
        """
        Check if a specific memory type should be extracted for this character.
        
        Args:
            memory_type: The memory type to check
            character: Character configuration
            
        Returns:
            True if this type should be extracted
        """
        profile = self.get_extraction_profile(character)
        
        # Map memory types to profile fields
        type_map = {
            MemoryType.CORE: True,  # Always extract (loaded from YAML)
            MemoryType.EXPLICIT: True,  # Always extract (user-created)
            MemoryType.FACT: profile["extract_facts"],
            MemoryType.PROJECT: profile["extract_projects"],
            MemoryType.EXPERIENCE: profile["extract_experiences"],
            MemoryType.STORY: profile["extract_stories"],
            MemoryType.RELATIONSHIP: profile["extract_relationship"],
            MemoryType.EPHEMERAL: True,  # Always extract (temporary)
        }
        
        # Handle backward compatibility alias
        if memory_type == MemoryType.IMPLICIT:
            return profile["extract_facts"]
        
        return type_map.get(memory_type, False)
    
    def get_allowed_types(self, character: CharacterConfig) -> List[MemoryType]:
        """
        Get list of all memory types allowed for this character.
        
        Args:
            character: Character configuration
            
        Returns:
            List of allowed MemoryType values
        """
        profile = self.get_extraction_profile(character)
        
        allowed = [
            MemoryType.CORE,  # Always allowed
            MemoryType.EXPLICIT,  # Always allowed
        ]
        
        if profile["extract_facts"]:
            allowed.append(MemoryType.FACT)
        
        if profile["extract_projects"]:
            allowed.append(MemoryType.PROJECT)
        
        if profile["extract_experiences"]:
            allowed.append(MemoryType.EXPERIENCE)
        
        if profile["extract_stories"]:
            allowed.append(MemoryType.STORY)
        
        if profile["extract_relationship"]:
            allowed.append(MemoryType.RELATIONSHIP)
        
        # Ephemeral is usually temporary, but include for completeness
        allowed.append(MemoryType.EPHEMERAL)
        
        logger.debug(
            f"Allowed memory types for {character.id}: {[t.value for t in allowed]}"
        )
        
        return allowed
    
    def get_extraction_instructions(self, character: CharacterConfig) -> str:
        """
        Generate human-readable extraction instructions for prompts.
        
        Args:
            character: Character configuration
            
        Returns:
            Instructions string for LLM prompts
        """
        profile = self.get_extraction_profile(character)
        
        # Build list of enabled types
        enabled_types = []
        
        if profile["extract_facts"]:
            enabled_types.append("FACT (factual information)")
        
        if profile["extract_projects"]:
            enabled_types.append("PROJECT (goals, plans, ongoing work)")
        
        if profile["extract_experiences"]:
            enabled_types.append("EXPERIENCE (shared activities and events)")
        
        if profile["extract_stories"]:
            enabled_types.append("STORY (narratives and anecdotes)")
        
        if profile["extract_relationship"]:
            enabled_types.append("RELATIONSHIP (relationship dynamics and bonds)")
        
        if not enabled_types:
            return "Extract only basic factual information."
        
        instructions = f"Extract memories of the following types: {', '.join(enabled_types)}."
        
        # Add emotional tracking note
        if profile["track_emotional_weight"]:
            instructions += " Include emotional_weight (0.0-1.0) for significant moments."
        
        if profile["track_participants"]:
            instructions += " Track participants involved in memories."
        
        return instructions
    
    def format_immersion_summary(self, character: CharacterConfig) -> str:
        """
        Generate a human-readable summary of the character's memory profile.
        
        Args:
            character: Character configuration
            
        Returns:
            Summary string for display/logging
        """
        profile = self.get_extraction_profile(character)
        
        level = character.immersion_level.upper()
        enabled = []
        disabled = []
        
        type_labels = {
            "extract_facts": "Facts",
            "extract_projects": "Projects",
            "extract_experiences": "Experiences",
            "extract_stories": "Stories",
            "extract_relationship": "Relationships",
        }
        
        for key, label in type_labels.items():
            if profile[key]:
                enabled.append(label)
            else:
                disabled.append(label)
        
        summary = f"[{level}] Extracts: {', '.join(enabled)}"
        
        if disabled:
            summary += f" | Skips: {', '.join(disabled)}"
        
        return summary
