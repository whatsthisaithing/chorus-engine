"""
SillyTavern Adapter
==================

Converts between SillyTavern and Chorus Engine character formats.
"""

import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _build_personality_summary(chorus_data: Dict[str, Any]) -> str:
    """Build a personality summary for SillyTavern from Chorus character data."""
    parts = []
    
    # Include personality traits if present
    if chorus_data.get("personality_traits"):
        traits = ", ".join(chorus_data["personality_traits"])
        parts.append(f"Personality: {traits}")
    
    # Include emotional range if present
    if chorus_data.get("emotional_range"):
        emotional = chorus_data["emotional_range"]
        if emotional.get("baseline"):
            parts.append(f"Baseline mood: {emotional['baseline']}")
    
    return " | ".join(parts) if parts else ""


def _build_system_prompt_from_st(data: Dict[str, Any]) -> str:
    """
    Build a Chorus-style system prompt from SillyTavern fields.
    Combines description, personality, and system_prompt into character-specific instructions.
    Does NOT add Chorus framework guidance - that's handled by immersion_level settings.
    """
    parts = []
    
    name = data.get("name", "Character")
    description = data.get("description", "")
    personality = data.get("personality", "")
    system_prompt = data.get("system_prompt", "")
    
    # Start with character introduction using description
    if description:
        parts.append(f"You're {name}. {description}")
    else:
        parts.append(f"You're {name}.")
    
    # Add personality as behavioral guidance if present and distinct from description
    if personality and personality not in description:
        parts.append(f"\n\nPersonality & Behavior: {personality}")
    
    # Add ST's system prompt if provided (usually contains special instructions)
    if system_prompt:
        parts.append(f"\n\n{system_prompt}")
    
    return "".join(parts)


def _extract_role_summary(description: str) -> str:
    """Extract a brief role summary from description (first sentence or ~100 chars)."""
    if not description:
        return "Imported Roleplay Character"
    
    # Try to get first sentence
    match = re.match(r'^([^.!?]+[.!?])', description)
    if match:
        role = match.group(1).strip()
        if len(role) <= 150:
            return role
    
    # Otherwise truncate at word boundary
    if len(description) <= 150:
        return description
    
    truncated = description[:147]
    last_space = truncated.rfind(' ')
    if last_space > 100:
        return truncated[:last_space] + "..."
    return truncated + "..."


def _extract_personality_traits(personality: str) -> List[str]:
    """
    Extract trait keywords from SillyTavern personality field.
    Looks for common adjectives and trait descriptors.
    """
    if not personality:
        return []
    
    # Common personality trait keywords
    trait_patterns = [
        r'\b(kind|caring|compassionate|empathetic|sympathetic)\b',
        r'\b(confident|assertive|bold|brave|courageous)\b',
        r'\b(shy|timid|reserved|introverted|quiet)\b',
        r'\b(outgoing|extroverted|social|friendly|gregarious)\b',
        r'\b(intelligent|smart|clever|wise|analytical)\b',
        r'\b(creative|imaginative|artistic|innovative)\b',
        r'\b(playful|fun|cheerful|jovial|humorous)\b',
        r'\b(serious|solemn|grave|stern)\b',
        r'\b(loyal|faithful|devoted|dedicated)\b',
        r'\b(curious|inquisitive|questioning)\b',
        r'\b(calm|peaceful|serene|tranquil)\b',
        r'\b(energetic|enthusiastic|lively|vibrant)\b',
        r'\b(mysterious|enigmatic|secretive)\b',
        r'\b(protective|defensive|guardian)\b',
        r'\b(dominant|commanding|authoritative)\b',
        r'\b(submissive|obedient|compliant)\b',
        r'\b(romantic|affectionate|loving|tender)\b',
        r'\b(ambitious|driven|determined|motivated)\b',
        r'\b(patient|tolerant|understanding)\b',
        r'\b(impulsive|spontaneous|reckless)\b',
    ]
    
    traits = set()
    personality_lower = personality.lower()
    
    for pattern in trait_patterns:
        matches = re.findall(pattern, personality_lower, re.IGNORECASE)
        traits.update(matches)
    
    # Limit to 8 most relevant traits
    return sorted(list(traits))[:8]


def _convert_description_to_memories(description: str, name: str) -> List[Dict[str, Any]]:
    """
    Convert description paragraphs into core memories.
    Each paragraph becomes a memory with appropriate tags.
    """
    if not description:
        return []
    
    memories = []
    paragraphs = [p.strip() for p in description.split('\n\n') if p.strip()]
    
    # Take up to 5 paragraphs as memories
    for i, paragraph in enumerate(paragraphs[:5]):
        if len(paragraph) < 20:  # Skip very short paragraphs
            continue
        
        memory = {
            "content": paragraph,
            "tags": ["background", "imported", f"detail-{i+1}"],
            "embedding_priority": "high" if i == 0 else "medium"
        }
        memories.append(memory)
    
    return memories


class SillyTavernAdapter:
    """Convert between SillyTavern and Chorus Engine character formats."""
    
    @staticmethod
    def from_sillytavern_v2(st_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert SillyTavern V2/V3 card to Chorus Engine character format.
        
        SillyTavern cards are assumed to be roleplay characters, so we:
        - Set immersion_level to "unbounded" (full roleplay mode)
        - Set role_type to "companion" (conversational roleplay)
        - Build system_prompt from description + personality + system_prompt
        - Extract personality traits from personality field
        - Convert description paragraphs into core memories
        
        Args:
            st_data: SillyTavern card data (parsed JSON)
            
        Returns:
            Chorus Engine character configuration dict
        """
        # Extract data section (V2/V3 format) or use root (V1 format)
        data = st_data.get("data", st_data)
        
        name = data.get("name", "Unnamed Character")
        description = data.get("description", "")
        personality = data.get("personality", "")
        
        # Build Chorus character structure with intelligent mapping
        character = {
            "name": name,
            "role": _extract_role_summary(description),
            "role_type": "companion",  # ST cards are roleplay companions
            "system_prompt": _build_system_prompt_from_st(data),
            "immersion_level": "unbounded",  # Full roleplay immersion
            "scenario": data.get("scenario", ""),
            "greeting": data.get("first_mes", ""),
            "example_messages": data.get("mes_example", ""),
        }
        
        # Extract personality traits
        traits = _extract_personality_traits(personality)
        if traits:
            character["personality_traits"] = traits
        
        # Convert description to core memories
        memories = _convert_description_to_memories(description, name)
        if memories:
            character["core_memories"] = memories
        
        # Handle alternate greetings - use first as primary greeting if main is empty
        alternate_greetings = data.get("alternate_greetings", [])
        if not character["greeting"] and alternate_greetings:
            character["greeting"] = alternate_greetings[0]
            alternate_greetings = alternate_greetings[1:]
        
        # Store remaining alternate greetings in extensions
        if alternate_greetings:
            character.setdefault("extensions", {})
            character["extensions"]["alternate_greetings"] = alternate_greetings
        
        # Metadata
        character["creator"] = data.get("creator", "")
        character["tags"] = data.get("tags", [])
        
        # Store original SillyTavern fields in extensions for reference
        # This allows users to see the original data if needed
        extensions = character.setdefault("extensions", {})
        extensions["sillytavern_import"] = {
            "original_description": description,
            "original_personality": personality,
            "import_date": None  # Will be set by importer
        }
        
        if data.get("creator_notes"):
            extensions["creator_notes"] = data["creator_notes"]
        
        if data.get("system_prompt"):
            extensions["system_prompt"] = data["system_prompt"]
        
        if data.get("post_history_instructions"):
            extensions["post_history_instructions"] = data["post_history_instructions"]
        
        if data.get("character_book"):
            extensions["character_book"] = data["character_book"]
        
        # Merge any existing extensions
        if data.get("extensions"):
            extensions.update(data["extensions"])
        
        # Try to extract temperature from extensions if present
        st_extensions = data.get("extensions", {})
        if "temperature" in st_extensions:
            character["temperature"] = st_extensions["temperature"]
        
        logger.info(f"Converted SillyTavern card '{character['name']}' to Chorus format")
        return character
    
    @staticmethod
    def to_sillytavern_v2(chorus_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Chorus Engine character to SillyTavern V2 format.
        
        Args:
            chorus_data: Chorus character configuration dict
            
        Returns:
            SillyTavern V2 card data dict
        """
        # Build SillyTavern V2 structure
        st_card = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": chorus_data.get("name", "Unnamed Character"),
                # Map Chorus fields to ST equivalents
                "description": chorus_data.get("role", ""),  # Use role as brief description
                "personality": _build_personality_summary(chorus_data),  # Combine personality info
                "scenario": chorus_data.get("scenario", ""),
                "first_mes": chorus_data.get("greeting", ""),
                "mes_example": chorus_data.get("example_messages", ""),
                "creator_notes": "",
                "system_prompt": chorus_data.get("system_prompt", ""),
                "post_history_instructions": "",
                "alternate_greetings": [],
                "character_book": None,
                "tags": chorus_data.get("tags", []),
                "creator": chorus_data.get("creator", ""),
                "character_version": chorus_data.get("character_version", "1.0"),
                "extensions": {}
            }
        }
        
        # Extract fields from extensions if present
        extensions = chorus_data.get("extensions", {})
        
        if "creator_notes" in extensions:
            st_card["data"]["creator_notes"] = extensions["creator_notes"]
        
        if "system_prompt" in extensions:
            st_card["data"]["system_prompt"] = extensions["system_prompt"]
        
        if "post_history_instructions" in extensions:
            st_card["data"]["post_history_instructions"] = extensions["post_history_instructions"]
        
        if "alternate_greetings" in extensions:
            st_card["data"]["alternate_greetings"] = extensions["alternate_greetings"]
        
        if "character_book" in extensions:
            st_card["data"]["character_book"] = extensions["character_book"]
        
        # Add temperature to extensions if present
        if "temperature" in chorus_data:
            st_card["data"]["extensions"]["temperature"] = chorus_data["temperature"]
        
        # Merge remaining extensions
        st_card["data"]["extensions"].update(extensions)
        
        # Clean up - remove Chorus-specific extensions
        for key in ["creator_notes", "system_prompt", "post_history_instructions", 
                    "alternate_greetings", "character_book"]:
            st_card["data"]["extensions"].pop(key, None)
        
        logger.info(f"Converted Chorus character '{st_card['data']['name']}' to SillyTavern V2 format")
        return st_card
