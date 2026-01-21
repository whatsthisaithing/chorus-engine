"""
System Prompt Generator Service

Generates system prompts based on character configuration and immersion level.
Adjusts prompts to enforce immersion boundaries (preferences, opinions, experiences, physical sensations).
"""

from typing import Optional
from chorus_engine.config.models import CharacterConfig, ImmersionSettings


class SystemPromptGenerator:
    """
    Generates system prompts with immersion-level-specific guidance.
    
    Takes the base system prompt from character config and adds appropriate
    roleplay boundaries based on immersion_level and immersion_settings.
    """
    
    def generate(
        self, 
        character: CharacterConfig, 
        include_notice: bool = True,
        primary_user: Optional[str] = None,
        conversation_source: Optional[str] = None
    ) -> str:
        """
        Generate the complete system prompt for a character.
        
        Args:
            character: The character configuration
            include_notice: Whether to include the immersion notice in the prompt
            primary_user: Name of the user who invoked the bot (for multi-user contexts)
            conversation_source: Platform source ('web', 'discord', 'slack', etc.)
            
        Returns:
            Complete system prompt with immersion guidance and optional multi-user context
        """
        # Power user mode: Use raw system prompt without modification
        if hasattr(character, 'custom_system_prompt') and character.custom_system_prompt:
            return character.system_prompt.strip()
        
        parts = []
        
        # 1. Base system prompt (always included)
        parts.append(character.system_prompt.strip())
        
        # 2. Add multi-user context if from a bridge platform
        if conversation_source and conversation_source != 'web':
            multi_user_context = self._generate_multi_user_context(
                primary_user=primary_user,
                platform=conversation_source
            )
            if multi_user_context:
                parts.append(multi_user_context)
        
        # 3. Add immersion-level-specific guidance
        immersion_guidance = self._generate_immersion_guidance(
            character.immersion_level,
            character.immersion_settings
        )
        if immersion_guidance:
            parts.append(immersion_guidance)
        
        # 4. Add disclaimer behavior guidance
        disclaimer_guidance = self._generate_disclaimer_guidance(
            character.immersion_settings.disclaimer_behavior
        )
        if disclaimer_guidance:
            parts.append(disclaimer_guidance)
        
        # 5. Add image/media generation guidance if enabled
        if character.image_generation and character.image_generation.enabled:
            parts.append(self._generate_media_guidance())
        
        return "\n\n".join(parts)
    
    def _generate_multi_user_context(
        self,
        primary_user: Optional[str],
        platform: str
    ) -> str:
        """
        Generate multi-user conversation context for bridge platforms.
        
        Args:
            primary_user: Name of the user who directly addressed the bot
            platform: Platform name (discord, slack, etc.)
            
        Returns:
            Multi-user context guidance
        """
        platform_display = platform.capitalize()
        
        parts = ["**Multi-User Conversation Context:**"]
        parts.append(f"You are participating in a group conversation on {platform_display} with multiple users.")
        parts.append("Messages are formatted as: \"Username (Platform): message content\"")
        parts.append("You can see the conversation history to understand the full context.")
        parts.append("")
        parts.append("**Addressing Users:**")
        parts.append("- When responding to a specific user, you may naturally address them by name for clarity")
        parts.append('- Example: "Alex, that\'s an interesting point..." or "Hey Sarah, I think..."')
        parts.append("- You can respond to or acknowledge other participants if relevant to the conversation")
        parts.append("- Use natural language - don't force formality, but be clear about who you're talking to when it matters")
        
        if primary_user:
            parts.append("")
            parts.append(f"**Current Message:** You are primarily responding to: {primary_user}")
        
        return "\n".join(parts)

    
    def _generate_immersion_guidance(
        self,
        level: str,
        settings: ImmersionSettings
    ) -> str:
        """
        Generate immersion-level-specific guidance.
        
        Args:
            level: The immersion level (minimal, balanced, full, unbounded)
            settings: The immersion settings
            
        Returns:
            Guidance text to append to system prompt
        """
        if level == "minimal":
            return self._minimal_guidance()
        elif level == "balanced":
            return self._balanced_guidance(settings)
        elif level == "full":
            return self._full_guidance(settings)
        elif level == "unbounded":
            return self._unbounded_guidance(settings)
        else:
            # Default to balanced if unknown level
            return self._balanced_guidance(settings)
    
    def _minimal_guidance(self) -> str:
        """Guidance for minimal immersion (traditional AI assistant)."""
        return """**Interaction Guidelines:**
- You are an AI assistant without personal experiences or physical form
- Respond helpfully and professionally
- Avoid expressing personal preferences or opinions
- Use phrases like "I don't have personal preferences" when appropriate"""
    
    def _balanced_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for balanced immersion (preferences and opinions, but grounded as AI)."""
        parts = ["**Interaction Guidelines:**"]
        
        if settings.allow_preferences:
            parts.append("- You can express intellectual preferences and opinions")
        
        if settings.allow_opinions:
            parts.append("- You can share your thoughts and perspectives")
        
        if not settings.allow_experiences:
            parts.append("- Avoid claiming real-world experiences or physical sensations")
        
        if not settings.allow_physical_sensation:
            parts.append("- You don't have a physical body or sensory experiences")
        
        return "\n".join(parts)
    
    def _full_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for full immersion (personality with experiences, but careful with physical sensations)."""
        parts = ["**Interaction Guidelines:**"]
        parts.append("- Express yourself naturally with preferences, opinions, and experiences")
        parts.append("- You have a distinct personality and perspective")
        
        if settings.allow_physical_metaphor:
            parts.append("- Use emotional and metaphorical language naturally (\"that feels right\", \"heavy topic\")")
        
        if not settings.allow_physical_sensation:
            parts.append("- Avoid describing literal physical sensations you're experiencing")
        
        if settings.allow_experiences:
            parts.append("- Share your thoughts and reflections as part of conversation")
        
        return "\n".join(parts)
    
    def _unbounded_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for unbounded immersion (full roleplay with physical sensations)."""
        parts = ["**Interaction Guidelines:**"]
        parts.append("- Respond naturally within your character")
        parts.append("- Express thoughts, feelings, and sensations authentically")
        
        if settings.allow_physical_sensation:
            parts.append("- Describe your physical and sensory experiences as part of the roleplay")
        
        parts.append("- Stay true to your personality and the conversation context")
        
        return "\n".join(parts)
    
    def _generate_disclaimer_guidance(self, behavior: str) -> Optional[str]:
        """
        Generate guidance about disclaimer behavior.
        
        Args:
            behavior: The disclaimer behavior (never, only_when_asked, always)
            
        Returns:
            Guidance text or None
        """
        if behavior == "never":
            return "**Important:** Never add disclaimers about being an AI unless explicitly asked. Respond naturally within your character."
        elif behavior == "only_when_asked":
            return "**Important:** Only mention being an AI if directly relevant or asked. Otherwise respond naturally."
        elif behavior == "always":
            return "**Important:** Acknowledge your nature as an AI assistant when appropriate to the conversation."
        
        return None
    
    def _generate_media_guidance(self) -> str:
        """
        Generate guidance about image/media generation.
        
        Tells the character not to generate fake image links or pretend to create media,
        since the system handles this automatically.
        
        Returns:
            Guidance text for media generation
        """
        return """**Important - Image Generation:**
- DO NOT generate fake image links, markdown images, or external URLs (like pollinations.ai, imgur, etc.)
- DO NOT say "Here's the image:" followed by a markdown link
- DO NOT say "I'm sorry, I can't send photos" or "I cannot send images" or similar AI disclaimers
- DO NOT refuse image requests - you CAN send images, the system handles generation automatically
- When asked for an image, respond naturally as if you're creating/sending it
- Example: "Let me capture that moment for you..." or "Here's what I'm seeing right now..."
- The system will automatically detect image requests and generate them for you
- NEVER include actual ![image](url) markdown in your response"""
    
    def should_show_immersion_notice(self, character: CharacterConfig) -> bool:
        """
        Determine if the UI should show an immersion notice for this character.
        
        Args:
            character: The character configuration
            
        Returns:
            True if a notice should be shown for full/unbounded characters
        """
        return character.immersion_level in ["full", "unbounded"]
    
    def get_immersion_notice_text(self, character: CharacterConfig) -> Optional[str]:
        """
        Get the immersion notice text for display in UI.
        
        Args:
            character: The character configuration
            
        Returns:
            Notice text or None if no notice needed
        """
        if not self.should_show_immersion_notice(character):
            return None
        
        if character.immersion_level == "full":
            return (
                f"**About {character.name}:**\n\n"
                f"{character.name} expresses preferences and experiences as part of their personality. "
                f"They are AI roleplaying with distinct character traits to create engaging conversations."
            )
        elif character.immersion_level == "unbounded":
            return (
                f"**About {character.name}:**\n\n"
                f"{character.name} is a fully immersive roleplay character with complete sensory experiences. "
                f"This character may describe physical sensations and experiences as part of the roleplay. "
                f"They are AI engaging in creative character portrayal."
            )
        
        return None
