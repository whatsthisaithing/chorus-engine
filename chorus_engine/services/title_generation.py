"""Service for generating conversation titles automatically."""

import asyncio
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

from chorus_engine.models.conversation import Message, MessageRole
from chorus_engine.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TitleGenerationResult:
    """Result of title generation attempt."""
    success: bool
    title: Optional[str] = None
    error: Optional[str] = None


class TitleGenerationService:
    """
    Generates conversation titles automatically using the character's loaded LLM model.
    
    Respects ComfyUI lock to avoid VRAM conflicts - waits if ComfyUI operation in progress.
    Uses the same LLM model already loaded for the character (no model swapping).
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        min_words: int = 3,
        max_words: int = 8,
        max_chars: int = 90,
        temperature: float = 0.7
    ):
        """
        Initialize the title generation service.
        
        Args:
            llm_client: LLM client for generation
            min_words: Minimum words in title
            max_words: Maximum words in title
            max_chars: Maximum characters in title
            temperature: Temperature for generation (0.7 = creative but controlled)
        """
        self.llm_client = llm_client
        self.min_words = min_words
        self.max_words = max_words
        self.max_chars = max_chars
        self.temperature = temperature
    
    def _build_title_prompt(self, messages: List[Message], character_name: str) -> tuple[str, str]:
        """
        Build the prompt for title generation.
        
        Args:
            messages: List of messages from conversation
            character_name: Name of the character
        
        Returns:
            Tuple of (system_prompt, user_content)
        """
        # Build conversation text from messages
        conversation_lines = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else character_name
            content = msg.content[:500]  # Truncate long messages
            conversation_lines.append(f"{role}: {content}")
        
        conversation_text = "\n\n".join(conversation_lines)
        
        system_prompt = f"""You are a title generator. Your ONLY job is to create SHORT conversation titles.

RULES:
1. Generate {self.min_words}-{self.max_words} words maximum
2. Under {self.max_chars} characters total
3. Capture the main topic or theme
4. Use proper capitalization (title case)
5. NO quotes, NO punctuation at the end
6. Be specific and descriptive
7. OUTPUT ONLY THE TITLE - nothing else

GOOD EXAMPLES:
- "Planning Weekend Hiking Trip"
- "Debugging Python Memory Leak"
- "Discussing Renaissance Art History"
- "Recipe for Homemade Pizza"

BAD EXAMPLES:
- "Conversation about various topics" (too vague)
- "The user and I discussed their plans for the upcoming weekend adventure" (too long)
- "debugging" (too short, not descriptive)
- "What is your favorite color?" (question format)"""

        user_content = f"""Generate a title for this conversation:

{conversation_text}

Title:"""
        
        return system_prompt, user_content
    
    def _validate_title(self, title: str) -> Optional[str]:
        """
        Validate and clean the generated title.
        
        Args:
            title: Generated title to validate
        
        Returns:
            Cleaned title if valid, None if invalid
        """
        if not title:
            return None
        
        # Clean up the title
        title = title.strip()
        
        # Remove quotes if present
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1].strip()
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1].strip()
        
        # Remove trailing punctuation
        while title and title[-1] in '.!?,:;':
            title = title[:-1].strip()
        
        # Check length constraints
        word_count = len(title.split())
        if word_count < self.min_words or word_count > self.max_words:
            logger.warning(f"Title word count ({word_count}) outside range [{self.min_words}, {self.max_words}]: {title}")
            # Still allow it if close enough
            if word_count < self.min_words - 1 or word_count > self.max_words + 2:
                return None
        
        if len(title) > self.max_chars:
            logger.warning(f"Title too long ({len(title)} chars): {title}")
            # Truncate to max length at word boundary
            title = title[:self.max_chars].rsplit(' ', 1)[0]
        
        # Ensure it's not just generic phrases
        generic_phrases = [
            "conversation about",
            "discussion about",
            "chat about",
            "talking about",
            "various topics"
        ]
        title_lower = title.lower()
        if any(phrase in title_lower for phrase in generic_phrases):
            logger.warning(f"Title too generic: {title}")
            return None
        
        return title
    
    async def generate_title(
        self,
        messages: List[Message],
        character_name: str,
        model: str,
        comfyui_lock: Optional[asyncio.Lock] = None
    ) -> TitleGenerationResult:
        """
        Generate a title for the conversation.
        
        Args:
            messages: List of messages from conversation
            character_name: Name of the character
            model: Model name to use (should match character's loaded model)
            comfyui_lock: Optional lock to respect ComfyUI operations
        
        Returns:
            TitleGenerationResult with success status and title/error
        """
        try:
            # Check if ComfyUI operation is in progress
            if comfyui_lock and comfyui_lock.locked():
                logger.info("[TITLE GEN] ComfyUI operation in progress, waiting...")
                # Wait for lock to be available (timeout after 30 seconds)
                try:
                    async with asyncio.timeout(30.0):
                        async with comfyui_lock:
                            # Lock acquired, ComfyUI done, we can proceed
                            pass
                except asyncio.TimeoutError:
                    logger.warning("[TITLE GEN] Timeout waiting for ComfyUI lock")
                    return TitleGenerationResult(
                        success=False,
                        error="Timeout waiting for ComfyUI operation"
                    )
            
            # Build prompt
            system_prompt, user_content = self._build_title_prompt(messages, character_name)
            
            logger.info(f"[TITLE GEN] Generating title with model {model}")
            
            # Generate title using character's already-loaded model
            response = await self.llm_client.generate(
                prompt=user_content,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=30,  # Titles are short
                model=model
            )
            
            # Extract and validate title
            title_text = response.content if hasattr(response, 'content') else str(response)
            title_text = title_text.strip()
            
            # Validate and clean
            validated_title = self._validate_title(title_text)
            
            if validated_title:
                logger.info(f"[TITLE GEN] Generated title: {validated_title}")
                return TitleGenerationResult(
                    success=True,
                    title=validated_title
                )
            else:
                logger.warning(f"[TITLE GEN] Title validation failed: {title_text}")
                return TitleGenerationResult(
                    success=False,
                    error=f"Title validation failed: {title_text}"
                )
        
        except Exception as e:
            logger.error(f"[TITLE GEN] Failed to generate title: {e}", exc_info=True)
            return TitleGenerationResult(
                success=False,
                error=str(e)
            )
