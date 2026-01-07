"""
Image prompt generation service.

Phase 5.2: Generates appropriate prompts for ComfyUI image generation
using LLM to analyze conversation context.
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List

from chorus_engine.llm.client import LLMClient
from chorus_engine.config.models import CharacterConfig, ImageGenerationConfig

logger = logging.getLogger(__name__)


class ImagePromptError(Exception):
    """Base exception for image prompt generation errors."""
    pass


class ImagePromptService:
    """
    Service for generating image prompts based on conversation context.
    
    Handles:
    - LLM-based prompt generation from context
    - Trigger word detection and injection
    - Style and negative prompt application
    - Character self-portrait detection
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.3  # Lower for more consistent prompts
    ):
        """
        Initialize image prompt service.
        
        Args:
            llm_client: LLM client for prompt generation
            temperature: Temperature for LLM (lower = more consistent)
        """
        self.llm_client = llm_client
        self.temperature = temperature
        
        logger.info("Image prompt service initialized")
    
    async def detect_image_request(
        self,
        message: str,
        character: CharacterConfig
    ) -> bool:
        """
        Detect if a user message is requesting an image.
        
        Uses keyword detection for fast, reliable detection without LLM call.
        
        Args:
            message: User's message text
            character: Character configuration
        
        Returns:
            True if message appears to request an image
        """
        if not character.image_generation.enabled:
            return False
        
        message_lower = message.lower()
        
        # Keywords that indicate image request
        # NOTE: These should stay in sync with KeywordIntentDetector.IMAGE_KEYWORDS
        image_keywords = [
            "show me",
            "can you show",
            "picture",
            "image",
            "photo",
            "selfie",
            "snapshot",
            "portrait",
            "drawing",
            "sketch",
            "illustration",
            "draw",
            "generate",
            "create an image",
            "what do you look like",
            "what does",
            "look like",
            "appearance",
            "visualize",
            "illustrate"
        ]
        
        # Check for keywords
        for keyword in image_keywords:
            if keyword in message_lower:
                logger.debug(f"Image request detected: found '{keyword}'")
                return True
        
        return False
    
    async def generate_prompt(
        self,
        user_request: str,
        character: CharacterConfig,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        workflow_config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Generate an image prompt based on user request and context.
        
        Args:
            user_request: The user's request for an image
            character: Character configuration
            conversation_context: Optional recent conversation messages
            model: Optional model to use for prompt generation (defaults to character's preferred model)
            workflow_config: Optional workflow configuration from database
        
        Returns:
            Dictionary with:
                - prompt: Generated image prompt
                - negative_prompt: Negative prompt
                - needs_trigger: Whether trigger word should be included
                - reasoning: Brief explanation
        
        Raises:
            ImagePromptError: If prompt generation fails
        """
        if not character.image_generation.enabled:
            raise ImagePromptError(f"Image generation not enabled for character {character.id}")
        
        # Build context string
        context_str = self._build_context_string(conversation_context) if conversation_context else "No context"
        
        # Build system prompt
        system_prompt = self._build_system_prompt(character, workflow_config)
        
        # Build user prompt with emphasis on the actual request
        user_prompt = f"""USER'S IMAGE REQUEST: {user_request}

RECENT CONVERSATION CONTEXT:
{context_str}

Generate an image prompt based on the USER'S IMAGE REQUEST above. Focus on creating a detailed, vivid prompt that captures what the user is asking for."""
        
        try:
            # Call LLM with character's preferred model (if available)
            logger.info(f"[IMAGE PROMPT SERVICE] Calling LLM with model: {model}")
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                model=model
            )
            logger.info(f"[IMAGE PROMPT SERVICE] LLM returned, used model: {response.model}")
            
            # Parse JSON response
            result = self._parse_llm_response(response.content)
            
            # Detect if trigger word is needed
            needs_trigger = self._should_include_trigger(
                user_request=user_request,
                generated_prompt=result["prompt"],
                character=character
            )
            
            result["needs_trigger"] = needs_trigger
            
            # Apply default negative prompt if not provided
            if not result.get("negative_prompt") and workflow_config:
                result["negative_prompt"] = workflow_config.get("negative_prompt")
            
            logger.info(f"Generated prompt for {character.id}: {result['prompt'][:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate image prompt: {e}")
            raise ImagePromptError(f"Prompt generation failed: {str(e)}")
    
    def _build_system_prompt(self, character: CharacterConfig, workflow_config: dict = None) -> str:
        """Build system prompt for LLM.
        
        Args:
            character: Character configuration
            workflow_config: Optional workflow configuration from database
        """
        # Use workflow config if provided, otherwise use placeholders
        default_style = workflow_config.get("default_style", "Not specified") if workflow_config else "Not specified"
        self_description = workflow_config.get("self_description", "Not specified") if workflow_config else "Not specified"
        negative_prompt = workflow_config.get("negative_prompt", "Not specified") if workflow_config else "Not specified"
        trigger_word = workflow_config.get("trigger_word", "TRIGGER") if workflow_config else "TRIGGER"
        
        prompt = f"""You are an expert at creating detailed, vivid image generation prompts for Stable Diffusion/ComfyUI.

Character: {character.name}
Character Description: {character.role}

Image Generation Settings:
- Default Style: {default_style}
- Character Appearance: {self_description}
- Default Negative Prompt: {negative_prompt}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt. 
It will be added automatically when needed.

Your task is to generate a HIGHLY DETAILED, descriptive image prompt based on the user's request.

CONTEXT USAGE:
You will receive recent conversation context. Use it intelligently to enhance your image prompt:
- EXTRACT visual details mentioned in the context (scenes, objects, settings, descriptions, stories)
- UNDERSTAND references ("that moment", "what you described", "from the story", "your favorite")
- FOCUS on visually describable elements (ignore conversational mechanics and meta-discussion)
- SYNTHESIZE details from multiple messages if the user references earlier discussion
- If the user says "show me X from [earlier topic]", pull ALL relevant visual details from context
- When a story or description was shared, extract specific visual elements (locations, objects, people, actions, atmosphere)

CRITICAL - CHARACTER DEPICTION RULES:
When the user requests an image involving the character (you/yourself):
- ALWAYS depict the character at their CURRENT age and appearance (use Character Appearance description above)
- Even if the request references a past memory/story, show the character NOW unless explicitly asked for a historical depiction
- Example: "show me that moment from your childhood story" → Show current character in a scene inspired by the story, NOT as a child
- Example: "photo of you eating ramen" → Show current character appearance eating ramen
- Only depict the character at a different age/appearance if explicitly requested: "show yourself as a child in that scene"
- When in doubt, default to current character appearance

Context Usage Examples:
1. Context: "ASSISTANT: I remember climbing the old oak tree as a child and watching the sunset..."
   Request: "Show me that sunset moment"
   → Show CURRENT character sitting in/near an oak tree at sunset, capturing the nostalgic mood (NOT a child)
   
2. Context: "USER: Tell me about your favorite place / ASSISTANT: My favorite place is a hidden beach with black sand..."
   Request: "Can you show me that beach?"
   → Extract: black sand beach, turquoise water, volcanic cliffs, hidden/secluded atmosphere

3. Context shows NO relevant visual details
   Request: "Draw a cat"
   → Use general knowledge and character style, no context needed

DETAIL REQUIREMENTS:
- Write 100-300 words of rich, specific description
- Include fine details: textures, colors, lighting quality, atmosphere
- Describe composition, perspective, and framing
- Add environmental details and mood
- If depicting the character, describe their appearance, pose, expression, clothing, and surroundings
- Use evocative, visual language that paints a clear picture
- Include artistic style keywords and technical photography/art terms

Example of detail level:
Bad: "woman in a garden"
Good: "A young woman stands in an enchanted garden at golden hour, soft sunlight filtering through ancient oak trees and casting dappled shadows across her white linen dress. Her hair flows freely in the gentle breeze. She holds a leather-bound book, her expression thoughtful and serene. Wildflowers in vibrant purples and yellows surround her feet, while butterflies dance in the warm, hazy air. The background shows a stone archway covered in climbing roses, slightly out of focus. Painted in the style of Pre-Raphaelite art, with rich colors, intricate details, and romantic lighting. Shot with shallow depth of field, 85mm lens, soft bokeh."

Return ONLY valid JSON in this format:
{{
  "prompt": "extremely detailed 100-300 word image description with comprehensive visual details, style, lighting, composition, atmosphere, and technical specifications",
  "negative_prompt": "things to avoid (or use character default)",
  "reasoning": "brief 1-2 sentence explanation of the image concept"
}}

Remember: More detail = better results. Be specific, visual, and evocative!
"""
        
        return prompt
    
    def _build_context_string(self, messages: List[Dict[str, str]], max_messages: int = 3) -> str:
        """Build context string from recent messages (limited to avoid context pollution)."""
        recent = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context_lines = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_lines.append(f"{role}: {content[:150]}")
        
        return "\n".join(context_lines)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # Extract content between ``` markers
                match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            result = json.loads(cleaned)
            
            # Validate required fields
            if "prompt" not in result:
                raise ValueError("Response missing 'prompt' field")
            
            return {
                "prompt": result["prompt"],
                "negative_prompt": result.get("negative_prompt", ""),
                "reasoning": result.get("reasoning", "")
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response, using fallback: {e}")
            
            # Fallback: extract prompt from text
            prompt = response_text.strip()
            if len(prompt) > 500:
                prompt = prompt[:500]
            
            return {
                "prompt": prompt,
                "negative_prompt": "",
                "reasoning": "Fallback extraction"
            }
    
    def _should_include_trigger(
        self,
        user_request: str,
        generated_prompt: str,
        character: CharacterConfig
    ) -> bool:
        """
        Determine if character trigger word should be included.
        
        Checks if the image is of the character themselves.
        
        Args:
            user_request: Original user request
            generated_prompt: Generated prompt
            character: Character config
        
        Returns:
            True if trigger word should be included
        """
        # Keywords that indicate character self-portrait
        self_references = [
            "yourself",
            "you look",
            "your appearance",
            "what you look like",
            "self-portrait",
            "selfie",
            "picture of you",
            "image of you",
            character.name.lower()
        ]
        
        # Check user request
        request_lower = user_request.lower()
        for ref in self_references:
            if ref in request_lower:
                logger.debug(f"Trigger word needed: found '{ref}' in request")
                return True
        
        # Check generated prompt
        prompt_lower = generated_prompt.lower()
        for ref in self_references:
            if ref in prompt_lower:
                logger.debug(f"Trigger word needed: found '{ref}' in prompt")
                return True
        
        return False
    
    def inject_trigger_word(
        self,
        prompt: str,
        character: CharacterConfig,
        workflow_config: Optional[dict] = None
    ) -> str:
        """
        Inject character trigger word into prompt.
        
        Args:
            prompt: Original prompt
            character: Character config
            workflow_config: Optional workflow config from database
        
        Returns:
            Prompt with trigger word injected at the start
        """
        trigger = workflow_config.get("trigger_word") if workflow_config else None
        
        if not trigger:
            return prompt
        
        # Check if trigger already present (case-insensitive)
        if trigger.lower() in prompt.lower():
            logger.debug("Trigger word already present in prompt")
            return prompt
        
        # Inject at start
        injected = f"{trigger}, {prompt}"
        logger.debug(f"Injected trigger word '{trigger}' into prompt")
        
        return injected
    
    def apply_style(
        self,
        prompt: str,
        character: CharacterConfig,
        workflow_config: Optional[dict] = None
    ) -> str:
        """
        Apply character's default style to prompt.
        
        Args:
            prompt: Original prompt
            character: Character config
            workflow_config: Optional workflow config from database
        
        Returns:
            Prompt with style applied
        """
        style = workflow_config.get("default_style") if workflow_config else None
        
        if not style:
            return prompt
        
        # Check if style keywords already present
        style_keywords = style.lower().split(',')
        prompt_lower = prompt.lower()
        
        # If some style keywords present, don't duplicate
        if any(kw.strip() in prompt_lower for kw in style_keywords):
            logger.debug("Style keywords already present")
            return prompt
        
        # Append style
        styled = f"{prompt}, {style}"
        logger.debug(f"Applied style: {style}")
        
        return styled
    
    def prepare_final_prompt(
        self,
        base_prompt: str,
        character: CharacterConfig,
        include_trigger: bool = False,
        apply_character_style: bool = True,
        workflow_config: Optional[dict] = None
    ) -> str:
        """
        Prepare final prompt with all enhancements.
        
        Args:
            base_prompt: Base prompt text
            character: Character config
            include_trigger: Whether to include trigger word
            apply_character_style: Whether to apply character style
            workflow_config: Optional workflow config from database
        
        Returns:
            Final enhanced prompt
        """
        final_prompt = base_prompt
        
        # Apply trigger word if needed
        if include_trigger:
            final_prompt = self.inject_trigger_word(final_prompt, character, workflow_config)
        
        # Apply character style if requested
        if apply_character_style:
            final_prompt = self.apply_style(final_prompt, character, workflow_config)
        
        return final_prompt
    
    async def generate_from_context(
        self,
        character: CharacterConfig,
        conversation_context: List[Dict[str, str]],
        explicit_request: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze context and suggest an image if appropriate.
        
        Args:
            character: Character config
            conversation_context: Recent conversation messages
            explicit_request: Optional explicit image request
        
        Returns:
            Prompt dict if image suggested, None otherwise
        """
        # For Phase 5, we only generate on explicit request
        # Future: Could add proactive image suggestions
        
        if explicit_request:
            return await self.generate_prompt(
                user_request=explicit_request,
                character=character,
                conversation_context=conversation_context
            )
        
        return None
