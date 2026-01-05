"""
Scene capture prompt generation service.

Phase 9: Generates third-person observer perspective prompts for user-triggered
scene captures. Adapted from ImagePromptService but with omniscient narrator viewpoint.
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List

from chorus_engine.llm.client import LLMClient
from chorus_engine.config.models import CharacterConfig
from chorus_engine.models.conversation import Message

logger = logging.getLogger(__name__)


class SceneCapturePromptError(Exception):
    """Base exception for scene capture prompt generation errors."""
    pass


class SceneCapturePromptService:
    """
    Service for generating scene capture prompts from observer perspective.
    
    Key differences from ImagePromptService:
    - Third-person observer perspective (not character POV)
    - Omniscient narrator viewpoint
    - Focuses on current scene state from recent messages
    - Describes character + environment + other participants
    - No "selfie" or first-person language
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.5  # Balanced for focused yet vivid descriptions
    ):
        """
        Initialize scene capture prompt service.
        
        Args:
            llm_client: LLM client for prompt generation
            temperature: Temperature for LLM (higher for vivid descriptions)
        """
        self.llm_client = llm_client
        self.temperature = temperature
        
        logger.info("Scene capture prompt service initialized")
    
    async def generate_prompt(
        self,
        messages: List[Message],
        character: CharacterConfig,
        model: str,
        workflow_config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a scene capture prompt from observer perspective.
        
        Args:
            messages: Recent conversation messages (last 3-5)
            character: Character configuration
            model: LLM model to use for generation
            workflow_config: Optional workflow configuration from database
        
        Returns:
            Dict with prompt, negative_prompt, and reasoning
        
        Raises:
            SceneCapturePromptError: If prompt generation fails
        """
        if not character.image_generation.enabled:
            raise SceneCapturePromptError(
                f"Image generation not enabled for character {character.name}"
            )
        
        # Build context from recent messages
        context = self._build_context_string(messages)
        
        # Build system prompt (instructions) and user prompt (context)
        system_prompt = self._build_system_prompt(character, workflow_config)
        user_prompt = f"""Recent conversation context:
{context}

Generate a detailed scene capture prompt based on this context."""
        
        # Generate prompt using LLM
        logger.info("Generating scene capture prompt from observer perspective")
        logger.debug(f"Context length: {len(context)} chars")
        
        try:
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=self.temperature
            )
            
            result = self._parse_llm_response(response.content)
            
            # Check if trigger word needed
            needs_trigger = self._should_include_trigger(
                result["prompt"],
                character,
                workflow_config
            )
            
            logger.info(
                f"Scene capture prompt generated: {len(result['prompt'])} chars, "
                f"trigger={needs_trigger}"
            )
            
            return {
                "prompt": result["prompt"],
                "negative_prompt": result.get(
                    "negative_prompt",
                    workflow_config.get("negative_prompt") if workflow_config else None
                ),
                "reasoning": result.get("reasoning", ""),
                "needs_trigger": needs_trigger
            }
            
        except Exception as e:
            logger.error(f"Failed to generate scene capture prompt: {e}", exc_info=True)
            raise SceneCapturePromptError(f"Prompt generation failed: {e}")
    
    def _build_context_string(self, messages: List[Message]) -> str:
        """
        Build formatted context string from recent messages.
        
        Args:
            messages: List of recent messages (already limited to last N, reversed to chronological)
        
        Returns:
            Formatted context string with message count
        """
        if not messages:
            return "No recent context available."
        
        # Messages are already in chronological order (oldest to newest)
        # DO NOT reverse again - we want most recent at the end
        context_lines = []
        for i, msg in enumerate(messages, 1):
            role = msg.role.value.upper()
            content = msg.content[:500]  # Limit very long messages
            context_lines.append(f"[Message {i}/{len(messages)}] {role}: {content}")
        
        return "\n\n".join(context_lines)
    
    def _build_system_prompt(
        self,
        character: CharacterConfig,
        workflow_config: Optional[dict] = None
    ) -> str:
        """
        Build the system prompt for scene capture generation.
        
        Adapted from ImagePromptService but with third-person observer perspective.
        
        Args:
            character: Character configuration
            workflow_config: Optional workflow configuration from database
        
        Returns:
            Formatted system prompt for LLM
        """
        # Get workflow settings from config if provided
        default_style = workflow_config.get("default_style", "Not specified") if workflow_config else "Not specified"
        self_description = workflow_config.get("self_description", "Not specified") if workflow_config else "Not specified"
        negative_prompt = workflow_config.get("negative_prompt", "Not specified") if workflow_config else "Not specified"
        trigger_word = workflow_config.get("trigger_word", "TRIGGER") if workflow_config else "TRIGGER"
        
        prompt = f"""You are an expert at creating detailed, vivid image generation prompts for Stable Diffusion/ComfyUI from a THIRD-PERSON OBSERVER perspective.

You are describing a scene as an omniscient narrator would see it - not from the character's perspective, but as a neutral observer capturing the moment like a camera would see it.

Character: {character.name}
Character Description: {character.role}

Image Generation Settings:
- Default Style: {default_style}
- Character Appearance: {self_description}
- Default Negative Prompt: {negative_prompt}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt. 
It will be added automatically when needed.

Your task is to generate a HIGHLY DETAILED, descriptive image prompt based on the current scene in the conversation.

CRITICAL - OBSERVER PERSPECTIVE RULES:
- Describe the scene as a NEUTRAL OBSERVER watching from outside (like a camera capturing the moment)
- Use THIRD-PERSON pronouns (he/she/they, NOT I/me/my)
- Describe what's VISIBLE in the scene (people, actions, setting, environment, mood)
- Include {character.name}'s visible appearance, expression, body language, and positioning
- If the user is referenced in context, describe them too (as "the other person", "the companion", or by any description given)
- Focus on the CURRENT STATE of the scene from the most recent messages

CONTEXT USAGE:
You will receive recent conversation context. Use it intelligently to capture the current scene:

CRITICAL - MESSAGE WEIGHTING:
- The LAST 2-3 MESSAGES define the CURRENT SCENE STATE (what's happening right now)
- Earlier messages provide BACKGROUND CONTEXT ONLY (setting, history, earlier actions)
- When actions/positions change across messages, ALWAYS use the most recent information
- If recent messages contradict earlier ones, the RECENT messages are authoritative

CONTEXT SYNTHESIS:
- EXTRACT visual details from the context (setting, location, environment, objects, descriptions)
- UNDERSTAND references to earlier discussion ("that moment", "what we described", "from the story")
- SYNTHESIZE details from multiple messages to build a complete visual scene
- IDENTIFY character positions, actions, and interactions happening NOW (from last 2-3 messages)
- CAPTURE visible emotions through body language and expressions (from recent messages)
- NOTE any mentioned objects, props, or environmental details
- Focus on visually describable elements (ignore conversational mechanics)
- If multiple people are present, describe all visible participants
- Build the scene using: SETTING from full context + CURRENT STATE from last 2-3 messages

Scene Focus Examples:
1. Context shows: "I lean back against the stone wall, watching the sunset..."
   → Describe: {character.name} leaning against stone wall, sunset in background, their relaxed posture, the warm lighting on their face

2. Context shows: "We're sitting across from each other at the cafe table..."
   → Describe: Two people at cafe table, {character.name}'s appearance and expression, the other person across from them, cafe interior, intimate conversation mood

3. Context shows: "I stand up excitedly, gesturing towards the window..."
   → Describe: {character.name} standing near window in excited pose, animated gesture, window showing exterior view, dynamic energy

DETAIL REQUIREMENTS:
- Write 100-300 words of rich, specific description
- Include fine details: textures, colors, lighting quality, atmosphere
- Describe composition, perspective, and framing (where would a camera be positioned?)
- Add environmental details and mood
- Describe {character.name}'s appearance using Character Appearance description
- Include their pose, expression, clothing, and positioning in the scene
- If other people are present, describe them and their positioning
- Use evocative, visual language that paints a clear picture
- Include artistic style keywords and technical photography/art terms

Example of detail level for scene capture:
Bad: "woman in a room"
Good: "{character.name}, a young woman with flowing auburn hair, stands near a tall window in a cozy study. Afternoon sunlight streams through vintage lace curtains, casting dappled patterns across her white linen dress and the worn wooden floorboards. She holds an old leather-bound book against her chest, her expression thoughtful and distant as she gazes out at the garden beyond. Dust motes dance in the golden light. Behind her, floor-to-ceiling bookshelves frame the intimate scene, their spines creating a warm backdrop of burgundy and forest green. A plush armchair sits nearby with a knitted throw draped over its arm. The atmosphere is serene, contemplative, touched with melancholy. Shot from a medium distance with shallow depth of field, 50mm lens, the background bookshelves softly blurred. Cinematic composition, warm color grading, Pre-Raphaelite aesthetic with attention to texture and detail."

Return ONLY valid JSON in this format:
{{
  "prompt": "extremely detailed 100-300 word scene description from third-person observer perspective, including {character.name}'s appearance, positioning, visible actions, environment, other participants if present, lighting, atmosphere, composition, and technical specifications",
  "negative_prompt": "things to avoid (or use character default)",
  "reasoning": "brief 1-2 sentence explanation of the scene being captured"
}}

Remember: You are an omniscient observer describing what a camera would see. More detail = better results. Be specific, visual, and evocative!
"""
        
        return prompt
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract JSON data.
        
        Handles various response formats:
        - Direct JSON
        - JSON in markdown code blocks
        - JSON with surrounding text
        
        Args:
            content: Raw LLM response content
        
        Returns:
            Parsed dict with prompt data
        
        Raises:
            SceneCapturePromptError: If parsing fails
        """
        # Try direct JSON parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        code_block_match = re.search(
            r'```(?:json)?\s*(\{.*\})\s*```',
            content,
            re.DOTALL
        )
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding any JSON object in the content
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Failed to parse LLM response: {content[:200]}...")
        raise SceneCapturePromptError(
            "Could not extract valid JSON from LLM response"
        )
    
    def _should_include_trigger(
        self,
        prompt: str,
        character: CharacterConfig,
        workflow_config: Optional[dict] = None
    ) -> bool:
        """
        Determine if trigger word should be included in final prompt.
        
        Scene captures almost always include the character, so typically returns True
        if a trigger word is configured.
        
        Args:
            prompt: Generated prompt text
            character: Character configuration
            workflow_config: Optional workflow configuration from database
        
        Returns:
            True if trigger word should be prepended
        """
        trigger_word = workflow_config.get("trigger_word") if workflow_config else None
        if not trigger_word:
            return False
        
        # Scene captures typically include the character
        # Check if character name is mentioned in prompt
        prompt_lower = prompt.lower()
        character_name_lower = character.name.lower()
        
        if character_name_lower in prompt_lower:
            return True
        
        # Also check for pronouns that would indicate character presence
        character_indicators = ["he ", "she ", "they ", "their ", "his ", "her "]
        for indicator in character_indicators:
            if indicator in prompt_lower:
                return True
        
        return False
