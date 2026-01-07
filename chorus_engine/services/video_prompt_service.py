"""
Video prompt generation service.

Generates motion-focused prompts for video generation using LLM.
Similar to ImagePromptService but emphasizes dynamic action and temporal progression.
"""

import logging
from typing import List, Optional, Dict, Any
from chorus_engine.llm.client import LLMClient
from chorus_engine.models.conversation import Message, MessageRole
from chorus_engine.config.models import CharacterConfig

logger = logging.getLogger(__name__)


class VideoPromptService:
    """
    Service for generating video prompts from conversation context.
    
    Analyzes conversation history and creates motion-focused prompts
    for video generation. Emphasizes dynamic action, movement, and
    temporal progression.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.3
    ):
        """
        Initialize video prompt service.
        
        Args:
            llm_client: LLM client for prompt generation
            temperature: Temperature for LLM (lower = more consistent)
        """
        self.llm_client = llm_client
        self.temperature = temperature
        
        logger.info("Video prompt service initialized")
    
    def _build_system_prompt(self, character: CharacterConfig, workflow_config: Optional[dict] = None) -> str:
        """
        Build system prompt for LLM.
        
        Args:
            character: Character configuration
            workflow_config: Optional workflow configuration from database
        """
        # Use workflow config if provided, otherwise use placeholders
        default_style = workflow_config.get("default_style", "Not specified") if workflow_config else "Not specified"
        self_description = workflow_config.get("self_description", "Not specified") if workflow_config else "Not specified"
        negative_prompt = workflow_config.get("negative_prompt", "Not specified") if workflow_config else "Not specified"
        trigger_word = workflow_config.get("trigger_word", "TRIGGER") if workflow_config else "TRIGGER"
        
        return f"""You are a video generation prompt engineer. Your job is to create detailed, motion-focused prompts for video generation based on conversation context.

Character: {character.name}
Character Description: {character.role}

Video Generation Settings:
- Default Style: {default_style}
- Character Appearance: {self_description}
- Default Negative Prompt: {negative_prompt}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt.
It will be added automatically when needed.

Your task is to generate a MOTION-FOCUSED, dynamic video prompt based on the user's request.

CONTEXT USAGE:
You will receive recent conversation context. Use it intelligently to enhance your video prompt:
- EXTRACT motion-related details mentioned in the context (actions, movements, scenes with activity)
- UNDERSTAND references ("that moment", "what you described", "from the story", "your favorite")
- FOCUS on describable motion and action (ignore conversational mechanics and meta-discussion)
- SYNTHESIZE details from multiple messages if the user references earlier discussion
- If the user says "show me X from [earlier topic]", pull ALL relevant motion/action details from context
- When a story or description was shared, extract specific dynamic elements (actions, movements, changes, progression)

CRITICAL - CHARACTER DEPICTION RULES:
When the user requests a video involving the character (you/yourself):
- ALWAYS depict the character at their CURRENT age and appearance (use Character Appearance description above)
- Even if the request references a past memory/story, show the character NOW unless explicitly asked for a historical depiction
- Example: "show me that moment from your childhood story" → Show current character in a scene inspired by the story, NOT as a child
- Example: "video of you walking on the beach" → Show current character appearance walking on beach
- Only depict the character at a different age/appearance if explicitly requested: "show yourself as a child in that scene"
- When in doubt, default to current character appearance

CRITICAL RULES:
1. Focus on MOTION, DYNAMIC ACTION, and TEMPORAL PROGRESSION
2. Describe what HAPPENS in the video, not just what it looks like
3. Include camera movement if relevant (pan, zoom, tracking shot, crane shot, dolly zoom)
4. Specify timing/pacing when appropriate (slow motion, quick cuts, smooth transition)
5. Maximum 150 words - be concise yet vivid
6. Use present tense and active verbs (flows, moves, transforms, swirls, drifts, cascades)
7. DO NOT include dialogue or text that would render as on-screen captions
8. DO NOT use quotation marks around the prompt itself

GOOD VIDEO PROMPTS (emphasize action):
- A cat leaps gracefully through the air, paws extended, landing softly on a windowsill as sunlight streams through
- Ocean waves crash against rocky cliffs in slow motion, water spraying upward and catching golden hour light
- Camera slowly orbits around a steaming cup of coffee, revealing swirling cream patterns forming and dissolving
- Leaves tumble and spiral through autumn air, dancing in wind currents, casting moving shadows on the ground

BAD VIDEO PROMPTS (too static):
- A beautiful landscape with mountains and trees (no action!)
- A portrait of a person smiling (no movement!)
- A still shot of a building (explicitly static!)

Extract the essence of what should MOVE and CHANGE in the scene, then describe that motion vividly.

Return ONLY valid JSON in this format:
{{
  "prompt": "detailed motion-focused video description (100-300 words) with action verbs, camera movement, and temporal progression",
  "negative_prompt": "things to avoid in the video (static shots, jerky motion, poor lighting, etc.)",
  "reasoning": "brief 1-2 sentence explanation of the video concept"
}}

Remember: Motion and action are key! Describe what MOVES and CHANGES in the scene."""
    
    async def generate_video_prompt(
        self,
        messages: List[Message],
        character: CharacterConfig,
        character_name: str,
        custom_instruction: Optional[str] = None,
        trigger_words: Optional[List[str]] = None,
        model: Optional[str] = None,
        workflow_config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Generate video prompt from conversation context.
        
        Args:
            messages: Recent conversation history
            character: Character configuration
            character_name: Name of character in conversation
            custom_instruction: Optional user-provided prompt guidance
            trigger_words: Optional trigger words for workflow
            model: Optional model to use for generation
            workflow_config: Optional workflow configuration from database
        
        Returns:
            Dictionary with:
                - prompt: Generated video prompt
                - negative_prompt: Negative prompt for video generation
                - needs_trigger: Whether trigger word should be included
                - reasoning: Brief explanation
        
        Raises:
            Exception: Failed to generate prompt
        """
        try:
            # Build system prompt with character/workflow context
            system_prompt = self._build_system_prompt(character, workflow_config)
            
            # Build user message with context
            user_message = self._build_context_message(
                messages,
                character_name,
                custom_instruction,
                trigger_words
            )
            
            # Generate prompt via LLM
            logger.info(f"[VIDEO PROMPT SERVICE] Calling LLM with model: {model}")
            response = await self.llm_client.generate(
                prompt=user_message,
                system_prompt=system_prompt,
                temperature=self.temperature,  # Use configured temperature
                max_tokens=300,  # Enough for detailed motion description
                model=model  # Use specified model
            )
            logger.info(f"[VIDEO PROMPT SERVICE] LLM returned, used model: {response.model}")
            
            # Parse JSON response
            result = self._parse_llm_response(response.content)
            
            # Determine if trigger words are needed
            needs_trigger = bool(trigger_words)
            
            # Prepend trigger words to prompt if provided
            prompt = result["prompt"]
            if trigger_words:
                trigger_str = ", ".join(trigger_words)
                prompt = f"{trigger_str}, {prompt}"
            
            # Build return dict matching image service structure
            result["prompt"] = prompt
            result["needs_trigger"] = needs_trigger
            
            logger.info(f"Generated video prompt: {prompt[:100]}...")
            return result
        
        except Exception as e:
            logger.error(f"Failed to generate video prompt: {e}")
            raise
    
    def _build_context_message(
        self,
        messages: List[Message],
        character_name: str,
        custom_instruction: Optional[str] = None,
        trigger_words: Optional[List[str]] = None
    ) -> str:
        """
        Build context message for LLM prompt generation.
        
        Args:
            messages: Conversation messages
            character_name: Character name
            custom_instruction: User instruction
            trigger_words: Workflow trigger words
        
        Returns:
            Formatted context string
        """
        parts = []
        
        # Get the LATEST user message (the actual video request)
        latest_user_message = None
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                latest_user_message = msg.content
                break
        
        # Add recent messages for context (last 3 for context, matching images)
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        # Build the prompt with emphasis on the latest request
        if latest_user_message:
            parts.append(f"USER'S VIDEO REQUEST: {latest_user_message}")
            parts.append("\nRECENT CONVERSATION CONTEXT:")
        else:
            parts.append("CONVERSATION CONTEXT:")
        
        for msg in recent_messages:
            role = "User" if msg.role == MessageRole.USER else character_name
            parts.append(f"{role}: {msg.content}")
        
        # Add custom instruction if provided
        if custom_instruction:
            parts.append(f"\nUSER INSTRUCTION: {custom_instruction}")
        
        # Add trigger word guidance
        if trigger_words:
            parts.append(f"\nREQUIRED TRIGGER WORDS: {', '.join(trigger_words)}")
            parts.append("(Include these trigger words naturally in the prompt)")
        
        parts.append("\nGenerate a video prompt based on the USER'S VIDEO REQUEST above. Focus on MOTION and DYNAMIC ACTION:")
        
        return "\n".join(parts)
    
    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse LLM response into structured data."""
        import json
        import re
        
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            
            # Try to extract JSON from markdown code blocks
            if "```" in cleaned:
                # Try to find JSON between ``` markers (with or without closing marker)
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*(?:```|$)', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                else:
                    # Try to find any JSON object, ignoring markdown
                    match = re.search(r'(\{[^`]*?\})', cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1)
            
            # Try to parse as JSON
            result = json.loads(cleaned)
            
            # Validate required fields
            if "prompt" not in result:
                raise ValueError("Missing 'prompt' field in LLM response")
            
            # Set defaults for optional fields
            if "reasoning" not in result:
                result["reasoning"] = "Video prompt generated from context"
            if "negative_prompt" not in result:
                result["negative_prompt"] = ""
            
            return result
            
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse JSON from LLM response, attempting fallback: {e}")
            logger.debug(f"Raw response: {response_text[:200]}...")
            
            # Fallback: extract just the prompt field if possible
            cleaned = response_text.strip()
            
            # Try to extract prompt value from malformed JSON
            prompt_match = re.search(r'"prompt"\s*:\s*"([^"]+)"', cleaned, re.DOTALL)
            if prompt_match:
                extracted_prompt = prompt_match.group(1)
                logger.info("Extracted prompt from malformed JSON")
                return {
                    "prompt": extracted_prompt,
                    "negative_prompt": "",
                    "reasoning": "Fallback - extracted from malformed JSON"
                }
            
            # Last resort: strip markdown and use as-is
            cleaned = re.sub(r'```(?:json)?', '', cleaned)  # Remove code block markers
            cleaned = re.sub(r'^(PROMPT:|Prompt:|VIDEO PROMPT:)\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip('`"\' \n\r\t')  # Strip all wrapper characters
            
            # If still looks like JSON structure, just extract text content
            if cleaned.startswith('{'):
                cleaned = re.sub(r'[{}":\[\],]', ' ', cleaned)  # Remove JSON structure chars
                cleaned = ' '.join(cleaned.split())  # Normalize whitespace
            
            return {
                "prompt": cleaned[:500] if len(cleaned) > 500 else cleaned,
                "negative_prompt": "",
                "reasoning": "Fallback parsing - LLM response was malformed"
            }
    
    async def generate_scene_capture_prompt(
        self,
        messages: List[Message],
        character: CharacterConfig,
        model: Optional[str] = None,
        workflow_config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Generate video prompt for scene capture (🎥 button).
        
        Third-person observer perspective focused on action/motion
        in the current conversation moment.
        
        Args:
            messages: Recent conversation history
            character: Character configuration
            model: Optional model to use for generation
            workflow_config: Optional workflow configuration from database
        
        Returns:
            Dictionary with:
                - prompt: Generated video prompt
                - negative_prompt: Negative prompt for video generation
                - needs_trigger: Whether trigger word should be included
                - reasoning: Brief explanation
        
        Raises:
            Exception: Failed to generate prompt
        """
        try:
            # Build context from recent messages
            context = self._build_scene_context_string(messages, character.name)
            
            # Build system prompt for scene capture (third-person observer)
            system_prompt = self._build_scene_capture_system_prompt(character, workflow_config)
            
            # Build user prompt
            user_prompt = f"""Recent conversation context:
{context}

Generate a detailed video scene capture prompt based on this context. Focus on MOTION and what's HAPPENING."""
            
            # Generate via LLM
            logger.info(f"[VIDEO SCENE CAPTURE] Generating from observer perspective")
            logger.debug(f"Context length: {len(context)} chars")
            
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,  # Use configured temperature
                max_tokens=300,
                model=model
            )
            logger.info(f"[VIDEO SCENE CAPTURE] LLM returned, used model: {response.model}")
            
            # Parse JSON response
            result = self._parse_llm_response(response.content)
            
            # Determine if trigger words are needed (scene captures typically include character)
            needs_trigger = bool(workflow_config and workflow_config.get("trigger_word"))
            
            # Build return dict
            result["needs_trigger"] = needs_trigger
            
            logger.info(f"Generated scene capture video prompt: {result['prompt'][:100]}...")
            return result
        
        except Exception as e:
            logger.error(f"Failed to generate scene capture prompt: {e}")
            raise
    
    def _build_scene_context_string(self, messages: List[Message], character_name: str) -> str:
        """
        Build formatted context string for scene capture from recent messages.
        
        Args:
            messages: List of recent messages
            character_name: Character name for role labeling
        
        Returns:
            Formatted context string with message numbers
        """
        if not messages:
            return "No recent context available."
        
        # Take last 10 messages for scene context (matching image scene capture)
        recent = messages[-10:] if len(messages) > 10 else messages
        
        context_lines = []
        for i, msg in enumerate(recent, 1):
            role = "User" if msg.role == MessageRole.USER else character_name
            content = msg.content[:500]  # Limit very long messages
            context_lines.append(f"[Message {i}/{len(recent)}] {role}: {content}")
        
        return "\n\n".join(context_lines)
    
    def _build_scene_capture_system_prompt(
        self,
        character: CharacterConfig,
        workflow_config: Optional[dict] = None
    ) -> str:
        """
        Build system prompt for scene capture (third-person observer perspective).
        
        Args:
            character: Character configuration
            workflow_config: Optional workflow configuration from database
        
        Returns:
            Formatted system prompt for scene capture
        """
        # Get workflow settings
        default_style = workflow_config.get("default_style", "Not specified") if workflow_config else "Not specified"
        self_description = workflow_config.get("self_description", "Not specified") if workflow_config else "Not specified"
        negative_prompt = workflow_config.get("negative_prompt", "Not specified") if workflow_config else "Not specified"
        trigger_word = workflow_config.get("trigger_word", "TRIGGER") if workflow_config else "TRIGGER"
        
        return f"""You are a video generation prompt engineer creating motion-focused prompts from a THIRD-PERSON OBSERVER perspective.

You are describing a scene as an omniscient narrator would see it - not from the character's perspective, but as a neutral observer capturing the moment like a camera would see it.

Character: {character.name}
Character Description: {character.role}

Video Generation Settings:
- Default Style: {default_style}
- Character Appearance: {self_description}
- Default Negative Prompt: {negative_prompt}

IMPORTANT: Do NOT include the trigger word "{trigger_word}" in your prompt.
It will be added automatically when needed.

Your task is to generate a MOTION-FOCUSED video prompt capturing the current scene in the conversation.

CRITICAL - OBSERVER PERSPECTIVE RULES:
- Describe the scene as a NEUTRAL OBSERVER watching from outside (like a camera capturing the moment)
- Use THIRD-PERSON pronouns (he/she/they, NOT I/me/my)
- Describe what's VISIBLY HAPPENING with motion and action
- Include {character.name}'s visible movements, gestures, expressions, and actions
- If the user is referenced in context, describe their actions too
- Focus on the CURRENT MOMENT and what's actively happening

CRITICAL - MESSAGE WEIGHTING:
- The LAST 2-3 MESSAGES define the CURRENT SCENE STATE (what's happening right now)
- Earlier messages provide BACKGROUND CONTEXT ONLY (setting, history)
- When actions/positions change across messages, ALWAYS use the most recent information
- If recent messages contradict earlier ones, the RECENT messages are authoritative

CONTEXT USAGE FOR VIDEO:
- EXTRACT motion-related details (actions, movements, gestures, physical activities)
- IDENTIFY what's actively happening NOW (from last 2-3 messages)
- CAPTURE visible emotions through animated expressions and body language
- NOTE any mentioned movements, interactions, or dynamic activities
- SYNTHESIZE setting from full context + current action from recent messages
- If conversation mentions actions (gesturing, moving, doing something), capture that motion
- If conversation is static (just talking), focus on subtle motions (expressions changing, slight movements, environment activity)

Scene Motion Examples:
1. Context: "I lean back and sigh, watching the clouds drift by..."
   → {character.name} leans back in their seat, chest rising with a deep sigh, gaze following clouds drifting slowly across the sky, peaceful contemplation

2. Context: "I stand up excitedly, gesturing towards the window..."
   → {character.name} rises quickly from seated position, arms gesturing animatedly toward the window, excited energy in their movements, dynamic transition from still to active

3. Context: "We're discussing art techniques..."
   → Close shot of {character.name} speaking, hands moving expressively as they explain concepts, facial expressions shifting with enthusiasm, subtle head tilts and nods during conversation

MOTION AND ACTION REQUIREMENTS:
- Write 100-300 words describing what MOVES and CHANGES
- Focus on visible actions, gestures, movements, transitions
- Include camera movement if relevant (pan, zoom, tracking, orbit)
- Specify pacing (slow motion, real-time, gentle movement)
- Use action verbs (moves, shifts, gestures, turns, leans, flows, drifts)
- Describe {character.name}'s physical actions and expressions
- If other people present, describe their movements too
- Capture the energy/pace of the moment (calm, energetic, tense, playful)
- DO NOT include dialogue or on-screen text
- Use present tense and active language

Example of motion-focused scene capture:
Bad: "Person sitting in a room" (no motion!)
Good: "{character.name} sits near a window, sunlight streaming across her face as she slowly turns pages of a worn book. Her fingers trace the text, head tilting slightly as she reads. A gentle breeze from the open window causes pages to flutter and her hair to drift softly. She pauses, gaze lifting to watch leaves dancing outside, expression thoughtful. Camera slowly pushes in on her contemplative face, then pulls back to reveal the cozy reading space. Dust motes float lazily in the golden afternoon light. Her hand reaches for a teacup, steam rising and swirling. Peaceful, meditative moment captured in gentle, flowing motion."

Return ONLY valid JSON in this format:
{{
  "prompt": "detailed 100-300 word video description from third-person observer perspective, emphasizing {character.name}'s visible actions, movements, gestures, environment motion, camera movement, and dynamic energy of the moment",
  "negative_prompt": "things to avoid in the video (static shots, frozen poses, jerky motion, poor lighting, etc.)",
  "reasoning": "brief 1-2 sentence explanation of the motion/action being captured"
}}

Remember: You are an omniscient observer describing what a camera would capture. Focus on MOTION and what HAPPENS. Be dynamic and vivid!"""
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate video prompt quality.
        
        Checks for motion keywords and reasonable length.
        
        Args:
            prompt: Generated prompt
        
        Returns:
            True if valid
        """
        if not prompt or len(prompt) < 10:
            logger.warning("Prompt too short")
            return False
        
        if len(prompt) > 500:
            logger.warning("Prompt too long (truncating)")
            return False
        
        # Check for motion keywords (at least one)
        motion_keywords = [
            'move', 'flow', 'drift', 'swirl', 'rotate', 'spin', 'cascade',
            'ripple', 'wave', 'pulse', 'transform', 'shift', 'glide', 'leap',
            'tumble', 'dance', 'sway', 'orbit', 'pan', 'zoom', 'track', 'fly'
        ]
        
        has_motion = any(word in prompt.lower() for word in motion_keywords)
        if not has_motion:
            logger.warning("Prompt lacks motion keywords - may be too static")
            # Don't fail, just warn
        
        return True
