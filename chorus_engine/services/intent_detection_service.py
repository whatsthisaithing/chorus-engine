"""
Intent detection service for unified multi-modal conversation handling.

Phase 7: Replaces keyword-based detection with LLM-based classification.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from chorus_engine.llm.client import LLMClient
from chorus_engine.config.models import CharacterConfig
from chorus_engine.utils.debug_logger import log_llm_call

logger = logging.getLogger(__name__)


class IntentDetectionError(Exception):
    """Base exception for intent detection errors."""
    pass



@dataclass
class IntentResult:
    """
    Result of intent detection analysis.
    
    Multiple intents can be true simultaneously.
    """
    
    # Image/Video/Audio Generation
    generate_image: bool = False
    generate_video: bool = False
    
    # Memory Detection (explicit requests only)
    record_memory: bool = False  # Explicit: "remember this" (needs confirmation)
    
    # Activities
    query_ambient: bool = False
    
    # Standard
    standard_conversation: bool = True
    
    # Metadata
    confidence: float = 0.0
    reasoning: str = ""
    processing_time_ms: float = 0.0
    
    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None


class IntentDetectionService:
    """
    LLM-based intent detection for multi-modal conversation.
    
    Uses dedicated 3B model for fast, consistent intent classification
    across all characters.
    
    Key Features:
    - Single pass detection of all intent types
    - Consistent results across all characters (same 3B model)
    - Fast classification (50-100ms target)
    """
    
    INTENT_DETECTION_PROMPT = """You are an intent classifier for a conversational AI system. Analyze ONLY the user's message to determine which intents are present.

USER MESSAGE:
{message}

AVAILABLE INTENT TYPES:

1. generate_image: User wants to see a picture, drawing, or visual representation
   Examples: "show me", "picture of", "what do you look like", "draw a"

2. generate_video: User wants to see a video, animation, or moving visual
   Examples: "make a video", "create an animation", "show me a movie of"

3. record_memory: User EXPLICITLY wants to remember something (needs confirmation)
   Examples: "remember that", "don't forget", "please remember my"
   Note: This is for EXPLICIT requests only.

4. query_ambient: User asks about character's background activities
   Examples: "what are you doing", "what have you been up to", "any recent activities"

5. standard_conversation: User wants normal conversation (almost always true)

INSTRUCTIONS:
- Analyze the message carefully for each intent type
- Multiple intents can be true simultaneously
- Only set an intent to true if you're confident (>70%)
- Provide reasoning for detected intents
- Return ONLY valid JSON, no other text

RESPONSE FORMAT:
{
  "generate_image": boolean,
  "generate_video": boolean,
  "record_memory": boolean,
  "query_ambient": boolean,
  "standard_conversation": boolean,
  "confidence": float (0.0-1.0),
  "reasoning": "Brief explanation of detected intents"
}
"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "qwen2.5:3b-instruct",
        temperature: float = 0.1  # Very low for consistent classification
    ):
        """
        Initialize intent detection service.
        
        Args:
            llm_client: LLM client for calling the intent model
            model: Model to use for intent detection (default: qwen2.5:3b-instruct)
            temperature: Temperature for LLM (lower = more consistent)
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        
        logger.info(
            "Intent detection service initialized",
            extra={
                "model": self.model,
                "temperature": self.temperature,
            }
        )
    
    async def detect_intents(
        self,
        message: str,
        character: CharacterConfig,
        model: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> IntentResult:
        """
        Detect all intents in a user message.
        
        This is the main entry point for intent detection. It analyzes the
        message and returns all detected intents.
        
        Args:
            message: User's message text
            character: Character configuration (for context)
            model: Model to use for intent detection (typically character's preferred model)
            context: Optional recent conversation history
            
        Returns:
            IntentResult with boolean flags for each intent type
        """
        start_time = time.time()
        
        logger.info(
            "Intent detection started",
            extra={
                "message_preview": message[:100],
                "message_length": len(message),
                "character_id": character.id,
                "model": model,
                "has_context": context is not None,
            }
        )
        
        try:
            # Build the prompt
            prompt = self._build_prompt(message, character, context)
            
            # Call the LLM using generate method
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=None,  # System instructions are in the prompt
                temperature=self.temperature,
                model=model
            )
            
            # Log to debug file
            log_llm_call(
                conversation_id="intent_detection",
                interaction_type="intent_detection",
                model=model,
                prompt=prompt,
                response=response.content,
                settings={"temperature": self.temperature},
                metadata={"character_id": character.id, "message_preview": message[:100]}
            )
            
            # Parse the response (response is LLMResponse object with .content)
            result = await self._parse_llm_response(response.content)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time_ms
            
            logger.info(
                "Intent detection complete",
                extra={
                    "intents": {
                        "generate_image": result.generate_image,
                        "generate_video": result.generate_video,
                        "record_memory": result.record_memory,
                        "query_ambient": result.query_ambient,
                        "standard_conversation": result.standard_conversation,
                    },
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "processing_time_ms": processing_time_ms,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Intent detection failed",
                extra={
                    "error": str(e),
                    "message_preview": message[:100],
                },
                exc_info=True
            )
            
            # Return default result (standard conversation only)
            return IntentResult(
                standard_conversation=True,
                confidence=0.0,
                reasoning=f"Intent detection failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    def _build_prompt(
        self,
        message: str,
        character: CharacterConfig,
        context: Optional[List[Dict[str, str]]]
    ) -> str:
        """
        Build the intent detection prompt.
        
        Args:
            message: User's message
            character: Character configuration (not used - kept for compatibility)
            context: Optional conversation context (not used - kept for compatibility)
            
        Returns:
            Formatted prompt string with ONLY the user's message
        """
        # Format the prompt with ONLY the user message - no character info or context
        return self.INTENT_DETECTION_PROMPT.format(message=message)
    
    async def _parse_llm_response(self, response: str) -> IntentResult:
        """
        Parse and validate the LLM JSON response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed IntentResult
            
        Raises:
            IntentDetectionError: If parsing fails
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLMs add extra text, so look for JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise IntentDetectionError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Create result
            result = IntentResult(
                generate_image=bool(data.get("generate_image", False)),
                generate_video=bool(data.get("generate_video", False)),
                record_memory=bool(data.get("record_memory", False)),
                query_ambient=bool(data.get("query_ambient", False)),
                standard_conversation=bool(data.get("standard_conversation", True)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=str(data.get("reasoning", "")),
                raw_response=data
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON response",
                extra={
                    "error": str(e),
                    "response": response[:500],
                }
            )
            raise IntentDetectionError(f"Invalid JSON in response: {e}")
        except Exception as e:
            logger.error(
                "Failed to parse LLM response",
                extra={
                    "error": str(e),
                    "response": response[:500],
                },
                exc_info=True
            )
            raise IntentDetectionError(f"Failed to parse response: {e}")
