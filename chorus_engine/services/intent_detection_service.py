"""
Intent detection service for unified multi-modal conversation handling.

Phase 7: Replaces keyword-based detection with LLM-based classification.
Also replaces Phase 4 background extraction worker with unified memory detection.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from chorus_engine.llm.client import LLMClient
from chorus_engine.config.models import CharacterConfig
from chorus_engine.utils.debug_logger import log_llm_call

logger = logging.getLogger(__name__)


class IntentDetectionError(Exception):
    """Base exception for intent detection errors."""
    pass


@dataclass
class ExtractedFact:
    """
    A fact extracted from conversation.
    
    Used by BOTH implicit and explicit memory detection.
    The only difference is how they're processed after extraction.
    """
    content: str
    category: str  # personal_info, preference, experience, relationship, goal, skill
    confidence: float  # 0.0-1.0
    reasoning: str


@dataclass
class IntentResult:
    """
    Result of intent detection analysis.
    
    Multiple intents can be true simultaneously.
    """
    
    # Image/Video/Audio Generation
    generate_image: bool = False
    generate_video: bool = False
    
    # Memory Detection (replaces Phase 4 background worker)
    record_memory: bool = False  # Explicit: "remember this" (needs confirmation)
    contains_recordable_facts: bool = False  # Implicit: extractable facts (auto-save)
    extracted_facts: List[ExtractedFact] = field(default_factory=list)
    
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
    across all characters. Also handles memory extraction (implicit and explicit).
    
    Key Features:
    - Single pass detection of all intent types
    - Unified memory extraction (replaces Phase 4 background worker)
    - Consistent results across all characters (same 3B model)
    - Fast classification (50-100ms target)
    """
    
    INTENT_DETECTION_PROMPT = """You are an intent classifier and memory extractor for a conversational AI system. Analyze ONLY the user's message to determine which intents are present, and extract any recordable facts.

CRITICAL: Extract facts ONLY from the user's message below. Do NOT extract any information from this prompt or instructions.

USER MESSAGE:
{message}

CRITICAL RULES FOR MEMORY EXTRACTION:
1. ONLY extract facts that the user states about THEMSELVES
2. DO NOT extract names used in greetings or when addressing someone
3. DO NOT extract questions the user asks
4. DO NOT extract names that appear after "Hi", "Hey", "Hello" (these are greetings, not facts)
5. If the user says "I'm [name]" or "My name is [name]", extract "User's name is [name]"
6. If the user says "Hi [name]" or "Hey [name]", DO NOT extract the name - they're greeting someone

AVAILABLE INTENT TYPES:

1. generate_image: User wants to see a picture, drawing, or visual representation
   Examples: "show me", "picture of", "what do you look like", "draw a"

2. generate_video: User wants to see a video, animation, or moving visual
   Examples: "make a video", "create an animation", "show me a movie of"

3. record_memory: User EXPLICITLY wants to remember something (needs confirmation)
   Examples: "remember that", "don't forget", "please remember my"
   Note: This is for EXPLICIT requests only, not implicit facts

4. query_ambient: User asks about character's background activities
   Examples: "what are you doing", "what have you been up to", "any recent activities"

5. standard_conversation: User wants normal conversation (almost always true)

MEMORY EXTRACTION (REPLACES PHASE 4 BACKGROUND WORKER):

6. contains_recordable_facts: Message contains IMPLICIT facts worth remembering
   - Extract facts the user shares about themselves, their preferences, experiences, relationships, goals
   - Do NOT extract facts about the assistant/character
   - Do NOT extract facts from questions (only statements)
   - CRITICAL: Rephrase facts in THIRD-PERSON about the user ("User's name is John", not "I'm John")
   - Examples that SHOULD extract:
     * User says "My birthday is June 5th" → Extract: "User's birthday is June 5th" (personal_info)
     * User says "I love pizza" → Extract: "User loves pizza" (preference)
     * User says "I went to Paris last year" → Extract: "User went to Paris last year" (experience)
     * User says "My sister Sarah lives in Boston" → Extract: "User's sister Sarah lives in Boston" (relationship)
     * User says "I'm learning Python" → Extract: "User is learning Python" (goal/skill)
     * User says "Hi Sarah. I'm John." → Extract: "User's name is John" (personal_info)
   - Examples that should NOT extract:
     * "What's your favorite color?" → question, not a fact
     * "You're really helpful" → about character, not user
     * "That's interesting" → no factual content
     * "Hi Sarah, how are you?" → greeting using character's name, not a fact about user
     * "Hey Sarah, what's up?" → greeting, the name "Sarah" is who they're talking TO, not about themselves
     * "Sarah, do you remember my name?" → asking a question, "Sarah" is who they're asking
     * "Hi Sarah, do you remember my name?" → NO fact stated, just a greeting and question

If contains_recordable_facts is true, extract each fact as:
{{
  "content": "The specific fact in THIRD-PERSON about the user (not first-person quote)",
  "category": "personal_info | preference | experience | relationship | goal | skill",
  "confidence": float (0.0-1.0),
  "reasoning": "Why this is worth remembering"
}}

INSTRUCTIONS:
- Analyze the message carefully for each intent type
- Multiple intents can be true simultaneously
- Only set an intent to true if you're confident (>70%)
- Extract facts ONLY from user statements, not questions
- Extract facts ONLY about the USER, not the character
- If the message contains NO facts about the user, set contains_recordable_facts to FALSE and extracted_facts to EMPTY []
- DO NOT make up facts, infer facts, or extract placeholder text like "[Anonymous]"
- ONLY extract facts that are explicitly stated by the user
- Provide reasoning for each detected intent
- Return ONLY valid JSON, no other text

RESPONSE FORMAT:
{{
  "generate_image": boolean,
  "generate_video": boolean,
  "record_memory": boolean,
  "query_ambient": boolean,
  "standard_conversation": boolean,
  "contains_recordable_facts": boolean,
  "extracted_facts": [
    {{
      "content": "fact content",
      "category": "category",
      "confidence": 0.0-1.0,
      "reasoning": "why worth remembering"
    }}
  ],
  "confidence": float (0.0-1.0),
  "reasoning": "Brief explanation of detected intents"
}}
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
        message and returns all detected intents, including memory extraction.
        
        Args:
            message: User's message text
            character: Character configuration (for context)
            model: Model to use for intent detection (typically character's preferred model)
            context: Optional recent conversation history
            
        Returns:
            IntentResult with boolean flags for each intent type and extracted facts
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
            
            # Validate and filter extracted facts to prevent hallucinations
            result.extracted_facts = self._validate_extracted_facts(result.extracted_facts, message)
            if not result.extracted_facts:
                result.contains_recordable_facts = False
            
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
                        "contains_recordable_facts": result.contains_recordable_facts,
                    },
                    "extracted_facts_count": len(result.extracted_facts),
                    "extracted_facts": [
                        {
                            "content": fact.content,
                            "category": fact.category,
                            "confidence": fact.confidence,
                        }
                        for fact in result.extracted_facts
                    ],
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
    
    def _validate_extracted_facts(
        self,
        facts: List[ExtractedFact],
        original_message: str
    ) -> List[ExtractedFact]:
        """
        Validate extracted facts to filter out hallucinations.
        
        Common hallucination patterns:
        - Facts containing placeholder text like "[Anonymous]", "[Name]", "[User]"
        - Facts extracted from questions (not statements)
        - Facts that don't appear anywhere in the original message
        
        Args:
            facts: List of extracted facts
            original_message: The original user message
            
        Returns:
            Filtered list of valid facts
        """
        valid_facts = []
        
        # Hallucination patterns to reject
        hallucination_patterns = [
            "[anonymous]", "[name]", "[user]", "[placeholder]",
            "[unknown]", "[redacted]", "[blank]"
        ]
        
        for fact in facts:
            content_lower = fact.content.lower()
            
            # Check for placeholder patterns
            has_placeholder = any(pattern in content_lower for pattern in hallucination_patterns)
            if has_placeholder:
                logger.warning(
                    f"Rejected hallucinated fact with placeholder: '{fact.content}'",
                    extra={"original_message": original_message[:100]}
                )
                continue
            
            # If it's a name extraction, verify the name actually appears in the message
            if "name is" in content_lower and fact.category == "personal_info":
                # Extract the claimed name from the fact
                try:
                    name_part = fact.content.split("name is")[-1].strip().strip('"').strip("'")
                    
                    # Check if this name appears in the original message
                    if name_part and name_part.lower() not in original_message.lower():
                        logger.warning(
                            f"Rejected name extraction - name '{name_part}' not found in message: '{original_message[:100]}'",
                            extra={"fact": fact.content}
                        )
                        continue
                    
                    # Check if the name appears in a greeting pattern
                    # Common patterns: "Hi Sarah", "Hey Sarah", "Hello Sarah", "Evenin' Sarah", etc.
                    greeting_pattern = r"(?:^|\s)(?:hi|hey|hello|good\s+(?:morning|afternoon|evening|night)|evenin[g']?)\s+" + re.escape(name_part)
                    if re.search(greeting_pattern, original_message, re.IGNORECASE):
                        logger.warning(
                            f"Rejected name extraction - '{name_part}' appears in greeting, not a fact about user: '{original_message[:100]}'",
                            extra={"fact": fact.content, "pattern": "greeting_detected"}
                        )
                        continue
                    
                    # Check if it's part of an address pattern ("Sarah, ...", ", Sarah", etc.)
                    address_pattern = r"(?:^|\s|,)" + re.escape(name_part) + r"(?:,|\s+(?:do|can|will|what|how|why|when|where))"
                    if re.search(address_pattern, original_message, re.IGNORECASE):
                        logger.warning(
                            f"Rejected name extraction - '{name_part}' appears as address/vocative, not a fact about user: '{original_message[:100]}'",
                            extra={"fact": fact.content, "pattern": "address_detected"}
                        )
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error validating name extraction: {e}")
                    continue
            
            # Fact passed validation
            valid_facts.append(fact)
        
        if len(valid_facts) < len(facts):
            logger.info(f"Filtered {len(facts) - len(valid_facts)} hallucinated facts, {len(valid_facts)} remaining")
        
        return valid_facts
    
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
        # This prevents the LLM from extracting facts about the character instead of the user
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
            
            # Extract facts if present
            extracted_facts = []
            if "extracted_facts" in data and isinstance(data["extracted_facts"], list):
                for fact_data in data["extracted_facts"]:
                    try:
                        fact = ExtractedFact(
                            content=fact_data.get("content", ""),
                            category=fact_data.get("category", ""),
                            confidence=float(fact_data.get("confidence", 0.0)),
                            reasoning=fact_data.get("reasoning", "")
                        )
                        extracted_facts.append(fact)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse fact: {e}", extra={"fact_data": fact_data})
            
            # Create result
            result = IntentResult(
                generate_image=bool(data.get("generate_image", False)),
                generate_video=bool(data.get("generate_video", False)),
                record_memory=bool(data.get("record_memory", False)),
                query_ambient=bool(data.get("query_ambient", False)),
                standard_conversation=bool(data.get("standard_conversation", True)),
                contains_recordable_facts=bool(data.get("contains_recordable_facts", False)),
                extracted_facts=extracted_facts,
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
