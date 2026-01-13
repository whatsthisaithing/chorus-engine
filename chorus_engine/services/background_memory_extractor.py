"""Background memory extraction worker for async implicit memory processing.

Phase 7.5: Extracts implicit memories from conversation using character's
loaded model to avoid VRAM overhead and latency.
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict

from chorus_engine.llm.client import LLMClient
from chorus_engine.services.memory_extraction import MemoryExtractionService, ExtractedMemory
from chorus_engine.models.conversation import Message
from chorus_engine.services.memory_profile_service import MemoryProfileService
from chorus_engine.config.models import CharacterConfig

logger = logging.getLogger(__name__)

# Debug log directory
DEBUG_LOG_DIR = Path("data/debug_logs")


@dataclass
class ExtractionTask:
    """Task for background memory extraction."""
    conversation_id: str
    character_id: str
    messages: List[Message]
    model: str
    character_name: str
    character: CharacterConfig


class BackgroundMemoryExtractor:
    """Extracts implicit memories from conversations in the background.
    
    This worker runs async memory extraction using the character's already-loaded
    model, so there's no VRAM overhead or model swapping delay. Memory extraction
    happens after the conversation response is sent, keeping the user experience fast.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        extraction_service: MemoryExtractionService,
        temperature: float = 0.1,
        llm_usage_lock: Optional[asyncio.Lock] = None
    ):
        self.llm_client = llm_client
        self.extraction_service = extraction_service
        self.temperature = temperature
        self.memory_profile_service = MemoryProfileService()
        self.llm_usage_lock = llm_usage_lock or asyncio.Lock()
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Ensure debug log directory exists
        DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
    async def start(self):
        """Start the background worker."""
        if self._running:
            return
            
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Background memory extraction worker started")
    
    async def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Background memory extraction worker stopped")
    
    async def queue_extraction(
        self,
        conversation_id: str,
        character_id: str,
        messages: List[Message],
        model: str,
        character_name: str,
        character: CharacterConfig
    ):
        """Queue a memory extraction task.
        
        Args:
            conversation_id: ID of the conversation
            character_id: ID of the character
            messages: List of messages to extract from
            model: Model name to use for extraction (character's model)
            character_name: Name of character for context
            character: Character configuration (for memory profile)
        """
        task = ExtractionTask(
            conversation_id=conversation_id,
            character_id=character_id,
            messages=messages,
            model=model,
            character_name=character_name,
            character=character
        )
        await self._task_queue.put(task)
        logger.debug(f"Queued memory extraction for conversation {conversation_id}")
    
    async def _worker_loop(self):
        """Main worker loop that processes extraction tasks."""
        while self._running:
            try:
                # Wait for next task with timeout
                try:
                    task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process the task
                await self._process_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory extraction worker: {e}", exc_info=True)
    
    async def _process_task(self, task: ExtractionTask):
        """Process a single extraction task."""
        
        extraction_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_id": task.conversation_id,
            "character_id": task.character_id,
            "character_name": task.character_name,
            "model": task.model,
            "temperature": self.temperature,
            "input_messages": [],
            "prompt": None,
            "llm_response": None,
            "extracted_memories": [],
            "saved_memories": [],
            "error": None
        }
        
        try:
            logger.info(
                f"[BACKGROUND MEMORY] Extracting from {len(task.messages)} message(s) "
                f"for character {task.character_name}"
            )
            
            # Log input message metadata (not content for privacy)
            for msg in task.messages:
                extraction_log["input_messages"].append({
                    "role": str(msg.role),
                    "content_length": len(msg.content),
                    "id": msg.id if hasattr(msg, 'id') else None
                })
            
            # Build extraction prompt with memory profile
            system_prompt, user_content = self._build_extraction_prompt(
                task.messages, 
                task.character_name,
                task.character
            )
            # Log prompt metadata only (not content for privacy)
            extraction_log["prompt_length"] = {
                "system": len(system_prompt) if system_prompt else 0,
                "user": len(user_content) if user_content else 0
            }
            
            if not system_prompt or not user_content:
                logger.info("[BACKGROUND MEMORY] Empty prompt after filtering - skipping LLM call")
                extraction_log["error"] = "No user messages to extract from"
                self._write_extraction_log(task.conversation_id, extraction_log)
                return
            
            logger.info(f"[BACKGROUND MEMORY] Calling LLM with extraction prompt (system: {len(system_prompt)} chars, user: {len(user_content)} chars)")
            
            # Acquire lock to prevent model unloading during extraction
            async with self.llm_usage_lock:
                logger.debug("[BACKGROUND MEMORY] Acquired LLM usage lock")
                
                # Call LLM to extract memories using character's model
                response = await self.llm_client.generate(
                    prompt=user_content,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                    max_tokens=1000,
                    model=task.model
                )
                
                logger.debug("[BACKGROUND MEMORY] Released LLM usage lock")
            
            # Extract text from LLMResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)
            # Log response length only (not content for privacy)
            extraction_log["llm_response_length"] = len(response_text)
            
            logger.info(f"[BACKGROUND MEMORY] LLM response ({len(response_text)} chars)")
            
            # Parse response and save memories
            memories = self._parse_extraction_response(response_text)
            extraction_log["extracted_memories"] = [
                {
                    "content": m.content,
                    "category": m.category,
                    "confidence": m.confidence,
                    "reasoning": m.reasoning
                }
                for m in memories
            ]
            
            saved_count = 0
            for memory in memories:
                try:
                    saved = await self.extraction_service.save_extracted_memory(
                        extracted=memory,
                        character_id=task.character_id,
                        conversation_id=task.conversation_id
                    )
                    if saved:
                        saved_count += 1
                        extraction_log["saved_memories"].append({
                            "content": memory.content,
                            "confidence": memory.confidence,
                            "status": saved.status if hasattr(saved, 'status') else 'unknown'
                        })
                        logger.info(
                            f"[BACKGROUND MEMORY] Saved memory: type={memory.category}, "
                            f"confidence={memory.confidence:.2f}, status={saved.status if hasattr(saved, 'status') else 'saved'}"
                        )
                except Exception as e:
                    logger.error(f"Failed to save memory: {e}", exc_info=True)
                    extraction_log["saved_memories"].append({
                        "content": memory.content,
                        "error": str(e)
                    })
            
            logger.info(
                f"[BACKGROUND MEMORY] Extraction complete for conversation {task.conversation_id}: "
                f"{saved_count}/{len(memories)} memories saved"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to extract memories for conversation {task.conversation_id}: {e}",
                exc_info=True
            )
            extraction_log["error"] = str(e)
        finally:
            # Always write the extraction log
            self._write_extraction_log(task.conversation_id, extraction_log)
    
    def _build_extraction_prompt(
        self, 
        messages: List[Message], 
        character_name: str,
        character: CharacterConfig
    ) -> tuple[str, str]:
        """Build system prompt and user content for memory extraction.
        
        Phase 8: Includes memory profile filtering and new memory types/fields.
        
        Args:
            messages: Messages to extract from (should be USER messages only)
            character_name: Name of character for context
            character: Character configuration (for memory profile)
            
        Returns:
            Tuple of (system_prompt, user_content)
        """
        # Log what messages we received (INFO level for visibility)
        logger.info(f"[EXTRACTION] Received {len(messages)} messages for extraction")
        for i, msg in enumerate(messages):
            role = getattr(msg, 'role', 'unknown')
            content_len = len(msg.content) if hasattr(msg, 'content') else 0
            logger.info(f"[EXTRACTION]   Msg {i+1}: role={role}, length={content_len} chars")
        
        # CRITICAL: Only process USER messages (filter out character responses and system messages)
        user_messages = [msg for msg in messages if msg.role == 'user']
        
        if not user_messages:
            # No user messages to extract from
            logger.info("[EXTRACTION] No user messages found after filtering - SKIPPING")
            return ("", "")
        
        logger.info(f"[EXTRACTION] Processing {len(user_messages)} user message(s) after filtering")
        
        # Format user messages (plain content, no role prefix)
        conversation_text = "\n".join([
            msg.content
            for msg in user_messages
        ])
        
        # Get extraction instructions from memory profile service
        extraction_instructions = self.memory_profile_service.get_extraction_instructions(character)
        allowed_types = self.memory_profile_service.get_allowed_types(character)
        allowed_type_names = [t.value for t in allowed_types]
        
        # System prompt with Phase 8 extraction instructions
        system_prompt = f"""You are a memory extraction system. Your job is to identify and extract information about the user from their messages.

IMPORTANT: The assistant in this conversation is named {character_name!r}. DO NOT confuse the assistant name with the user name.

TASK: Extract memories about the USER (not the assistant).

{extraction_instructions}

MEMORY TYPES (extract these types only):
- FACT: Factual information (name, location, job, preferences, opinions)
- PROJECT: Goals, plans, ongoing projects or objectives
- EXPERIENCE: Shared experiences, activities, events (if allowed)
- STORY: Narratives, anecdotes, past stories (if allowed)
- RELATIONSHIP: Relationship dynamics, emotional bonds (if allowed)

What NOT to extract:
- Unknown information or speculation
- Information about the assistant/character
- Conversational filler (greetings, small talk)
- Facts NOT mentioned by the user
- The assistant name as the user name
- Requests or questions from the user
- Physical descriptions unless explicitly stated by user
- Demographic assumptions (age, gender, ethnicity) unless explicitly stated

For each memory, provide a JSON object with:
- "content": Clear statement (e.g., "User name is John")
- "type": One of {allowed_type_names}
- "confidence": Float 0.0-1.0 (0.95 explicit, 0.8 clear, 0.7 inference)
- "reasoning": One sentence explaining extraction
- "emotional_weight": Optional float 0.0-1.0 for emotionally significant moments
- "participants": Optional list of people involved (including user)
- "key_moments": Optional list of significant moments in the memory

EXAMPLES OF GOOD EXTRACTIONS:
- "My name is Sarah" → {{"content": "User name is Sarah", "type": "fact", "confidence": 0.95, "reasoning": "User explicitly stated their name"}}
- "I love hiking" → {{"content": "User enjoys hiking", "type": "fact", "confidence": 0.9, "reasoning": "Direct statement of interest"}}
- "I'm working on building a game" → {{"content": "User is developing a game", "type": "project", "confidence": 0.9, "reasoning": "Ongoing project stated"}}
- "We went to the beach last summer" → {{"content": "User went to beach with companion", "type": "experience", "confidence": 0.85, "reasoning": "Shared experience mentioned", "participants": ["user", "companion"]}}
- "When I was a kid, I broke my arm" → {{"content": "User broke arm as child", "type": "story", "confidence": 0.9, "reasoning": "Past narrative shared", "emotional_weight": 0.6}}

EXAMPLES OF BAD EXTRACTIONS (DO NOT extract these):
- DO NOT extract unknowns
- DO NOT extract vague impressions
- DO NOT extract facts about assistant
- DO NOT confuse assistant name with user name
- DO NOT invent facts not mentioned
- DO NOT make assumptions from greetings
- DO NOT extract requests/actions
- DO NOT extract descriptions from character responses

CRITICAL RULES:
1. If the user has not explicitly mentioned something, DO NOT extract it
2. DO NOT confuse greetings with the user stating their own name
3. DO NOT invent hobbies, interests, or facts that were not discussed
4. DO NOT extract what the user is asking for or requesting - only extract facts about themselves
5. DO NOT extract physical descriptions unless the user explicitly states them about themselves
6. DO NOT infer gender, age, ethnicity, or other demographics from names or greetings
7. DO NOT extract demographic information unless the user EXPLICITLY states it about themselves
8. Only extract information that was ACTUALLY stated or clearly implied in the USER messages
9. DO NOT extract conversation actions like "User greeted", "User asked", "User said hello"
10. If the messages contain ONLY greetings or small talk with NO actual facts, return an empty array []
11. Questions from the user contain NO facts about the user - questions always return []
12. You are NOT answering the user's questions - you are extracting facts the user stated about THEMSELVES

RESPONSE FORMAT:
You MUST return ONLY a valid JSON array. NO explanations, NO apologies, NO markdown formatting, NO text before or after.

VALID responses:
- Facts found: [{{"content": "...", "category": "...", "confidence": 0.95, "reasoning": "..."}}]
- No facts: []

INVALID responses (DO NOT use these):
- "Sorry, I can't extract..."
- Text explanations
- Markdown code blocks
- Empty responses

If no memorable facts are present, return exactly: []

CRITICAL: Returning an empty array [] is the CORRECT and EXPECTED response when messages contain no facts. Simple greetings like "Hello" MUST return []. Questions MUST return []."""

        # User content is ONLY the messages - no instructions or prompting
        user_content = conversation_text
        
        return (system_prompt, user_content)
    
    def _write_extraction_log(self, conversation_id: str, log_data: dict) -> None:
        """Write extraction log to debug file.
        
        Only writes if debug mode is enabled in system config.
        
        Args:
            conversation_id: Conversation ID for directory name
            log_data: Extraction log data to write
        """
        # Check if debug logging is enabled
        from chorus_engine.utils.debug_logger import get_debug_logger
        if not get_debug_logger().enabled:
            return
        
        try:
            # Create conversation-specific directory under conversations/
            conversation_dir = DEBUG_LOG_DIR / "conversations" / conversation_id
            conversation_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to extractions.jsonl file
            log_file = conversation_dir / "extractions.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write extraction debug log: {e}")
    
    def _parse_extraction_response(self, response: str) -> List[ExtractedMemory]:
        """Parse LLM response into ExtractedMemory objects.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of ExtractedMemory objects
        """
        import json
        import re
        
        memories = []
        
        try:
            # Try to find JSON array first
            json_match = re.search(r'\[[\s\S]*\]', response)
            
            if json_match:
                # Found array
                facts_data = json.loads(json_match.group(0))
            else:
                # Try to find single JSON object
                json_match = re.search(r'\{[\s\S]*?\}', response)
                if json_match:
                    # Found single object, wrap in array
                    single_fact = json.loads(json_match.group(0))
                    facts_data = [single_fact]
                    logger.info("Found single JSON object instead of array, converted to array")
                else:
                    logger.warning(f"No JSON found in extraction response: {response[:200]}...")
                    return []
            
            for fact in facts_data:
                try:
                    content = fact['content']
                    content_lower = content.lower()
                    
                    # DEFENSIVE FILTERS: Catch bad extractions before saving
                    
                    # Filter 1: Conversation actions (not facts about user)
                    conversation_actions = [
                        "user greeted", "user said hello", "user said hi", "user initiated",
                        "user responded", "user asked", "user requested", "user thanked",
                        "user confirmed", "user agreed", "user disagreed", "user inquired",
                        "user asked about", "user requested a", "user wants to know"
                    ]
                    if any(action in content_lower for action in conversation_actions):
                        logger.warning(f"[FILTER] Blocked conversation action: {content}")
                        continue
                    
                    # Filter 2: Assistant/character facts (not user facts)
                    assistant_patterns = [
                        "assistant is", "assistant's", "assistant has", "assistant can",
                        "character is", "character's", "character has"
                    ]
                    if any(pattern in content_lower for pattern in assistant_patterns):
                        logger.warning(f"[FILTER] Blocked assistant fact: {content}")
                        continue
                    
                    # Filter 3: System prompt leaks
                    system_prompt_indicators = [
                        "is uncensored", "is unrestricted", "is a helpful", "is truthful",
                        "is unbiased", "is designed to", "follows instructions"
                    ]
                    if any(indicator in content_lower for indicator in system_prompt_indicators):
                        logger.warning(f"[FILTER] Blocked system prompt leak: {content}")
                        continue
                    
                    # Filter 4: Demographic hallucinations
                    demographic_assumptions = [
                        "is male", "is female", "is a man", "is a woman",
                        "years old", "age is", "ethnicity is", "race is"
                    ]
                    if any(assumption in content_lower for assumption in demographic_assumptions):
                        logger.warning(f"[FILTER] Blocked demographic hallucination: {content}")
                        continue
                    
                    # Filter 5: Unknown information
                    if "unknown" in content_lower or "not mentioned" in content_lower:
                        logger.warning(f"[FILTER] Blocked unknown information: {content}")
                        continue
                    
                    # Filter 6: Must start with "User" (ensures it's about the user)
                    if not content_lower.startswith("user"):
                        logger.warning(f"[FILTER] Blocked non-user content: {content}")
                        continue
                    
                    # Phase 8: Parse new fields and memory type
                    memory_type = fact.get('type', fact.get('category', 'fact'))  # Backward compatible
                    emotional_weight = fact.get('emotional_weight')
                    participants = fact.get('participants', [])
                    key_moments = fact.get('key_moments', [])
                    
                    # Validate emotional_weight if present
                    if emotional_weight is not None:
                        emotional_weight = float(emotional_weight)
                        if not (0.0 <= emotional_weight <= 1.0):
                            logger.warning(f"Invalid emotional_weight {emotional_weight}, clamping to [0, 1]")
                            emotional_weight = max(0.0, min(1.0, emotional_weight))
                    
                    # All filters passed, create memory object
                    memory = ExtractedMemory(
                        content=content,
                        category=memory_type,  # Will be converted to MemoryType in save_extracted_memory
                        confidence=float(fact['confidence']),
                        reasoning=fact.get('reasoning', ''),
                        source_message_ids=[],
                        emotional_weight=emotional_weight,
                        participants=participants if participants else None,
                        key_moments=key_moments if key_moments else None
                    )
                    memories.append(memory)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed fact: {e}")
                    continue
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}", exc_info=True)
        
        return memories
