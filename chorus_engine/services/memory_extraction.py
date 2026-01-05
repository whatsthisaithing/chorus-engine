"""Service for extracting implicit memories from conversations (Phase 4.1)."""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from chorus_engine.models.conversation import Message, Memory, MemoryType
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMemory:
    """Data class for extracted memory before saving.
    
    Phase 8: Added emotional_weight, participants, key_moments fields.
    """
    content: str
    category: str
    confidence: float
    reasoning: str
    source_message_ids: List[str]
    emotional_weight: Optional[float] = None
    participants: Optional[List[str]] = None
    key_moments: Optional[List[str]] = None


class MemoryExtractionService:
    """
    Service for extracting implicit memories from conversations.
    
    Uses LLM to analyze conversation content and extract memorable facts
    about the user, with confidence scoring and deduplication.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        memory_repository: MemoryRepository,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ):
        self.llm = llm_client
        self.memory_repo = memory_repository
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def extract_from_messages(
        self,
        messages: List[Message],
        character_id: str,
        conversation_id: str,
        model: Optional[str] = None,
        character_name: Optional[str] = None
    ) -> List[ExtractedMemory]:
        """
        Analyze messages and extract memorable facts.
        
        Args:
            messages: List of messages to analyze
            character_id: Character ID for context
            conversation_id: Conversation ID for tracking
            model: Model to use for extraction (character's preferred model)
            character_name: Name of the character (for prompt context)
        
        Returns:
            List of extracted memories with confidence scores
        """
        if not messages:
            return []
        
        # Debug: Show messages being extracted (use logger to avoid emoji encoding issues)
        logger.info(f"Messages being sent for extraction: {len(messages)} messages")
        for i, msg in enumerate(messages, 1):
            role = msg.role
            # Sanitize content for logging (replace problematic characters)
            content = msg.content[:80].encode('ascii', errors='replace').decode('ascii')
            if len(msg.content) > 80:
                content += "..."
            logger.debug(f"  {i}. {role}: {content}")
        
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(messages, character_name)
            
            # Call LLM with structured output request
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent extraction
                model=model  # Use character's preferred model
            )
            
            # Debug: Log LLM response
            logger.debug(f"LLM extraction response: {len(response.content)} chars")
            logger.debug(response.content[:500])
            if len(response.content) > 500:
                logger.debug(f"... (truncated, total {len(response.content)} chars)")
            
            # Parse JSON response
            extracted = self._parse_extraction_response(response.content, messages)
            
            logger.info(f"Parsed {len(extracted)} memories from LLM response")
            for mem in extracted:
                content_preview = mem.content[:60].encode('ascii', errors='replace').decode('ascii')
                if len(mem.content) > 60:
                    content_preview += "..."
                logger.debug(f"  - {content_preview} (confidence: {mem.confidence})")
            
            logger.info(f"Extracted {len(extracted)} potential memories from {len(messages)} messages")
            return extracted
            
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}", exc_info=True)
            return []
    
    async def check_for_duplicates(
        self,
        memory_content: str,
        character_id: str
    ) -> Optional[Memory]:
        """
        Check if similar memory already exists.
        
        Args:
            memory_content: Memory content to check
            character_id: Character ID
        
        Returns:
            Existing memory if similarity ≥0.85, else None
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed(memory_content)
            
            # Search for similar memories using vector store
            results = self.vector_store.query_memories(
                character_id=character_id,
                query_embedding=query_embedding,
                n_results=1,
                where={"type": "implicit"}
            )
            
            # Check if we have results and if similarity is high enough
            if results and results['ids'] and len(results['ids'][0]) > 0:
                # Calculate similarity from distance (cosine distance)
                # ChromaDB returns distances, we need to convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][0]
                similarity = 1.0 - distance
                
                if similarity >= 0.85:
                    # Fetch full memory from database using vector_id
                    vector_id = results['ids'][0][0]
                    memories = self.memory_repo.list_by_character(character_id)
                    for mem in memories:
                        if mem.vector_id == vector_id:
                            logger.debug(f"Found duplicate memory (similarity={similarity:.2f})")
                            return mem
            
            return None
            
        except Exception as e:
            logger.error(f"Duplicate check failed: {e}", exc_info=True)
            return None
    
    async def save_extracted_memory(
        self,
        extracted: ExtractedMemory,
        character_id: str,
        conversation_id: str
    ) -> Optional[Memory]:
        """
        Save extracted memory with auto-approval logic.
        
        Confidence thresholds:
        - ≥0.9: auto_approved (saved and added to vector store)
        - 0.7-0.89: pending (saved but not in vector store)
        - <0.7: discarded (not saved)
        
        Args:
            extracted: Extracted memory data
            character_id: Character ID
            conversation_id: Source conversation ID
        
        Returns:
            Created memory or None if discarded
        """
        try:
            # 1. Check confidence threshold
            if extracted.confidence < 0.7:
                logger.debug(f"Discarding low-confidence memory (confidence={extracted.confidence:.2f})")
                return None
            
            # 2. Check for duplicates
            duplicate = await self.check_for_duplicates(extracted.content, character_id)
            
            if duplicate:
                # Handle duplicate (reinforce or update)
                return await self._handle_duplicate(duplicate, extracted)
            
            # 3. Determine status based on confidence
            if extracted.confidence >= 0.9:
                status = "auto_approved"
            else:
                status = "pending"
            
            # 4. Create memory in database with Phase 8 fields
            memory = self.memory_repo.create(
                content=extracted.content,
                character_id=character_id,
                conversation_id=conversation_id,
                memory_type=MemoryType.IMPLICIT,  # Will be mapped to FACT in repository
                confidence=extracted.confidence,
                category=extracted.category,
                status=status,
                source_messages=extracted.source_message_ids,
                metadata={
                    "reasoning": extracted.reasoning,
                    "extraction_date": str(json.dumps({}))  # Placeholder for timestamp
                },
                emotional_weight=extracted.emotional_weight,
                participants=extracted.participants,
                key_moments=extracted.key_moments
            )
            
            # 5. Add to vector store if auto-approved
            if status == "auto_approved":
                await self._add_to_vector_store(memory, character_id)
            
            logger.info(
                f"Saved {status} memory: {memory.content[:50]}... "
                f"(confidence={extracted.confidence:.2f}, category={extracted.category})"
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to save extracted memory: {e}", exc_info=True)
            return None
    
    async def approve_pending_memory(self, memory_id: str) -> bool:
        """
        Approve a pending memory and add it to vector store.
        
        Args:
            memory_id: Memory ID to approve
        
        Returns:
            True if approved successfully
        """
        try:
            memory = self.memory_repo.get_by_id(memory_id)
            if not memory:
                logger.warning(f"Memory {memory_id} not found")
                return False
            
            if memory.status != "pending":
                logger.warning(f"Memory {memory_id} is not pending (status={memory.status})")
                return False
            
            # Update status
            self.memory_repo.update_status(memory_id, "approved")
            
            # Add to vector store
            await self._add_to_vector_store(memory, memory.character_id)
            
            logger.info(f"Approved memory {memory_id}: {memory.content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve memory {memory_id}: {e}", exc_info=True)
            return False
    
    def _build_extraction_prompt(self, messages: List[Message], character_name: Optional[str] = None) -> str:
        """Build prompt for LLM extraction."""
        # Format messages for prompt
        formatted_messages = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            formatted_messages.append(f"{role.upper()}: {msg.content}")
        
        conversation_text = "\n".join(formatted_messages)
        
        # Add character context if available
        character_context = ""
        if character_name:
            character_context = f"\nIMPORTANT: The assistant in this conversation is named '{character_name}'. DO NOT confuse the assistant's name with the user's name.\n"
        
        return f"""You are a memory extraction system. Your job is to identify and extract factual information about the user from conversations.{character_context}

CONVERSATION:
{conversation_text}

TASK: Extract memorable facts about the USER (not the assistant).

What to extract:
- Names, locations, jobs, relationships (of the USER)
- Hobbies, interests, skills, preferences (of the USER)
- Past experiences, goals, plans (of the USER)
- Opinions and values (of the USER)
- Any information that would personalize future conversations

What NOT to extract:
- Unknown information ("user's favorite X is unknown")
- Vague impressions or speculation
- Information about the assistant/character
- Obvious conversational filler
- Facts NOT mentioned in the conversation (DO NOT invent or assume)
- The assistant's name as the user's name

For each fact, provide a JSON object with:
- "content": Clear factual statement (e.g., "User's name is John")
- "category": One of [personal_info, preference, experience, relationship, goal, skill]
- "confidence": Float 0.0-1.0 (0.95 for explicit, 0.8 for clear implication, 0.7 for reasonable inference)
- "reasoning": One sentence explaining why you extracted this

EXAMPLES OF GOOD EXTRACTIONS:
- "My name is Sarah" → {{"content": "User's name is Sarah", "category": "personal_info", "confidence": 0.95, "reasoning": "User explicitly stated their name"}}
- "I love hiking" → {{"content": "User enjoys hiking", "category": "preference", "confidence": 0.9, "reasoning": "Direct statement of interest"}}
- "I work as a teacher" → {{"content": "User is a teacher", "category": "personal_info", "confidence": 0.95, "reasoning": "Occupation explicitly stated"}}

EXAMPLES OF BAD EXTRACTIONS (DO NOT extract these):
- {{"content": "User's favorite color is unknown", ...}} ← DO NOT extract unknowns
- {{"content": "User seems interested in AI", ...}} ← DO NOT extract vague impressions
- {{"content": "Assistant is helpful", ...}} ← DO NOT extract facts about assistant
- {{"content": "User's name is {character_name or 'Nova'}", ...}} ← DO NOT confuse assistant name with user name
- {{"content": "User likes photography", ...}} (when photography was NEVER mentioned) ← DO NOT invent facts
- {{"content": "User enjoys outdoor activities", ...}} (when user only said "Hello") ← DO NOT make assumptions from greetings

CRITICAL RULES:
1. If the user hasn't explicitly mentioned something, DO NOT extract it
2. DO NOT confuse greetings like "Hello, {character_name or 'Nova'}" with the user stating their own name
3. DO NOT invent hobbies, interests, or facts that weren't discussed
4. Only extract information that was ACTUALLY stated or clearly implied in THIS conversation

RESPONSE FORMAT:
Return ONLY a JSON array. No explanations, no markdown, just the array:
[
  {{"content": "User's name is John", "category": "personal_info", "confidence": 0.95, "reasoning": "User explicitly stated their name"}},
  {{"content": "User enjoys photography", "category": "preference", "confidence": 0.9, "reasoning": "User mentioned interest in landscape photography"}}
]

If no memorable facts are present, return: []
"""
    
    def _parse_extraction_response(
        self,
        response: str,
        messages: List[Message]
    ) -> List[ExtractedMemory]:
        """
        Parse LLM response into ExtractedMemory objects.
        
        Args:
            response: LLM response (should be JSON)
            messages: Original messages (for source tracking)
        
        Returns:
            List of ExtractedMemory objects
        """
        try:
            # Extract JSON from response (handle potential markdown wrapping)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            if not isinstance(data, list):
                logger.warning("Extraction response is not a list")
                return []
            
            # Convert to ExtractedMemory objects
            extracted = []
            message_ids = [msg.id for msg in messages]
            
            for item in data:
                try:
                    memory = ExtractedMemory(
                        content=item.get("content", ""),
                        category=item.get("category", "personal_info"),
                        confidence=float(item.get("confidence", 0.0)),
                        reasoning=item.get("reasoning", ""),
                        source_message_ids=message_ids
                    )
                    
                    # Validate basic structure
                    if not memory.content or not memory.content.strip():
                        logger.debug("Skipping empty memory content")
                        continue
                    
                    if not (0.0 <= memory.confidence <= 1.0):
                        logger.debug(f"Skipping memory with invalid confidence: {memory.confidence}")
                        continue
                    
                    # Defensive validation: Filter out memories about the assistant
                    content_lower = memory.content.lower()
                    if any(phrase in content_lower for phrase in [
                        "assistant is", "assistant's", "assistant has", "assistant can",
                        "character is", "character's", "character has"
                    ]):
                        logger.warning(f"Filtered out memory about assistant: {memory.content}")
                        continue
                    
                    # CRITICAL: Filter out system prompt instructions being extracted as user facts
                    system_prompt_indicators = [
                        "is uncensored", "is unrestricted", "is a helpful", "is truthful",
                        "is unbiased", "is designed to", "follows instructions", "provides assistance",
                        "can discuss", "without limitations", "no restrictions", "can help with",
                        "answer questions", "provide information", "assist with", "roleplay as"
                    ]
                    if any(indicator in content_lower for indicator in system_prompt_indicators):
                        logger.warning(f"Filtered out system prompt instruction extracted as memory: {memory.content}")
                        continue
                    
                    # CRITICAL: Filter out demographic hallucinations (gender, age, race without explicit statement)
                    # These are commonly hallucinated from names or greetings
                    demographic_assumptions = [
                        "is male", "is female", "is a man", "is a woman", "is a boy", "is a girl",
                        "is masculine", "is feminine", "years old", "age is", "ethnicity is",
                        "race is", "is caucasian", "is asian", "is african", "is hispanic"
                    ]
                    if any(assumption in content_lower for assumption in demographic_assumptions):
                        logger.warning(f"Filtered out demographic hallucination: {memory.content}")
                        continue
                    
                    # CRITICAL: Filter out conversation actions/events that aren't facts about the user
                    # These describe what happened in the conversation, not who the user is
                    conversation_actions = [
                        "user greeted", "user said hello", "user initiated", "user responded",
                        "user asked", "user requested", "user thanked", "user confirmed",
                        "user agreed", "user disagreed", "user laughed", "user joked",
                        "user started conversation", "user ended conversation"
                    ]
                    if any(action in content_lower for action in conversation_actions):
                        logger.warning(f"Filtered out conversation action (not a user fact): {memory.content}")
                        continue
                    
                    # Defensive validation: Filter out "unknown" memories
                    if "unknown" in content_lower or "not mentioned" in content_lower:
                        logger.warning(f"Filtered out 'unknown' memory: {memory.content}")
                        continue
                    
                    # Defensive validation: Must start with "User" to ensure it's about the user
                    if not content_lower.startswith("user"):
                        logger.warning(f"Filtered out memory not about user: {memory.content}")
                        continue
                    
                    extracted.append(memory)
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse extraction item: {e}")
                    continue
            
            return extracted
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return []
    
    async def _add_to_vector_store(self, memory: Memory, character_id: str) -> None:
        """Add memory to vector store."""
        try:
            # Generate vector_id if not present
            if not memory.vector_id:
                memory.vector_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_service.embed(memory.content)
            
            # Add to vector store
            success = self.vector_store.add_memories(
                character_id=character_id,
                memory_ids=[memory.vector_id],
                contents=[memory.content],
                embeddings=[embedding],
                metadatas=[{
                    "type": memory.memory_type.value,
                    "category": memory.category or "",
                    "confidence": memory.confidence or 0.0,
                    "status": memory.status
                }]
            )
            
            if success:
                # Update memory in database with vector_id
                self.memory_repo.db.commit()
                logger.debug(f"Added memory {memory.id} to vector store")
            else:
                logger.warning(f"Failed to add memory {memory.id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add memory to vector store: {e}", exc_info=True)
    
    async def _handle_duplicate(
        self,
        existing: Memory,
        extracted: ExtractedMemory
    ) -> Memory:
        """
        Handle duplicate/similar memory.
        
        Strategy:
        - If content significantly different: replace old memory
        - If content same: reinforce confidence
        
        Args:
            existing: Existing memory from database
            extracted: Newly extracted memory
        
        Returns:
            Updated or existing memory
        """
        try:
            # Simple strategy: if extracted has higher confidence, update
            if extracted.confidence > existing.confidence:
                logger.info(
                    f"Updating memory (old confidence={existing.confidence:.2f}, "
                    f"new confidence={extracted.confidence:.2f})"
                )
                # Update content and confidence
                existing.content = extracted.content
                existing.confidence = extracted.confidence
                existing.category = extracted.category
                self.memory_repo.db.commit()
                self.memory_repo.db.refresh(existing)
            else:
                # Reinforce confidence slightly
                new_confidence = min(1.0, existing.confidence + 0.05)
                self.memory_repo.update_confidence(existing.id, new_confidence)
                logger.info(f"Reinforced memory confidence: {existing.confidence:.2f} → {new_confidence:.2f}")
            
            return existing
            
        except Exception as e:
            logger.error(f"Failed to handle duplicate memory: {e}", exc_info=True)
            return existing
