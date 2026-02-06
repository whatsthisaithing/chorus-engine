"""
Conversation Analysis Service for Phase 8.

Analyzes complete conversations to extract:
- Conversation summary (narrative)
- Archivist memories (durable, assistant-neutral)
- Participants and emotional arc
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session

from chorus_engine.llm.client import LLMClient
from chorus_engine.models.conversation import (
    Conversation, Message, Thread, ConversationSummary, 
    Memory, MemoryType
)
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.repositories.conversation_repository import ConversationRepository
from chorus_engine.services.token_counter import TokenCounter
from chorus_engine.services.memory_profile_service import MemoryProfileService
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.json_extraction import extract_json_block
from chorus_engine.config.models import CharacterConfig

logger = logging.getLogger(__name__)

# Debug log directory
DEBUG_LOG_DIR = Path("data/debug_logs/conversations")
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalyzedMemory:
    """Memory extracted from conversation analysis."""
    content: str
    memory_type: MemoryType
    confidence: float
    reasoning: str
    durability: str
    pattern_eligible: bool
    emotional_weight: Optional[float] = None
    participants: Optional[List[str]] = None
    key_moments: Optional[List[str]] = None


@dataclass
class ConversationAnalysis:
    """Result of conversation analysis."""
    memories: List[AnalyzedMemory]
    summary: str
    emotional_arc: str
    participants: List[str]
    open_questions: List[str]


class ConversationAnalysisService:
    """
    Analyzes complete conversations for memories and summaries.
    
    Used when conversations reach completion thresholds:
    - ≥10,000 tokens (comprehensive conversation)
    - ≥2,500 tokens + 24h inactive (shorter but complete)
    """
    
    def __init__(
        self,
        db: Session,
        llm_client: LLMClient,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        temperature: float = 0.1,
        summary_vector_store: Optional[ConversationSummaryVectorStore] = None,
        llm_usage_lock: Optional[Any] = None,
        archivist_model: Optional[str] = None,
        analysis_max_tokens_summary: int = 4096,
        analysis_max_tokens_memories: int = 4096
    ):
        self.db = db
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.temperature = temperature
        self.summary_vector_store = summary_vector_store
        self.llm_usage_lock = llm_usage_lock
        self.archivist_model = archivist_model
        self.analysis_max_tokens_summary = analysis_max_tokens_summary
        self.analysis_max_tokens_memories = analysis_max_tokens_memories
        
        # Services
        self.token_counter = TokenCounter()
        self.memory_profile_service = MemoryProfileService()
        self.memory_repo = MemoryRepository(db)
        self.conv_repo = ConversationRepository(db)
        self._summary_parse_mode: str = "unknown"
        self._archivist_parse_mode: str = "unknown"
    async def analyze_conversation(
        self,
        conversation_id: str,
        character: CharacterConfig,
        manual: bool = False
    ) -> Optional[ConversationAnalysis]:
        """
        Analyze a complete conversation.
        
        Args:
            conversation_id: Conversation to analyze
            character: Character configuration
            manual: Whether this is a manual analysis (from "Analyze Now" button)
            
        Returns:
            ConversationAnalysis if successful, None otherwise
        """
        try:
            # Get conversation and messages
            conversation = self.conv_repo.get_by_id(conversation_id)
            if not conversation:
                logger.error(f"Conversation {conversation_id} not found")
                return None
            
            messages = self._get_all_messages(conversation_id)
            if not messages:
                logger.warning(f"No messages in conversation {conversation_id}")
                return None
            
            token_count = self._count_tokens(messages)
            logger.info(f"Analyzing conversation {conversation_id[:8]}... ({token_count} tokens)")
            
            if token_count < 500:
                logger.warning(
                    f"Conversation {conversation_id[:8]}... too short ({token_count} tokens), skipping"
                )
                return None
            
            conversation_text = self._format_conversation(messages)
            model_primary, model_fallback = self._select_models(character)
            summary_model = self.archivist_model or model_primary or model_fallback
            if self.archivist_model and not summary_model:
                logger.warning(
                    "Archivist model configured but no fallback model available; "
                    "analysis may fail if the model is unavailable."
                )
            if not self.archivist_model and not summary_model:
                logger.warning(
                    "Archivist model not configured and no preferred model found; "
                    "falling back to system default."
                )
            
            # Step 1: Conversation summary
            summary_system_prompt, summary_user_prompt = self._build_summary_prompt(
                conversation_text=conversation_text,
                token_count=token_count
            )
            summary_data, summary_response = await self._call_and_parse(
                system_prompt=summary_system_prompt,
                user_prompt=summary_user_prompt,
                model_primary=summary_model,
                model_fallback=summary_model,
                max_tokens=self.analysis_max_tokens_summary,
                parser=self._parse_summary_response,
                label="summary",
                temperature=0.0
            )
            
            if not summary_data:
                logger.error("Summary parsing failed after retries")
                return None
            
            # Step 2: Archivist memory extraction
            archivist_system_prompt, archivist_user_prompt = self._build_archivist_prompt(
                conversation_text=conversation_text,
                token_count=token_count
            )
            memories, archivist_response = await self._call_and_parse(
                system_prompt=archivist_system_prompt,
                user_prompt=archivist_user_prompt,
                model_primary=summary_model,
                model_fallback=summary_model,
                max_tokens=self.analysis_max_tokens_memories,
                parser=lambda response: self._parse_archivist_response(response, character),
                label="archivist",
                temperature=0.0
            )
            
            memories = memories or []
            
            analysis = ConversationAnalysis(
                memories=memories,
                summary=summary_data.get("summary", ""),
                emotional_arc=summary_data.get("emotional_arc", ""),
                participants=summary_data.get("participants", []),
                open_questions=summary_data.get("open_questions", [])
            )
            
            self._write_debug_log(
                conversation_id=conversation_id,
                summary_prompt=summary_user_prompt,
                summary_system_prompt=summary_system_prompt,
                summary_response=summary_response or "",
                archivist_prompt=archivist_user_prompt,
                archivist_system_prompt=archivist_system_prompt,
                archivist_response=archivist_response or "",
                analysis=analysis,
                token_count=token_count
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing conversation {conversation_id}: {e}", exc_info=True)
            return None

    async def save_analysis(
        self,
        conversation_id: str,
        character_id: str,
        analysis: ConversationAnalysis,
        manual: bool = False
    ) -> bool:
        """
        Save analysis results to database.
        
        Args:
            conversation_id: Conversation ID
            character_id: Character ID
            analysis: Analysis results
            manual: Whether this is a manual analysis
            
        Returns:
            True if successful
        """
        try:
            # Save memories
            saved_memories = await self._save_extracted_memories(
                conversation_id=conversation_id,
                character_id=character_id,
                memories=analysis.memories
            )
            
            # Save conversation summary to database
            summary = self._save_conversation_summary(
                conversation_id=conversation_id,
                analysis=analysis,
                manual=manual
            )
            
            # Save summary to vector store for semantic search
            await self._save_summary_to_vector_store(
                conversation_id=conversation_id,
                character_id=character_id,
                summary=summary,
                analysis=analysis
            )
            
            # Update conversation last_analyzed_at
            conversation = self.conv_repo.get_by_id(conversation_id)
            if conversation:
                conversation.last_analyzed_at = datetime.utcnow()
                self.db.commit()
            
            logger.info(
                f"Saved analysis for {conversation_id[:8]}...: "
                f"{len(saved_memories)} memories, summary created"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}", exc_info=True)
            self.db.rollback()
            return False
    
    def _get_all_messages(self, conversation_id: str) -> List[Message]:
        """Get all messages in conversation across all threads."""
        threads = (
            self.db.query(Thread)
            .filter(Thread.conversation_id == conversation_id)
            .all()
        )
        
        if not threads:
            return []
        
        thread_ids = [t.id for t in threads]
        
        messages = (
            self.db.query(Message)
            .filter(Message.thread_id.in_(thread_ids))
            .order_by(Message.created_at)
            .all()
        )
        
        return messages
    
    def _count_tokens(self, messages: List[Message]) -> int:
        """Count total tokens in messages."""
        message_dicts = [
            {
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content
            }
            for msg in messages
        ]
        return self.token_counter.count_messages(message_dicts)
    def _build_summary_prompt(
        self,
        conversation_text: str,
        token_count: int
    ) -> tuple[str, str]:
        system_prompt = """You are a conversation analysis engine.

Your task is to produce a clear, concise, narrative summary of the conversation provided.

PURPOSE
- Capture the themes, insights, tensions, and shifts that occurred in the conversation.
- Preserve a human-readable understanding of what was explored and why it mattered.
- Support later reflection or review.

SCOPE AND CONSTRAINTS
- This is not a memory extraction task.
- Do not output facts, preferences, or durable memories.
- Do not speculate beyond what occurred in the conversation.

STYLE RULES (CRITICAL)
- Focus on outcomes, themes, and changes, not techniques.
- Do not foreground assistant style, rhetoric, or personality.
- Avoid describing how the assistant responded unless it is necessary to explain an effect on the conversation.
- The summary must remain valid if the assistant or model were replaced.

ALLOWED CONTENT
- Topics discussed
- Emotional or cognitive shifts
- Questions raised or resolved
- Reframes or insights acknowledged by the user
- Open threads or unresolved tensions

DISALLOWED CONTENT
- Assistant-specific traits or behaviors
- Commentary on assistant strategy or skill
- Diagnostic judgments
- New information not present in the conversation

OUTPUT FORMAT
Return a single JSON object:

{
  "summary": "A concise narrative summary of the conversation",
  "participants": ["user", "assistant"],
  "emotional_arc": "optional brief description",
  "open_questions": ["optional", "list"]
}

All fields except summary are optional.

Return only valid JSON. Do not include commentary or formatting."""

        user_prompt = f"""CONVERSATION ({token_count} tokens):
---
{conversation_text}
---
"""

        return system_prompt, user_prompt

    def _build_archivist_prompt(
        self,
        conversation_text: str,
        token_count: int
    ) -> tuple[str, str]:
        system_prompt = """You are an archivist system responsible for extracting durable, assistant-neutral memories from a completed conversation.

Your role is to identify information that may be useful in the future without freezing transient states, assistant behavior, or stylistic artifacts.

CORE PRINCIPLES (MANDATORY)
1. Assistant Neutrality
   - All memories must remain true if the assistant or model is replaced.
   - Do not store assistant behaviors, styles, or techniques as traits.
   - Prefer storing effects or outcomes, not causes rooted in assistant behavior.

2. Temporal Discipline
   - Write all memories in the past tense.
   - Avoid language that implies permanence unless explicitly justified.

3. Durability Awareness
   - Every memory must be classified by durability.
   - Default to conservative classifications.

4. Ephemeral State Exclusion
   - Transient states (current mood, sleep, immediate plans, location) must not be persisted.

5. Pattern Separation
   - A single memory is never a pattern.
   - Some memories may be marked as pattern-eligible, but patterns are inferred elsewhere.

MEMORY TYPES
- FACT: explicit user-stated facts or preferences
- PROJECT: ongoing or completed projects or goals
- EXPERIENCE: meaningful reflections, struggles, or insights
- STORY: personal narratives shared by the user
- RELATIONSHIP: explicitly described relationships or dynamics

DURABILITY CLASSIFICATION
- ephemeral: transient state (DO NOT PERSIST)
- situational: context-bound or time-limited relevance
- long_term: stable unless contradicted
- identity: explicitly self-asserted, core to self-description

RULES:
- Default to situational unless durability is clearly signaled.
- Use identity sparingly and only when the user explicitly self-identifies.
- Any memory classified as ephemeral should still be output but will be excluded from persistence by the system.

PATTERN-ELIGIBLE TAGGING
- Set pattern_eligible = true only if the memory could meaningfully contribute to a future pattern hypothesis across multiple conversations.
- Do not assert patterns or generalizations.

CONFIDENCE SCORING
- 0.9-1.0: Explicit user statement or very clear evidence
- 0.7-0.89: Reasonable inference grounded in context
- <0.7: Weak or speculative (avoid if possible)

OUTPUT FORMAT (REQUIRED)
Return a JSON array of memory objects:

[
  {
    "content": "memory text written in past tense",
    "type": "fact | project | experience | story | relationship",
    "confidence": 0.0,
    "durability": "ephemeral | situational | long_term | identity",
    "pattern_eligible": true,
    "reasoning": "brief explanation of why this was extracted"
  }
]

If no valid durable memories are found, return an empty array.

Return only valid JSON. Do not include commentary or formatting."""

        user_prompt = f"""CONVERSATION ({token_count} tokens):
---
{conversation_text}
---
"""

        return system_prompt, user_prompt

    async def _call_and_parse(
        self,
        system_prompt: str,
        user_prompt: str,
        model_primary: Optional[str],
        model_fallback: Optional[str],
        max_tokens: int,
        parser,
        label: str,
        temperature: Optional[float] = None
    ) -> tuple[Optional[Any], Optional[str]]:
        attempt_models = [model_primary]
        if model_fallback != model_primary:
            attempt_models.append(model_fallback)
        else:
            attempt_models.append(model_primary)

        last_response = None
        for attempt, model in enumerate(attempt_models, start=1):
            response_text = await self._call_llm_with_lock(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            last_response = response_text
            parsed = parser(response_text)
            if parsed is not None:
                return parsed, response_text
            logger.warning(
                f"{label.capitalize()} parsing failed on attempt {attempt}/2; retrying with fallback"
            )

        return None, last_response

    async def _call_llm_with_lock(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: Optional[float] = None
    ) -> str:
        temperature_to_use = self.temperature if temperature is None else temperature
        if self.llm_usage_lock is not None:
            async with self.llm_usage_lock:
                response = await self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature_to_use,
                    max_tokens=max_tokens
                )
        else:
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature_to_use,
                max_tokens=max_tokens
            )

        return response.content

    def _select_models(self, character: CharacterConfig) -> tuple[Optional[str], Optional[str]]:
        primary = character.preferred_llm.model if character.preferred_llm.model else None
        fallback = self.llm_client.model if not primary else None
        return primary, fallback

    def _parse_summary_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            data, parse_mode = extract_json_block(response, "object")
            self._summary_parse_mode = parse_mode
            if data is None:
                return None

            summary = str(data.get("summary", "")).strip()
            if not summary:
                return None

            participants = data.get("participants", [])
            if isinstance(participants, str):
                participants = [participants]
            if not isinstance(participants, list):
                participants = []

            emotional_arc = data.get("emotional_arc", "")
            if isinstance(emotional_arc, list):
                emotional_arc = " | ".join([str(item) for item in emotional_arc])
            emotional_arc = str(emotional_arc) if emotional_arc is not None else ""

            open_questions = data.get("open_questions", [])
            if isinstance(open_questions, str):
                open_questions = [open_questions]
            if not isinstance(open_questions, list):
                open_questions = []

            return {
                "summary": summary,
                "participants": participants,
                "emotional_arc": emotional_arc,
                "open_questions": open_questions
            }
        except Exception as e:
            logger.error(f"Error parsing summary response: {e}", exc_info=True)
            return None

    def _parse_archivist_response(
        self,
        response: str,
        character: CharacterConfig
    ) -> Optional[List[AnalyzedMemory]]:
        try:
            data, parse_mode = extract_json_block(response, "array")
            self._archivist_parse_mode = parse_mode
            if data is None:
                return None
            if not isinstance(data, list):
                return None

            profile = self.memory_profile_service.get_extraction_profile(character)
            allowed_types = set(self.memory_profile_service.get_allowed_types(character))
            allowed_types = {
                t for t in allowed_types
                if t in {
                    MemoryType.FACT,
                    MemoryType.PROJECT,
                    MemoryType.EXPERIENCE,
                    MemoryType.STORY,
                    MemoryType.RELATIONSHIP
                }
            }
            track_emotional_weight = profile.get("track_emotional_weight", True)
            track_participants = profile.get("track_participants", True)

            valid_durabilities = {"ephemeral", "situational", "long_term", "identity"}
            memories: List[AnalyzedMemory] = []

            for mem_data in data:
                try:
                    raw_type = str(mem_data.get("type", "")).strip().lower()
                    if "|" in raw_type:
                        raw_type = raw_type.split("|", 1)[0].strip()
                    memory_type = MemoryType(raw_type)
                    if memory_type not in allowed_types:
                        continue

                    durability = str(mem_data.get("durability", "situational")).strip().lower()
                    if durability not in valid_durabilities:
                        durability = "situational"

                    memory = AnalyzedMemory(
                        content=str(mem_data.get("content", "")).strip(),
                        memory_type=memory_type,
                        confidence=float(mem_data.get("confidence", 0.0)),
                        reasoning=str(mem_data.get("reasoning", "")),
                        durability=durability,
                        pattern_eligible=bool(mem_data.get("pattern_eligible", False)),
                        emotional_weight=float(mem_data.get("emotional_weight")) if track_emotional_weight and mem_data.get("emotional_weight") is not None else None,
                        participants=mem_data.get("participants") if track_participants else None,
                        key_moments=mem_data.get("key_moments")
                    )

                    if memory.content:
                        memories.append(memory)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid memory entry: {e}")
                    continue

            return memories
        except Exception as e:
            logger.error(f"Error parsing archivist response: {e}", exc_info=True)
            return None

    def _format_conversation(self, messages: List[Message]) -> str:
        """Format messages for analysis prompt."""
        lines = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            lines.append(f"{role.upper()}: {msg.content}")
        return "\n\n".join(lines)

    async def _save_extracted_memories(
        self,
        conversation_id: str,
        character_id: str,
        memories: List[AnalyzedMemory]
    ) -> List[Memory]:
        """Save extracted memories to database and vector store."""
        saved_memories = []
        
        for mem in memories:
            try:
                if mem.durability == "ephemeral":
                    logger.debug("Skipping ephemeral memory")
                    continue
                
                # Confidence thresholding
                if mem.confidence < 0.7:
                    logger.debug(f"Discarding low-confidence memory (confidence={mem.confidence:.2f})")
                    continue
                
                # Check for duplicates
                if self._is_duplicate_memory(character_id, mem.content):
                    logger.debug(f"Skipping duplicate memory: {mem.content[:50]}...")
                    continue
                
                # Determine status based on confidence
                if mem.confidence >= 0.9:
                    status = "auto_approved"
                else:
                    status = "pending"
                
                vector_id = None
                if status == "auto_approved":
                    # Generate embedding and save to vector store
                    embedding = self.embedding_service.embed(mem.content)
                    import uuid
                    vector_id = str(uuid.uuid4())
                    
                    metadata = {
                        "character_id": character_id,
                        "conversation_id": conversation_id,
                        "type": mem.memory_type.value,
                        "confidence": mem.confidence,
                        "durability": mem.durability,
                        "pattern_eligible": mem.pattern_eligible
                    }
                    self.vector_store.add_memories(
                        character_id=character_id,
                        memory_ids=[vector_id],
                        contents=[mem.content],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                
                # Create memory in database
                memory = self.memory_repo.create(
                    character_id=character_id,
                    content=mem.content,
                    memory_type=mem.memory_type,
                    vector_id=vector_id,
                    conversation_id=conversation_id,
                    status=status,
                    confidence=mem.confidence,
                    emotional_weight=mem.emotional_weight,
                    participants=mem.participants,
                    key_moments=mem.key_moments,
                    durability=mem.durability,
                    pattern_eligible=mem.pattern_eligible,
                    metadata={
                        "reasoning": mem.reasoning
                    },
                    category=None  # Analysis doesn't extract categories
                )
                
                saved_memories.append(memory)
                
            except Exception as e:
                logger.error(f"Error saving memory: {e}", exc_info=True)
                continue
        
        return saved_memories
    
    def _is_duplicate_memory(self, character_id: str, content: str) -> bool:
        """Check if memory already exists using vector similarity."""
        try:
            query_embedding = self.embedding_service.embed(content)
            results = self.vector_store.query_memories(
                character_id=character_id,
                query_embedding=query_embedding,
                n_results=1
            )
            
            if results and results.get("ids") and results["ids"][0]:
                distance = results["distances"][0][0]
                similarity = 1.0 - distance
                if similarity >= 0.85:
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking duplicates for {character_id}: {e}")
            return False
    
    def _save_conversation_summary(
        self,
        conversation_id: str,
        analysis: ConversationAnalysis,
        manual: bool = False
    ) -> ConversationSummary:
        """Save conversation summary to database."""
        # Get message count
        messages = self._get_all_messages(conversation_id)
        message_count = len(messages)
        
        # Create summary
        summary = ConversationSummary(
            conversation_id=conversation_id,
            summary=analysis.summary,
            message_range_start=0,
            message_range_end=message_count - 1,
            message_count=message_count,
            key_topics=None,
            participants=analysis.participants,
            emotional_arc=analysis.emotional_arc,
            tone=None,
            open_questions=analysis.open_questions,
            manual="true" if manual else "false"
        )
        
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        
        return summary
    
    async def _save_summary_to_vector_store(
        self,
        conversation_id: str,
        character_id: str,
        summary: ConversationSummary,
        analysis: ConversationAnalysis
    ) -> bool:
        """
        Save conversation summary to vector store for semantic search.
        
        Creates an embedded representation of the summary with rich metadata
        to enable finding relevant past conversations.
        
        Args:
            conversation_id: Conversation ID
            character_id: Character ID  
            summary: The ConversationSummary database object
            analysis: The ConversationAnalysis with summary metadata
            
        Returns:
            True if successful, False otherwise
        """
        if self.summary_vector_store is None:
            logger.debug("No summary vector store configured, skipping vector save")
            return True  # Not an error, just not configured
        
        try:
            # Get conversation for additional metadata
            conversation = self.conv_repo.get_by_id(conversation_id)
            
            # Build searchable text combining summary and open questions
            # This becomes the embedded document for semantic search
            searchable_text = f"{analysis.summary}"
            if analysis.open_questions:
                searchable_text += f"\nOpen Questions: {', '.join(analysis.open_questions)}"
            
            # Generate embedding
            embedding = self.embedding_service.embed(searchable_text)
            
            # Build metadata
            metadata = {
                "conversation_id": conversation_id,
                "character_id": character_id,
                "title": conversation.title if conversation else "Untitled",
                "created_at": conversation.created_at.isoformat() if conversation and conversation.created_at else "",
                "updated_at": conversation.updated_at.isoformat() if conversation and conversation.updated_at else "",
                "message_count": summary.message_count,
                "emotional_arc": analysis.emotional_arc or "",
                "participants": analysis.participants,  # Will be JSON serialized
                "open_questions": analysis.open_questions,  # Will be JSON serialized
                "source": conversation.source if conversation and hasattr(conversation, 'source') else "web",
                "analyzed_at": datetime.utcnow().isoformat(),
                "manual_analysis": summary.manual == "true"
            }
            
            # Upsert to vector store
            success = self.summary_vector_store.add_summary(
                character_id=character_id,
                conversation_id=conversation_id,
                summary_text=searchable_text,
                embedding=embedding,
                metadata=metadata
            )
            
            if success:
                logger.debug(f"Saved summary to vector store for conversation {conversation_id[:8]}...")
            else:
                logger.warning(f"Failed to save summary to vector store for {conversation_id[:8]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving summary to vector store: {e}", exc_info=True)
            return False
    
    def _write_debug_log(
        self,
        conversation_id: str,
        summary_prompt: str,
        summary_system_prompt: str,
        summary_response: str,
        archivist_prompt: str,
        archivist_system_prompt: str,
        archivist_response: str,
        analysis: Optional[ConversationAnalysis],
        token_count: int
    ):
        """Write debug log for analysis."""
        try:
            # Create conversation-specific directory
            conv_dir = DEBUG_LOG_DIR / conversation_id
            conv_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = conv_dir / f"analysis_{timestamp}.jsonl"
            
            # Write log entries
            with open(log_file, 'w', encoding='utf-8') as f:
                # Metadata
                f.write(json.dumps({
                    "type": "metadata",
                    "conversation_id": conversation_id,
                    "timestamp": timestamp,
                    "token_count": token_count
                }) + "\n")
                
                # Summary prompt + response
                f.write(json.dumps({
                    "type": "summary_prompt",
                    "system_prompt": summary_system_prompt,
                    "content": summary_prompt
                }) + "\n")
                
                f.write(json.dumps({
                    "type": "summary_response",
                    "content": summary_response,
                    "parse_mode": self._summary_parse_mode
                }) + "\n")
                
                # Archivist prompt + response
                f.write(json.dumps({
                    "type": "archivist_prompt",
                    "system_prompt": archivist_system_prompt,
                    "content": archivist_prompt
                }) + "\n")
                
                f.write(json.dumps({
                    "type": "archivist_response",
                    "content": archivist_response,
                    "parse_mode": self._archivist_parse_mode
                }) + "\n")
                
                # Parsed analysis
                if analysis:
                    f.write(json.dumps({
                        "type": "analysis",
                        "memory_count": len(analysis.memories),
                        "participants": analysis.participants,
                        "open_questions": analysis.open_questions
                    }) + "\n")
            
            logger.debug(f"Debug log written: {log_file}")
            
        except Exception as e:
            logger.error(f"Error writing debug log: {e}", exc_info=True)
