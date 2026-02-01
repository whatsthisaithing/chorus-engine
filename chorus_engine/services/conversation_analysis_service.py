"""
Conversation Analysis Service for Phase 8.

Analyzes complete conversations to extract:
- Comprehensive memories across all types
- Conversation summary
- Key themes and emotional arc
- Participants and significant moments
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
    emotional_weight: Optional[float] = None
    participants: Optional[List[str]] = None
    key_moments: Optional[List[str]] = None


@dataclass
class ConversationAnalysis:
    """Result of conversation analysis."""
    memories: List[AnalyzedMemory]
    summary: str
    themes: List[str]
    tone: str
    emotional_arc: str
    participants: List[str]
    key_topics: List[str]


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
        summary_vector_store: Optional[ConversationSummaryVectorStore] = None
    ):
        self.db = db
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.temperature = temperature
        self.summary_vector_store = summary_vector_store
        
        # Services
        self.token_counter = TokenCounter()
        self.memory_profile_service = MemoryProfileService()
        self.memory_repo = MemoryRepository(db)
        self.conv_repo = ConversationRepository(db)
    
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
            
            # Get all messages
            messages = self._get_all_messages(conversation_id)
            if not messages:
                logger.warning(f"No messages in conversation {conversation_id}")
                return None
            
            # Count tokens
            token_count = self._count_tokens(messages)
            logger.info(f"Analyzing conversation {conversation_id[:8]}... ({token_count} tokens)")
            
            # Quality check for very short conversations
            if token_count < 500:
                logger.warning(f"Conversation {conversation_id[:8]}... too short ({token_count} tokens), skipping")
                return None
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(
                messages=messages,
                character=character,
                token_count=token_count
            )
            
            # Get model to use (prefer character's model, fall back to client default)
            model = character.preferred_llm.model if character.preferred_llm.model else None
            
            # Call LLM for analysis
            llm_response = await self.llm_client.generate(
                prompt=prompt,
                model=model,
                temperature=self.temperature,
                max_tokens=4000  # Allow room for comprehensive analysis
            )
            
            # Extract text from LLMResponse
            response_text = llm_response.content
            
            # Parse response
            analysis = self._parse_analysis_response(response_text, character)
            
            # Write debug log
            self._write_debug_log(
                conversation_id=conversation_id,
                prompt=prompt,
                response=response_text,
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
    
    def _build_analysis_prompt(
        self,
        messages: List[Message],
        character: CharacterConfig,
        token_count: int
    ) -> str:
        """
        Build prompt for conversation analysis.
        
        Includes:
        - Full conversation
        - Memory type definitions
        - Extraction guidelines
        - Output schema
        """
        # Get memory profile
        profile = self.memory_profile_service.get_extraction_profile(character)
        
        # Format conversation
        conversation_text = self._format_conversation(messages)
        
        # Build memory type instructions
        type_instructions = self._build_type_instructions(profile)
        
        prompt = f"""You are analyzing a complete conversation to extract comprehensive memories and create a summary.

CONVERSATION ({token_count} tokens):
---
{conversation_text}
---

{type_instructions}

EXTRACTION GUIDELINES:
1. Extract ALL significant information across enabled memory types
2. Look for patterns, themes, and relationships
3. Identify key moments and emotional turning points
4. Note participants and their roles
5. Be thorough - this is a complete conversation analysis

OUTPUT FORMAT (JSON):
{{
  "memories": [
    {{
      "content": "Clear, specific memory statement",
      "type": "fact|project|experience|story|relationship",
      "confidence": 0.0-1.0,
      "reasoning": "Why this is significant",
      "emotional_weight": 0.0-1.0 (optional),
      "participants": ["person1", "person2"] (optional),
      "key_moments": ["moment1", "moment2"] (optional)
    }}
  ],
  "summary": "2-3 sentence conversation summary",
  "themes": ["theme1", "theme2", "theme3"],
  "tone": "overall emotional tone",
  "emotional_arc": ["start: emotion", "middle: emotion", "end: emotion"],
  "participants": ["all people mentioned"],
  "key_topics": ["topic1", "topic2", "topic3"]
}}

Analyze the conversation and respond with the JSON object:"""
        
        return prompt
    
    def _format_conversation(self, messages: List[Message]) -> str:
        """Format messages for analysis prompt."""
        lines = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            lines.append(f"{role.upper()}: {msg.content}")
        return "\n\n".join(lines)
    
    def _build_type_instructions(self, profile: Dict[str, bool]) -> str:
        """Build memory type instructions based on profile."""
        type_defs = {
            "fact": "FACT: Factual information (name, preferences, simple statements)",
            "project": "PROJECT: Goals, plans, ongoing work, future intentions",
            "experience": "EXPERIENCE: Shared activities, events, interactions",
            "story": "STORY: Narratives, anecdotes, personal stories",
            "relationship": "RELATIONSHIP: Emotional bonds, dynamics, connection evolution"
        }
        
        enabled_types = []
        for type_name, enabled in profile.items():
            # Extract type from profile key (e.g., "extract_facts" -> "fact")
            type_key = type_name.replace("extract_", "").rstrip("s")
            if enabled and type_key in type_defs:
                enabled_types.append(type_defs[type_key])
        
        if not enabled_types:
            return "MEMORY TYPES: Extract facts and projects only."
        
        return "MEMORY TYPES:\n" + "\n".join(enabled_types)
    
    def _parse_analysis_response(
        self,
        response: str,
        character: CharacterConfig
    ) -> Optional[ConversationAnalysis]:
        """Parse LLM response into ConversationAnalysis."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                logger.error("No JSON found in analysis response")
                return None
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Parse memories
            memories = []
            for mem_data in data.get("memories", []):
                try:
                    memory = AnalyzedMemory(
                        content=mem_data["content"],
                        memory_type=MemoryType(mem_data["type"]),
                        confidence=float(mem_data["confidence"]),
                        reasoning=mem_data.get("reasoning", ""),
                        emotional_weight=float(mem_data["emotional_weight"]) if mem_data.get("emotional_weight") else None,
                        participants=mem_data.get("participants"),
                        key_moments=mem_data.get("key_moments")
                    )
                    memories.append(memory)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid memory: {e}")
                    continue
            
            # Build analysis
            emotional_arc_data = data.get("emotional_arc", [])
            # Convert to JSON string for storage
            emotional_arc_str = json.dumps(emotional_arc_data) if emotional_arc_data else "[]"
            
            analysis = ConversationAnalysis(
                memories=memories,
                summary=data.get("summary", ""),
                themes=data.get("themes", []),
                tone=data.get("tone", ""),
                emotional_arc=emotional_arc_str,
                participants=data.get("participants", []),
                key_topics=data.get("key_topics", [])
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}", exc_info=True)
            return None
    
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
                # Check for duplicates
                if self._is_duplicate_memory(character_id, mem.content):
                    logger.debug(f"Skipping duplicate memory: {mem.content[:50]}...")
                    continue
                
                # Generate embedding
                embedding = self.embedding_service.embed(mem.content)
                
                # Generate unique vector ID
                import uuid
                vector_id = str(uuid.uuid4())
                
                # Save to vector store first
                metadata = {
                    "character_id": character_id,
                    "conversation_id": conversation_id,
                    "type": mem.memory_type.value,
                    "confidence": mem.confidence
                }
                self.vector_store.add_memories(
                    character_id=character_id,
                    memory_ids=[vector_id],
                    contents=[mem.content],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                
                # Determine status based on confidence
                if mem.confidence >= 0.9:
                    status = "auto_approved"
                elif mem.confidence >= 0.7:
                    status = "approved"
                else:
                    status = "pending"
                
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
                    category=None  # Analysis doesn't extract categories
                )
                
                saved_memories.append(memory)
                
            except Exception as e:
                logger.error(f"Error saving memory: {e}", exc_info=True)
                continue
        
        return saved_memories
    
    def _is_duplicate_memory(self, character_id: str, content: str) -> bool:
        """Check if memory already exists (simple content match)."""
        try:
            existing = self.memory_repo.list_by_character(character_id)
            content_lower = content.lower().strip()
            
            # Check first 100 memories (performance optimization)
            for mem in existing[:100]:
                if mem.content.lower().strip() == content_lower:
                    return True
            
            return False
        except Exception as e:
            # If there's an error (e.g., old IMPLICIT enum values), log and skip
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
            key_topics=analysis.key_topics,
            participants=analysis.participants,
            emotional_arc=analysis.emotional_arc,
            tone=analysis.tone,
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
            analysis: The ConversationAnalysis with themes and other data
            
        Returns:
            True if successful, False otherwise
        """
        if self.summary_vector_store is None:
            logger.debug("No summary vector store configured, skipping vector save")
            return True  # Not an error, just not configured
        
        try:
            # Get conversation for additional metadata
            conversation = self.conv_repo.get_by_id(conversation_id)
            
            # Build searchable text combining summary and key topics
            # This becomes the embedded document for semantic search
            searchable_text = f"{analysis.summary}"
            if analysis.themes:
                searchable_text += f"\nThemes: {', '.join(analysis.themes)}"
            if analysis.key_topics:
                searchable_text += f"\nTopics: {', '.join(analysis.key_topics)}"
            
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
                "themes": analysis.themes,  # Will be JSON serialized
                "tone": analysis.tone or "",
                "emotional_arc": analysis.emotional_arc or "",
                "key_topics": analysis.key_topics,  # Will be JSON serialized
                "participants": analysis.participants,  # Will be JSON serialized
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
        prompt: str,
        response: str,
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
                
                # Prompt
                f.write(json.dumps({
                    "type": "prompt",
                    "content": prompt
                }) + "\n")
                
                # Response
                f.write(json.dumps({
                    "type": "response",
                    "content": response
                }) + "\n")
                
                # Parsed analysis
                if analysis:
                    f.write(json.dumps({
                        "type": "analysis",
                        "memory_count": len(analysis.memories),
                        "themes": analysis.themes,
                        "tone": analysis.tone,
                        "participants": analysis.participants
                    }) + "\n")
            
            logger.debug(f"Debug log written: {log_file}")
            
        except Exception as e:
            logger.error(f"Error writing debug log: {e}", exc_info=True)
