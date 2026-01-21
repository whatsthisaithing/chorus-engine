"""
Memory Retrieval Service

Intelligent memory retrieval with semantic search, priority ranking,
and token budget management.

Memory retrieval hierarchy (Phase 3):
1. CORE memories (immutable character backstory) - highest priority
2. EXPLICIT memories (user-created facts) - high priority  
3. IMPLICIT memories (extracted context) - medium priority
4. EPHEMERAL memories (temporary working memory) - lowest priority

Within each type, memories are ranked by:
- Semantic similarity to query
- Priority score (0-100)
- Recency (newer = higher)
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Memory, MemoryType, Conversation
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.temporal_weighting_service import TemporalWeightingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievedMemory:
    """A memory with retrieval metadata."""
    memory: Memory
    similarity: float
    rank_score: float  # Combined score for sorting


class MemoryRetrievalService:
    """
    Retrieves relevant memories using semantic search and priority ranking.
    
    Features:
    - Semantic search via vector embeddings
    - Token budget management
    - Memory type hierarchy (core > explicit > implicit > ephemeral)
    - Similarity thresholding
    - Priority-based ranking
    """
    
    def __init__(
        self,
        db: Session,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        temporal_weighting_service: Optional[TemporalWeightingService] = None,
        similarity_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize memory retrieval service.
        
        Args:
            db: Database session
            vector_store: Vector store for semantic search
            embedder: Embedding service for query encoding
            temporal_weighting_service: Optional temporal weighting service for recency boost
            similarity_thresholds: Min similarity by memory type (default uses config)
        """
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
        self.temporal_service = temporal_weighting_service or TemporalWeightingService()
        
        # Default similarity thresholds (can be overridden)
        # Phase 7.5: Lowered thresholds to catch more relevant results (tuned after testing)
        self.similarity_thresholds = similarity_thresholds or {
            MemoryType.CORE: 0.45,      # Very lenient for core memories (always want them)
            MemoryType.EXPLICIT: 0.55,  # Lowered from 0.65 - catch more explicit facts
            MemoryType.IMPLICIT: 0.50,  # Lowered from 0.60 - catch more extracted memories ("New Year's" query)
            MemoryType.FACT: 0.50,      # Phase 8: Same as IMPLICIT (renamed)
            MemoryType.PROJECT: 0.55,   # Phase 8: Slightly higher threshold for projects
            MemoryType.EXPERIENCE: 0.50, # Phase 8: Experiences can be fuzzy matched
            MemoryType.STORY: 0.55,     # Phase 8: Stories need decent similarity
            MemoryType.RELATIONSHIP: 0.60,  # Phase 8: Relationship memories need good match
            MemoryType.EPHEMERAL: 0.60  # Lowered from 0.70 - less strict for temporary context
        }
    
    def retrieve_memories(
        self,
        query: str,
        character_id: str,
        conversation_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        token_budget: int = 1000,
        max_memories: int = 20,
        include_types: Optional[List[MemoryType]] = None,
        conversation_source: Optional[str] = None  # Phase 3: Filter by source (web, discord, etc.)
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: User's message or context
            character_id: Character ID for core memories
            conversation_id: Optional conversation scope
            thread_id: Optional thread scope
            token_budget: Max tokens to use for memories
            max_memories: Max number of memories to return
            include_types: Memory types to include (default: all)
            
        Returns:
            List of RetrievedMemory objects, ranked by relevance
        """
        logger.info(f"Retrieving memories for query: '{query[:50]}...'")
        
        # Default to all memory types
        if include_types is None:
            include_types = [MemoryType.CORE, MemoryType.EXPLICIT, MemoryType.IMPLICIT, MemoryType.EPHEMERAL]
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Retrieve candidates from vector store (generous initial fetch)
        vector_results = self.vector_store.query_memories(
            character_id=character_id,
            query_embedding=query_embedding,
            n_results=max_memories * 3  # Fetch more than needed for filtering
        )
        
        # Get vector IDs and similarities
        if not vector_results or not vector_results.get('ids'):
            logger.info("No vector results found")
            return []
        
        vector_ids = vector_results['ids'][0]
        distances = vector_results['distances'][0]
        
        # Convert distances to similarities (cosine: 0=identical, 2=opposite)
        similarities = [1 - (d / 2) for d in distances]
        
        # Map vector IDs to similarities
        vector_similarity_map = dict(zip(vector_ids, similarities))
        
        logger.info(f"Vector store returned {len(vector_ids)} candidates")
        
        # Fetch full Memory objects from database
        memories = (
            self.db.query(Memory)
            .filter(Memory.vector_id.in_(vector_ids))
            .all()
        )
        
        # Filter and score memories
        retrieved = []
        for memory in memories:
            # Skip if wrong type
            if memory.memory_type not in include_types:
                continue
            
            # Skip if wrong character (for core memories)
            if memory.memory_type == MemoryType.CORE and memory.character_id != character_id:
                continue
            
            # Phase 4.1: Skip pending implicit memories (only include approved/auto_approved)
            if memory.memory_type == MemoryType.IMPLICIT:
                if memory.status not in ["approved", "auto_approved"]:
                    continue
            
            # Phase 3: Filter by conversation source (segregate Discord/web memories)
            if conversation_source and hasattr(memory, 'source'):
                # Only include memories from the same source (discord memories for discord, web for web)
                if memory.source != conversation_source:
                    continue
            
            # Skip if wrong conversation/thread (for non-core memories)
            # Phase 4.1: Core and Implicit memories are character-scoped (not conversation-scoped)
            if memory.memory_type not in [MemoryType.CORE, MemoryType.IMPLICIT]:
                if conversation_id and memory.conversation_id != conversation_id:
                    continue
                if thread_id and memory.thread_id != thread_id:
                    continue
            
            # Get similarity
            similarity = vector_similarity_map.get(memory.vector_id, 0.0)
            
            # Check similarity threshold
            threshold = self.similarity_thresholds.get(memory.memory_type, 0.70)
            if similarity < threshold:
                continue
            
            # Calculate rank score (combines similarity, priority, type hierarchy, temporal boost)
            rank_score = self._calculate_rank_score(
                memory, 
                similarity,
                conversation_id=conversation_id,
                character_id=character_id
            )
            
            retrieved.append(RetrievedMemory(
                memory=memory,
                similarity=similarity,
                rank_score=rank_score
            ))
        
        # Sort by rank score (highest first)
        retrieved.sort(key=lambda x: x.rank_score, reverse=True)
        
        # Apply token budget
        budgeted = self._apply_token_budget(retrieved, token_budget)
        
        # Limit to max memories
        final = budgeted[:max_memories]
        
        logger.info(
            f"Retrieved {len(final)} memories "
            f"(filtered from {len(retrieved)} candidates, "
            f"fetched {len(memories)} from DB)"
        )
        
        return final
    
    def _calculate_rank_score(
        self, 
        memory: Memory, 
        similarity: float,
        conversation_id: Optional[str] = None,
        character_id: Optional[str] = None
    ) -> float:
        """
        Calculate combined ranking score for a memory.
        
        Phase 8 Factors:
        - Semantic similarity (50%)
        - Priority score (30%)
        - Type hierarchy (15%)
        - Temporal boost (5% - recency for conversation-scoped memories)
        
        Args:
            memory: Memory object
            similarity: Semantic similarity (0-1)
            conversation_id: Optional conversation for temporal context
            character_id: Optional character for temporal context
            
        Returns:
            Combined score (higher = more relevant)
        """
        # Type weights (core memories always rank highest)
        type_weights = {
            MemoryType.CORE: 1.5,         # 150% weight - always highest
            MemoryType.EXPLICIT: 1.2,     # 120% weight - user-defined facts
            MemoryType.FACT: 1.0,         # 100% weight - extracted facts (renamed from IMPLICIT)
            MemoryType.PROJECT: 1.1,      # 110% weight - ongoing projects
            MemoryType.EXPERIENCE: 0.95,  # 95% weight - shared experiences
            MemoryType.STORY: 0.9,        # 90% weight - narratives
            MemoryType.RELATIONSHIP: 1.05, # 105% weight - relationship dynamics
            MemoryType.EPHEMERAL: 0.8     # 80% weight - temporary context
        }
        
        type_weight = type_weights.get(memory.memory_type, 1.0)
        
        # Normalize priority (0-100 to 0-1)
        priority_normalized = (memory.priority or 50) / 100.0
        
        # Phase 8: Calculate temporal boost (for conversation-scoped memories)
        temporal_boost = 1.0
        if conversation_id and memory.conversation_id == conversation_id and character_id:
            # Get recent conversations for this character
            recent_conversations = self.temporal_service.get_recent_conversations(
                self.db,
                character_id,
                limit=10
            )
            # Calculate boost based on conversation recency
            temporal_boost = self.temporal_service.calculate_recency_boost(
                conversation_id,
                recent_conversations
            )
        
        # Phase 8: Combined score with temporal weighting
        # - 50% similarity (how relevant to query)
        # - 30% priority (how important in general)
        # - 15% type hierarchy (core > explicit > fact > project > ...)
        # - 5% temporal boost (recency of conversation)
        score = (
            similarity * 0.5 +
            priority_normalized * 0.3 +
            type_weight * 0.15 +
            (temporal_boost - 1.0) * 0.05  # Boost above 1.0 contributes up to 5%
        )
        
        return score
    
    def _apply_token_budget(
        self,
        memories: List[RetrievedMemory],
        token_budget: int
    ) -> List[RetrievedMemory]:
        """
        Filter memories to fit within token budget.
        
        Uses rough estimation: ~4 chars per token (English average).
        
        Args:
            memories: Sorted list of memories
            token_budget: Maximum tokens to use
            
        Returns:
            Filtered list that fits budget
        """
        if token_budget <= 0:
            return []
        
        budgeted = []
        tokens_used = 0
        
        for retrieved in memories:
            # Rough token estimate (4 chars/token)
            memory_tokens = len(retrieved.memory.content) // 4
            
            if tokens_used + memory_tokens <= token_budget:
                budgeted.append(retrieved)
                tokens_used += memory_tokens
            else:
                # Budget exhausted
                break
        
        logger.debug(f"Token budget: {tokens_used}/{token_budget} used, {len(budgeted)} memories")
        
        return budgeted
    
    def retrieve_core_memories(
        self,
        character_id: str,
        query: Optional[str] = None,
        max_memories: int = 10
    ) -> List[RetrievedMemory]:
        """
        Retrieve core memories for a character.
        
        If query provided, uses semantic search. Otherwise returns all core
        memories sorted by priority.
        
        Args:
            character_id: Character ID
            query: Optional query for semantic filtering
            max_memories: Max memories to return
            
        Returns:
            List of core memories
        """
        if query:
            # Semantic search
            return self.retrieve_memories(
                query=query,
                character_id=character_id,
                include_types=[MemoryType.CORE],
                max_memories=max_memories,
                token_budget=10000  # Generous budget for core memories
            )
        else:
            # Get all core memories, sorted by priority
            memories = (
                self.db.query(Memory)
                .filter(
                    Memory.character_id == character_id,
                    Memory.memory_type == MemoryType.CORE
                )
                .order_by(Memory.priority.desc())
                .limit(max_memories)
                .all()
            )
            
            return [
                RetrievedMemory(
                    memory=m,
                    similarity=1.0,  # Not using similarity
                    rank_score=m.priority / 100.0
                )
                for m in memories
            ]
    
    def format_memories_for_prompt(
        self,
        memories: List[RetrievedMemory],
        include_metadata: bool = False
    ) -> str:
        """
        Format retrieved memories as a string for LLM prompt.
        
        Args:
            memories: List of retrieved memories
            include_metadata: Include similarity/priority in output
            
        Returns:
            Formatted string
        """
        if not memories:
            return ""
        
        lines = []
        
        for i, retrieved in enumerate(memories, 1):
            mem = retrieved.memory
            
            if include_metadata:
                metadata = f" [type={mem.memory_type.value}, priority={mem.priority}, sim={retrieved.similarity:.2f}]"
            else:
                metadata = ""
            
            lines.append(f"{i}. {mem.content}{metadata}")
        
        return "\n".join(lines)
