"""Service for memory storage and persistence (Phase 4.1).

NOTE: LLM extraction now happens during analysis cycles. This service focuses
on storage, deduplication, and vector store integration.

This service provides:
- Memory storage and persistence layer
- Duplicate checking and deduplication
- Vector store integration
- Memory approval workflow
"""

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
    Service for memory storage, persistence, and management.
    
    Provides the storage layer for extracted memories.
    Handles:
    - Saving extracted memories with confidence-based approval
    - Duplicate detection via vector similarity
    - Vector store integration
    - Memory approval workflow
    
    NOTE: Extraction logic is handled elsewhere (analysis cycles).
    """
    
    def __init__(
        self,
        llm_client: LLMClient,  # Kept for backward compatibility, but no longer used
        memory_repository: MemoryRepository,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ):
        self.llm = llm_client  # No longer used - kept for backward compatibility
        self.memory_repo = memory_repository
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def check_for_duplicates(
        self,
        memory_content: str,
        character_id: str,
        where: Optional[Dict[str, Any]] = None
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
                where=where
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
        conversation_id: str,
        source: str = 'web'
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
                    durability="situational",
                    pattern_eligible=False,
                    emotional_weight=extracted.emotional_weight,
                    participants=extracted.participants,
                    key_moments=extracted.key_moments,
                    source=source  # Platform source: web, discord, slack, etc.
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
    
    async def _add_to_vector_store(self, memory: Memory, character_id: str) -> None:
        """Add memory to vector store."""
        try:
            logger.info(f"[VECTOR STORE] Starting add for memory {memory.id[:8]}... status={memory.status}")
            
            # Generate vector_id if not present
            if not memory.vector_id:
                vector_id = str(uuid.uuid4())
                logger.info(f"[VECTOR STORE] Generated new vector_id: {vector_id[:8]}...")
            else:
                vector_id = memory.vector_id
                logger.info(f"[VECTOR STORE] Using existing vector_id: {vector_id[:8]}...")
            
            # Generate embedding
            logger.info(f"[VECTOR STORE] Generating embedding for: {memory.content[:50]}...")
            embedding = self.embedding_service.embed(memory.content)
            logger.info(f"[VECTOR STORE] Embedding generated, size: {len(embedding)}")
            
            # Add to vector store
            logger.info(f"[VECTOR STORE] Adding to ChromaDB for character {character_id}")
            success = self.vector_store.add_memories(
                character_id=character_id,
                memory_ids=[vector_id],
                contents=[memory.content],
                embeddings=[embedding],
                metadatas=[{
                    "type": memory.memory_type.value,
                    "category": memory.category or "",
                    "confidence": memory.confidence or 0.0,
                    "status": memory.status,
                    "durability": memory.durability if hasattr(memory, "durability") else "situational",
                    "pattern_eligible": bool(getattr(memory, "pattern_eligible", 0))
                }]
            )
            
            logger.info(f"[VECTOR STORE] ChromaDB add result: {success}")
            
            if success:
                # Re-query the memory to get a fresh attached instance, then update vector_id
                logger.info(f"[VECTOR STORE] Re-querying memory {memory.id} to update vector_id in DB")
                fresh_memory = self.memory_repo.get_by_id(memory.id)
                if fresh_memory:
                    logger.info(f"[VECTOR STORE] Fresh memory retrieved, current vector_id: {fresh_memory.vector_id}")
                    fresh_memory.vector_id = vector_id
                    logger.info(f"[VECTOR STORE] Set vector_id to {vector_id[:8]}..., committing...")
                    self.memory_repo.db.commit()
                    logger.info(f"[VECTOR STORE] ✓ Committed! Memory {memory.id[:8]}... now has vector_id {vector_id[:8]}...")
                else:
                    logger.error(f"[VECTOR STORE] ✗ Could not re-query memory {memory.id} to update vector_id!")
            else:
                logger.warning(f"[VECTOR STORE] ✗ Failed to add memory {memory.id[:8]}... to vector store")
            
        except Exception as e:
            logger.error(f"[VECTOR STORE] ✗ Exception adding memory to vector store: {e}", exc_info=True)
    
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
