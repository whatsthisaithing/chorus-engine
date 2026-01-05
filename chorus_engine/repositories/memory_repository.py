"""Repository for memory operations."""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Memory, MemoryType


class MemoryRepository:
    """Handle database operations for memories."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        content: str,
        character_id: str,  # Phase 4.1: Now required for all memories
        memory_type: MemoryType = MemoryType.EXPLICIT,
        conversation_id: Optional[str] = None,  # Phase 4.1: Now optional
        thread_id: Optional[str] = None,
        vector_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        priority: Optional[int] = None,
        tags: Optional[List[str]] = None,
        confidence: Optional[float] = None,  # Phase 4.1: For implicit memories
        category: Optional[str] = None,  # Phase 4.1: Memory category
        status: str = "approved",  # Phase 4.1: pending | approved | auto_approved
        source_messages: Optional[List[str]] = None,  # Phase 4.1: Source message IDs
        metadata: Optional[Dict[str, Any]] = None,
        emotional_weight: Optional[float] = None,  # Phase 8: Emotional significance
        participants: Optional[List[str]] = None,  # Phase 8: People involved
        key_moments: Optional[List[str]] = None  # Phase 8: Significant moments
    ) -> Memory:
        """
        Create a new memory.
        
        Phase 4.1 updates:
        - character_id is now required (primary scope)
        - conversation_id is now optional (for tracking origin)
        - Added confidence, category, status, source_messages for implicit memories
        
        Phase 8 updates:
        - Added emotional_weight, participants, key_moments fields
        
        Args:
            content: Memory content
            character_id: Character ID (required - primary scope)
            memory_type: Type of memory (core/explicit/implicit/ephemeral)
            conversation_id: Optional conversation ID (tracks origin conversation)
            thread_id: Optional associated thread
            vector_id: Optional vector database ID
            embedding_model: Optional embedding model name
            priority: Optional priority (0-100, default 50)
            tags: Optional list of tags
            confidence: Optional confidence score (0.0-1.0, for implicit memories)
            category: Optional memory category (personal_info, preference, etc.)
            status: Memory status (pending | approved | auto_approved)
            source_messages: Optional list of source message IDs
            metadata: Optional metadata (for additional flexible storage)
            emotional_weight: Optional emotional significance (0.0-1.0)
            participants: Optional list of people involved in the memory
            key_moments: Optional list of significant moments in the memory
        
        Returns:
            Created memory
        """
        memory = Memory(
            content=content,
            character_id=character_id,
            memory_type=memory_type,
            conversation_id=conversation_id,
            thread_id=thread_id,
            vector_id=vector_id,
            embedding_model=embedding_model,
            priority=priority if priority is not None else 50,
            tags=tags,
            confidence=confidence,
            category=category,
            status=status,
            source_messages=source_messages,
            meta_data=metadata or {},
            emotional_weight=emotional_weight,
            participants=participants,
            key_moments=key_moments
        )
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        return memory
    
    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Get memory by ID.
        
        Args:
            memory_id: Memory ID
        
        Returns:
            Memory or None if not found
        """
        return self.db.query(Memory).filter(Memory.id == memory_id).first()
    
    def list_by_conversation(self, conversation_id: str) -> List[Memory]:
        """
        List all memories for a conversation.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            List of memories, ordered by creation time
        """
        return (
            self.db.query(Memory)
            .filter(Memory.conversation_id == conversation_id)
            .order_by(Memory.created_at.desc())
            .all()
        )
    
    def list_by_thread(self, thread_id: str) -> List[Memory]:
        """
        List all memories for a thread.
        
        Args:
            thread_id: Thread ID
        
        Returns:
            List of memories, ordered by creation time
        """
        return (
            self.db.query(Memory)
            .filter(Memory.thread_id == thread_id)
            .order_by(Memory.created_at.desc())
            .all()
        )
    
    def list_by_character(
        self, 
        character_id: str, 
        memory_type: Optional[MemoryType] = None
    ) -> List[Memory]:
        """
        List all memories for a character.
        
        Args:
            character_id: Character ID
            memory_type: Optional filter by memory type
        
        Returns:
            List of memories, ordered by priority then creation time
        """
        query = self.db.query(Memory).filter(Memory.character_id == character_id)
        
        if memory_type:
            query = query.filter(Memory.memory_type == memory_type)
        
        return (
            query
            .order_by(Memory.priority.desc().nullslast(), Memory.created_at.desc())
            .all()
        )
    
    def get_by_character_and_type(
        self,
        character_id: str,
        memory_type: MemoryType,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Get memories by character and type (Phase 8).
        
        Args:
            character_id: Character ID
            memory_type: Memory type to filter by
            limit: Optional limit on results
        
        Returns:
            List of memories matching filters, ordered by creation date (newest first)
        """
        query = self.db.query(Memory).filter(
            Memory.character_id == character_id,
            Memory.memory_type == memory_type
        ).order_by(Memory.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
        
        Returns:
            True if deleted, False if not found
        """
        memory = self.get_by_id(memory_id)
        if not memory:
            return False
        
        self.db.delete(memory)
        self.db.commit()
        return True    
    def get_pending(self, character_id: Optional[str] = None) -> List[Memory]:
        """
        Get pending memories awaiting user review (Phase 4.1).
        
        Args:
            character_id: Optional filter by character
        
        Returns:
            List of pending memories, ordered by creation time
        """
        query = self.db.query(Memory).filter(Memory.status == "pending")
        
        if character_id:
            query = query.filter(Memory.character_id == character_id)
        
        return query.order_by(Memory.created_at.desc()).all()
    
    def update_status(self, memory_id: str, status: str) -> Optional[Memory]:
        """
        Update memory status (Phase 4.1).
        
        Args:
            memory_id: Memory ID
            status: New status (pending | approved | auto_approved)
        
        Returns:
            Updated memory or None if not found
        """
        memory = self.get_by_id(memory_id)
        if not memory:
            return None
        
        memory.status = status
        self.db.commit()
        self.db.refresh(memory)
        return memory
    
    def update_confidence(self, memory_id: str, confidence: float) -> Optional[Memory]:
        """
        Update memory confidence score (Phase 4.1).
        
        Args:
            memory_id: Memory ID
            confidence: New confidence (0.0-1.0)
        
        Returns:
            Updated memory or None if not found
        """
        memory = self.get_by_id(memory_id)
        if not memory:
            return None
        
        memory.confidence = confidence
        self.db.commit()
        self.db.refresh(memory)
        return memory
    
    def get_by_character_and_status(
        self,
        character_id: str,
        status: Optional[str] = None,
        memory_type: Optional[MemoryType] = None
    ) -> List[Memory]:
        """
        Get memories by character with optional status and type filters (Phase 4.1).
        
        Args:
            character_id: Character ID
            status: Optional status filter (pending | approved | auto_approved)
            memory_type: Optional memory type filter
        
        Returns:
            List of memories matching filters
        """
        query = self.db.query(Memory).filter(Memory.character_id == character_id)
        
        if status:
            query = query.filter(Memory.status == status)
        
        if memory_type:
            query = query.filter(Memory.memory_type == memory_type)
        
        return (
            query
            .order_by(Memory.priority.desc().nullslast(), Memory.created_at.desc())
            .all()
        )
    
    def get_memories_by_message(self, message_id: str) -> List[Memory]:
        """
        Get memories extracted from a specific message (Phase 8).
        
        Args:
            message_id: Message ID to check
            
        Returns:
            List of memories associated with this message
        """
        # source_messages is a JSON array field
        # We need to check if message_id is in the array
        memories = (
            self.db.query(Memory)
            .filter(Memory.source_messages.contains([message_id]))
            .all()
        )
        
        return memories
    
    def orphan_conversation_memories(self, conversation_id: str) -> int:
        """
        Orphan all memories from a conversation by setting conversation_id to null.
        Memories remain in the character's knowledge base but lose conversation context.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Number of memories orphaned
        """
        count = (
            self.db.query(Memory)
            .filter(Memory.conversation_id == conversation_id)
            .update({"conversation_id": None})
        )
        self.db.commit()
        return count
    
    def delete_by_conversation(self, conversation_id: str) -> int:
        """
        Delete all memories associated with a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Number of memories deleted
        """
        count = (
            self.db.query(Memory)
            .filter(Memory.conversation_id == conversation_id)
            .delete()
        )
        self.db.commit()
        return count
    
    def count_by_conversation(self, conversation_id: str) -> int:
        """
        Count memories associated with a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Number of memories
        """
        return (
            self.db.query(Memory)
            .filter(Memory.conversation_id == conversation_id)
            .count()
        )
