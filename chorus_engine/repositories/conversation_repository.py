"""Repository for conversation operations."""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Conversation


class ConversationRepository:
    """Handle database operations for conversations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, character_id: str, title: Optional[str] = None, source: str = "web") -> Conversation:
        """
        Create a new conversation.
        
        Args:
            character_id: The character this conversation is with
            title: Optional title (auto-generated if not provided)
            source: Source platform ("web", "discord", etc.)
        
        Returns:
            Created conversation
        """
        if not title:
            title = f"Conversation with {character_id} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        
        conversation = Conversation(
            character_id=character_id,
            title=title,
            source=source
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get conversation by ID.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Conversation or None if not found
        """
        return self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    def list_all(self, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """
        List all conversations.
        
        Args:
            skip: Number of conversations to skip (pagination)
            limit: Maximum number to return
        
        Returns:
            List of conversations
        """
        return (
            self.db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def list_by_character(self, character_id: str, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """
        List conversations for a specific character.
        
        Args:
            character_id: Character ID
            skip: Number to skip (pagination)
            limit: Maximum number to return
        
        Returns:
            List of conversations
        """
        return (
            self.db.query(Conversation)
            .filter(Conversation.character_id == character_id)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update(self, conversation_id: str, title: Optional[str] = None) -> Optional[Conversation]:
        """
        Update conversation.
        
        Args:
            conversation_id: Conversation ID
            title: New title
        
        Returns:
            Updated conversation or None if not found
        """
        conversation = self.get_by_id(conversation_id)
        if not conversation:
            return None
        
        if title is not None:
            conversation.title = title
            # Mark as manually set (user edited the title)
            conversation.title_auto_generated = 0
        
        conversation.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def delete(self, conversation_id: str) -> bool:
        """
        Delete conversation (cascades to threads and messages).
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            True if deleted, False if not found
        """
        conversation = self.get_by_id(conversation_id)
        if not conversation:
            return False
        
        self.db.delete(conversation)
        self.db.commit()
        return True    
    def set_private(self, conversation_id: str, is_private: bool) -> Optional[Conversation]:
        """
        Set conversation privacy flag (Phase 4.1).
        
        When a conversation is private, no implicit memory extraction occurs.
        
        Args:
            conversation_id: Conversation ID
            is_private: True to mark private, False to mark public
        
        Returns:
            Updated conversation or None if not found
        """
        conversation = self.get_by_id(conversation_id)
        if not conversation:
            return None
        
        conversation.is_private = "true" if is_private else "false"
        conversation.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def is_private(self, conversation_id: str) -> bool:
        """
        Check if conversation is private (Phase 4.1).
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            True if private, False if public or not found
        """
        conversation = self.get_by_id(conversation_id)
        if not conversation:
            return False
        
        return conversation.is_private == "true"