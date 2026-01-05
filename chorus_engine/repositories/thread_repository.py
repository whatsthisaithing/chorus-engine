"""Repository for thread operations."""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Thread


class ThreadRepository:
    """Handle database operations for threads."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, conversation_id: str, title: str = "New Thread") -> Thread:
        """
        Create a new thread.
        
        Args:
            conversation_id: Parent conversation ID
            title: Thread title
        
        Returns:
            Created thread
        """
        thread = Thread(
            conversation_id=conversation_id,
            title=title
        )
        self.db.add(thread)
        self.db.commit()
        self.db.refresh(thread)
        return thread
    
    def get_by_id(self, thread_id: str) -> Optional[Thread]:
        """
        Get thread by ID.
        
        Args:
            thread_id: Thread ID
        
        Returns:
            Thread or None if not found
        """
        return self.db.query(Thread).filter(Thread.id == thread_id).first()
    
    def list_by_conversation(self, conversation_id: str) -> List[Thread]:
        """
        List all threads in a conversation.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            List of threads, ordered by creation time
        """
        return (
            self.db.query(Thread)
            .filter(Thread.conversation_id == conversation_id)
            .order_by(Thread.created_at.asc())
            .all()
        )
    
    def update(self, thread_id: str, title: Optional[str] = None) -> Optional[Thread]:
        """
        Update thread.
        
        Args:
            thread_id: Thread ID
            title: New title
        
        Returns:
            Updated thread or None if not found
        """
        thread = self.get_by_id(thread_id)
        if not thread:
            return None
        
        if title is not None:
            thread.title = title
        
        thread.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(thread)
        return thread
    
    def delete(self, thread_id: str) -> bool:
        """
        Delete thread (cascades to messages).
        
        Args:
            thread_id: Thread ID
        
        Returns:
            True if deleted, False if not found
        """
        thread = self.get_by_id(thread_id)
        if not thread:
            return False
        
        self.db.delete(thread)
        self.db.commit()
        return True
