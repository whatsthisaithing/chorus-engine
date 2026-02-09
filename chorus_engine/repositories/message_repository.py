"""Repository for message operations."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Message, MessageRole

logger = logging.getLogger(__name__)


class MessageRepository:
    """Handle database operations for messages."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        thread_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_private: bool = False
    ) -> Message:
        """
        Create a new message.
        
        Args:
            thread_id: Parent thread ID
            role: Message role (system/user/assistant)
            content: Message content
            metadata: Optional metadata (tokens, finish_reason, etc.)
            is_private: Whether message was sent during privacy mode (Phase 4.1)
        
        Returns:
            Created message
        """
        message = Message(
            thread_id=thread_id,
            role=role,
            content=content,
            meta_data=metadata or {},
            is_private="true" if is_private else "false"
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message
    
    def get_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get message by ID.
        
        Args:
            message_id: Message ID
        
        Returns:
            Message or None if not found
        """
        return self.db.query(Message).filter(Message.id == message_id).first()
    
    def list_by_thread(self, thread_id: str, skip: int = 0, limit: int = 1000) -> List[Message]:
        """
        List all messages in a thread.
        
        Args:
            thread_id: Thread ID
            skip: Number to skip (pagination)
            limit: Maximum number to return (gets most recent N messages)
        
        Returns:
            List of messages, ordered by creation time (oldest first)
        """
        # Get the most recent N messages by ordering DESC, limiting, then reversing
        messages = (
            self.db.query(Message)
            .filter(Message.thread_id == thread_id)
            .filter(Message.deleted_at.is_(None))
            .order_by(Message.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        # Reverse to return in chronological order (oldest first)
        return list(reversed(messages))
    
    def get_thread_history(self, thread_id: str, limit: int = None) -> List[Dict[str, str]]:
        """
        Get message history formatted for LLM.
        
        Args:
            thread_id: Thread ID
            limit: Optional limit for number of most recent messages to return
        
        Returns:
            List of messages in LLM format: [{"role": "...", "content": "..."}]
        """
        if limit:
            # Get total count first
            total = (
                self.db.query(Message)
                .filter(Message.thread_id == thread_id)
                .filter(Message.deleted_at.is_(None))
                .count()
            )
            skip = max(0, total - limit)
            messages = self.list_by_thread(thread_id, skip=skip, limit=limit)
        else:
            messages = self.list_by_thread(thread_id)
        
        print(f"\n[MESSAGE REPO] Retrieved {len(messages)} messages from thread {thread_id}")
        result = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        print(f"[MESSAGE REPO] Formatted {len(result)} messages for LLM\n")
        return result
    
    def delete(self, message_id: str) -> bool:
        """
        Delete a message.
        
        Args:
            message_id: Message ID
        
        Returns:
            True if deleted, False if not found
        """
        message = self.get_by_id(message_id)
        if not message:
            return False
        
        self.db.delete(message)
        self.db.commit()
        return True
    
    def count_thread_messages(self, thread_id: str) -> int:
        """
        Count total messages in a thread.
        
        Args:
            thread_id: Thread ID
        
        Returns:
            Number of messages
        """
        return (
            self.db.query(Message)
            .filter(Message.thread_id == thread_id)
            .filter(Message.deleted_at.is_(None))
            .count()
        )
    
    def get_thread_history_objects(self, thread_id: str) -> List[Message]:
        """
        Get message history as Message objects (Phase 4.1).
        
        Args:
            thread_id: Thread ID
        
        Returns:
            List of Message objects, ordered by creation time
        """
        return self.list_by_thread(thread_id)

    def soft_delete(self, message_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        Soft delete messages by setting deleted_at.
        
        Args:
            message_ids: List of message IDs
        
        Returns:
            Tuple of (deleted_ids, skipped_ids)
        """
        if not message_ids:
            return [], []
        
        messages = (
            self.db.query(Message)
            .filter(Message.id.in_(message_ids))
            .all()
        )
        message_map = {msg.id: msg for msg in messages}
        
        deleted_ids: List[str] = []
        skipped_ids: List[str] = []
        
        now = datetime.utcnow()
        for message_id in message_ids:
            message = message_map.get(message_id)
            if not message:
                skipped_ids.append(message_id)
                continue
            if message.deleted_at is not None:
                skipped_ids.append(message_id)
                continue
            message.deleted_at = now
            deleted_ids.append(message_id)
        
        if deleted_ids:
            self.db.commit()
        
        return deleted_ids, skipped_ids
