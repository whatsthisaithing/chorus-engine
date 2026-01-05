"""
Temporal Weighting Service for Memory Intelligence (Phase 8).

Provides recency-based boosting for memory retrieval and conversation context analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc

from chorus_engine.models.conversation import Conversation

logger = logging.getLogger(__name__)


class TemporalWeightingService:
    """
    Service for applying temporal weighting to memory retrieval.
    
    Features:
    - Recency boost for most recent conversations
    - Time gap detection (continuing, recent, catching_up, welcoming_back)
    - Conversation position tracking (0-5 most recent get boost)
    """
    
    def __init__(self):
        """Initialize temporal weighting service."""
        logger.info("Temporal weighting service initialized")
    
    def get_recent_conversations(
        self,
        db: Session,
        character_id: str,
        limit: int = 10
    ) -> List[Conversation]:
        """
        Get the N most recent conversations for a character, ordered by last activity.
        
        Args:
            db: Database session
            character_id: Character to get conversations for
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversations ordered by most recent first
        """
        conversations = (
            db.query(Conversation)
            .filter(Conversation.character_id == character_id)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
            .all()
        )
        
        logger.debug(
            f"Retrieved {len(conversations)} recent conversations for {character_id}"
        )
        
        return conversations
    
    def calculate_recency_boost(
        self,
        conversation_id: str,
        recent_conversations: List[Conversation]
    ) -> float:
        """
        Calculate recency boost multiplier based on conversation position.
        
        Boost factors:
        - Position 0 (most recent): 1.2x
        - Position 1: 1.15x
        - Position 2: 1.1x
        - Position 3-5: 1.05x
        - Position 6+: 1.0x (neutral)
        
        Args:
            conversation_id: ID of conversation to check
            recent_conversations: List of recent conversations (already sorted)
            
        Returns:
            Boost multiplier (1.0 = neutral, >1.0 = boost)
        """
        try:
            # Find position of this conversation
            position = next(
                (i for i, conv in enumerate(recent_conversations) if conv.id == conversation_id),
                None
            )
            
            if position is None:
                # Conversation not in recent list, no boost
                return 1.0
            
            # Apply boost based on position
            if position == 0:
                boost = 1.2
            elif position == 1:
                boost = 1.15
            elif position == 2:
                boost = 1.1
            elif position <= 5:
                boost = 1.05
            else:
                boost = 1.0
            
            logger.debug(
                f"Conversation {conversation_id[:8]} at position {position}: {boost}x boost"
            )
            
            return boost
            
        except Exception as e:
            logger.warning(f"Error calculating recency boost: {e}")
            return 1.0
    
    def calculate_time_gap_context(
        self,
        current_conversation: Conversation,
        db: Session
    ) -> Tuple[str, Optional[timedelta]]:
        """
        Determine the time gap context for conversation opening.
        
        Contexts:
        - "continuing": <= 1 day since last interaction
        - "recent": <= 7 days since last interaction
        - "catching_up": <= 30 days since last interaction
        - "welcoming_back": 30+ days since last interaction
        - "first_time": No previous conversations
        
        Args:
            current_conversation: The conversation being started/continued
            db: Database session
            
        Returns:
            Tuple of (context_type, time_gap)
        """
        character_id = current_conversation.character_id
        
        # Get previous conversation (most recent before this one)
        previous = (
            db.query(Conversation)
            .filter(
                Conversation.character_id == character_id,
                Conversation.id != current_conversation.id,
                Conversation.updated_at < current_conversation.created_at
            )
            .order_by(desc(Conversation.updated_at))
            .first()
        )
        
        if not previous:
            logger.debug(f"First conversation with {character_id}")
            return ("first_time", None)
        
        # Calculate time gap
        time_gap = current_conversation.created_at - previous.updated_at
        
        # Determine context
        if time_gap <= timedelta(days=1):
            context = "continuing"
        elif time_gap <= timedelta(days=7):
            context = "recent"
        elif time_gap <= timedelta(days=30):
            context = "catching_up"
        else:
            context = "welcoming_back"
        
        logger.debug(
            f"Time gap context for {character_id}: {context} "
            f"({time_gap.days} days since last interaction)"
        )
        
        return (context, time_gap)
    
    def get_conversation_context_summary(
        self,
        conversation: Conversation,
        db: Session
    ) -> dict:
        """
        Get a full context summary for a conversation including temporal data.
        
        Args:
            conversation: The conversation to analyze
            db: Database session
            
        Returns:
            Dictionary with context information
        """
        recent_conversations = self.get_recent_conversations(
            db,
            conversation.character_id,
            limit=10
        )
        
        recency_boost = self.calculate_recency_boost(
            conversation.id,
            recent_conversations
        )
        
        time_gap_context, time_gap = self.calculate_time_gap_context(
            conversation,
            db
        )
        
        return {
            "conversation_id": conversation.id,
            "character_id": conversation.character_id,
            "recency_position": next(
                (i for i, c in enumerate(recent_conversations) if c.id == conversation.id),
                None
            ),
            "recency_boost": recency_boost,
            "time_gap_context": time_gap_context,
            "time_gap_days": time_gap.days if time_gap else None,
            "is_recent_conversation": recency_boost > 1.0,
            "total_recent_conversations": len(recent_conversations)
        }
