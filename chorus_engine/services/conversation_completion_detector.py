"""
Conversation Completion Detector for Phase 8.

Detects when conversations are "complete" and ready for analysis:
- Active threshold: ≥10,000 tokens (comprehensive conversation)
- Inactive threshold: ≥2,500 tokens + 24h inactive (shorter but complete)
- Sweep analysis: Periodically check all conversations

Character-scoped triggers prevent duplicate analysis across conversations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from chorus_engine.models.conversation import Conversation, Message, Thread
from chorus_engine.repositories.conversation_repository import ConversationRepository
from chorus_engine.repositories.message_repository import MessageRepository
from chorus_engine.services.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ConversationCompletionDetector:
    """
    Detects when conversations are complete and ready for whole-conversation analysis.
    
    Thresholds:
    - ACTIVE_THRESHOLD: 10,000 tokens - triggers immediate analysis
    - INACTIVE_THRESHOLD: 2,500 tokens + 24h inactivity - triggers on sweep
    - SWEEP_COOLDOWN: 3 hours between character-wide sweeps
    - INACTIVITY_HOURS: 24 hours without activity = inactive
    
    Triggers:
    - on_conversation_activation(): Check this conversation if ≥10K tokens
    - on_character_load(): Check all ≥10K conversations + sweep ≥2.5K inactive (if >3h cooldown)
    - on_timer_tick(): Check active conversation periodically
    """
    
    # Token thresholds
    ACTIVE_THRESHOLD = 10000  # Tokens - immediate analysis trigger
    INACTIVE_THRESHOLD = 2500  # Tokens - must also be inactive for 24h
    
    # Time thresholds
    SWEEP_COOLDOWN_HOURS = 3  # Hours between character-wide sweeps
    INACTIVITY_HOURS = 24  # Hours of inactivity to consider conversation "complete"
    
    # Timer interval
    TIMER_CHECK_MINUTES = 5  # Check active conversations every 5 minutes
    
    def __init__(
        self,
        db: Session,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    ):
        """
        Initialize completion detector.
        
        Args:
            db: Database session
            model_name: Model name for token counting
        """
        self.db = db
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        self.token_counter = TokenCounter(model_name)
    
    def on_conversation_activation(
        self,
        conversation_id: str
    ) -> bool:
        """
        Check if conversation should be analyzed when user opens it.
        
        Triggers analysis if:
        - Conversation has ≥10,000 tokens
        - Conversation hasn't been analyzed yet (or was analyzed long ago)
        
        Args:
            conversation_id: Conversation to check
            
        Returns:
            True if conversation should be analyzed
        """
        try:
            conversation = self.conversation_repo.get_by_id(conversation_id)
            if not conversation:
                return False
            
            # Check if already analyzed recently
            if self._was_recently_analyzed(conversation):
                logger.debug(f"[COMPLETION] Conversation {conversation_id} recently analyzed, skipping")
                return False
            
            # Count tokens in conversation
            token_count = self._count_conversation_tokens(conversation_id)
            
            if token_count >= self.ACTIVE_THRESHOLD:
                logger.info(
                    f"[COMPLETION] Conversation {conversation_id} ready for analysis "
                    f"({token_count} tokens ≥ {self.ACTIVE_THRESHOLD})"
                )
                return True
            
            logger.debug(
                f"[COMPLETION] Conversation {conversation_id} not ready "
                f"({token_count} tokens < {self.ACTIVE_THRESHOLD})"
            )
            return False
            
        except Exception as e:
            logger.error(f"Error checking conversation activation: {e}", exc_info=True)
            return False
    
    def on_character_load(
        self,
        character_id: str
    ) -> List[str]:
        """
        Check all conversations for a character when character is loaded.
        
        Triggers analysis for:
        1. All conversations ≥10,000 tokens (not recently analyzed)
        2. Sweep: All conversations ≥2,500 tokens + inactive 24h (if >3h since last sweep)
        
        Args:
            character_id: Character to check conversations for
            
        Returns:
            List of conversation IDs that should be analyzed
        """
        try:
            conversations_to_analyze = []
            
            # Get all conversations for this character
            # Note: We need to query directly since repository might not have get_by_character
            all_conversations = (
                self.db.query(Conversation)
                .filter(Conversation.character_id == character_id)
                .all()
            )
            
            logger.info(
                f"[COMPLETION] Checking {len(all_conversations)} conversations "
                f"for character {character_id}"
            )
            
            # Check if we should do a sweep (last sweep >3h ago)
            should_sweep = self._should_do_sweep(character_id)
            
            for conversation in all_conversations:
                # Skip recently analyzed
                if self._was_recently_analyzed(conversation):
                    continue
                
                # Count tokens
                token_count = self._count_conversation_tokens(conversation.id)
                
                # Check active threshold (always check these)
                if token_count >= self.ACTIVE_THRESHOLD:
                    conversations_to_analyze.append(conversation.id)
                    logger.info(
                        f"[COMPLETION] {conversation.id}: {token_count} tokens "
                        f"≥ {self.ACTIVE_THRESHOLD} (active threshold)"
                    )
                    continue
                
                # Check inactive threshold (only during sweep)
                if should_sweep and token_count >= self.INACTIVE_THRESHOLD:
                    if self._is_inactive(conversation):
                        conversations_to_analyze.append(conversation.id)
                        logger.info(
                            f"[COMPLETION] {conversation.id}: {token_count} tokens "
                            f"≥ {self.INACTIVE_THRESHOLD} + inactive (sweep)"
                        )
            
            if conversations_to_analyze:
                logger.info(
                    f"[COMPLETION] Character {character_id}: "
                    f"{len(conversations_to_analyze)} conversations ready for analysis"
                )
            
            return conversations_to_analyze
            
        except Exception as e:
            logger.error(f"Error checking character load: {e}", exc_info=True)
            return []
    
    def on_timer_tick(
        self,
        active_conversation_id: Optional[str] = None
    ) -> bool:
        """
        Periodic check for active conversation (every 5 minutes).
        
        Args:
            active_conversation_id: Currently active conversation (if any)
            
        Returns:
            True if conversation should be analyzed
        """
        if not active_conversation_id:
            return False
        
        try:
            # Same logic as on_conversation_activation
            return self.on_conversation_activation(active_conversation_id)
            
        except Exception as e:
            logger.error(f"Error checking timer tick: {e}", exc_info=True)
            return False
    
    def _count_conversation_tokens(self, conversation_id: str) -> int:
        """
        Count total tokens in a conversation.
        
        Args:
            conversation_id: Conversation to count
            
        Returns:
            Total token count
        """
        try:
            # Get all threads for this conversation
            threads = (
                self.db.query(Thread)
                .filter(Thread.conversation_id == conversation_id)
                .all()
            )
            
            if not threads:
                return 0
            
            thread_ids = [t.id for t in threads]
            
            # Get all messages across all threads
            messages = (
                self.db.query(Message)
                .filter(Message.thread_id.in_(thread_ids))
                .filter(Message.deleted_at.is_(None))
                .order_by(Message.created_at)
                .all()
            )
            
            # Format messages for token counting
            message_dicts = [
                {"role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role), 
                 "content": msg.content}
                for msg in messages
            ]
            
            # Count tokens
            token_count = self.token_counter.count_messages(message_dicts)
            
            return token_count
            
        except Exception as e:
            logger.warning(f"Failed to count tokens for conversation {conversation_id}: {e}")
            return 0
    
    def _was_recently_analyzed(self, conversation: Conversation) -> bool:
        """
        Check if conversation was analyzed recently.
        
        Args:
            conversation: Conversation to check
            
        Returns:
            True if analyzed within last 24 hours
        """
        if not hasattr(conversation, 'last_analyzed_at') or not conversation.last_analyzed_at:
            return False
        
        time_since_analysis = datetime.utcnow() - conversation.last_analyzed_at
        return time_since_analysis < timedelta(hours=24)
    
    def _is_inactive(self, conversation: Conversation) -> bool:
        """
        Check if conversation has been inactive for INACTIVITY_HOURS.
        
        Args:
            conversation: Conversation to check
            
        Returns:
            True if inactive for ≥24 hours
        """
        if not conversation.updated_at:
            return False
        
        time_since_update = datetime.utcnow() - conversation.updated_at
        return time_since_update >= timedelta(hours=self.INACTIVITY_HOURS)
    
    def _should_do_sweep(self, character_id: str) -> bool:
        """
        Check if we should do a sweep for this character.
        
        A sweep checks all conversations ≥2.5K tokens that are inactive.
        We only do sweeps every 3 hours to avoid overhead.
        
        Args:
            character_id: Character to check
            
        Returns:
            True if sweep should be performed
        """
        try:
            # Get last sweep time for this character
            # We'll store this in the most recently analyzed conversation
            last_sweep = (
                self.db.query(Conversation)
                .filter(
                    Conversation.character_id == character_id,
                    Conversation.last_analyzed_at.isnot(None)
                )
                .order_by(Conversation.last_analyzed_at.desc())
                .first()
            )
            
            if not last_sweep or not last_sweep.last_analyzed_at:
                # Never swept, do it now
                return True
            
            time_since_sweep = datetime.utcnow() - last_sweep.last_analyzed_at
            return time_since_sweep >= timedelta(hours=self.SWEEP_COOLDOWN_HOURS)
            
        except Exception as e:
            logger.warning(f"Failed to check sweep cooldown: {e}")
            return True  # Default to doing sweep on error
