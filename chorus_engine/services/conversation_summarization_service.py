"""
Conversation Summarization Service

Handles smart summarization of long conversations to stay within token budgets.
Implements selective preservation of important messages and compression of filler.

Phase 8 - Day 8: Message Compression
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

from chorus_engine.models.conversation import Conversation, Message, Memory
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.services.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ConversationSummarizationService:
    """
    Manages conversation summarization for long conversations.
    
    Strategy:
    1. Always preserve recent N messages (window)
    2. Always preserve first message (establishes context)
    3. Preserve emotionally significant messages (high emotional_weight)
    4. Preserve user messages with extracted memories
    5. Discard pure filler (ok, yeah, thanks, etc.)
    6. Summarize everything else in groups
    
    Thresholds are dynamically calculated based on model context window:
    - TOKEN_THRESHOLD: 75% of context window
    - TARGET_TOKENS: 75% of threshold (to allow room for compression)
    - RECENT_MESSAGE_WINDOW: ~20% of threshold in tokens (~50 tokens/msg avg)
    """
    
    # Preservation strategy
    PRESERVE_EMOTIONAL_THRESHOLD = 0.7  # Preserve if emotional_weight >= 0.7
    
    # Recent window bounds
    MIN_RECENT_MESSAGES = 10   # Minimum recent messages to preserve (even tiny models)
    MAX_RECENT_MESSAGES = 100  # Maximum recent messages (practical limit)
    AVG_TOKENS_PER_MESSAGE = 50  # Average message length for window calculation
    
    def __init__(
        self,
        memory_repository: MemoryRepository,
        context_window: int,
        token_counter: Optional[TokenCounter] = None
    ):
        """
        Initialize summarization service.
        
        Args:
            memory_repository: For checking if messages have extracted memories
            context_window: Model's context window size in tokens
            token_counter: For calculating conversation tokens (optional, will create if None)
        """
        self.memory_repository = memory_repository
        self.token_counter = token_counter or TokenCounter()
        
        # Calculate dynamic thresholds based on context window
        # Start summarizing at 75% of context window
        self.token_threshold = int(context_window * 0.75)
        # Target 75% of threshold after compression (56.25% of context window)
        self.target_tokens = int(self.token_threshold * 0.75)
        
        # Calculate recent message window: ~20% of threshold in tokens
        # This scales naturally with model capability
        recent_token_budget = int(self.token_threshold * 0.20)
        self.recent_message_window = int(recent_token_budget / self.AVG_TOKENS_PER_MESSAGE)
        
        # Apply bounds
        self.recent_message_window = max(
            self.MIN_RECENT_MESSAGES,
            min(self.recent_message_window, self.MAX_RECENT_MESSAGES)
        )
    
    @property
    def TOKEN_THRESHOLD(self) -> int:
        """Get the current token threshold (for backwards compatibility)."""
        return self.token_threshold
    
    @property
    def TARGET_TOKENS(self) -> int:
        """Get the current target tokens (for backwards compatibility)."""
        return self.target_tokens
    
    @property
    def RECENT_MESSAGE_WINDOW(self) -> int:
        """Get the current recent message window (for backwards compatibility)."""
        return self.recent_message_window
    
    def should_summarize(self, conversation: Conversation) -> bool:
        """
        Check if conversation needs summarization.
        
        Args:
            conversation: Conversation to check
            
        Returns:
            True if total tokens exceed threshold
        """
        total_tokens = self._calculate_conversation_tokens(conversation)
        
        should_compress = total_tokens > self.TOKEN_THRESHOLD
        
        if should_compress:
            logger.info(
                f"Conversation {conversation.id} needs summarization: "
                f"{total_tokens} tokens > {self.TOKEN_THRESHOLD} threshold"
            )
        
        return should_compress
    
    def selective_preservation(self, conversation: Conversation) -> Dict[str, List[str]]:
        """
        Determine which messages to preserve, summarize, or discard.
        
        Strategy:
        - preserve_full: Recent messages, first message, emotional moments, messages with memories
        - summarize: Everything else (grouped for efficiency)
        - discard: Pure filler (ok, yeah, thanks)
        
        Args:
            conversation: Conversation to analyze
            
        Returns:
            Dict with three lists of message IDs:
            - preserve_full: Keep full message text
            - summarize: Compress into summaries
            - discard: Remove entirely
        """
        # Get all messages across all threads
        messages = self._get_all_messages(conversation)
        
        if len(messages) <= self.RECENT_MESSAGE_WINDOW:
            # Short conversation - preserve everything
            return {
                "preserve_full": [msg.id for msg in messages],
                "summarize": [],
                "discard": []
            }
        
        preserve_full = []
        summarize = []
        discard = []
        
        # Get recent message window (always preserve)
        recent_count = self.recent_message_window
        recent_messages = set(msg.id for msg in messages[-recent_count:])
        
        logger.info(
            f"Analyzing {len(messages)} messages for preservation strategy. "
            f"Recent window: {recent_count} messages"
        )
        
        for i, msg in enumerate(messages):
            # Always preserve recent messages
            if msg.id in recent_messages:
                preserve_full.append(msg.id)
                continue
            
            # Preserve first message (establishes context)
            if i == 0:
                preserve_full.append(msg.id)
                continue
            
            # Preserve emotionally significant messages
            if msg.emotional_weight and msg.emotional_weight >= self.PRESERVE_EMOTIONAL_THRESHOLD:
                preserve_full.append(msg.id)
                logger.debug(
                    f"Preserving message {msg.id} - high emotional weight: {msg.emotional_weight}"
                )
                continue
            
            # Preserve user messages with extracted memories
            if msg.role == 'user' and self._has_extracted_memories(msg.id):
                preserve_full.append(msg.id)
                logger.debug(f"Preserving message {msg.id} - has extracted memories")
                continue
            
            # Discard pure filler
            if self._is_filler(msg.content):
                discard.append(msg.id)
                logger.debug(f"Discarding message {msg.id} - pure filler")
                continue
            
            # Everything else: summarize
            summarize.append(msg.id)
        
        logger.info(
            f"Preservation strategy: {len(preserve_full)} preserve, "
            f"{len(summarize)} summarize, {len(discard)} discard"
        )
        
        return {
            "preserve_full": preserve_full,
            "summarize": summarize,
            "discard": discard
        }
    
    def _calculate_conversation_tokens(self, conversation: Conversation) -> int:
        """
        Calculate total tokens in conversation.
        
        Args:
            conversation: Conversation to count
            
        Returns:
            Total token count across all messages
        """
        messages = self._get_all_messages(conversation)
        
        total_tokens = 0
        for msg in messages:
            # Each message has role + content
            total_tokens += self.token_counter.count_tokens(f"{msg.role}: {msg.content}")
        
        logger.debug(f"Conversation {conversation.id} has {total_tokens} tokens across {len(messages)} messages")
        
        return total_tokens
    
    def _get_all_messages(self, conversation: Conversation) -> List[Message]:
        """
        Get all messages from conversation across all threads.
        
        Args:
            conversation: Conversation to get messages from
            
        Returns:
            Flat list of all messages, sorted by created_at timestamp
        """
        all_messages = []
        
        for thread in conversation.threads:
            all_messages.extend(thread.messages)
        
        # Sort by created_at timestamp to maintain chronological order
        all_messages.sort(key=lambda m: m.created_at)
        
        return all_messages
    
    def _has_extracted_memories(self, message_id: str) -> bool:
        """
        Check if message has any extracted memories associated with it.
        
        Args:
            message_id: Message ID to check
            
        Returns:
            True if message has memories
        """
        # Query memory repository for memories linked to this message
        memories = self.memory_repository.get_memories_by_message(message_id)
        return len(memories) > 0
    
    def _is_filler(self, content: str) -> bool:
        """
        Check if message is pure filler (can be safely discarded).
        
        Filler includes: ok, yeah, thanks, got it, sure, alright, yes, no
        
        Args:
            content: Message content to check
            
        Returns:
            True if message is pure filler
        """
        # Patterns for pure filler messages
        filler_patterns = [
            r'^(ok|okay|thanks|thank you|got it|sure|alright)\.?$',
            r'^(yes|yeah|yep|yup|no|nope)\.?$',
            r'^(cool|nice|great|awesome)\.?$',
            r'^(uh|um|hmm|hm)\.?$',
        ]
        
        content_lower = content.lower().strip()
        
        # Check if matches any filler pattern
        for pattern in filler_patterns:
            if re.match(pattern, content_lower):
                return True
        
        return False
