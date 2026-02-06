"""
Conversation Context Retrieval Service.

This service retrieves relevant past conversation summaries to enrich
the character's context when responding to new messages. It uses semantic
search to find conversations that are relevant to the current topic.

Two modes:
1. TRIGGERED: User explicitly references past conversations (e.g., "remember when...")
   - Lower threshold (0.55) - more inclusive matching
   
2. PASSIVE: No explicit reference, but topic similarity detected
   - Higher threshold (0.75) - only strong matches

This allows the character to naturally recall relevant past discussions
without being prompted, while being more liberal when the user asks.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.token_counter import TokenCounter

logger = logging.getLogger(__name__)


# Trigger phrases that indicate user is referencing past conversations
TRIGGER_PHRASES = [
    # Direct memory references
    r"remember when",
    r"you remember",
    r"do you remember",
    r"recall when",
    r"you mentioned",
    r"you said",
    r"you told me",
    r"we talked about",
    r"we discussed",
    r"we were discussing",
    r"last time",
    r"before when",
    r"earlier you",
    r"previously",
    r"in the past",
    r"back when",
    # Questions about past
    r"what did you say about",
    r"what did you think about",
    r"what was your opinion on",
    r"didn't you say",
    r"wasn't it you who",
    # Continuity references
    r"as we discussed",
    r"as you said",
    r"like you mentioned",
    r"following up on",
    r"going back to",
    r"returning to",
    r"about that thing",
]

# Compile patterns for efficiency
TRIGGER_PATTERNS = [re.compile(phrase, re.IGNORECASE) for phrase in TRIGGER_PHRASES]


@dataclass
class ConversationContextConfig:
    """Configuration for conversation context retrieval."""
    enabled: bool = True
    passive_threshold: float = 0.75  # High bar for unsolicited context
    triggered_threshold: float = 0.55  # Lower bar when user asks
    max_summaries: int = 2  # Max summaries to include
    token_budget_ratio: float = 0.05  # 5% of context for summaries


@dataclass
class RetrievedConversationContext:
    """A retrieved conversation summary with relevance info."""
    conversation_id: str
    title: str
    summary: str
    similarity: float
    tone: Optional[str] = None
    key_topics: Optional[List[str]] = None
    open_questions: Optional[List[str]] = None
    message_count: int = 0
    created_at: Optional[str] = None


class ConversationContextRetrievalService:
    """
    Service for retrieving relevant past conversation summaries.
    
    Enriches character responses by including relevant historical context
    when the topic matches or when the user explicitly asks about past discussions.
    """
    
    def __init__(
        self,
        summary_vector_store: ConversationSummaryVectorStore,
        embedding_service: EmbeddingService,
        token_counter: Optional[TokenCounter] = None,
        config: Optional[ConversationContextConfig] = None
    ):
        """
        Initialize the conversation context retrieval service.
        
        Args:
            summary_vector_store: Vector store for conversation summaries
            embedding_service: Service for generating embeddings
            token_counter: Optional token counter for budget management
            config: Optional configuration (defaults used if not provided)
        """
        self.summary_vector_store = summary_vector_store
        self.embedding_service = embedding_service
        self.token_counter = token_counter
        self.config = config or ConversationContextConfig()
        
        logger.debug(
            f"ConversationContextRetrievalService initialized: "
            f"passive_threshold={self.config.passive_threshold}, "
            f"triggered_threshold={self.config.triggered_threshold}"
        )
    
    def should_include_summaries(self, user_message: str) -> Tuple[bool, float, bool]:
        """
        Determine if conversation summaries should be included based on message.
        
        Checks for trigger phrases that indicate the user is referencing
        past conversations.
        
        Args:
            user_message: The user's message text
            
        Returns:
            Tuple of (should_include, threshold_to_use, is_triggered)
            - should_include: Always True (we always try retrieval)
            - threshold_to_use: The similarity threshold (lower if triggered)
            - is_triggered: Whether a trigger phrase was detected
        """
        if not self.config.enabled:
            return False, 1.0, False
        
        # Check for trigger phrases
        message_lower = user_message.lower()
        is_triggered = any(pattern.search(message_lower) for pattern in TRIGGER_PATTERNS)
        
        if is_triggered:
            logger.debug(f"Trigger phrase detected in message")
            return True, self.config.triggered_threshold, True
        
        # No trigger - use passive threshold
        return True, self.config.passive_threshold, False
    
    def retrieve_relevant_summaries(
        self,
        character_id: str,
        user_message: str,
        current_conversation_id: Optional[str] = None,
        max_summaries: Optional[int] = None,
        token_budget: Optional[int] = None
    ) -> Tuple[List[RetrievedConversationContext], int]:
        """
        Retrieve conversation summaries relevant to the user's message.
        
        Args:
            character_id: Character ID to search within
            user_message: The user's message to match against
            current_conversation_id: Current conversation ID (to exclude from results)
            max_summaries: Maximum summaries to return (uses config default if None)
            token_budget: Optional token budget for summaries
            
        Returns:
            Tuple of (list of RetrievedConversationContext, tokens_used)
        """
        if not self.config.enabled:
            return [], 0
        
        max_results = max_summaries or self.config.max_summaries
        
        # Determine threshold based on message
        should_include, threshold, is_triggered = self.should_include_summaries(user_message)
        
        if not should_include:
            return [], 0
        
        try:
            # Generate embedding for user message
            query_embedding = self.embedding_service.embed(user_message)
            
            # Search with extra results to account for filtering
            search_limit = max_results * 3  # Get extra in case of filtering
            
            results = self.summary_vector_store.search_conversations(
                character_id=character_id,
                query_embedding=query_embedding,
                n_results=search_limit
            )
            
            if not results.get('ids') or not results['ids'][0]:
                logger.debug(f"No conversation summaries found for character {character_id}")
                return [], 0
            
            # Process results
            retrieved = []
            
            ids = results['ids'][0]
            distances = results.get('distances', [[]])[0]
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            for i, conv_id in enumerate(ids):
                # Skip current conversation
                if current_conversation_id and conv_id == current_conversation_id:
                    logger.debug(f"Excluding current conversation {conv_id[:8]}...")
                    continue
                
                # Calculate similarity (ChromaDB returns L2 distance for cosine)
                # For cosine similarity: similarity = 1 - (distance / 2)
                distance = distances[i] if i < len(distances) else 1.0
                similarity = max(0, 1 - (distance / 2))
                
                # Apply threshold
                if similarity < threshold:
                    logger.debug(
                        f"Conversation {conv_id[:8]}... below threshold "
                        f"({similarity:.3f} < {threshold})"
                    )
                    continue
                
                # Extract metadata
                metadata = metadatas[i] if i < len(metadatas) else {}
                summary_text = documents[i] if i < len(documents) else ""
                
                # Handle key_topics which may be a JSON string or list
                key_topics = metadata.get('key_topics', [])
                if isinstance(key_topics, str):
                    import json
                    try:
                        key_topics = json.loads(key_topics)
                    except:
                        key_topics = []
                
                # Handle open_questions which may be a JSON string or list
                open_questions = metadata.get('open_questions', [])
                if isinstance(open_questions, str):
                    import json
                    try:
                        open_questions = json.loads(open_questions)
                    except:
                        open_questions = []

                retrieved.append(RetrievedConversationContext(
                    conversation_id=conv_id,
                    title=metadata.get('title', 'Untitled'),
                    summary=summary_text,
                    similarity=similarity,
                    tone=metadata.get('tone'),
                    key_topics=key_topics,
                    open_questions=open_questions,
                    message_count=metadata.get('message_count', 0),
                    created_at=metadata.get('created_at')
                ))
                
                # Stop if we have enough
                if len(retrieved) >= max_results:
                    break
            
            # Fit to token budget if specified
            tokens_used = 0
            if token_budget and self.token_counter and retrieved:
                retrieved, tokens_used = self._fit_to_budget(retrieved, token_budget)
            elif retrieved and self.token_counter:
                # Just count tokens
                formatted = self.format_summaries_for_prompt(retrieved)
                tokens_used = self.token_counter.count_tokens(formatted)
            
            if retrieved:
                logger.info(
                    f"Retrieved {len(retrieved)} relevant conversation(s) for context "
                    f"(triggered={is_triggered}, threshold={threshold})"
                )
            
            return retrieved, tokens_used
            
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return [], 0
    
    def _fit_to_budget(
        self,
        summaries: List[RetrievedConversationContext],
        token_budget: int
    ) -> Tuple[List[RetrievedConversationContext], int]:
        """
        Fit summaries to token budget, removing lowest-similarity items first.
        
        Args:
            summaries: List of retrieved summaries (already sorted by similarity)
            token_budget: Maximum tokens allowed
            
        Returns:
            Tuple of (fitted summaries list, tokens used)
        """
        if not self.token_counter:
            return summaries, 0
        
        # Start with all summaries
        fitted = list(summaries)
        
        while fitted:
            formatted = self.format_summaries_for_prompt(fitted)
            tokens = self.token_counter.count_tokens(formatted)
            
            if tokens <= token_budget:
                return fitted, tokens
            
            # Remove lowest similarity item
            fitted.pop()
        
        return [], 0
    
    def format_summaries_for_prompt(
        self,
        summaries: List[RetrievedConversationContext],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved summaries for inclusion in the system prompt.
        
        Args:
            summaries: List of retrieved conversation contexts
            include_metadata: Whether to include topic/tone metadata
            
        Returns:
            Formatted string for prompt injection
        """
        if not summaries:
            return ""
        
        lines = ["**RELEVANT PAST CONVERSATIONS:**"]
        lines.append("(You may reference these if relevant to the current topic)\n")
        
        for i, ctx in enumerate(summaries, 1):
            lines.append(f"--- Past Conversation {i}: \"{ctx.title}\" ---")
            lines.append(ctx.summary)
            
            if include_metadata:
                if ctx.tone:
                    lines.append(f"Tone: {ctx.tone}")
                if ctx.key_topics:
                    lines.append(f"Topics: {', '.join(ctx.key_topics[:5])}")
                if ctx.open_questions:
                    lines.append(f"Open Questions: {', '.join(ctx.open_questions[:5])}")
            
            lines.append("")  # Blank line between summaries
        
        return "\n".join(lines)
    
    def get_trigger_phrases(self) -> List[str]:
        """Return the list of trigger phrases (for documentation/debugging)."""
        return TRIGGER_PHRASES.copy()
