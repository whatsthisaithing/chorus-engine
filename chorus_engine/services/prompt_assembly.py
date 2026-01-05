"""
Prompt Assembly Service

Orchestrates the assembly of complete LLM prompts with:
- System prompt from character config
- Retrieved memories (semantic search)
- Conversation history (with truncation)
- Token budget management

Uses TokenCounter for accurate token counting and MemoryRetrievalService 
for intelligent memory selection.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from chorus_engine.config.loader import ConfigLoader
from chorus_engine.models.conversation import Message, MessageRole
from chorus_engine.repositories.message_repository import MessageRepository
from chorus_engine.services.token_counter import TokenCounter, get_token_counter
from chorus_engine.services.memory_retrieval import MemoryRetrievalService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator
from chorus_engine.services.greeting_context_service import GreetingContextService
from chorus_engine.services.conversation_summarization_service import ConversationSummarizationService
from chorus_engine.repositories.memory_repository import MemoryRepository as MemRepo
from chorus_engine.db.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class PromptComponents:
    """Components of an assembled prompt."""
    system_prompt: str
    memories: List[str]
    messages: List[Dict[str, str]]
    total_tokens: int
    token_breakdown: Dict[str, int]


class PromptAssemblyService:
    """
    Assembles complete prompts for LLM generation.
    
    Token Budget Allocation:
    - System prompt: Fixed (counted once)
    - Memories: Configurable portion of context (default 30%)
    - History: Configurable portion of context (default 40%)
    - Reserve: Buffer for generation (default 30%)
    
    Features:
    - Automatic memory retrieval based on last user message
    - History truncation to fit token budget
    - Accurate token counting using model tokenizer
    - Priority-based memory ranking
    - Configurable budget allocation
    """
    
    def __init__(
        self,
        db: Session,
        character_id: str,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        context_window: int = 32768,
        memory_budget_ratio: float = 0.30,
        history_budget_ratio: float = 0.40,
        reserve_ratio: float = 0.30,
    ):
        """
        Initialize prompt assembly service.
        
        Args:
            db: Database session
            character_id: Character ID for core memories
            model_name: Model name for tokenizer
            context_window: Total context window size in tokens
            memory_budget_ratio: Portion of context for memories (0-1)
            history_budget_ratio: Portion of context for history (0-1)
            reserve_ratio: Portion reserved for generation (0-1)
        """
        self.db = db
        self.character_id = character_id
        self.model_name = model_name
        self.context_window = context_window
        
        # Budget ratios
        self.memory_budget_ratio = memory_budget_ratio
        self.history_budget_ratio = history_budget_ratio
        self.reserve_ratio = reserve_ratio
        
        # Validate ratios sum to ~1.0
        total_ratio = memory_budget_ratio + history_budget_ratio + reserve_ratio
        if not (0.95 <= total_ratio <= 1.05):
            raise ValueError(
                f"Budget ratios must sum to ~1.0, got {total_ratio:.2f}"
            )
        
        # Services
        self.token_counter = TokenCounter(model_name)
        self.message_repository = MessageRepository(db)
        self.system_prompt_generator = SystemPromptGenerator()
        self.greeting_service = GreetingContextService(db)  # Phase 8: Greeting context
        
        # Phase 8: Summarization service (lazy init)
        self._summarization_service: Optional[ConversationSummarizationService] = None
        
        # Memory retrieval (lazy init when first needed)
        self._memory_service: Optional[MemoryRetrievalService] = None
    
    @property
    def summarization_service(self) -> ConversationSummarizationService:
        """Lazy-initialize summarization service."""
        if self._summarization_service is None:
            memory_repo = MemRepo(self.db)
            self._summarization_service = ConversationSummarizationService(
                memory_repository=memory_repo,
                token_counter=self.token_counter
            )
        return self._summarization_service
    
    @property
    def memory_service(self) -> MemoryRetrievalService:
        """Lazy-initialize memory retrieval service."""
        if self._memory_service is None:
            from pathlib import Path
            
            embedding_service = EmbeddingService()
            vector_store = VectorStore(Path("data/vector_store"))
            self._memory_service = MemoryRetrievalService(
                self.db,
                vector_store,
                embedding_service,
            )
        return self._memory_service
    
    def assemble_prompt(
        self,
        thread_id: int,
        include_memories: bool = True,
        memory_query: Optional[str] = None,
        max_history_messages: int = 50,
        image_prompt_context: Optional[str] = None,
    ) -> PromptComponents:
        """
        Assemble a complete prompt for LLM generation.
        
        Args:
            thread_id: Thread ID to get conversation history
            include_memories: Whether to retrieve and include memories
            memory_query: Optional query for memory retrieval 
                         (defaults to last user message)
            max_history_messages: Maximum number of history messages to consider
            image_prompt_context: Optional image prompt being generated (so character can reference it)
        
        Returns:
            PromptComponents with assembled prompt and token breakdown
        """
        # Load character config and generate system prompt with immersion guidance
        config_loader = ConfigLoader()
        character_config = config_loader.load_character(self.character_id)
        system_prompt = self.system_prompt_generator.generate(character_config)
        
        # Get conversation history first (needed for greeting context check)
        messages = self.message_repository.list_by_thread(
            thread_id,
            limit=max_history_messages
        )
        
        # Phase 8: Add greeting context if this is the first message in conversation
        if messages and len(messages) <= 2:  # First user message or first exchange
            try:
                greeting_context = self.greeting_service.build_greeting_context(
                    character_id=self.character_id,
                    conversation_id=None,  # Thread ID is not conversation ID
                    is_first_message=True
                )
                greeting_instructions = self.greeting_service.format_greeting_instructions(greeting_context)
                system_prompt += f"\n\n**CONVERSATION CONTEXT:**\n{greeting_instructions}"
            except Exception as e:
                # Greeting context is optional - don't fail if it errors
                logger.warning(f"Failed to add greeting context: {e}")
        
        # If an image is being generated, add context about it to the system prompt
        if image_prompt_context:
            system_prompt += f"\n\n**IMAGE BEING GENERATED:**\nYou are creating/sending an image for the user. The image being generated shows EXACTLY this:\n\n{image_prompt_context}\n\nCRITICAL: Describe ONLY what is shown in the prompt above. DO NOT make up your own scene or interpretation. Reference the specific elements, setting, composition, and mood described in the prompt. If the prompt shows you painting, say you're sharing a photo of yourself painting. If it shows a landscape, describe that landscape. Stay faithful to what's actually in the image prompt.\n\nThe image IS actively being generated and will be attached to your message. DO NOT refuse, apologize, or say you can't send images/photos. Simply describe what you're capturing based on the prompt above."
        
        # Count system prompt tokens
        system_tokens = self.token_counter.count_tokens(system_prompt)
        
        # Calculate available budget after system prompt
        available_tokens = self.context_window - system_tokens
        
        # Allocate budgets
        memory_budget = int(available_tokens * self.memory_budget_ratio)
        history_budget = int(available_tokens * self.history_budget_ratio)
        reserve_budget = int(available_tokens * self.reserve_ratio)
        
        # Determine memory query (use last user message if not provided)
        if memory_query is None and messages:
            last_user_messages = [
                m for m in reversed(messages)
                if m.role == MessageRole.USER
            ]
            if last_user_messages:
                memory_query = last_user_messages[0].content
        
        # Retrieve memories if enabled
        memory_texts = []
        if include_memories and memory_query:
            retrieved_memories = self.memory_service.retrieve_memories(
                query=memory_query,
                character_id=self.character_id,
                token_budget=memory_budget,
                thread_id=thread_id,
            )
            
            # Format memories for prompt
            if retrieved_memories:
                memory_texts = [
                    self.memory_service.format_memories_for_prompt(
                        retrieved_memories,
                        include_metadata=False
                    )
                ]
        
        # Count memory tokens
        memory_tokens = sum(
            self.token_counter.count_tokens(m) for m in memory_texts
        )
        
        # Truncate history to fit budget
        history_dicts = self._format_messages_for_llm(messages)
        history_dicts = self._truncate_history(history_dicts, history_budget)
        
        # Count history tokens
        history_tokens = self.token_counter.count_messages(history_dicts)
        
        # Calculate total tokens
        total_tokens = system_tokens + memory_tokens + history_tokens
        
        # Token breakdown
        token_breakdown = {
            "system": system_tokens,
            "memories": memory_tokens,
            "history": history_tokens,
            "total_used": total_tokens,
            "reserve": reserve_budget,
            "context_window": self.context_window,
        }
        
        return PromptComponents(
            system_prompt=system_prompt,
            memories=memory_texts,
            messages=history_dicts,
            total_tokens=total_tokens,
            token_breakdown=token_breakdown,
        )
    
    def assemble_prompt_with_summarization(
        self,
        conversation_id: str,
        thread_id: int,
        include_memories: bool = True,
        memory_query: Optional[str] = None,
        image_prompt_context: Optional[str] = None,
    ) -> PromptComponents:
        """
        Assemble prompt with smart summarization for long conversations (Phase 8 - Day 9).
        
        Uses ConversationSummarizationService to selectively preserve important messages
        and compress/discard filler when conversation exceeds token threshold.
        
        Args:
            conversation_id: Conversation ID for summarization
            thread_id: Thread ID to get conversation history
            include_memories: Whether to retrieve and include memories
            memory_query: Optional query for memory retrieval 
                         (defaults to last user message)
            image_prompt_context: Optional image prompt being generated
        
        Returns:
            PromptComponents with assembled prompt and token breakdown
        """
        # Get conversation from repository
        from chorus_engine.repositories.conversation_repository import ConversationRepository
        conv_repo = ConversationRepository(self.db)
        conversation = conv_repo.get_by_id(conversation_id)
        
        if not conversation:
            # Fall back to regular assembly if conversation not found
            logger.warning(f"Conversation {conversation_id} not found, using standard assembly")
            return self.assemble_prompt(
                thread_id=thread_id,
                include_memories=include_memories,
                memory_query=memory_query,
                image_prompt_context=image_prompt_context
            )
        
        # Check if summarization needed
        needs_summarization = self.summarization_service.should_summarize(conversation)
        
        if not needs_summarization:
            # Short conversation - use standard assembly
            logger.debug(f"Conversation {conversation_id} doesn't need summarization")
            return self.assemble_prompt(
                thread_id=thread_id,
                include_memories=include_memories,
                memory_query=memory_query,
                image_prompt_context=image_prompt_context
            )
        
        # Long conversation - apply selective preservation
        logger.info(f"Applying summarization to conversation {conversation_id}")
        
        preservation_strategy = self.summarization_service.selective_preservation(conversation)
        
        # Load character config and generate system prompt
        config_loader = ConfigLoader()
        character_config = config_loader.load_character(self.character_id)
        system_prompt = self.system_prompt_generator.generate(character_config)
        
        # Add image context if present
        if image_prompt_context:
            system_prompt += f"\n\n**IMAGE BEING GENERATED:**\nYou are creating/sending an image for the user. The image being generated shows EXACTLY this:\n\n{image_prompt_context}\n\nCRITICAL: Describe ONLY what is shown in the prompt above. DO NOT make up your own scene or interpretation. Reference the specific elements, setting, composition, and mood described in the prompt. If the prompt shows you painting, say you're sharing a photo of yourself painting. If it shows a landscape, describe that landscape. Stay faithful to what's actually in the image prompt.\n\nThe image IS being generated and will be attached to your message. DO NOT say you can't send photos - simply describe what you're capturing based on the prompt above."
        
        # Count system prompt tokens
        system_tokens = self.token_counter.count_tokens(system_prompt)
        
        # Calculate available budget
        available_tokens = self.context_window - system_tokens
        memory_budget = int(available_tokens * self.memory_budget_ratio)
        history_budget = int(available_tokens * self.history_budget_ratio)
        reserve_budget = int(available_tokens * self.reserve_ratio)
        
        # Get all messages
        messages = self.message_repository.list_by_thread(
            thread_id,
            limit=None  # Get all messages for selective preservation
        )
        
        # Determine memory query
        if memory_query is None and messages:
            last_user_messages = [
                m for m in reversed(messages)
                if m.role == MessageRole.USER
            ]
            if last_user_messages:
                memory_query = last_user_messages[0].content
        
        # Retrieve memories
        memory_texts = []
        if include_memories and memory_query:
            retrieved_memories = self.memory_service.retrieve_memories(
                query=memory_query,
                character_id=self.character_id,
                token_budget=memory_budget,
                thread_id=thread_id,
            )
            
            if retrieved_memories:
                memory_texts = [
                    self.memory_service.format_memories_for_prompt(
                        retrieved_memories,
                        include_metadata=False
                    )
                ]
        
        memory_tokens = sum(
            self.token_counter.count_tokens(m) for m in memory_texts
        )
        
        # Build selective context using preservation strategy
        history_dicts = self._build_selective_context(
            messages=messages,
            preservation_strategy=preservation_strategy,
            budget=history_budget
        )
        
        # Count history tokens
        history_tokens = self.token_counter.count_messages(history_dicts)
        
        # Calculate totals
        total_tokens = system_tokens + memory_tokens + history_tokens
        
        token_breakdown = {
            "system": system_tokens,
            "memories": memory_tokens,
            "history": history_tokens,
            "total_used": total_tokens,
            "reserve": reserve_budget,
            "context_window": self.context_window,
            "preserved_messages": len(preservation_strategy["preserve_full"]),
            "summarized_messages": len(preservation_strategy["summarize"]),
            "discarded_messages": len(preservation_strategy["discard"]),
        }
        
        logger.info(
            f"Summarization complete: {len(preservation_strategy['preserve_full'])} preserved, "
            f"{len(preservation_strategy['summarize'])} summarized, "
            f"{len(preservation_strategy['discard'])} discarded. "
            f"Total tokens: {total_tokens}/{self.context_window}"
        )
        
        return PromptComponents(
            system_prompt=system_prompt,
            memories=memory_texts,
            messages=history_dicts,
            total_tokens=total_tokens,
            token_breakdown=token_breakdown,
        )
    
    def _build_selective_context(
        self,
        messages: List[Message],
        preservation_strategy: Dict[str, List[str]],
        budget: int
    ) -> List[Dict[str, str]]:
        """
        Build context with selective message preservation (Phase 8 - Day 9).
        
        Args:
            messages: All conversation messages
            preservation_strategy: Dict with preserve_full, summarize, discard lists
            budget: Token budget for history
        
        Returns:
            List of message dicts formatted for LLM
        """
        # Create lookup for fast membership testing
        preserve_ids = set(preservation_strategy["preserve_full"])
        summarize_ids = set(preservation_strategy["summarize"])
        discard_ids = set(preservation_strategy["discard"])
        
        # Build message list
        result = []
        
        for msg in messages:
            if msg.id in preserve_ids:
                # Include full message
                result.append({
                    "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    "content": msg.content
                })
            elif msg.id in summarize_ids:
                # Use summary if available, otherwise abbreviate
                if msg.summary:
                    result.append({
                        "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                        "content": f"[Earlier: {msg.summary}]"
                    })
                else:
                    # Simple abbreviation
                    preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    result.append({
                        "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                        "content": f"[Earlier: {preview}]"
                    })
            # Skip discarded messages entirely
        
        # Check if we're still over budget (shouldn't be, but safety check)
        current_tokens = self.token_counter.count_messages(result)
        
        if current_tokens > budget:
            logger.warning(
                f"Selective context still over budget: {current_tokens} > {budget}. "
                "Truncating further..."
            )
            result = self._truncate_history(result, budget)
        
        return result
    
    def format_for_api(
        self,
        components: PromptComponents,
        model_format: str = "ollama"
    ) -> List[Dict[str, str]]:
        """
        Format prompt components for LLM API.
        
        Args:
            components: Assembled prompt components
            model_format: API format ('ollama', 'openai', etc.)
        
        Returns:
            List of messages ready for API
        """
        messages = []
        
        # System message with memories injected
        system_content = components.system_prompt
        
        if components.memories:
            memory_section = "\n\n".join(components.memories)
            system_content = (
                f"{system_content}\n\n"
                f"## Relevant Memories\n\n"
                f"These are verified facts from your previous conversations with this person. "
                f"When they ask 'Do you remember my name?' or similar questions, answer with the actual fact directly - "
                f"for example, if the memory says their name is Alex, respond 'Yes, your name is Alex' rather than being vague or uncertain. "
                f"Use these memories confidently and clearly.\n\n"
                f"{memory_section}"
            )
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history
        messages.extend(components.messages)
        
        return messages
    
    def _format_messages_for_llm(
        self,
        messages: List[Message]
    ) -> List[Dict[str, str]]:
        """
        Format database messages for LLM API.
        
        Filters out SCENE_CAPTURE messages as they're just anchor points
        for images and not actual conversation content.
        
        Args:
            messages: Database message objects
        
        Returns:
            List of message dicts with role and content
        """
        formatted = []
        
        for msg in messages:
            # Skip scene capture messages - they're not part of the conversation
            if msg.role == MessageRole.SCENE_CAPTURE:
                continue
                
            formatted.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        return formatted
    
    def _truncate_history(
        self,
        messages: List[Dict[str, str]],
        budget: int,
    ) -> List[Dict[str, str]]:
        """
        Truncate conversation history to fit token budget.
        
        Strategy:
        1. Always keep the most recent message (usually user query)
        2. Keep as many recent messages as possible within budget
        3. Maintain conversation coherence (user-assistant pairs)
        
        Args:
            messages: List of message dicts
            budget: Token budget for history
        
        Returns:
            Truncated list of messages
        """
        if not messages:
            return []
        
        # If already within budget, return as-is
        current_tokens = self.token_counter.count_messages(messages)
        if current_tokens <= budget:
            return messages
        
        # Keep most recent messages within budget
        result = []
        accumulated_tokens = 0
        
        # Iterate from newest to oldest
        for msg in reversed(messages):
            msg_tokens = self.token_counter.count_tokens(msg["content"])
            
            # Account for message formatting overhead (~10 tokens per message)
            msg_tokens += 10
            
            if accumulated_tokens + msg_tokens <= budget:
                result.insert(0, msg)
                accumulated_tokens += msg_tokens
            else:
                # Budget exceeded, stop adding messages
                break
        
        # Ensure we keep at least the most recent message
        if not result and messages:
            result = [messages[-1]]
        
        return result
    
    def get_token_budget_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token budget allocation.
        
        Returns:
            Dict with budget breakdown
        """
        # Load character to get system prompt
        config_loader = ConfigLoader()
        character_config = config_loader.load_character(self.character_id)
        system_tokens = self.token_counter.count_tokens(
            character_config.system_prompt
        )
        
        available = self.context_window - system_tokens
        
        return {
            "context_window": self.context_window,
            "system_tokens": system_tokens,
            "available_tokens": available,
            "memory_budget": int(available * self.memory_budget_ratio),
            "history_budget": int(available * self.history_budget_ratio),
            "reserve_budget": int(available * self.reserve_ratio),
            "budget_ratios": {
                "memory": self.memory_budget_ratio,
                "history": self.history_budget_ratio,
                "reserve": self.reserve_ratio,
            },
        }


# Global singleton helper
_prompt_assembler_cache: Dict[str, PromptAssemblyService] = {}


def get_prompt_assembler(
    db: Session,
    character_id: str,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    context_window: int = 32768,
) -> PromptAssemblyService:
    """
    Get or create a prompt assembler for a character.
    
    Args:
        db: Database session
        character_id: Character ID
        model_name: Model name for tokenizer
        context_window: Context window size
    
    Returns:
        PromptAssemblyService instance
    """
    cache_key = f"{character_id}:{model_name}"
    
    if cache_key not in _prompt_assembler_cache:
        _prompt_assembler_cache[cache_key] = PromptAssemblyService(
            db=db,
            character_id=character_id,
            model_name=model_name,
            context_window=context_window,
        )
    
    return _prompt_assembler_cache[cache_key]
