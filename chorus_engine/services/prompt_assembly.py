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
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from chorus_engine.config.loader import ConfigLoader
from chorus_engine.models.conversation import Message, MessageRole, Conversation
from chorus_engine.repositories.message_repository import MessageRepository
from chorus_engine.services.token_counter import TokenCounter, get_token_counter
from chorus_engine.services.memory_retrieval import MemoryRetrievalService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator
from chorus_engine.services.greeting_context_service import GreetingContextService
from chorus_engine.services.conversation_summarization_service import ConversationSummarizationService
from chorus_engine.services.conversation_context_retrieval import (
    ConversationContextRetrievalService,
    ConversationContextConfig as ServiceContextConfig
)
from chorus_engine.repositories.memory_repository import MemoryRepository as MemRepo
from chorus_engine.repositories.continuity_repository import ContinuityRepository
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore

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
    - Conversation Context: Past conversation summaries (default 5%)
    - Memories: Character memories from vector store (default 20%)
    - History: Conversation messages (default 50%)
    - Documents: Injected document chunks (default 15%)
    - Reserve: Buffer for generation (default 10%)
    
    Budget Cascade:
    Unused budget from conversation context flows to memory retrieval.
    This ensures that when no relevant past conversations are found,
    the character can recall more memories instead.
    
    Features:
    - Automatic memory retrieval based on last user message
    - Past conversation context retrieval with semantic search
    - History truncation to fit token budget
    - Accurate token counting using model tokenizer
    - Priority-based memory ranking
    - Configurable budget allocation with cascade
    """
    
    def __init__(
        self,
        db: Session,
        character_id: str,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        context_window: int = 32768,
        memory_budget_ratio: float = 0.20,
        history_budget_ratio: float = 0.50,
        document_budget_ratio: float = 0.15,
        reserve_ratio: float = 0.10,
        conversation_context_budget_ratio: float = 0.05,
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
            document_budget_ratio: Portion of context for document chunks (0-1)
            reserve_ratio: Portion reserved for generation (0-1)
            conversation_context_budget_ratio: Portion for past conversation summaries (0-1)
        """
        self.db = db
        self.character_id = character_id
        self.model_name = model_name
        self.context_window = context_window
        
        # Budget ratios
        self.memory_budget_ratio = memory_budget_ratio
        self.history_budget_ratio = history_budget_ratio
        self.document_budget_ratio = document_budget_ratio
        self.reserve_ratio = reserve_ratio
        self.conversation_context_budget_ratio = conversation_context_budget_ratio
        
        # Validate ratios sum to ~1.0
        total_ratio = (memory_budget_ratio + history_budget_ratio + document_budget_ratio + 
                      reserve_ratio + conversation_context_budget_ratio)
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
        
        # Conversation context retrieval (lazy init)
        self._conversation_context_service: Optional[ConversationContextRetrievalService] = None
        self._summary_vector_store: Optional[ConversationSummaryVectorStore] = None
    
    @property
    def summarization_service(self) -> ConversationSummarizationService:
        """Lazy-initialize summarization service."""
        if self._summarization_service is None:
            memory_repo = MemRepo(self.db)
            self._summarization_service = ConversationSummarizationService(
                memory_repository=memory_repo,
                context_window=self.context_window,
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
    
    @property
    def conversation_context_service(self) -> ConversationContextRetrievalService:
        """Lazy-initialize conversation context retrieval service."""
        if self._conversation_context_service is None:
            from pathlib import Path
            
            embedding_service = EmbeddingService()
            
            if self._summary_vector_store is None:
                self._summary_vector_store = ConversationSummaryVectorStore(
                    Path("data/vector_store")
                )
            
            # Load config if available
            try:
                config_loader = ConfigLoader()
                system_config = config_loader.load_system_config()
                context_config = getattr(system_config, 'conversation_context', None)
                
                if context_config:
                    service_config = ServiceContextConfig(
                        enabled=context_config.enabled,
                        passive_threshold=context_config.passive_threshold,
                        triggered_threshold=context_config.triggered_threshold,
                        max_summaries=context_config.max_summaries,
                        token_budget_ratio=context_config.token_budget_ratio
                    )
                else:
                    service_config = None
            except Exception:
                service_config = None
            
            self._conversation_context_service = ConversationContextRetrievalService(
                summary_vector_store=self._summary_vector_store,
                embedding_service=embedding_service,
                token_counter=self.token_counter,
                config=service_config
            )
        return self._conversation_context_service
    
    def get_document_token_budget(self) -> int:
        """
        Calculate available token budget for document chunks.
        
        Returns:
            Number of tokens available for document content
        """
        # Rough estimate of system prompt size (will be refined during actual assembly)
        estimated_system_tokens = 1000  # Conservative estimate
        available_tokens = self.context_window - estimated_system_tokens
        document_budget = int(available_tokens * self.document_budget_ratio)
        return document_budget
    
    def assemble_prompt(
        self,
        thread_id: int,
        include_memories: bool = True,
        memory_query: Optional[str] = None,
        max_history_messages: int = 10000,
        image_prompt_context: Optional[str] = None,
        video_prompt_context: Optional[str] = None,
        document_context: Optional[Any] = None,
        primary_user: Optional[str] = None,
        conversation_source: Optional[str] = None,
        conversation_id: Optional[str] = None,
        include_conversation_context: bool = True,
        allowed_media_tools: Optional[set[str]] = None,
        allow_proactive_media_offers: Optional[bool] = None,
        media_gate_context: Optional[dict] = None,
    ) -> PromptComponents:
        """
        Assemble a complete prompt for LLM generation.
        
        Args:
            thread_id: Thread ID to get conversation history
            include_memories: Whether to retrieve and include memories
            memory_query: Optional query for memory retrieval 
                         (defaults to last user message)
            max_history_messages: Safety limit for message query (default 10000)
                                 Actual truncation is handled by token budget
            image_prompt_context: Optional image prompt being generated (so character can reference it)
            video_prompt_context: Optional video prompt being generated (so character can reference it)
            document_context: Optional DocumentContext with retrieved document chunks
            primary_user: Name of the user who invoked the bot (for multi-user contexts)
            conversation_source: Platform source ('web', 'discord', 'slack', etc.)
            conversation_id: Current conversation ID (for excluding from context search)
            include_conversation_context: Whether to include relevant past conversation summaries
            allowed_media_tools: Optional per-turn allowed tool names for payload contract injection
        
        Returns:
            PromptComponents with assembled prompt and token breakdown
        """
        # Load character config and generate system prompt with immersion guidance
        config_loader = ConfigLoader()
        character_config = config_loader.load_character(self.character_id)
        system_config = config_loader.load_system_config()
        media_interpretation = bool(image_prompt_context or video_prompt_context)
        system_prompt = self.system_prompt_generator.generate(
            character_config,
            primary_user=primary_user,
            conversation_source=conversation_source,
            include_chatbot_guidance=not media_interpretation,
            allowed_media_tools=allowed_media_tools,
            allow_proactive_media_offers=allow_proactive_media_offers,
            media_gate_context=media_gate_context,
        )
        
        # Inject identity/time headers before other system prompt additions
        system_prompt = self._prepend_identity_time_headers(
            system_prompt=system_prompt,
            system_config=system_config,
            character_config=character_config,
            conversation_source=conversation_source
        )
        
        # CRITICAL: When interpreting a generated image/video prompt, skip conversation history
        # The character's ONLY job is to describe the pre-generated prompt, not respond to conversation
        # Step 1 (prompt generation) already used full context, Step 2 (interpretation) needs NONE
        skip_history_for_media_interpretation = bool(image_prompt_context or video_prompt_context)
        
        # Get conversation history (unless we're just interpreting a media prompt)
        if skip_history_for_media_interpretation:
            messages = []  # No conversation history needed for prompt interpretation
            logger.debug("Skipping conversation history for media prompt interpretation")
        else:
            messages = self.message_repository.list_by_thread(
                thread_id,
                limit=max_history_messages
            )
            
            # Task 1.8: Enrich user messages with vision observations from attached images
            try:
                from chorus_engine.models.conversation import ImageAttachment
                
                for message in messages:
                    if message.role == MessageRole.USER:
                        # Check if this message has image attachments with vision analysis
                        attachments = self.db.query(ImageAttachment).filter(
                            ImageAttachment.message_id == message.id,
                            ImageAttachment.vision_processed == "true"
                        ).all()
                        
                        if attachments:
                            # Add vision observations to message content
                            vision_contexts = []
                            for attachment in attachments:
                                if attachment.vision_observation:
                                    vision_contexts.append(
                                        f"[VISUAL CONTEXT: {attachment.vision_observation}]"
                                    )
                            
                            if vision_contexts:
                                # Append vision context to message content
                                message.content += "\n\n" + "\n\n".join(vision_contexts)
            except Exception as e:
                logger.error(f"Failed to enrich messages with vision observations: {e}")
                # Continue without vision enrichment if it fails
        
        # Continuity bootstrap injection (only before first assistant response)
        has_assistant = any(
            (msg.role == MessageRole.ASSISTANT or str(msg.role) == MessageRole.ASSISTANT.value)
            for msg in messages
        )
        bootstrap_injected = False
        if messages and not has_assistant and conversation_id:
            try:
                continuity_repo = ContinuityRepository(self.db)
                conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
                if conversation and conversation.continuity_mode == "use":
                    allow_injection = True
                    if conversation.source != "web":
                        if not primary_user or conversation.primary_user != primary_user:
                            allow_injection = False
                    if allow_injection:
                        cache = continuity_repo.get_cache(self.character_id)
                        if cache and cache.bootstrap_packet_internal:
                            system_prompt += f"\n\n{cache.bootstrap_packet_internal}"
                            bootstrap_injected = True
            except Exception as e:
                logger.warning(f"Failed to inject continuity bootstrap: {e}")

        # Phase 8: Add greeting context if this is the first message in conversation
        # Skip greeting context if continuity bootstrap is injected to avoid redundancy.
        if not bootstrap_injected and messages and len(messages) <= 2:  # First user message or first exchange
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
            system_prompt += f"\n\n**IMAGE BEING GENERATED:**\nYou are creating/sending an image for the user. The image being generated shows EXACTLY this:\n\n{image_prompt_context}\n\nCRITICAL:\n- You MUST describe the image in 2–5 sentences.\n- Include at least 5 concrete visual details from the prompt (subjects, setting, lighting, composition, mood, colors, clothing, props).\n- Do NOT add new elements or change the scene.\n- Speak as if you are sharing the image right now.\n- Do NOT evaluate or comment on the image quality or whether it \"captures\" someone well; just describe what is visible.\n- Return only the description. Do not include preambles, lead-ins, or questions.\n\nThe image IS actively being generated and will be attached to your message. DO NOT refuse, apologize, or say you can't send images/photos. Do NOT respond with only a short opener. Your reply must include the descriptive summary above."
        
        # If a video is being generated, add context about it to the system prompt
        if video_prompt_context:
            system_prompt += f"\n\n**VIDEO BEING GENERATED:**\nYou are creating/sending a video for the user. The video being generated shows EXACTLY this:\n\n{video_prompt_context}\n\nCRITICAL:\n- You MUST describe the video in 2–5 sentences, in FIRST-PERSON.\n- Include at least 5 concrete visual or motion details from the prompt (setting, action, camera movement, lighting, mood).\n- Do NOT add new elements or change the scene.\n- Speak as if you are sharing the video right now.\n- Do NOT evaluate or comment on the video quality or whether it \"captures\" someone well; just describe what is visible.\n- Return only the description. Do not include preambles, lead-ins, or questions.\n\nThe video IS actively being generated and will be attached to your message. DO NOT refuse, apologize, or say you can't send videos. Do NOT respond with only a short opener. Your reply must include the descriptive summary above."
        
        # Count base system prompt tokens (before document injection)
        base_system_tokens = self.token_counter.count_tokens(system_prompt)
        
        # If document context is provided, add it to the system prompt
        document_tokens = 0
        if document_context:
            try:
                doc_injection = document_context.format_context_injection()
                if doc_injection:
                    system_prompt += doc_injection
                    document_tokens = self.token_counter.count_tokens(doc_injection)
                    logger.info(f"Added document context: {len(document_context.chunks)} chunks ({document_tokens} tokens)")
            except Exception as e:
                logger.error(f"Error injecting document context: {e}", exc_info=True)
                # Continue without document context if injection fails
        
        # Calculate total system tokens
        system_tokens = base_system_tokens + document_tokens
        
        # Calculate available budget after system prompt AND documents
        # Document tokens are pre-allocated (document_budget_ratio), so exclude them from "available" space
        # This prevents document usage from reducing memory/history budgets
        available_tokens = self.context_window - base_system_tokens - document_tokens
        
        # Allocate budgets
        conversation_context_budget = int(available_tokens * self.conversation_context_budget_ratio)
        memory_budget = int(available_tokens * self.memory_budget_ratio)
        history_budget = int(available_tokens * self.history_budget_ratio)
        document_budget = int(available_tokens * self.document_budget_ratio)
        reserve_budget = int(available_tokens * self.reserve_ratio)
        
        logger.debug(
            f"Token budgets - Available: {available_tokens}, "
            f"ConvContext: {conversation_context_budget}, Memory: {memory_budget}, "
            f"History: {history_budget}, Document: {document_budget}, Reserve: {reserve_budget}"
        )
        
        # Determine memory query (use last user message if not provided)
        if memory_query is None and messages:
            last_user_messages = [
                m for m in reversed(messages)
                if m.role == MessageRole.USER
            ]
            if last_user_messages:
                memory_query = last_user_messages[0].content
        
        # Retrieve relevant past conversation summaries FIRST (budget cascade)
        # Unused conversation context budget flows to memory retrieval
        conversation_context_tokens = 0
        unused_context_budget = conversation_context_budget  # Start with full allocation
        
        if include_conversation_context and memory_query:
            try:
                retrieved_contexts, context_tokens_used = self.conversation_context_service.retrieve_relevant_summaries(
                    character_id=self.character_id,
                    user_message=memory_query,
                    current_conversation_id=conversation_id,
                    token_budget=conversation_context_budget
                )
                
                if retrieved_contexts:
                    # Format and append to system prompt
                    context_text = self.conversation_context_service.format_summaries_for_prompt(
                        retrieved_contexts
                    )
                    system_prompt += f"\n\n{context_text}"
                    conversation_context_tokens = context_tokens_used
                    system_tokens += conversation_context_tokens
                    unused_context_budget = conversation_context_budget - context_tokens_used
                    
                    logger.info(
                        f"Added {len(retrieved_contexts)} past conversation context(s) "
                        f"({conversation_context_tokens} tokens, {unused_context_budget} unused)"
                    )
                else:
                    logger.debug(
                        f"No relevant conversation context found, "
                        f"cascading {unused_context_budget} tokens to memory budget"
                    )
            except Exception as e:
                logger.warning(f"Failed to retrieve conversation context: {e}")
                # Continue without conversation context if it fails
        
        # Cascade unused context budget to memory retrieval
        effective_memory_budget = memory_budget + unused_context_budget
        if unused_context_budget > 0:
            logger.debug(
                f"Budget cascade: memory budget {memory_budget} + {unused_context_budget} unused context "
                f"= {effective_memory_budget} effective"
            )
        
        # Retrieve memories (using effective budget with cascade from unused context)
        memory_texts = []
        if include_memories and memory_query:
            retrieved_memories = self.memory_service.retrieve_memories(
                query=memory_query,
                character_id=self.character_id,
                token_budget=effective_memory_budget,
                thread_id=thread_id,
                conversation_source=conversation_source
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
        history_dicts = self._format_messages_for_llm(
            messages,
            conversation_source=conversation_source
        )
        history_dicts = self._truncate_history(history_dicts, history_budget)
        
        # Media interpretation: inject an explicit user instruction even when history is skipped
        if media_interpretation:
            if video_prompt_context:
                user_instruction = "Describe the video above in 2–5 sentences, following the instructions."
            else:
                user_instruction = "Describe the image above in 2–5 sentences, following the instructions."
            history_dicts.append({
                "role": "user",
                "content": user_instruction
            })
        
        # Count history tokens
        history_tokens = self.token_counter.count_messages(history_dicts)
        
        # Calculate total tokens
        total_tokens = system_tokens + memory_tokens + history_tokens
        
        # Token breakdown
        token_breakdown = {
            "system": system_tokens,
            "memories": memory_tokens,
            "conversation_context": conversation_context_tokens,
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
        video_prompt_context: Optional[str] = None,
        document_context: Optional[Any] = None,
        conversation_source: Optional[str] = None,
        include_conversation_context: bool = True,
        allowed_media_tools: Optional[set[str]] = None,
        allow_proactive_media_offers: Optional[bool] = None,
        media_gate_context: Optional[dict] = None
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
            document_context: Optional document context to inject
            include_conversation_context: Whether to include relevant past conversation summaries
        
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
                image_prompt_context=image_prompt_context,
                conversation_id=conversation_id,
                include_conversation_context=include_conversation_context,
                allowed_media_tools=allowed_media_tools,
                allow_proactive_media_offers=allow_proactive_media_offers,
                media_gate_context=media_gate_context,
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
                image_prompt_context=image_prompt_context,
                document_context=document_context,
                conversation_id=conversation_id,
                include_conversation_context=include_conversation_context,
                allowed_media_tools=allowed_media_tools,
                allow_proactive_media_offers=allow_proactive_media_offers,
                media_gate_context=media_gate_context,
            )
        
        # Long conversation - apply selective preservation
        logger.info(f"Applying summarization to conversation {conversation_id}")
        
        preservation_strategy = self.summarization_service.selective_preservation(conversation)
        
        # Load character config and generate system prompt
        config_loader = ConfigLoader()
        character_config = config_loader.load_character(self.character_id)
        system_config = config_loader.load_system_config()
        media_interpretation = bool(image_prompt_context or video_prompt_context)
        system_prompt = self.system_prompt_generator.generate(
            character_config,
            primary_user=primary_user,
            conversation_source=conversation_source,
            include_chatbot_guidance=not media_interpretation,
            allowed_media_tools=allowed_media_tools,
            allow_proactive_media_offers=allow_proactive_media_offers,
            media_gate_context=media_gate_context,
        )
        
        # Inject identity/time headers before other system prompt additions
        system_prompt = self._prepend_identity_time_headers(
            system_prompt=system_prompt,
            system_config=system_config,
            character_config=character_config,
            conversation_source=conversation_source
        )
        
        # Add image context if present
        if image_prompt_context:
            system_prompt += f"\n\n**IMAGE BEING GENERATED:**\nYou are creating/sending an image for the user. The image being generated shows EXACTLY this:\n\n{image_prompt_context}\n\nCRITICAL:\n- You MUST describe the image in 2–5 sentences.\n- Include at least 5 concrete visual details from the prompt (subjects, setting, lighting, composition, mood, colors, clothing, props).\n- Do NOT add new elements or change the scene.\n- Speak as if you are sharing the image right now.\n- Do NOT evaluate or comment on the image quality or whether it \"captures\" someone well; just describe what is visible.\n- Return only the description. Do not include preambles, lead-ins, or questions.\n\nThe image IS being generated and will be attached to your message. DO NOT say you can't send photos. Do NOT respond with only a short opener. Your reply must include the descriptive summary above."
        
        # Add video context if present
        if video_prompt_context:
            system_prompt += f"\n\n**VIDEO BEING GENERATED:**\nYou are creating/sending a video for the user. The video being generated shows EXACTLY this:\n\n{video_prompt_context}\n\nCRITICAL:\n- You MUST describe the video in 2–5 sentences, in FIRST-PERSON.\n- Include at least 5 concrete visual or motion details from the prompt (setting, action, camera movement, lighting, mood).\n- Do NOT add new elements or change the scene.\n- Speak as if you are sharing the video right now.\n- Do NOT evaluate or comment on the video quality or whether it \"captures\" someone well; just describe what is visible.\n- Return only the description. Do not include preambles, lead-ins, or questions.\n\nThe video IS being generated and will be attached to your message. DO NOT say you can't send videos. Do NOT respond with only a short opener. Your reply must include the descriptive summary above."
        
        # Count base system prompt tokens (before document injection)
        base_system_tokens = self.token_counter.count_tokens(system_prompt)
        
        # If document context is provided, add it to the system prompt
        document_tokens = 0
        if document_context:
            try:
                doc_injection = document_context.format_context_injection()
                if doc_injection:
                    system_prompt += doc_injection
                    document_tokens = self.token_counter.count_tokens(doc_injection)
                    logger.info(f"Added document context: {len(document_context.chunks)} chunks ({document_tokens} tokens)")
            except Exception as e:
                logger.error(f"Error injecting document context: {e}", exc_info=True)
        
        # Calculate available budget after system prompt AND documents
        # Document tokens are pre-allocated, so exclude them from "available" space
        available_tokens = self.context_window - base_system_tokens - document_tokens
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
                conversation_source=conversation_source
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

        # Media interpretation: inject an explicit user instruction even when history is skipped
        if media_interpretation:
            if video_prompt_context:
                user_instruction = "Describe the video above in 2–5 sentences, following the instructions."
            else:
                user_instruction = "Describe the image above in 2–5 sentences, following the instructions."
            history_dicts.append({
                "role": "user",
                "content": user_instruction
            })
        
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

    def _prepend_identity_time_headers(
        self,
        system_prompt: str,
        system_config,
        character_config,
        conversation_source: Optional[str]
    ) -> str:
        identity_header = self._build_identity_header(
            system_config=system_config,
            character_config=character_config,
            conversation_source=conversation_source
        )
        
        time_header = self._build_time_header(system_config)
        
        user_info_lines = []
        if identity_header:
            user_info_lines.append(f"- {identity_header}")
        if time_header:
            user_info_lines.append(f"- {time_header}")
        
        user_info_section = "User Info (authoritative):"
        if user_info_lines:
            user_info_section = "\n".join([user_info_section] + user_info_lines)
        
        character_section = "Your Character Context:\n" + system_prompt
        
        return "\n\n".join([user_info_section, character_section])

    def _build_identity_header(
        self,
        system_config,
        character_config,
        conversation_source: Optional[str]
    ) -> Optional[str]:
        # Only inject identity for web (single-user) conversations
        if conversation_source != "web":
            return None
        
        identity_override = getattr(character_config, "user_identity", None)
        mode = getattr(identity_override, "mode", "canonical") if identity_override else "canonical"
        
        if mode == "masked":
            return (
                "User identity is intentionally withheld for this conversation. "
                "Do not guess the user's name or identity."
            )
    
        if mode == "role":
            role_name = getattr(identity_override, "role_name", None) or "User"
            role_aliases = getattr(identity_override, "role_aliases", []) or []
            if role_aliases:
                aliases_str = ", ".join(role_aliases)
                return (
                    f"In-universe user identity (authoritative for this conversation): "
                    f"Name = {role_name}. Aliases: {aliases_str}. "
                    "Address the user by the role name by default. "
                    "Only use a role alias if the current message explicitly shows that alias as "
                    "the speaker (e.g., a username/handle in the message metadata or prefix)."
                )
            return (
                f"In-universe user identity (authoritative for this conversation): "
                f"Name = {role_name}. "
                "Address the user by the role name by default. "
                "Only use a role alias if the current message explicitly shows that alias as "
                "the speaker (e.g., a username/handle in the message metadata or prefix)."
            )
        
        # Canonical mode
        system_identity = getattr(system_config, "user_identity", None)
        display_name = getattr(system_identity, "display_name", "") if system_identity else ""
        aliases = getattr(system_identity, "aliases", []) if system_identity else []
        
        display_name = display_name.strip() if display_name else ""
        if not display_name:
            display_name = "User"
        
        if aliases:
            aliases_str = ", ".join(aliases)
            return (
                f"Canonical user identity (authoritative): Name = {display_name}. "
                f"Aliases: {aliases_str}. "
                "Address the user by the canonical name by default. "
                "Only use an alias if the current message explicitly shows that alias as "
                "the speaker (e.g., a username/handle in the message metadata or prefix)."
            )
        
        return f"Canonical user identity (authoritative): Name = {display_name}."

    def _build_time_header(self, system_config) -> Optional[str]:
        time_context = getattr(system_config, "time_context", None)
        if not time_context or not getattr(time_context, "enabled", False):
            return None
        
        tz_name = getattr(time_context, "timezone", None)
        tzinfo = None
        
        if tz_name:
            try:
                tzinfo = ZoneInfo(tz_name)
            except Exception:
                logger.warning(f"Invalid time_context.timezone '{tz_name}', using server local time")
                tzinfo = None
        
        if tzinfo:
            now = datetime.now(tzinfo)
            tz_label = tzinfo.key if hasattr(tzinfo, "key") else tz_name
        else:
            now = datetime.now().astimezone()
            tz_label = now.tzname() or "local"
        
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now.strftime("%A")
        
        return f"Current local time (server): {timestamp} {tz_label} ({weekday})"
    
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
        messages: List[Message],
        conversation_source: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format database messages for LLM API.
        
        For multi-user contexts (Discord, Slack, etc.), includes username
        in message content to show who said what.
        
        Filters out SCENE_CAPTURE messages as they're just anchor points
        for images and not actual conversation content.
        
        Args:
            messages: Database message objects
            conversation_source: Platform source ('web', 'discord', 'slack', etc.)
        
        Returns:
            List of message dicts with role and content
        """
        formatted = []
        is_multi_user = conversation_source and conversation_source != 'web'
        
        for msg in messages:
            # Skip scene capture messages - they're not part of the conversation
            if msg.role == MessageRole.SCENE_CAPTURE:
                continue
            
            content = msg.content
            
            # Filter out error messages from assistant responses to prevent LLM pattern learning
            # Error messages like "Sorry, I encountered an error..." were being echoed by the LLM
            # because it learned them as response patterns from conversation history
            if msg.role == MessageRole.ASSISTANT:
                # Remove common error message patterns that shouldn't be in training context
                error_patterns = [
                    "Sorry, I encountered an error communicating with my brain:",
                    "Sorry, I encountered an error rebuilding my memory.",
                    "Sorry, something went wrong",
                    "Connection error: HTTPConnectionPool",
                ]
                
                # Check if message is primarily an error message
                content_stripped = content.strip()
                is_error_message = any(
                    content_stripped.startswith(pattern) or content_stripped.endswith(pattern)
                    for pattern in error_patterns
                )
                
                if is_error_message:
                    # Skip this message entirely - it's just error noise
                    logger.debug(f"Filtering out error message from history: {content[:100]}...")
                    continue
                
                # Also clean error suffixes from otherwise valid responses
                # (in case error was appended to a good response)
                for pattern in error_patterns:
                    if pattern in content:
                        # Split on the error pattern and keep only the part before it
                        content = content.split(pattern)[0].strip()
                        if content:  # Only log if there's actual content remaining
                            logger.debug(f"Removed error suffix from response, keeping: {content[:100]}...")
            
            # For multi-user contexts, prepend username to user messages
            if is_multi_user and msg.role == MessageRole.USER:
                # Extract username from metadata
                username = None
                if msg.meta_data:
                    username = msg.meta_data.get('username') or msg.meta_data.get('user_name')
                
                # Format with username if available
                if username:
                    platform_display = conversation_source.capitalize() if conversation_source else 'Platform'
                    content = f"{username} ({platform_display}): {content}"
            
            # Only add message if it has content after error filtering
            if content.strip():
                formatted.append({
                    "role": msg.role.value,
                    "content": content,
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
        
        # CRITICAL: Ensure history ends with a user message for proper LLM response
        # If last message is assistant, the model will immediately output a stop token
        if result and result[-1]["role"] != "user":
            logger.warning(
                f"History truncation resulted in assistant as last message. "
                f"Removing trailing assistant messages to fix. "
                f"History length: {len(result)} -> ",
                extra={"end": ""}
            )
            # Remove trailing assistant messages until we find a user message or run out
            while result and result[-1]["role"] != "user":
                result.pop()
            logger.warning(f"{len(result)}", extra={"end": "\n"})
        
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
