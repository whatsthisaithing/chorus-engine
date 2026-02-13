"""Database models for conversations, threads, and messages."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Enum, JSON, Integer, Float
from sqlalchemy.orm import relationship
import enum
import uuid

from chorus_engine.db.database import Base


def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())


class MessageRole(str, enum.Enum):
    """Message role types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    SCENE_CAPTURE = "scene_capture"  # Phase 9: User-triggered scene capture images


class MemoryType(str, enum.Enum):
    """Memory types for hierarchical retrieval."""
    CORE = "core"           # Immutable character backstory (highest priority)
    EXPLICIT = "explicit"   # User-created facts (high priority)
    FACT = "fact"           # Extracted factual information (renamed from implicit)
    PROJECT = "project"     # Ongoing projects, goals, or objectives
    EXPERIENCE = "experience"  # Shared experiences and stories
    STORY = "story"         # Narratives and anecdotes
    RELATIONSHIP = "relationship"  # Relationship dynamics and interactions
    EPHEMERAL = "ephemeral" # Temporary context (low priority)
    
    # Backward compatibility alias
    IMPLICIT = "fact"       # For migration compatibility
    
    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive lookup for backwards compatibility."""
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class Conversation(Base):
    """
    A conversation represents a full interaction context with a character.
    
    Each conversation:
    - Is associated with one character
    - Can have multiple threads
    - Has a title (auto-generated or user-set)
    - Tracks creation and update times
    - Can be marked as private (Phase 4.1 - no memory extraction)
    """
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    character_id = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Phase 4.1: Privacy mode
    is_private = Column(String(10), nullable=False, default="false")  # "true" or "false" as string for SQLite compatibility
    
    # Phase 4.1: Track last extracted message count to prevent duplicate extraction
    last_extracted_message_count = Column(Integer, nullable=False, default=0)
    
    # Phase 5: Image generation confirmation preference
    image_confirmation_disabled = Column(String(10), nullable=False, default="false")  # "true" or "false"
    
    # Video generation confirmation preference
    video_confirmation_disabled = Column(String(10), nullable=False, default="false")  # "true" or "false"

    # Proactive offer controls (conversation-level)
    allow_image_offers = Column(String(10), nullable=False, default="true")
    allow_video_offers = Column(String(10), nullable=False, default="true")
    last_image_offer_at = Column(DateTime, nullable=True, default=None)
    last_video_offer_at = Column(DateTime, nullable=True, default=None)
    last_image_offer_message_count = Column(Integer, nullable=True, default=None)
    last_video_offer_message_count = Column(Integer, nullable=True, default=None)
    image_offer_count = Column(Integer, nullable=False, default=0)
    video_offer_count = Column(Integer, nullable=False, default=0)
    
    # Phase 6: TTS enabled for this conversation (NULL = use character default, 0 = off, 1 = on)
    tts_enabled = Column(Integer, nullable=True, default=None)
    
    # Phase 8: Conversation analysis tracking
    last_analyzed_at = Column(DateTime, nullable=True, default=None)
    last_summary_analyzed_at = Column(DateTime, nullable=True, default=None)
    last_memories_analyzed_at = Column(DateTime, nullable=True, default=None)

    # Continuity bootstrapping (single-user)
    continuity_mode = Column(String(10), nullable=False, default="ask")  # ask | use | fresh
    continuity_choice_remembered = Column(String(10), nullable=False, default="false")
    primary_user = Column(String(200), nullable=True, default=None)
    
    # Auto-generated title tracking (1 = auto-generated, 0 = user-set, NULL = old data)
    title_auto_generated = Column(Integer, nullable=True, default=1)
    
    # Source platform (web, discord, etc.)
    source = Column(String(20), nullable=False, default="web")
    
    # Relationships
    threads = relationship("Thread", back_populates="conversation", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, character={self.character_id}, title={self.title})>"


class ConversationSummary(Base):
    """
    Conversation summary for long-term context compression.
    
    Phase 8: Memory Intelligence & Conversation Management
    - Stores progressive summaries of conversation threads
    - Enables efficient context management for very long conversations
    - Tracks key themes, participants, and emotional arc
    - Supports multi-turn summarization (summary of summaries)
    """
    __tablename__ = "conversation_summaries"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(36), ForeignKey("threads.id", ondelete="CASCADE"), nullable=True)
    
    # Summary content
    summary = Column(Text, nullable=False)  # The actual summary text
    summary_type = Column(String(20), nullable=False, default="progressive")  # progressive | final | meta
    
    # What this summary covers
    message_range_start = Column(Integer, nullable=False)  # Starting message index
    message_range_end = Column(Integer, nullable=False)  # Ending message index
    message_count = Column(Integer, nullable=False)  # How many messages summarized
    
    # Extracted metadata
    key_topics = Column(JSON, nullable=True)  # List of main topics discussed
    participants = Column(JSON, nullable=True)  # List of who was involved
    emotional_arc = Column(String(200), nullable=True)  # Brief emotional progression description
    tone = Column(String(200), nullable=True)  # Overall tone of the conversation
    open_questions = Column(JSON, nullable=True)  # Optional open questions for follow-up
    
    # Phase 8 Day 10: Track if analysis was manually triggered
    manual = Column(String(10), nullable=False, default="false")  # "true" or "false" for SQLite compatibility
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        preview = self.summary[:50] + "..." if len(self.summary) > 50 else self.summary
        return f"<ConversationSummary(id={self.id}, conv={self.conversation_id}, msgs={self.message_range_start}-{self.message_range_end}, summary={preview})>"


class Thread(Base):
    """
    A thread represents a sub-conversation within a larger conversation.
    
    Threads allow:
    - Branching discussions
    - Exploring different topics
    - Organizing message history
    """
    __tablename__ = "threads"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    title = Column(String(200), nullable=False, default="New Thread")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="threads")
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan", order_by="Message.created_at")
    
    def __repr__(self):
        return f"<Thread(id={self.id}, conversation={self.conversation_id}, title={self.title})>"


class Message(Base):
    """
    A message represents a single exchange in a conversation.
    
    Messages have:
    - A role (system, user, assistant)
    - Content (the actual message text)
    - Optional metadata (tokens, finish reason, etc.)
    - Timestamps
    """
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    thread_id = Column(String(36), ForeignKey("threads.id"), nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    meta_data = Column("metadata", JSON, nullable=True)  # Store token counts, finish reason, etc.
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Phase 4.1: Privacy flag - marks messages sent during privacy mode
    is_private = Column(String(10), nullable=False, default="false")  # "true" or "false" as string for SQLite compatibility
    
    # Phase 8: Enhanced message fields for compression
    emotional_weight = Column(Float, nullable=True)  # 0.0-1.0 emotional significance
    summary = Column(String(500), nullable=True)  # Brief summary for compressed context
    preserve_full_text = Column(String(10), nullable=False, default="true")  # Whether to keep full text in context
    
    # Soft delete timestamp (NULL = active)
    deleted_at = Column(DateTime, nullable=True, default=None)
    
    # Relationships
    thread = relationship("Thread", back_populates="messages")
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role}, content={preview})>"


class Memory(Base):
    """
    A memory represents stored information about the conversation.
    
    Memories have a hierarchical type system (Phase 3):
    - CORE: Immutable character backstory (highest priority, loaded from character YAML)
    - EXPLICIT: User-created facts ("remember this")
    - IMPLICIT: Extracted from conversation context (Phase 4.1)
    - EPHEMERAL: Temporary working memory
    
    Phase 3 adds:
    - Vector embeddings for semantic search
    - Character association for core memories
    - Priority scoring for retrieval ranking
    - Tag-based organization
    
    Phase 4.1 adds:
    - Character-scoped storage (conversation_id now optional)
    - Confidence scoring for implicit memories
    - Status tracking (pending/approved/auto_approved)
    - Memory categories for implicit memories
    - Source message tracking
    """
    __tablename__ = "memories"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Phase 4.1: Make conversation_id optional, character_id required
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True)  # Optional - tracks origin conversation
    thread_id = Column(String(36), ForeignKey("threads.id"), nullable=True)
    character_id = Column(String(50), nullable=False, index=True)  # Required - primary scope for all memories
    
    memory_type = Column(Enum(MemoryType, values_callable=lambda obj: [e.value for e in obj]), nullable=False, default=MemoryType.EXPLICIT)
    content = Column(Text, nullable=False)
    
    # Phase 3: Vector embedding fields
    vector_id = Column(String(36), nullable=True)  # ID in ChromaDB collection
    embedding_model = Column(String(100), nullable=True, default="all-MiniLM-L6-v2")  # Track which model generated embedding
    
    # Phase 3: Retrieval metadata
    priority = Column(Integer, nullable=False, default=50)  # 0-100, higher = more important
    tags = Column(JSON, nullable=True)  # List of tags for filtering
    
    # Phase 4.1: Implicit memory extraction fields
    confidence = Column(Float, nullable=True)  # 0.0-1.0 confidence score (for implicit memories)
    category = Column(String(50), nullable=True)  # personal_info, preference, experience, relationship, goal, skill
    status = Column(String(20), nullable=False, default="approved")  # pending | approved | auto_approved
    source_messages = Column(JSON, nullable=True)  # List of message IDs that contributed to this memory
    durability = Column(String(20), nullable=False, default="situational")  # ephemeral | situational | long_term | identity
    pattern_eligible = Column(Integer, nullable=False, default=0)  # 0/1 for SQLite compatibility
    
    # Phase 8: Enhanced memory fields
    emotional_weight = Column(Float, nullable=True)  # 0.0-1.0 emotional significance
    participants = Column(JSON, nullable=True)  # List of participants involved in this memory
    key_moments = Column(JSON, nullable=True)  # List of key moment references/timestamps
    summary = Column(String(500), nullable=True)  # Brief summary for quick context
    
    # Phase 3: Platform source field (Discord Bridge Phase 3)
    source = Column(String(20), nullable=False, default='web', server_default='web')  # web, discord, slack, etc.
    
    # Existing metadata field (for backwards compatibility)
    meta_data = Column("metadata", JSON, nullable=True)  # Additional flexible storage
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="memories")
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Memory(id={self.id}, type={self.memory_type}, char={self.character_id}, priority={self.priority}, content={preview})>"


class MomentPin(Base):
    """
    Moment pin stores bounded conversational moments with both hot and cold layers.

    Hot layer fields are used for retrieval/injection while transcript_snapshot is
    archival cold data only loaded via explicit tool recall.
    """
    __tablename__ = "moment_pins"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(200), nullable=False, index=True)
    character_id = Column(String(50), nullable=False, index=True)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    selected_message_ids = Column(JSON, nullable=False)  # [message_id, ...]
    transcript_snapshot = Column(Text, nullable=False)  # cold archival snapshot

    what_happened = Column(Text, nullable=False)
    why_model = Column(Text, nullable=False)
    why_user = Column(Text, nullable=True)
    quote_snippet = Column(Text, nullable=True)
    tags = Column(JSON, nullable=False, default=list)

    reinforcement_score = Column(Float, nullable=False, default=1.0)
    turns_since_reinforcement = Column(Integer, nullable=False, default=0)
    archived = Column(Integer, nullable=False, default=0)  # 0/1 for SQLite compatibility

    telemetry_flags = Column(
        JSON,
        nullable=False,
        default=lambda: {
            "contains_roleplay": False,
            "contains_directives": False,
            "contains_sensitive_content": False,
        },
    )

    vector_id = Column(String(36), nullable=True)  # ID in moment pin vector store

    def __repr__(self):
        preview = self.what_happened[:50] + "..." if len(self.what_happened) > 50 else self.what_happened
        return f"<MomentPin(id={self.id}, char={self.character_id}, user={self.user_id}, archived={bool(self.archived)}, what={preview})>"


class GeneratedImage(Base):
    """
    A generated image represents an AI-generated image from ComfyUI.
    
    Phase 5: Image Generation
    - Tracks images generated during conversations
    - Stores prompts, workflow info, and file paths
    - Links to specific messages/conversations
    - Includes generation metadata (seed, dimensions, time)
    """
    __tablename__ = "generated_images"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(36), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    character_id = Column(String(50), nullable=False, index=True)
    
    # Generation parameters
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=True)
    workflow_file = Column(String(200), nullable=False)  # e.g., "workflows/nova/workflow.json"
    
    # File paths
    file_path = Column(String(500), nullable=False)  # e.g., "data/images/123/456_full.png"
    thumbnail_path = Column(String(500), nullable=True)
    
    # Image metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    seed = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)  # User notes about the image
    
    # Timestamps and performance
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    generation_time = Column(Float, nullable=True)  # Seconds
    
    def __repr__(self):
        prompt_preview = self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt
        return f"<GeneratedImage(id={self.id}, char={self.character_id}, prompt={prompt_preview})>"


class VoiceSample(Base):
    """
    Voice sample for TTS voice cloning.
    
    Phase 6: Text-to-Speech Integration
    - Stores voice samples uploaded by users
    - Each character can have multiple samples
    - One sample marked as default per character
    - Includes transcript for voice cloning workflows
    """
    __tablename__ = "voice_samples"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(String(50), nullable=False, index=True)
    filename = Column(String(200), nullable=False)  # e.g., "nova_sample_001.wav"
    transcript = Column(Text, nullable=False)  # Exact words spoken in sample
    is_default = Column(Integer, nullable=False, default=0)  # 0 or 1
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<VoiceSample(id={self.id}, char={self.character_id}, file={self.filename}, default={bool(self.is_default)})>"


class AudioMessage(Base):
    """
    TTS-generated audio for a message.
    
    Phase 6: Text-to-Speech Integration
    - Tracks audio generated for assistant messages
    - Stores generation metadata and workflow info
    - Links to voice sample used (if any)
    - Includes preprocessed text sent to TTS
    """
    __tablename__ = "audio_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True)
    audio_filename = Column(String(200), nullable=False)  # e.g., "msg_123_20250101_120000.wav"
    workflow_name = Column(String(200), nullable=True)  # Which workflow was used
    generation_duration = Column(Float, nullable=True)  # Seconds
    text_preprocessed = Column(Text, nullable=True)  # Plain English sent to TTS
    voice_sample_id = Column(Integer, ForeignKey("voice_samples.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AudioMessage(id={self.id}, msg_id={self.message_id}, file={self.audio_filename})>"


class GeneratedVideo(Base):
    """
    A generated video represents an AI-generated video from ComfyUI.
    
    Video Generation Feature:
    - Tracks videos generated during conversations
    - Stores motion-focused prompts and workflow info
    - Workflow-agnostic (format, duration, resolution determined by ComfyUI)
    - Includes thumbnails (first frame extraction)
    - Links to conversations with metadata
    """
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # File paths
    file_path = Column(String(500), nullable=False)  # e.g., "data/videos/conv_id/1_video.webm"
    thumbnail_path = Column(String(500), nullable=True)  # First frame thumbnail
    
    # Video properties (workflow-agnostic, may be null)
    format = Column(String(10), nullable=True)  # webm, mp4, webp, gif, etc.
    duration_seconds = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Generation parameters
    prompt = Column(Text, nullable=False)  # Motion-focused prompt
    negative_prompt = Column(Text, nullable=True)
    workflow_file = Column(String(500), nullable=True)  # Path to workflow JSON
    
    # ComfyUI metadata
    comfy_prompt_id = Column(String(100), nullable=True)  # ComfyUI prompt tracking
    generation_time_seconds = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        prompt_preview = self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt
        duration_str = f"{self.duration_seconds:.1f}s" if self.duration_seconds else "unknown"
        return f"<GeneratedVideo(id={self.id}, format={self.format}, duration={duration_str}, prompt={prompt_preview})>"


class ImageAttachment(Base):
    """
    Image attachment for vision system.
    
    Stores image files uploaded by users and their vision analysis results.
    Supports the two-stage vision architecture:
    - Vision model observes (structured perception)
    - Character LLM interprets through personality
    """
    __tablename__ = "image_attachments"
    
    id = Column(String(50), primary_key=True, default=generate_uuid)
    message_id = Column(String(50), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(String(50), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    character_id = Column(String(50), nullable=False)
    
    # File storage
    original_path = Column(String(500), nullable=False)
    processed_path = Column(String(500), nullable=True)
    original_filename = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(50), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Vision analysis
    vision_processed = Column(String(10), nullable=False, default="false")  # "true" or "false"
    vision_skipped = Column(String(10), nullable=False, default="false")  # "true" or "false"
    vision_skip_reason = Column(String(100), nullable=True)
    
    vision_model = Column(String(100), nullable=True)
    vision_backend = Column(String(50), nullable=True)
    vision_processed_at = Column(DateTime, nullable=True)
    vision_processing_time_ms = Column(Integer, nullable=True)
    
    vision_observation = Column(Text, nullable=True)
    vision_confidence = Column(Float, nullable=True)
    vision_tags = Column(Text, nullable=True)  # JSON string
    
    # User metadata
    description = Column(Text, nullable=True)
    source = Column(String(20), nullable=False, default="web")
    
    def __repr__(self):
        filename = self.original_filename or "unknown"
        status = "processed" if self.vision_processed == "true" else ("skipped" if self.vision_skipped == "true" else "pending")
        return f"<ImageAttachment(id={self.id}, file={filename}, status={status})>"
