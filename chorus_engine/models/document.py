"""Database models for document analysis feature.

This module contains SQLAlchemy models for storing and managing
documents, document chunks, and document access logs.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    Boolean,
    JSON,
)
from sqlalchemy.orm import relationship

from chorus_engine.db.database import Base


class Document(Base):
    """Main document table storing document metadata."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)  # Original filename
    storage_key = Column(String(500), nullable=False, unique=True)  # Unique storage identifier
    file_type = Column(String(50), nullable=False)  # pdf, csv, xlsx, txt, docx, md
    file_size_bytes = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=True)  # For PDFs, null for other types
    
    # Metadata
    title = Column(String(500), nullable=True)  # User-provided or extracted title
    description = Column(Text, nullable=True)  # User-provided description
    author = Column(String(200), nullable=True)  # Extracted from document metadata
    character_id = Column(String(100), nullable=True)  # Character who uploaded/owns document (NULL = global)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_accessed = Column(DateTime, nullable=True)
    
    # Document scoping (Phase 9 - Conversation-level isolation)
    document_scope = Column(String(50), nullable=False, default="conversation")  # conversation, character, global
    conversation_id = Column(String(200), nullable=True)  # Required if scope='conversation'
    
    # Processing status
    processing_status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)  # Error message if processing failed
    chunk_count = Column(Integer, nullable=False, default=0)  # Number of chunks created
    
    # Vector store integration
    vector_collection_id = Column(String(200), nullable=True)  # ChromaDB collection ID
    
    # Additional metadata (JSON field for extensibility)
    metadata_json = Column(JSON, nullable=True)  # Custom metadata, tags, etc.
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    access_logs = relationship("DocumentAccessLog", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', type='{self.file_type}')>"


class DocumentChunk(Base):
    """Document chunks for vector storage and retrieval."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Chunk identification
    chunk_index = Column(Integer, nullable=False)  # Sequential position in document (0-based)
    chunk_id = Column(String(200), nullable=False, unique=True)  # Unique identifier for vector store
    
    # Content
    content = Column(Text, nullable=False)  # The actual text content
    content_length = Column(Integer, nullable=False)  # Character count
    
    # Location in original document
    page_numbers = Column(String(200), nullable=True)  # "5-7" or "12" for PDFs
    start_line = Column(Integer, nullable=True)  # For text files
    end_line = Column(Integer, nullable=True)  # For text files
    
    # Chunking metadata
    chunk_method = Column(String(50), nullable=False)  # semantic, fixed_size, paragraph, etc.
    overlap_tokens = Column(Integer, nullable=False, default=0)  # Overlap with previous chunk
    
    # Vector embedding
    embedding_model = Column(String(100), nullable=True)  # Model used for embedding
    embedding_created_at = Column(DateTime, nullable=True)  # When embedding was created
    
    # Additional metadata
    metadata_json = Column(JSON, nullable=True)  # Headers, context, etc.
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class DocumentAccessLog(Base):
    """Track when documents are accessed in conversations."""

    __tablename__ = "document_access_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Conversation context
    conversation_id = Column(String(200), nullable=True)  # Which conversation accessed it
    message_id = Column(Integer, nullable=True)  # Which message triggered the access
    
    # Access details
    access_type = Column(String(50), nullable=False)  # retrieval, reference, citation
    chunks_retrieved = Column(Integer, nullable=False, default=0)  # How many chunks were used
    chunk_ids = Column(Text, nullable=True)  # Comma-separated list of chunk IDs
    
    # Search query that triggered retrieval
    query = Column(Text, nullable=True)  # The search query used
    relevance_score = Column(Float, nullable=True)  # Average relevance score
    
    # Timestamps
    accessed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="access_logs")

    def __repr__(self):
        return f"<DocumentAccessLog(id={self.id}, document_id={self.document_id}, type='{self.access_type}')>"


class CodeExecutionLog(Base):
    """Track code execution attempts for security and debugging."""

    __tablename__ = "code_execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Conversation context
    conversation_id = Column(String(200), nullable=True)
    message_id = Column(Integer, nullable=True)
    
    # Code details
    language = Column(String(50), nullable=False)  # python, javascript, sql, etc.
    code = Column(Text, nullable=False)  # The actual code executed
    code_hash = Column(String(64), nullable=False)  # SHA-256 hash for deduplication
    
    # Execution details
    execution_status = Column(String(50), nullable=False)  # pending, running, success, error, timeout
    execution_time_ms = Column(Integer, nullable=True)  # How long it took to run
    
    # Results
    stdout = Column(Text, nullable=True)  # Standard output
    stderr = Column(Text, nullable=True)  # Standard error
    return_value = Column(Text, nullable=True)  # Serialized return value
    error_message = Column(Text, nullable=True)  # Error message if failed
    
    # Security
    approved_by_user = Column(Boolean, nullable=False, default=False)  # User approved execution
    sandboxed = Column(Boolean, nullable=False, default=True)  # Ran in sandbox
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Additional metadata
    metadata_json = Column(JSON, nullable=True)  # Execution environment, dependencies, etc.

    def __repr__(self):
        return f"<CodeExecutionLog(id={self.id}, language='{self.language}', status='{self.execution_status}')>"
