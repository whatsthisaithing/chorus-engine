"""Repository for document database operations.

This module provides CRUD operations for documents, chunks, and access logs.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from chorus_engine.models.document import (
    Document,
    DocumentChunk,
    DocumentAccessLog,
    CodeExecutionLog
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document-related database operations."""
    
    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db
    
    # ==================== Document Operations ====================
    
    def create_document(
        self,
        filename: str,
        storage_key: str,
        file_type: str,
        file_size_bytes: int,
        page_count: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        character_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: str = "conversation",
        metadata_json: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Create a new document record.
        
        Args:
            filename: Original filename
            storage_key: Unique storage identifier
            file_type: File extension (pdf, csv, txt, etc.)
            file_size_bytes: File size in bytes
            page_count: Number of pages (for PDFs)
            title: Document title
            description: User-provided description
            author: Document author
            character_id: Character who owns document (None = global)
            conversation_id: Conversation scope (required if scope='conversation')
            document_scope: Scope level ('conversation', 'character', 'global')
            metadata_json: Additional metadata
            
        Returns:
            Created Document object
        """
        # Validation: conversation scope requires conversation_id
        if document_scope == "conversation" and not conversation_id:
            raise ValueError("conversation_id required for conversation-scoped documents")
        if document_scope == "character" and not character_id:
            raise ValueError("character_id required for character-scoped documents")
        
        document = Document(
            filename=filename,
            storage_key=storage_key,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            page_count=page_count,
            title=title or filename,  # Default to filename if no title
            description=description,
            author=author,
            character_id=character_id,
            conversation_id=conversation_id,
            document_scope=document_scope,
            created_at=datetime.utcnow(),
            uploaded_at=datetime.utcnow(),
            processing_status="pending",
            chunk_count=0,
            metadata_json=metadata_json
        )
        
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Created document: {document.id} ({filename}) with {document_scope} scope")
        return document
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID."""
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def get_document_by_storage_key(self, storage_key: str) -> Optional[Document]:
        """Get document by storage key."""
        return self.db.query(Document).filter(Document.storage_key == storage_key).first()
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        character_id: Optional[str] = None,
        include_global: bool = True
    ) -> List[Document]:
        """
        List documents with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            file_type: Filter by file type
            status: Filter by processing status
            character_id: Filter by character (None = all)
            include_global: Include documents with NULL character_id
            
        Returns:
            List of Document objects
        """
        query = self.db.query(Document)
        
        # Character filtering
        if character_id is not None:
            if include_global:
                # Show character's documents AND global documents
                query = query.filter(
                    (Document.character_id == character_id) | 
                    (Document.character_id.is_(None))
                )
            else:
                # Only this character's documents
                query = query.filter(Document.character_id == character_id)
        
        if file_type:
            query = query.filter(Document.file_type == file_type)
        if status:
            query = query.filter(Document.processing_status == status)
        
        return query.order_by(desc(Document.uploaded_at)).offset(offset).limit(limit).all()
        
        return query.order_by(desc(Document.uploaded_at)).limit(limit).offset(offset).all()
    
    def update_document_status(
        self,
        document_id: int,
        status: str,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None
    ) -> Optional[Document]:
        """
        Update document processing status.
        
        Args:
            document_id: Document ID
            status: New status (pending, processing, completed, failed)
            error: Error message if status is failed
            chunk_count: Number of chunks created
            
        Returns:
            Updated Document object or None if not found
        """
        document = self.get_document(document_id)
        if not document:
            return None
        
        document.processing_status = status
        if error:
            document.processing_error = error
        if chunk_count is not None:
            document.chunk_count = chunk_count
        
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Updated document {document_id} status: {status}")
        return document
    
    def update_document_metadata(
        self,
        document_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """
        Update document metadata.
        
        Args:
            document_id: Document ID
            title: New title
            description: New description
            metadata_json: Additional metadata
            
        Returns:
            Updated Document object or None if not found
        """
        document = self.get_document(document_id)
        if not document:
            return None
        
        if title is not None:
            document.title = title
        if description is not None:
            document.description = description
        if metadata_json is not None:
            document.metadata_json = metadata_json
        
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Updated document {document_id} metadata")
        return document
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete document and all related chunks/logs.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        document = self.get_document(document_id)
        if not document:
            return False
        
        # Cascading delete will remove chunks and logs automatically
        self.db.delete(document)
        self.db.commit()
        
        logger.info(f"Deleted document {document_id}")
        return True
    
    def update_last_accessed(self, document_id: int):
        """Update document's last accessed timestamp."""
        document = self.get_document(document_id)
        if document:
            document.last_accessed = datetime.utcnow()
            self.db.commit()
    
    # ==================== Scope-Aware Document Operations (Phase 9) ====================
    
    def get_accessible_documents(
        self,
        character_id: str,
        conversation_id: Optional[str] = None,
        include_character_scope: bool = True,
        include_global: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """
        Get documents accessible in the given context (scope-aware).
        
        This is the PRIMARY method for retrieving documents with proper privacy isolation.
        
        Args:
            character_id: Character accessing the documents
            conversation_id: Conversation context (None = show all accessible)
            include_character_scope: Include character-wide documents
            include_global: Include global documents
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of accessible documents
            
        Example:
            # In conversation context - show all accessible documents
            docs = repo.get_accessible_documents(
                character_id="marcus",
                conversation_id="conv-123"
            )
            # Returns: conversation-scoped + character-scoped + global
        """
        query = self.db.query(Document).filter(Document.processing_status == "completed")
        
        # Build scope filter conditions
        conditions = []
        
        # Conversation-scoped documents (if conversation_id provided)
        if conversation_id:
            conditions.append(
                (Document.document_scope == "conversation") &
                (Document.conversation_id == conversation_id) &
                (Document.character_id == character_id)
            )
        
        # Character-scoped documents
        if include_character_scope:
            conditions.append(
                (Document.document_scope == "character") &
                (Document.character_id == character_id)
            )
        
        # Global documents
        if include_global:
            conditions.append(Document.document_scope == "global")
        
        # Combine with OR
        if conditions:
            from sqlalchemy import or_
            query = query.filter(or_(*conditions))
        else:
            # No conditions = no results
            return []
        
        return query.order_by(desc(Document.uploaded_at)).offset(offset).limit(limit).all()
    
    def verify_document_access(
        self,
        document_id: int,
        character_id: str,
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Verify if a document is accessible in the given context.
        
        Use this before allowing document operations (edit, delete, retrieve).
        
        Args:
            document_id: Document to check
            character_id: Character attempting access
            conversation_id: Conversation context
            
        Returns:
            True if access allowed, False otherwise
        """
        document = self.get_document(document_id)
        if not document:
            return False
        
        # Global documents: always accessible
        if document.document_scope == "global":
            return True
        
        # Character-scoped: must match character
        if document.document_scope == "character":
            return document.character_id == character_id
        
        # Conversation-scoped: must match character AND conversation
        if document.document_scope == "conversation":
            return (
                document.character_id == character_id and
                document.conversation_id == conversation_id
            )
        
        # Unknown scope
        return False
    
    # ==================== Chunk Operations ====================
    
    def create_chunk(
        self,
        document_id: int,
        chunk_index: int,
        chunk_id: str,
        content: str,
        chunk_method: str,
        page_numbers: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        overlap_tokens: int = 0,
        embedding_model: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Create a document chunk.
        
        Args:
            document_id: Parent document ID
            chunk_index: Sequential position in document
            chunk_id: Unique identifier for vector store
            content: Text content
            chunk_method: Chunking method used (semantic, fixed_size, etc.)
            page_numbers: Page range for PDFs
            start_line: Starting line for text files
            end_line: Ending line for text files
            overlap_tokens: Overlap with previous chunk
            embedding_model: Model used for embedding
            metadata_json: Additional metadata
            
        Returns:
            Created DocumentChunk object
        """
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            content=content,
            content_length=len(content),
            page_numbers=page_numbers,
            start_line=start_line,
            end_line=end_line,
            chunk_method=chunk_method,
            overlap_tokens=overlap_tokens,
            embedding_model=embedding_model,
            embedding_created_at=datetime.utcnow() if embedding_model else None,
            metadata_json=metadata_json,
            created_at=datetime.utcnow()
        )
        
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        
        return chunk
    
    def get_chunks_for_document(self, document_id: int) -> List[DocumentChunk]:
        """Get all chunks for a document, ordered by chunk_index."""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by unique chunk_id."""
        return self.db.query(DocumentChunk).filter(DocumentChunk.chunk_id == chunk_id).first()
    
    def delete_chunks_for_document(self, document_id: int) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks deleted
        """
        count = (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .delete()
        )
        self.db.commit()
        return count
    
    # ==================== Access Log Operations ====================
    
    def log_document_access(
        self,
        document_id: int,
        access_type: str,
        conversation_id: Optional[str] = None,
        message_id: Optional[int] = None,
        chunks_retrieved: int = 0,
        chunk_ids: Optional[str] = None,
        query: Optional[str] = None,
        relevance_score: Optional[float] = None
    ) -> DocumentAccessLog:
        """
        Log document access in a conversation.
        
        Args:
            document_id: Document ID
            access_type: Type of access (retrieval, reference, citation)
            conversation_id: Conversation ID
            message_id: Message ID that triggered access
            chunks_retrieved: Number of chunks used
            chunk_ids: Comma-separated chunk IDs
            query: Search query used
            relevance_score: Average relevance score
            
        Returns:
            Created DocumentAccessLog object
        """
        log = DocumentAccessLog(
            document_id=document_id,
            conversation_id=conversation_id,
            message_id=message_id,
            access_type=access_type,
            chunks_retrieved=chunks_retrieved,
            chunk_ids=chunk_ids,
            query=query,
            relevance_score=relevance_score,
            accessed_at=datetime.utcnow()
        )
        
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        
        # Update document's last_accessed timestamp
        self.update_last_accessed(document_id)
        
        return log
    
    def get_access_logs_for_document(
        self,
        document_id: int,
        limit: int = 100
    ) -> List[DocumentAccessLog]:
        """Get access logs for a document."""
        return (
            self.db.query(DocumentAccessLog)
            .filter(DocumentAccessLog.document_id == document_id)
            .order_by(desc(DocumentAccessLog.accessed_at))
            .limit(limit)
            .all()
        )
    
    def get_access_logs_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100
    ) -> List[DocumentAccessLog]:
        """Get all document accesses in a conversation."""
        return (
            self.db.query(DocumentAccessLog)
            .filter(DocumentAccessLog.conversation_id == conversation_id)
            .order_by(desc(DocumentAccessLog.accessed_at))
            .limit(limit)
            .all()
        )
    
    def get_document_usage_stats(self, document_id: int) -> Dict[str, Any]:
        """
        Get usage statistics for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary with usage stats
        """
        stats = {
            "total_accesses": 0,
            "unique_conversations": 0,
            "total_chunks_retrieved": 0,
            "avg_relevance_score": 0.0,
            "last_accessed": None
        }
        
        # Total accesses
        stats["total_accesses"] = (
            self.db.query(func.count(DocumentAccessLog.id))
            .filter(DocumentAccessLog.document_id == document_id)
            .scalar()
        ) or 0
        
        # Unique conversations
        stats["unique_conversations"] = (
            self.db.query(func.count(func.distinct(DocumentAccessLog.conversation_id)))
            .filter(DocumentAccessLog.document_id == document_id)
            .filter(DocumentAccessLog.conversation_id.isnot(None))
            .scalar()
        ) or 0
        
        # Total chunks retrieved
        stats["total_chunks_retrieved"] = (
            self.db.query(func.sum(DocumentAccessLog.chunks_retrieved))
            .filter(DocumentAccessLog.document_id == document_id)
            .scalar()
        ) or 0
        
        # Average relevance score
        avg_score = (
            self.db.query(func.avg(DocumentAccessLog.relevance_score))
            .filter(DocumentAccessLog.document_id == document_id)
            .filter(DocumentAccessLog.relevance_score.isnot(None))
            .scalar()
        )
        if avg_score:
            stats["avg_relevance_score"] = float(avg_score)
        
        # Last accessed
        document = self.get_document(document_id)
        if document and document.last_accessed:
            stats["last_accessed"] = document.last_accessed.isoformat()
        
        return stats
    
    # ==================== Code Execution Logs ====================
    
    def create_code_execution_log(
        self,
        language: str,
        code: str,
        code_hash: str,
        conversation_id: Optional[str] = None,
        message_id: Optional[int] = None,
        approved_by_user: bool = False,
        sandboxed: bool = True,
        metadata_json: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionLog:
        """
        Create a code execution log entry.
        
        Args:
            language: Programming language
            code: Code to execute
            code_hash: SHA-256 hash of code
            conversation_id: Associated conversation
            message_id: Associated message
            approved_by_user: Whether user approved execution
            sandboxed: Whether code runs in sandbox
            metadata_json: Additional metadata
            
        Returns:
            Created CodeExecutionLog object
        """
        log = CodeExecutionLog(
            conversation_id=conversation_id,
            message_id=message_id,
            language=language,
            code=code,
            code_hash=code_hash,
            execution_status="pending",
            approved_by_user=approved_by_user,
            sandboxed=sandboxed,
            created_at=datetime.utcnow(),
            metadata_json=metadata_json
        )
        
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        
        logger.info(f"Created code execution log: {log.id} ({language})")
        return log
    
    def update_code_execution_result(
        self,
        log_id: int,
        status: str,
        execution_time_ms: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        return_value: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[CodeExecutionLog]:
        """
        Update code execution results.
        
        Args:
            log_id: Log ID
            status: Execution status (running, success, error, timeout)
            execution_time_ms: Execution duration
            stdout: Standard output
            stderr: Standard error
            return_value: Serialized return value
            error_message: Error message if failed
            
        Returns:
            Updated CodeExecutionLog or None if not found
        """
        log = self.db.query(CodeExecutionLog).filter(CodeExecutionLog.id == log_id).first()
        if not log:
            return None
        
        log.execution_status = status
        log.execution_time_ms = execution_time_ms
        log.stdout = stdout
        log.stderr = stderr
        log.return_value = return_value
        log.error_message = error_message
        
        if not log.executed_at and status == "running":
            log.executed_at = datetime.utcnow()
        
        if status in ("success", "error", "timeout"):
            log.completed_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(log)
        
        return log
    
    def get_code_execution_log(self, log_id: int) -> Optional[CodeExecutionLog]:
        """Get code execution log by ID."""
        return self.db.query(CodeExecutionLog).filter(CodeExecutionLog.id == log_id).first()
    
    def get_code_execution_logs_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100
    ) -> List[CodeExecutionLog]:
        """Get all code execution logs for a conversation."""
        return (
            self.db.query(CodeExecutionLog)
            .filter(CodeExecutionLog.conversation_id == conversation_id)
            .order_by(desc(CodeExecutionLog.created_at))
            .limit(limit)
            .all()
        )
