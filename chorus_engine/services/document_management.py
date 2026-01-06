"""Document management service orchestrating upload, processing, and storage."""

import logging
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.services.document_loader import DocumentLoader, LoadedDocument
from chorus_engine.services.document_chunking import ChunkingService, ChunkMethod
from chorus_engine.services.document_vector_store import DocumentVectorStore
from chorus_engine.repositories.document_repository import DocumentRepository
from chorus_engine.models.document import Document

logger = logging.getLogger(__name__)


class DocumentManagementService:
    """Service for managing document upload, processing, and retrieval."""
    
    def __init__(
        self,
        storage_dir: str = "data/documents",
        vector_store_dir: str = "data/vector_store"
    ):
        """
        Initialize document management service.
        
        Args:
            storage_dir: Directory for document file storage
            vector_store_dir: Directory for vector store persistence
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = DocumentLoader()
        self.chunker = ChunkingService()
        self.vector_store = DocumentVectorStore(vector_store_dir)
        
        logger.info(f"DocumentManagementService initialized (storage: {storage_dir})")
    
    def upload_document(
        self,
        db: Session,
        file_path: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        chunk_method: ChunkMethod = ChunkMethod.SEMANTIC,
        character_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Upload and process a document with conversation-level scoping (Phase 9).
        
        This method:
        1. Validates and loads the document
        2. Stores the file
        3. Creates database record with scope
        4. Chunks the content
        5. Adds chunks to vector store with scope metadata
        6. Updates processing status
        
        Args:
            db: Database session
            file_path: Path to document file
            title: Optional document title (defaults to filename)
            description: Optional description
            chunk_method: Chunking strategy to use
            character_id: Character who owns document
            conversation_id: Conversation scope (required if scope='conversation')
            document_scope: Scope level ('conversation', 'character', 'global')
            metadata: Additional metadata
            
        Returns:
            Created Document object
            
        Raises:
            ValueError: If file type not supported or scope invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        repo = DocumentRepository(db)
        
        logger.info(f"Uploading document: {file_path.name}")
        
        # Validate file type
        if not self.loader.is_supported(str(file_path)):
            supported = ", ".join(self.loader.get_supported_extensions())
            raise ValueError(f"Unsupported file type. Supported: {supported}")
        
        # Generate unique storage key
        storage_key = self._generate_storage_key(file_path)
        stored_path = self.storage_dir / storage_key
        
        # Load document content
        try:
            loaded_doc = self.loader.load(str(file_path))
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            raise
        
        # Get file info
        file_size = file_path.stat().st_size
        file_type = file_path.suffix.lstrip('.').lower()
        
        # Extract metadata from loaded document
        doc_metadata = metadata or {}
        doc_metadata.update(loaded_doc.metadata)
        
        # Use title from loaded metadata if not provided
        if not title and 'title' in loaded_doc.metadata:
            title = loaded_doc.metadata['title']
        
        # Use author from loaded metadata
        author = loaded_doc.metadata.get('author')
        
        # Create database record with scope
        document = repo.create_document(
            filename=file_path.name,
            storage_key=storage_key,
            file_type=file_type,
            file_size_bytes=file_size,
            page_count=loaded_doc.page_count,
            title=title,
            description=description,
            author=author,
            character_id=character_id,
            conversation_id=conversation_id,
            document_scope=document_scope,
            metadata_json=doc_metadata
        )
        
        try:
            # Copy file to storage
            import shutil
            shutil.copy2(file_path, stored_path)
            logger.info(f"Stored file: {storage_key}")
            
            # Update status to processing
            repo.update_document_status(document.id, "processing")
            
            # Chunk document
            chunks = self.chunker.chunk_document(
                content=loaded_doc.content,
                method=chunk_method,
                metadata={"document_id": document.id}
            )
            
            logger.info(f"Created {len(chunks)} chunks using {chunk_method.value}")
            
            # Prepare chunk data for database and vector store
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for chunk in chunks:
                # Generate unique chunk ID
                chunk_id = f"doc_{document.id}_chunk_{chunk.index}"
                
                # Create chunk in database
                repo.create_chunk(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    chunk_id=chunk_id,
                    content=chunk.content,
                    chunk_method=chunk_method.value,
                    metadata_json=chunk.metadata
                )
                
                # Prepare for vector store with scope metadata
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk.content)
                
                # Build metadata dict, excluding None values (ChromaDB doesn't accept None)
                metadata = {
                    "document_id": document.id,
                    "document_title": document.title or document.filename,
                    "chunk_index": chunk.index,
                    "chunk_method": chunk_method.value,
                    "document_scope": document_scope,  # Scope level: conversation/character/global
                }
                
                # Add optional fields only if not None
                if character_id:
                    metadata["character_id"] = character_id
                if conversation_id:
                    metadata["conversation_id"] = conversation_id
                
                # Merge with chunk-specific metadata (excluding None values)
                for key, value in chunk.metadata.items():
                    if value is not None:
                        metadata[key] = value
                
                chunk_metadatas.append(metadata)
            
            # Add chunks to vector store
            self.vector_store.add_chunks(
                chunk_ids=chunk_ids,
                texts=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            # Update document status to completed
            repo.update_document_status(
                document.id,
                "completed",
                chunk_count=len(chunks)
            )
            
            logger.info(f"Document processing completed: {document.id}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            
            # Update status to failed
            repo.update_document_status(
                document.id,
                "failed",
                error=str(e)
            )
            
            # Clean up stored file
            if stored_path.exists():
                stored_path.unlink()
            
            raise
        
        return document
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """
        Delete document and all associated data.
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        repo = DocumentRepository(db)
        document = repo.get_document(document_id)
        
        if not document:
            logger.warning(f"Document {document_id} not found")
            return False
        
        logger.info(f"Deleting document: {document_id} ({document.filename})")
        
        # Delete from vector store
        deleted_chunks = self.vector_store.delete_chunks_by_document(document_id)
        logger.info(f"Deleted {deleted_chunks} chunks from vector store")
        
        # Delete stored file
        stored_path = self.storage_dir / document.storage_key
        if stored_path.exists():
            stored_path.unlink()
            logger.info(f"Deleted stored file: {document.storage_key}")
        
        # Delete from database (cascades to chunks and logs)
        repo.delete_document(document_id)
        
        return True
    
    def search_documents(
        self,
        db: Session,
        query: str,
        n_results: int = 5,
        document_id: Optional[int] = None,
        character_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.
        
        Args:
            db: Database session
            query: Search query
            n_results: Number of results
            document_id: Optional filter by document
            character_id: Optional filter by character (includes global docs)
            
        Returns:
            List of search results with chunk content and metadata
        """
        repo = DocumentRepository(db)
        
        # Search vector store with character filtering
        chunk_ids, texts, metadatas, distances = self.vector_store.search(
            query=query,
            n_results=n_results,
            document_id=document_id,
            character_id=character_id
        )
        
        # Build results
        results = []
        for i in range(len(chunk_ids)):
            # Get chunk from database for complete info
            chunk = repo.get_chunk_by_id(chunk_ids[i])
            
            result = {
                "chunk_id": chunk_ids[i],
                "content": texts[i],
                "metadata": metadatas[i],
                "relevance_score": 1.0 - distances[i],  # Convert distance to score
                "document_id": metadatas[i].get("document_id"),
                "document_title": metadatas[i].get("document_title"),
                "chunk_index": metadatas[i].get("chunk_index")
            }
            
            # Add database info if available
            if chunk:
                result["page_numbers"] = chunk.page_numbers
                result["start_line"] = chunk.start_line
                result["end_line"] = chunk.end_line
            
            results.append(result)
        
        return results
    
    def get_document_info(self, db: Session, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed document information.
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            Dictionary with document info or None if not found
        """
        repo = DocumentRepository(db)
        document = repo.get_document(document_id)
        
        if not document:
            return None
        
        # Get usage stats
        stats = repo.get_document_usage_stats(document_id)
        
        return {
            "id": document.id,
            "filename": document.filename,
            "title": document.title,
            "description": document.description,
            "file_type": document.file_type,
            "file_size_bytes": document.file_size_bytes,
            "page_count": document.page_count,
            "author": document.author,
            "chunk_count": document.chunk_count,
            "processing_status": document.processing_status,
            "processing_error": document.processing_error,
            "created_at": document.created_at.isoformat(),
            "uploaded_at": document.uploaded_at.isoformat(),
            "last_accessed": document.last_accessed.isoformat() if document.last_accessed else None,
            "metadata": document.metadata_json,
            "usage_stats": stats
        }
    
    def list_documents(
        self,
        db: Session,
        limit: int = 100,
        offset: int = 0,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        character_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> List[Document]:
        """
        List documents with filtering (Phase 9: scope-aware).
        
        Uses scope-aware retrieval to respect document privacy boundaries:
        - conversation scope: Only accessible in specific conversation
        - character scope: Accessible across all conversations with character
        - global scope: Accessible system-wide
        
        Args:
            db: Database session
            limit: Maximum number of results
            offset: Number of results to skip
            file_type: Filter by file type
            status: Filter by processing status
            character_id: Filter by character (required for scope filtering)
            conversation_id: Current conversation context (for conversation-scoped docs)
            
        Returns:
            List of Document objects accessible in the given context
        """
        repo = DocumentRepository(db)
        
        # Use scope-aware retrieval if character_id provided
        if character_id:
            documents = repo.get_accessible_documents(
                character_id=character_id,
                conversation_id=conversation_id,
                include_character_scope=True,
                include_global=True,
                limit=limit,
                offset=offset
            )
            
            # Apply additional filters if specified
            if file_type:
                documents = [d for d in documents if d.file_type == file_type]
            if status:
                documents = [d for d in documents if d.processing_status == status]
                
            return documents
        else:
            # Fallback to basic list (shows only global documents)
            return repo.list_documents(
                limit=limit,
                offset=offset,
                file_type=file_type,
                status=status,
                character_id=None,
                include_global=True
            )
    
    def get_document(self, db: Session, document_id: int) -> Optional[Document]:
        """
        Get a single document by ID.
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            Document object or None
        """
        repo = DocumentRepository(db)
        return repo.get_document(document_id)
    
    def _generate_storage_key(self, file_path: Path) -> str:
        """
        Generate unique storage key for file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Unique storage key
        """
        # Generate hash from file content
        hasher = hashlib.sha256()
        hasher.update(str(datetime.utcnow()).encode())
        hasher.update(file_path.name.encode())
        hasher.update(str(uuid.uuid4()).encode())
        
        hash_prefix = hasher.hexdigest()[:16]
        
        # Preserve file extension
        ext = file_path.suffix
        
        return f"{hash_prefix}{ext}"
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_collection_stats()
