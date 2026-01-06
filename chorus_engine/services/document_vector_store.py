"""Document vector store for semantic search over document chunks.

This service manages a separate ChromaDB collection for document chunks,
enabling semantic retrieval during conversations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class DocumentVectorStore:
    """Vector store for document chunks with semantic search."""
    
    COLLECTION_NAME = "document_library"
    
    def __init__(self, persist_directory: str = "data/vector_store"):
        """
        Initialize document vector store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create document collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Document chunks for semantic retrieval"}
        )
        
        logger.info(f"DocumentVectorStore initialized: {self.persist_directory}")
        logger.info(f"Collection '{self.COLLECTION_NAME}' has {self.collection.count()} chunks")
    
    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Add document chunks to vector store.
        
        Args:
            chunk_ids: Unique identifiers for chunks
            texts: Chunk text content
            metadatas: Chunk metadata (document_id, page_numbers, etc.)
            embeddings: Pre-computed embeddings (optional, will auto-generate if None)
        """
        if not chunk_ids:
            logger.warning("No chunks to add")
            return
        
        if embeddings:
            self.collection.add(
                ids=chunk_ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            # Let ChromaDB auto-generate embeddings
            self.collection.add(
                ids=chunk_ids,
                documents=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Added {len(chunk_ids)} chunks to vector store")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        document_id: Optional[int] = None,
        character_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            document_id: Filter by specific document ID
            character_id: Filter by character ID (includes global docs with None)
            filter_metadata: Additional metadata filters
            
        Returns:
            Tuple of (chunk_ids, texts, metadatas, distances)
        """
        # Build filter
        where_filter = {}
        if document_id is not None:
            where_filter["document_id"] = document_id
        
        # Character filtering: include character's docs AND global docs (character_id = None)
        # Note: ChromaDB doesn't support OR conditions easily, so we'll fetch more results
        # and filter in Python if character_id is specified
        if character_id is not None:
            # For now, include character_id in metadata but filter after retrieval
            # ChromaDB's where clause doesn't support OR, so we need to handle this
            pass  # Will filter in post-processing
        
        if filter_metadata:
            where_filter.update(filter_metadata)
        
        # Perform search with extra results if character filtering is needed
        search_n = n_results * 3 if character_id is not None else n_results
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=search_n,
            where=where_filter if where_filter else None
        )
        
        # Extract results
        chunk_ids = results["ids"][0] if results["ids"] else []
        texts = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        # Post-filter by character_id if specified
        if character_id is not None and chunk_ids:
            filtered_ids = []
            filtered_texts = []
            filtered_metadatas = []
            filtered_distances = []
            
            for i in range(len(chunk_ids)):
                meta = metadatas[i]
                # Include if matches character OR is global (None)
                if meta.get("character_id") == character_id or meta.get("character_id") is None:
                    filtered_ids.append(chunk_ids[i])
                    filtered_texts.append(texts[i])
                    filtered_metadatas.append(meta)
                    filtered_distances.append(distances[i])
                    
                    # Stop when we have enough results
                    if len(filtered_ids) >= n_results:
                        break
            
            chunk_ids = filtered_ids
            texts = filtered_texts
            metadatas = filtered_metadatas
            distances = filtered_distances
        
        logger.debug(f"Search query: '{query[:50]}...' returned {len(chunk_ids)} results")
        
        return chunk_ids, texts, metadatas, distances
    
    def search_with_scope(
        self,
        query: str,
        character_id: str,
        conversation_id: Optional[str] = None,
        n_results: int = 5,
        document_ids: Optional[List[int]] = None
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for chunks with scope-aware filtering (Phase 9).
        
        This prevents cross-conversation document leakage by respecting scope boundaries.
        
        Args:
            query: Search query text
            character_id: Character accessing documents
            conversation_id: Conversation context (None = all accessible)
            n_results: Number of results to return
            document_ids: Optional list of accessible document IDs (pre-filtered)
            
        Returns:
            Tuple of (chunk_ids, texts, metadatas, distances)
            
        Example:
            # Search within conversation scope
            results = vector_store.search_with_scope(
                query="disparate impact analysis",
                character_id="marcus",
                conversation_id="conv-123",
                n_results=10
            )
        """
        # If document_ids provided (already filtered by scope), use them directly
        if document_ids is not None:
            # Fetch more results for filtering
            search_n = n_results * 3
            
            results = self.collection.query(
                query_texts=[query],
                n_results=search_n
            )
            
            # Filter by document_ids
            chunk_ids = []
            texts = []
            metadatas = []
            distances = []
            
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                if meta.get("document_id") in document_ids:
                    chunk_ids.append(results["ids"][0][i])
                    texts.append(results["documents"][0][i])
                    metadatas.append(meta)
                    distances.append(results["distances"][0][i])
                    
                    if len(chunk_ids) >= n_results:
                        break
            
            logger.debug(
                f"Scope-aware search: '{query[:50]}...' "
                f"returned {len(chunk_ids)} results from {len(document_ids)} accessible documents"
            )
            
            return chunk_ids, texts, metadatas, distances
        
        # Otherwise, use metadata filtering (less precise, kept for backwards compatibility)
        # Build scope filter in metadata
        where_filter = {}
        
        # Note: ChromaDB metadata filtering has limitations with OR conditions
        # For production, it's better to pre-filter document_ids and pass them in
        logger.warning(
            "search_with_scope called without document_ids - "
            "recommend pre-filtering accessible documents for better scope isolation"
        )
        
        # Fallback: search all character's documents
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Get extra for filtering
        )
        
        # Post-filter by character_id
        chunk_ids = []
        texts = []
        metadatas = []
        distances = []
        
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            if meta.get("character_id") == character_id:
                chunk_ids.append(results["ids"][0][i])
                texts.append(results["documents"][0][i])
                metadatas.append(meta)
                distances.append(results["distances"][0][i])
                
                if len(chunk_ids) >= n_results:
                    break
        
        return chunk_ids, texts, metadatas, distances
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Dictionary with chunk data or None if not found
        """
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return None
        
        return {
            "id": results["ids"][0],
            "text": results["documents"][0],
            "metadata": results["metadatas"][0]
        }
    
    def delete_chunks_by_document(self, document_id: int) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=[]  # Only need IDs
        )
        
        chunk_ids = results["ids"]
        
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
        
        return len(chunk_ids)
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete specific chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[chunk_id])
            logger.debug(f"Deleted chunk: {chunk_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete chunk {chunk_id}: {e}")
            return False
    
    def update_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            metadata: New metadata
            
        Returns:
            True if updated, False if not found
        """
        try:
            self.collection.update(
                ids=[chunk_id],
                metadatas=[metadata]
            )
            logger.debug(f"Updated metadata for chunk: {chunk_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update chunk {chunk_id}: {e}")
            return False
    
    def get_chunks_by_document(
        self,
        document_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk dictionaries
        """
        results = self.collection.get(
            where={"document_id": document_id},
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        
        return chunks
    
    def count_chunks(self, document_id: Optional[int] = None) -> int:
        """
        Count chunks in vector store.
        
        Args:
            document_id: Optional filter by document ID
            
        Returns:
            Number of chunks
        """
        if document_id is not None:
            results = self.collection.get(
                where={"document_id": document_id},
                include=[]
            )
            return len(results["ids"])
        else:
            return self.collection.count()
    
    def clear_collection(self) -> None:
        """
        Clear all chunks from collection.
        
        WARNING: This deletes all document chunks!
        """
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Document chunks for semantic retrieval"}
        )
        logger.warning(f"Cleared collection '{self.COLLECTION_NAME}'")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection stats
        """
        total_chunks = self.collection.count()
        
        # Get unique document count
        results = self.collection.get(
            include=["metadatas"]
        )
        
        document_ids = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
        
        return {
            "collection_name": self.COLLECTION_NAME,
            "total_chunks": total_chunks,
            "unique_documents": len(document_ids),
            "persist_directory": str(self.persist_directory)
        }
