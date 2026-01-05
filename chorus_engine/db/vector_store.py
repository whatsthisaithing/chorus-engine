"""Vector database wrapper for semantic memory storage."""

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import os

# Disable ChromaDB telemetry to avoid noisy warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper for ChromaDB vector database."""
    
    def __init__(self, persist_directory: Path):
        """
        Initialize vector store with persistent storage.
        
        Args:
            persist_directory: Path to store ChromaDB data
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"VectorStore initialized at {persist_directory}")
    
    def get_or_create_collection(
        self, 
        character_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get or create a collection for a character.
        
        Args:
            character_id: Unique character identifier
            metadata: Optional metadata for the collection
            
        Returns:
            ChromaDB Collection object
        """
        collection_name = f"character_{character_id}"
        
        # Default metadata
        if metadata is None:
            metadata = {
                "hnsw:space": "cosine",  # Use cosine similarity
                "character_id": character_id
            }
        
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata
            )
            logger.debug(f"Collection '{collection_name}' ready (count: {collection.count()})")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection '{collection_name}': {e}")
            raise
    
    def get_collection(self, character_id: str) -> Optional[Any]:
        """
        Get existing collection for a character.
        
        Args:
            character_id: Unique character identifier
            
        Returns:
            ChromaDB Collection object or None if not found
        """
        collection_name = f"character_{character_id}"
        
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection
        except Exception:
            logger.debug(f"Collection '{collection_name}' not found")
            return None
    
    def delete_collection(self, character_id: str) -> bool:
        """
        Delete a character's collection.
        
        Args:
            character_id: Unique character identifier
            
        Returns:
            True if deleted, False if not found
        """
        collection_name = f"character_{character_id}"
        
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector store.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def add_memories(
        self,
        character_id: str,
        memory_ids: List[str],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add memories to a character's collection.
        
        Args:
            character_id: Character identifier
            memory_ids: List of unique memory IDs
            contents: List of memory content strings
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            
        Returns:
            True if successful
        """
        collection = self.get_or_create_collection(character_id)
        
        try:
            collection.add(
                ids=memory_ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.debug(f"Added {len(memory_ids)} memories to '{character_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add memories: {e}")
            return False
    
    def query_memories(
        self,
        character_id: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query memories using semantic search.
        
        Args:
            character_id: Character identifier
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document filter conditions
            
        Returns:
            Query results with ids, distances, documents, metadatas
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            logger.warning(f"No collection found for character '{character_id}'")
            return {
                'ids': [[]],
                'distances': [[]],
                'documents': [[]],
                'metadatas': [[]]
            }
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query memories: {e}")
            return {
                'ids': [[]],
                'distances': [[]],
                'documents': [[]],
                'metadatas': [[]]
            }
    
    def delete_memories(
        self,
        character_id: str,
        memory_ids: List[str]
    ) -> bool:
        """
        Delete specific memories from a collection.
        
        Args:
            character_id: Character identifier
            memory_ids: List of memory IDs to delete
            
        Returns:
            True if successful
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            logger.warning(f"No collection found for character '{character_id}'")
            return False
        
        try:
            collection.delete(ids=memory_ids)
            logger.debug(f"Deleted {len(memory_ids)} memories from '{character_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")
            return False
    
    def get_collection_count(self, character_id: str) -> int:
        """
        Get number of memories in a character's collection.
        
        Args:
            character_id: Character identifier
            
        Returns:
            Number of memories in collection
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            return 0
        
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0
    
    def reset(self) -> None:
        """Reset the vector store (delete all collections). Use with caution!"""
        try:
            self.client.reset()
            logger.warning("Vector store reset - all collections deleted")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
