"""Vector database for conversation summary storage and semantic search.

This module provides a separate ChromaDB collection for conversation summaries,
enabling semantic search across past conversations. Each character has their
own collection to maintain isolation.

Unlike the memory vector store (which stores individual memories), this stores
one summary per conversation with rich metadata (themes, tone, participants, etc.).
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from chorus_engine.db.chroma_config_fix import normalize_collection_configs

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


class ConversationSummaryVectorStore:
    """
    Vector store for conversation summaries with semantic search.
    
    Maintains one collection per character: `conversation_summaries_{character_id}`
    Each conversation has at most one summary entry (upserted by conversation_id).
    
    Metadata schema:
    - conversation_id: str (also used as vector ID)
    - character_id: str
    - title: str
    - created_at: str (ISO timestamp)
    - updated_at: str (ISO timestamp)
    - message_count: int
    - key_topics: str (JSON array)
    - tone: str
    - emotional_arc: str
    - participants: str (JSON array)
    - open_questions: str (JSON array)
    - source: str (web, discord, etc.)
    - analyzed_at: str (ISO timestamp)
    - manual_analysis: bool
    """
    
    def __init__(self, persist_directory: Path):
        """
        Initialize conversation summary vector store.
        
        Args:
            persist_directory: Path to store ChromaDB data (shared with other collections)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        normalize_collection_configs(self.persist_directory)
        
        # Initialize ChromaDB client with persistence
        # Uses same directory as memory vector store - collections are separate
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"ConversationSummaryVectorStore initialized at {persist_directory}")
    
    def _collection_name(self, character_id: str) -> str:
        """Generate collection name for a character."""
        return f"conversation_summaries_{character_id}"
    
    def get_or_create_collection(
        self, 
        character_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get or create a conversation summary collection for a character.
        
        Args:
            character_id: Unique character identifier
            metadata: Optional metadata for the collection
            
        Returns:
            ChromaDB Collection object
        """
        collection_name = self._collection_name(character_id)
        
        # Default metadata
        if metadata is None:
            metadata = {
                "hnsw:space": "cosine",  # Use cosine similarity
                "character_id": character_id,
                "type": "conversation_summaries"
            }
        
        try:
            # Prefer get-then-create; get_or_create can fail on some persisted config variants.
            try:
                collection = self.client.get_collection(name=collection_name)
                return collection
            except Exception:
                pass
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata
            )
            logger.debug(f"Collection '{collection_name}' created (count: {collection.count()})")
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
        collection_name = self._collection_name(character_id)
        
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection
        except Exception:
            logger.debug(f"Collection '{collection_name}' not found")
            return None
    
    def add_summary(
        self,
        character_id: str,
        conversation_id: str,
        summary_text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add or update a conversation summary in the vector store.
        
        Uses upsert semantics - if a summary for this conversation_id exists,
        it will be replaced (for re-analysis scenarios).
        
        Args:
            character_id: Character identifier (determines collection)
            conversation_id: Unique conversation ID (used as vector ID)
            summary_text: The summary text to embed
            embedding: Pre-computed embedding vector
            metadata: Metadata dict with themes, tone, etc.
            
        Returns:
            True if successful
        """
        collection = self.get_or_create_collection(character_id)
        
        # Ensure required metadata fields and serialize lists
        processed_metadata = self._process_metadata_for_storage(metadata)
        
        try:
            # Use upsert to handle both insert and update cases
            collection.upsert(
                ids=[conversation_id],
                documents=[summary_text],
                embeddings=[embedding],
                metadatas=[processed_metadata]
            )
            logger.debug(f"Upserted summary for conversation '{conversation_id[:8]}...' in '{character_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add/update summary: {e}")
            return False
    
    def search_conversations(
        self,
        character_id: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search conversation summaries using semantic search.
        
        Args:
            character_id: Character identifier
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter conditions
            
        Returns:
            Query results with ids, distances, documents, metadatas
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            logger.debug(f"No summary collection found for character '{character_id}'")
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
                where=where
            )
            
            # Deserialize JSON fields in metadata
            if results.get('metadatas') and results['metadatas'][0]:
                results['metadatas'][0] = [
                    self._process_metadata_from_storage(meta)
                    for meta in results['metadatas'][0]
                ]
            
            return results
        except Exception as e:
            logger.error(
                f"[VECTOR_HEALTH][SUMMARY_QUERY_ERROR] Failed to search conversation summaries "
                f"for '{character_id}': {e}"
            )
            return {
                'ids': [[]],
                'distances': [[]],
                'documents': [[]],
                'metadatas': [[]]
            }
    
    def delete_summary(
        self,
        character_id: str,
        conversation_id: str
    ) -> bool:
        """
        Delete a conversation summary from the vector store.
        
        Called when a conversation is deleted.
        
        Args:
            character_id: Character identifier
            conversation_id: Conversation ID to delete
            
        Returns:
            True if successful (or if not found - idempotent)
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            # No collection means no summary to delete - that's fine
            logger.debug(f"No summary collection for character '{character_id}', nothing to delete")
            return True
        
        try:
            collection.delete(ids=[conversation_id])
            logger.debug(f"Deleted summary for conversation '{conversation_id[:8]}...' from '{character_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete summary: {e}")
            return False
    
    def get_summary(
        self,
        character_id: str,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific conversation summary by ID.
        
        Args:
            character_id: Character identifier
            conversation_id: Conversation ID to fetch
            
        Returns:
            Dict with summary text and metadata, or None if not found
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            return None
        
        try:
            results = collection.get(
                ids=[conversation_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Check if we got any results - ids is always a list
            if not results.get('ids') or len(results['ids']) == 0:
                return None
            
            # Get first metadata (results are in parallel lists)
            metadata = {}
            if results.get('metadatas') and len(results['metadatas']) > 0:
                metadata = results['metadatas'][0]
            
            # Get first document
            summary_text = ''
            if results.get('documents') and len(results['documents']) > 0:
                summary_text = results['documents'][0]
            
            # Get first embedding (avoid numpy array truth value issue)
            embedding = None
            if 'embeddings' in results and results['embeddings'] is not None:
                embeddings_list = results['embeddings']
                if len(embeddings_list) > 0:
                    embedding = embeddings_list[0]
            
            return {
                'conversation_id': conversation_id,
                'summary': summary_text,
                'metadata': self._process_metadata_from_storage(metadata),
                'embedding': embedding
            }
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return None
    
    def get_collection_count(self, character_id: str) -> int:
        """
        Get number of summaries in a character's collection.
        
        Args:
            character_id: Character identifier
            
        Returns:
            Number of summaries in collection
        """
        collection = self.get_collection(character_id)
        
        if collection is None:
            return 0
        
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0
    
    def list_collections(self) -> List[str]:
        """
        List all conversation summary collections.
        
        Returns:
            List of collection names (filtered to summary collections only)
        """
        try:
            collections = self.client.list_collections()
            return [
                col.name for col in collections 
                if col.name.startswith("conversation_summaries_")
            ]
        except Exception as e:
            if "_type" in str(e):
                try:
                    db_path = self.persist_directory / "chroma.sqlite3"
                    conn = sqlite3.connect(str(db_path))
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT name FROM collections "
                        "WHERE name LIKE 'conversation_summaries_%' ORDER BY name"
                    )
                    names = [row[0] for row in cur.fetchall()]
                    conn.close()
                    return names
                except Exception as fallback_error:
                    logger.error(f"Failed fallback summary list_collections: {fallback_error}")
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def delete_collection(self, character_id: str) -> bool:
        """
        Delete a character's entire summary collection.
        
        Use with caution - typically only for character deletion.
        
        Args:
            character_id: Character identifier
            
        Returns:
            True if deleted, False if not found or error
        """
        collection_name = self._collection_name(character_id)
        
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted summary collection '{collection_name}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    def _process_metadata_for_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata for ChromaDB storage.
        
        ChromaDB only supports str, int, float, bool values.
        Lists must be JSON-serialized to strings.
        
        Args:
            metadata: Raw metadata dict
            
        Returns:
            Processed metadata with lists serialized
        """
        processed = {}
        
        for key, value in metadata.items():
            if isinstance(value, list):
                # Serialize lists to JSON strings
                processed[key] = json.dumps(value)
            elif isinstance(value, bool):
                # ChromaDB handles bools fine
                processed[key] = value
            elif value is None:
                # Store None as empty string
                processed[key] = ""
            else:
                # Keep other types as-is (str, int, float)
                processed[key] = value
        
        return processed
    
    def _process_metadata_from_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata retrieved from ChromaDB storage.
        
        Deserializes JSON strings back to lists.
        
        Args:
            metadata: Stored metadata dict
            
        Returns:
            Processed metadata with lists deserialized
        """
        processed = {}
        
        # Fields known to be JSON-serialized lists
        list_fields = {'themes', 'key_topics', 'participants', 'open_questions'}
        
        for key, value in metadata.items():
            if key in list_fields and isinstance(value, str):
                try:
                    processed[key] = json.loads(value) if value else []
                except json.JSONDecodeError:
                    processed[key] = []
            elif value == "":
                # Convert empty strings back to None for cleaner API
                processed[key] = None
            else:
                processed[key] = value
        
        return processed
