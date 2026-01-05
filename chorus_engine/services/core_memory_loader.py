"""
Core Memory Loader Service

Loads immutable character backstory memories from character YAML files
and stores them in the database with vector embeddings.

Core memories:
- Are loaded once on character initialization
- Cannot be modified by users
- Have highest priority in retrieval
- Are character-specific (not conversation-specific)
- Are stored in both SQL database and vector store
"""

import logging
from typing import List, Optional
from pathlib import Path
from sqlalchemy.orm import Session
import uuid

from chorus_engine.config.loader import ConfigLoader
from chorus_engine.models.conversation import Memory, MemoryType
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class CoreMemoryLoader:
    """
    Loads and manages core memories for characters.
    
    Core memories are immutable character backstory loaded from YAML.
    They are:
    - Stored in SQL database with character_id
    - Embedded in vector store for semantic search
    - Given highest priority (90-100) for retrieval
    - Tagged for easy filtering
    """
    
    def __init__(
        self,
        db: Session,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingService] = None,
        persist_directory: Path = Path("data/vector_store")
    ):
        """
        Initialize core memory loader.
        
        Args:
            db: Database session
            vector_store: Optional vector store (created if not provided)
            embedder: Optional embedding service (created if not provided)
            persist_directory: Where to store vector database
        """
        self.db = db
        self.vector_store = vector_store or VectorStore(persist_directory=persist_directory)
        self.embedder = embedder or EmbeddingService()
        self.config_loader = ConfigLoader()
    
    def load_character_core_memories(self, character_id: str) -> int:
        """
        Load core memories for a character from their YAML configuration.
        
        This should be called once when a character is first initialized.
        If core memories already exist, they are skipped (idempotent).
        
        Args:
            character_id: ID of the character to load memories for
            
        Returns:
            Number of new core memories loaded
            
        Raises:
            ValueError: If character not found or has no core memories
        """
        logger.info(f"Loading core memories for character: {character_id}")
        
        # Load character config
        character = self.config_loader.load_character(character_id)
        if not character:
            raise ValueError(f"Character not found: {character_id}")
        
        core_memories = character.core_memories
        if not core_memories:
            logger.warning(f"Character {character_id} has no core memories defined")
            return 0
        
        # Check if core memories already exist
        existing_count = self.db.query(Memory).filter(
            Memory.character_id == character_id,
            Memory.memory_type == MemoryType.CORE
        ).count()
        
        if existing_count > 0:
            logger.info(f"Core memories already loaded for {character_id} ({existing_count} memories)")
            return 0
        
        logger.info(f"Found {len(core_memories)} core memories to load")
        
        # Prepare memory contents and metadata
        memory_contents = []
        memory_metadata_list = []
        
        # Priority mapping: low=60, medium=80, high=95
        priority_map = {"low": 60, "medium": 80, "high": 95}
        
        for idx, core_mem in enumerate(core_memories):
            memory_contents.append(core_mem.content)
            priority = priority_map.get(core_mem.embedding_priority, 80)
            
            # Convert tags list to comma-separated string for ChromaDB
            tags_str = ",".join(core_mem.tags) if core_mem.tags else ""
            
            memory_metadata_list.append({
                "tags": tags_str,  # ChromaDB only accepts string/int/float/bool
                "priority": priority,
                "source": "character_yaml",
                "index": idx
            })
        
        # Generate embeddings (batch for efficiency)
        logger.info("Generating embeddings for core memories...")
        embeddings = self.embedder.embed_batch(memory_contents)
        
        # Generate vector IDs (these will be stored in both SQL and vector store)
        vector_ids = [str(uuid.uuid4()) for _ in memory_contents]
        
        # Add to vector store
        logger.info("Adding core memories to vector store...")
        success = self.vector_store.add_memories(
            character_id=character_id,
            memory_ids=vector_ids,
            contents=memory_contents,
            embeddings=embeddings,
            metadatas=memory_metadata_list
        )
        
        if not success:
            raise RuntimeError(f"Failed to add memories to vector store for {character_id}")
        
        # Store in database
        logger.info("Storing core memories in database...")
        new_memories = []
        
        for idx, (content, vector_id, metadata, core_mem) in enumerate(zip(
            memory_contents, vector_ids, memory_metadata_list, core_memories
        )):
            memory = Memory(
                character_id=character_id,
                memory_type=MemoryType.CORE,
                content=content,
                vector_id=vector_id,
                embedding_model=self.embedder.model_name,
                priority=metadata["priority"],
                tags=core_mem.tags,  # Store original list in SQL
                meta_data={"source": "character_yaml", "yaml_index": idx}
            )
            new_memories.append(memory)
        
        self.db.add_all(new_memories)
        self.db.commit()
        
        logger.info(f"✓ Loaded {len(new_memories)} core memories for {character_id}")
        return len(new_memories)
    
    def get_core_memories(self, character_id: str) -> List[Memory]:
        """
        Get all core memories for a character.
        
        Args:
            character_id: ID of the character
            
        Returns:
            List of core Memory objects, ordered by priority (highest first)
        """
        return (
            self.db.query(Memory)
            .filter(
                Memory.character_id == character_id,
                Memory.memory_type == MemoryType.CORE
            )
            .order_by(Memory.priority.desc())
            .all()
        )
    
    def delete_core_memories(self, character_id: str) -> int:
        """
        Delete all core memories for a character.
        
        This removes memories from both the database and vector store.
        Use with caution - typically only needed for testing or character reset.
        
        Args:
            character_id: ID of the character
            
        Returns:
            Number of memories deleted
        """
        logger.warning(f"Deleting core memories for character: {character_id}")
        
        # Get all core memories
        memories = self.get_core_memories(character_id)
        
        if not memories:
            return 0
        
        # Extract vector IDs for deletion
        vector_ids = [m.vector_id for m in memories if m.vector_id]
        
        # Delete from vector store
        if vector_ids:
            self.vector_store.delete_memories(character_id, vector_ids)
            logger.info(f"Deleted {len(vector_ids)} vectors from vector store")
        
        # Delete from database
        count = (
            self.db.query(Memory)
            .filter(
                Memory.character_id == character_id,
                Memory.memory_type == MemoryType.CORE
            )
            .delete()
        )
        self.db.commit()
        
        logger.info(f"✓ Deleted {count} core memories for {character_id}")
        return count
    
    def reload_core_memories(self, character_id: str) -> int:
        """
        Reload core memories from character YAML.
        
        Deletes existing core memories and loads fresh from config.
        Useful after character YAML updates.
        
        Args:
            character_id: ID of the character
            
        Returns:
            Number of new core memories loaded
        """
        logger.info(f"Reloading core memories for character: {character_id}")
        self.delete_core_memories(character_id)
        return self.load_character_core_memories(character_id)
