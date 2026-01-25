"""
Vector Regeneration Service

Rebuilds vector embeddings for characters when ChromaDB data is corrupted or missing.
"""

import logging
from typing import Dict, Any, Generator
from pathlib import Path
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Memory
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorRegenerationService:
    """
    Service for regenerating vector embeddings from existing SQL memories.
    
    Use cases:
    - ChromaDB corruption recovery
    - After database migration
    - Restoring from backups with missing vectors
    """
    
    def __init__(
        self,
        db: Session,
        vector_store: VectorStore,
        embedder: EmbeddingService
    ):
        """
        Initialize vector regeneration service.
        
        Args:
            db: Database session
            vector_store: Vector store instance
            embedder: Embedding service
        """
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
    
    def check_vector_health(self, character_id: str) -> Dict[str, Any]:
        """
        Check if character has missing or corrupted vectors.
        
        Args:
            character_id: Character ID to check
            
        Returns:
            Dict with:
                - memory_count: Total memories in SQL
                - vector_count: Total vectors in ChromaDB
                - missing_vectors: Number of memories without vectors
                - needs_regeneration: Boolean flag
        """
        # Count memories in SQL
        memory_count = self.db.query(Memory).filter(
            Memory.character_id == character_id
        ).count()
        
        # Count vectors in ChromaDB
        try:
            collection = self.vector_store.get_collection(character_id)
            vector_count = collection.count() if collection else 0
        except Exception as e:
            logger.warning(f"Could not access collection for {character_id}: {e}")
            vector_count = 0
        
        missing = memory_count - vector_count
        
        return {
            'memory_count': memory_count,
            'vector_count': vector_count,
            'missing_vectors': max(0, missing),
            'needs_regeneration': missing > 0
        }
    
    def regenerate_vectors(
        self,
        character_id: str,
        progress_callback: Any = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Regenerate all vector embeddings for a character.
        
        This is a generator that yields progress updates.
        
        Args:
            character_id: Character ID
            progress_callback: Optional callback for progress updates
            
        Yields:
            Progress dictionaries with:
                - step: Current step description
                - current: Current item number
                - total: Total items
                - percent: Completion percentage
                - status: 'running', 'success', or 'error'
                - message: Status message
        """
        try:
            # Step 1: Fetch all memories
            yield {
                'step': 'fetch',
                'current': 0,
                'total': 0,
                'percent': 0,
                'status': 'running',
                'message': 'Fetching memories from database...'
            }
            
            memories = self.db.query(Memory).filter(
                Memory.character_id == character_id
            ).all()
            
            total = len(memories)
            
            if total == 0:
                yield {
                    'step': 'complete',
                    'current': 0,
                    'total': 0,
                    'percent': 100,
                    'status': 'success',
                    'message': 'No memories found - nothing to regenerate'
                }
                return
            
            yield {
                'step': 'fetch',
                'current': total,
                'total': total,
                'percent': 10,
                'status': 'running',
                'message': f'Found {total} memories'
            }
            
            # Step 2: Delete existing collection (full wipe)
            yield {
                'step': 'delete',
                'current': 0,
                'total': total,
                'percent': 15,
                'status': 'running',
                'message': 'Deleting existing vectors...'
            }
            
            try:
                collection = self.vector_store.get_collection(character_id)
                if collection:
                    # Delete the collection
                    self.vector_store.client.delete_collection(
                        name=f"character_{character_id}"
                    )
                    logger.info(f"Deleted existing collection for {character_id}")
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
            
            # Create fresh collection
            collection = self.vector_store.get_or_create_collection(character_id)
            
            yield {
                'step': 'delete',
                'current': 0,
                'total': total,
                'percent': 20,
                'status': 'running',
                'message': 'Created fresh collection'
            }
            
            # Step 3: Generate embeddings in batches
            batch_size = 10
            for i in range(0, total, batch_size):
                batch = memories[i:i + batch_size]
                
                # Prepare batch data
                ids = []
                embeddings = []
                documents = []
                metadatas = []
                
                for memory in batch:
                    # Generate embedding
                    embedding = self.embedder.embed(memory.content)
                    
                    # Use memory ID as vector ID
                    memory_id = str(memory.id)
                    
                    ids.append(memory_id)
                    embeddings.append(embedding)
                    documents.append(memory.content)
                    metadatas.append({
                        'type': memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                        'character_id': character_id,
                        'confidence': memory.confidence or 0.8,
                        'status': memory.status or 'approved',
                        'category': memory.category or ''
                    })
                    
                    # Update memory record with vector_id
                    memory.vector_id = memory_id
                
                # Add batch to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                # Commit SQL updates
                self.db.commit()
                
                current = min(i + batch_size, total)
                percent = 20 + int((current / total) * 75)
                
                yield {
                    'step': 'generate',
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'status': 'running',
                    'message': f'Generated embeddings for {current}/{total} memories'
                }
            
            # Step 4: Complete
            yield {
                'step': 'complete',
                'current': total,
                'total': total,
                'percent': 100,
                'status': 'success',
                'message': f'Successfully regenerated {total} vector embeddings'
            }
            
            logger.info(f"Regenerated {total} vectors for character {character_id}")
            
        except Exception as e:
            logger.error(f"Failed to regenerate vectors for {character_id}: {e}")
            yield {
                'step': 'error',
                'current': 0,
                'total': 0,
                'percent': 0,
                'status': 'error',
                'message': f'Error: {str(e)}'
            }
