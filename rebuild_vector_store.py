"""
Rebuild ChromaDB vector store from SQL database memories.

This will re-add all memories that have vector_ids back to the vector store.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.embedding_service import EmbeddingService

def main():
    db = SessionLocal()
    
    try:
        # Get all memories with vector_ids
        memories = db.query(Memory).filter(Memory.vector_id.isnot(None)).all()
        
        print(f"Found {len(memories)} memories to rebuild in vector store")
        print()
        
        # Initialize services
        print("Initializing embedding service...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore(Path("data/vector_store"))
        print("✓ Services ready")
        print()
        
        # Group by character
        memories_by_character = {}
        for memory in memories:
            if memory.character_id not in memories_by_character:
                memories_by_character[memory.character_id] = []
            memories_by_character[memory.character_id].append(memory)
        
        # Rebuild for each character
        for char_id, char_memories in memories_by_character.items():
            print(f"Rebuilding {char_id}: {len(char_memories)} memories")
            
            # Prepare data for batch insert
            vector_ids = []
            contents = []
            embeddings = []
            metadatas = []
            
            for i, memory in enumerate(char_memories, 1):
                if i % 10 == 0:
                    print(f"  Processing {i}/{len(char_memories)}...", end='\r')
                
                # Generate embedding
                embedding = embedding_service.embed(memory.content)
                
                vector_ids.append(memory.vector_id)
                contents.append(memory.content)
                embeddings.append(embedding)
                metadatas.append({
                    "type": memory.memory_type.value,
                    "category": memory.category or "",
                    "confidence": memory.confidence or 0.0,
                    "status": memory.status
                })
            
            # Add to vector store
            try:
                success = vector_store.add_memories(
                    character_id=char_id,
                    memory_ids=vector_ids,
                    contents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                if success:
                    print(f"  ✓ Added {len(char_memories)} memories to vector store")
                else:
                    print(f"  ✗ Failed to add memories for {char_id}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print()
        print("✓ Vector store rebuild complete!")
        print()
        print("Run 'python verify_vector_store.py' to verify.")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()
