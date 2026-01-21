"""
Verify vector store contents against SQL database.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory
from chorus_engine.db.vector_store import VectorStore
from sqlalchemy import func

def main():
    db = SessionLocal()
    
    try:
        # Count memories in SQL database with vector_ids
        total_memories = db.query(Memory).count()
        memories_with_vector_id = db.query(Memory).filter(Memory.vector_id.isnot(None)).count()
        memories_without_vector_id = total_memories - memories_with_vector_id
        
        print("SQL Database:")
        print(f"  Total memories: {total_memories}")
        print(f"  With vector_id: {memories_with_vector_id}")
        print(f"  Without vector_id: {memories_without_vector_id}")
        print()
        
        # Count by character
        from sqlalchemy import case
        character_counts = db.query(
            Memory.character_id,
            func.count(Memory.id).label('total'),
            func.sum(case((Memory.vector_id.isnot(None), 1), else_=0)).label('with_vector_id')
        ).group_by(Memory.character_id).all()
        
        print("By Character:")
        for char_id, total, with_vid in character_counts:
            print(f"  {char_id}: {with_vid}/{total} have vector_ids")
        print()
        
        # Check vector store
        print("Checking ChromaDB Vector Store...")
        vector_store = VectorStore(Path("data/vector_store"))
        
        # Get all unique character_ids from memories
        character_ids = db.query(Memory.character_id).distinct().all()
        character_ids = [c[0] for c in character_ids]
        
        vector_store_total = 0
        for char_id in character_ids:
            try:
                collection = vector_store.get_or_create_collection(char_id)
                count = collection.count()
                vector_store_total += count
                print(f"  {char_id}: {count} memories in vector store")
            except Exception as e:
                print(f"  {char_id}: ERROR - {e}")
        
        print()
        print(f"Vector Store Total: {vector_store_total}")
        print()
        
        if vector_store_total == memories_with_vector_id:
            print("✓ Vector store matches SQL database (all memories with vector_ids are present)")
        elif vector_store_total < memories_with_vector_id:
            print(f"⚠ Vector store missing {memories_with_vector_id - vector_store_total} memories")
        else:
            print(f"⚠ Vector store has {vector_store_total - memories_with_vector_id} extra memories")
        
        if memories_without_vector_id > 0:
            print(f"\n⚠ Warning: {memories_without_vector_id} memories in SQL don't have vector_ids")
            print("  These memories won't be retrievable until they're re-added to the vector store.")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()
