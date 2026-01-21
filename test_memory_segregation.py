"""Test script to verify memory segregation between Discord and Web UI.

Tests that:
1. Discord memories are only retrieved in Discord conversations
2. Web memories are only retrieved in Web conversations
3. No cross-contamination between platforms
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, '.')

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory, MemoryType
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.services.memory_retrieval import MemoryRetrievalService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.db.vector_store import VectorStore


def create_test_memories(db, character_id="nova"):
    """Create test memories with different sources."""
    print("\n" + "="*80)
    print("CREATING TEST MEMORIES")
    print("="*80)
    
    memory_repo = MemoryRepository(db)
    
    # Create Discord memories
    discord_memories = [
        ("fitzycodesthings enjoys sci-fi and fantasy books", "discord"),
        ("alex prefers Python over JavaScript", "discord"),
        ("sarah loves hiking and outdoor activities", "discord"),
    ]
    
    # Create Web memories
    web_memories = [
        ("User prefers dark mode interfaces", "web"),
        ("User is interested in machine learning", "web"),
        ("User works as a software engineer", "web"),
    ]
    
    created_discord = []
    created_web = []
    
    print("\nCreating Discord memories:")
    for content, source in discord_memories:
        memory = memory_repo.create(
            content=content,
            character_id=character_id,
            memory_type=MemoryType.IMPLICIT,
            confidence=0.95,
            category="personal_info",
            status="auto_approved",
            source=source
        )
        created_discord.append(memory)
        print(f"  ✓ Created (ID: {memory.id[:8]}...): {content}")
    
    print("\nCreating Web UI memories:")
    for content, source in web_memories:
        memory = memory_repo.create(
            content=content,
            character_id=character_id,
            memory_type=MemoryType.IMPLICIT,
            confidence=0.95,
            category="personal_info",
            status="auto_approved",
            source=source
        )
        created_web.append(memory)
        print(f"  ✓ Created (ID: {memory.id[:8]}...): {content}")
    
    db.commit()
    return created_discord, created_web


def add_memories_to_vector_store(memories, vector_store, embedding_service, character_id):
    """Add memories to vector store for retrieval."""
    print("\n" + "="*80)
    print("ADDING MEMORIES TO VECTOR STORE")
    print("="*80)
    
    # Batch process all memories
    memory_ids = []
    contents = []
    embeddings = []
    metadatas = []
    
    for memory in memories:
        # Generate embedding
        embedding = embedding_service.embed(memory.content)
        
        memory_ids.append(memory.id)
        contents.append(memory.content)
        embeddings.append(embedding)
        metadatas.append({"source": memory.source, "memory_type": memory.memory_type.value})
        
        print(f"  ✓ Prepared: {memory.content[:60]}...")
    
    # Add all memories to vector store in batch
    vector_store.add_memories(
        character_id=character_id,
        memory_ids=memory_ids,
        contents=contents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"\n✓ Added {len(memories)} memories to vector store")


def test_memory_retrieval(db, character_id="nova"):
    """Test memory retrieval with different conversation sources."""
    print("\n" + "="*80)
    print("TESTING MEMORY RETRIEVAL")
    print("="*80)
    
    # Initialize services
    vector_store = VectorStore(persist_directory=Path("data/vector_store"))
    embedding_service = EmbeddingService()
    memory_service = MemoryRetrievalService(db, vector_store, embedding_service)
    
    # Test query about books (should match Discord memory)
    test_query = "What kind of books does the user like?"
    
    print(f"\nTest Query: \"{test_query}\"")
    
    # Test 1: Retrieve from Discord context
    print("\n--- Test 1: Discord Context ---")
    discord_memories = memory_service.retrieve_memories(
        query=test_query,
        character_id=character_id,
        conversation_source="discord",
        max_memories=10
    )
    
    print(f"Retrieved {len(discord_memories)} memories from Discord context:")
    for mem in discord_memories:
        print(f"  • [{mem.memory.source}] {mem.memory.content} (similarity: {mem.similarity:.3f})")
    
    # Verify no web memories leaked
    web_leaked = [m for m in discord_memories if m.memory.source == "web"]
    if web_leaked:
        print(f"  ❌ ERROR: {len(web_leaked)} web memories leaked into Discord!")
    else:
        print("  ✓ No web memories leaked into Discord context")
    
    # Test 2: Retrieve from Web context
    print("\n--- Test 2: Web UI Context ---")
    web_memories = memory_service.retrieve_memories(
        query=test_query,
        character_id=character_id,
        conversation_source="web",
        max_memories=10
    )
    
    print(f"Retrieved {len(web_memories)} memories from Web context:")
    for mem in web_memories:
        print(f"  • [{mem.memory.source}] {mem.memory.content} (similarity: {mem.similarity:.3f})")
    
    # Verify no discord memories leaked
    discord_leaked = [m for m in web_memories if m.memory.source == "discord"]
    if discord_leaked:
        print(f"  ❌ ERROR: {len(discord_leaked)} Discord memories leaked into Web!")
    else:
        print("  ✓ No Discord memories leaked into Web context")
    
    # Test 3: Retrieve without source filter (should get both)
    print("\n--- Test 3: No Source Filter (should get both) ---")
    all_memories = memory_service.retrieve_memories(
        query=test_query,
        character_id=character_id,
        conversation_source=None,
        max_memories=10
    )
    
    print(f"Retrieved {len(all_memories)} memories without source filter:")
    discord_count = len([m for m in all_memories if m.memory.source == "discord"])
    web_count = len([m for m in all_memories if m.memory.source == "web"])
    print(f"  • Discord memories: {discord_count}")
    print(f"  • Web memories: {web_count}")
    
    return discord_memories, web_memories, all_memories


def cleanup_test_memories(db, discord_memories, web_memories):
    """Clean up test memories."""
    print("\n" + "="*80)
    print("CLEANING UP TEST MEMORIES")
    print("="*80)
    
    all_test_memories = discord_memories + web_memories
    
    for memory in all_test_memories:
        db.delete(memory)
        print(f"  ✓ Deleted: {memory.content[:60]}...")
    
    db.commit()
    print(f"\nDeleted {len(all_test_memories)} test memories")


def main():
    """Run memory segregation tests."""
    print("\n" + "="*80)
    print("MEMORY SEGREGATION TEST")
    print("="*80)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    character_id = "nova"
    db = SessionLocal()
    
    try:
        # Step 1: Create test memories
        discord_memories, web_memories = create_test_memories(db, character_id)
        
        # Step 2: Add to vector store
        vector_store = VectorStore(persist_directory=Path("data/vector_store"))
        embedding_service = EmbeddingService()
        
        all_memories = discord_memories + web_memories
        add_memories_to_vector_store(all_memories, vector_store, embedding_service, character_id)
        
        # Step 3: Test retrieval
        discord_results, web_results, all_results = test_memory_retrieval(db, character_id)
        
        # Step 4: Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        discord_clean = all(m.memory.source == "discord" for m in discord_results)
        web_clean = all(m.memory.source == "web" for m in web_results)
        
        if discord_clean and web_clean:
            print("✅ PASS: Memory segregation working correctly!")
            print("   - Discord memories isolated to Discord context")
            print("   - Web memories isolated to Web context")
            print("   - No cross-contamination detected")
        else:
            print("❌ FAIL: Memory segregation not working!")
            if not discord_clean:
                print("   - Web memories leaked into Discord context")
            if not web_clean:
                print("   - Discord memories leaked into Web context")
        
        # Step 5: Cleanup
        cleanup_test_memories(db, discord_memories, web_memories)
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()
