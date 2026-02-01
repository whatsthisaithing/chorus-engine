"""
Test for ConversationSummaryVectorStore.

This script tests the basic functionality of the conversation summary
vector store including:
- Collection creation
- Summary insertion and upsert
- Semantic search
- Summary retrieval and deletion

Usage:
    python testing/test_conversation_summary_vector_store.py

Note: This creates a test collection that will be cleaned up at the end.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.embedding_service import EmbeddingService


def test_vector_store():
    """Run tests for ConversationSummaryVectorStore."""
    
    print("=" * 70)
    print("CONVERSATION SUMMARY VECTOR STORE TESTS")
    print("=" * 70)
    print()
    
    # Create temporary directory for test
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    print()
    
    try:
        # Initialize services
        print("Initializing services...")
        vector_store = ConversationSummaryVectorStore(temp_dir)
        embedding_service = EmbeddingService()
        print("✓ Services initialized")
        print()
        
        # Test character
        character_id = "test_character"
        
        # Test 1: Create collection
        print("Test 1: Create collection")
        collection = vector_store.get_or_create_collection(character_id)
        assert collection is not None
        print(f"✓ Collection created: conversation_summaries_{character_id}")
        print()
        
        # Test 2: Add summary
        print("Test 2: Add summary")
        conversation_id = "test_conv_001"
        summary_text = "We discussed the user's preference for vegetarian food and their interest in Italian cuisine. They mentioned they love pasta and want to learn how to make homemade gnocchi."
        embedding = embedding_service.embed(summary_text)
        
        metadata = {
            "conversation_id": conversation_id,
            "character_id": character_id,
            "title": "Cooking Discussion",
            "created_at": "2024-01-15T10:00:00",
            "message_count": 25,
            "themes": ["cooking", "food preferences", "Italian cuisine"],
            "tone": "enthusiastic",
            "key_topics": ["vegetarian", "pasta", "gnocchi"],
            "participants": ["User"],
            "source": "web"
        }
        
        success = vector_store.add_summary(
            character_id=character_id,
            conversation_id=conversation_id,
            summary_text=summary_text,
            embedding=embedding,
            metadata=metadata
        )
        assert success
        print(f"✓ Summary added for conversation {conversation_id[:8]}...")
        print()
        
        # Test 3: Add more summaries for search testing
        print("Test 3: Add additional summaries")
        test_summaries = [
            {
                "id": "test_conv_002",
                "text": "The user asked about machine learning and AI concepts. We covered neural networks, transformers, and natural language processing techniques.",
                "title": "AI Discussion",
                "themes": ["technology", "AI", "machine learning"],
                "key_topics": ["neural networks", "transformers", "NLP"]
            },
            {
                "id": "test_conv_003",
                "text": "We talked about hiking trails in Colorado and camping gear recommendations. The user is planning a trip to Rocky Mountain National Park.",
                "title": "Outdoor Activities",
                "themes": ["outdoors", "travel", "hiking"],
                "key_topics": ["hiking", "camping", "Rocky Mountain"]
            },
            {
                "id": "test_conv_004",
                "text": "Discussion about pizza toppings and different regional pizza styles. User prefers New York style thin crust over deep dish.",
                "title": "Pizza Preferences",
                "themes": ["food", "preferences", "pizza"],
                "key_topics": ["pizza", "New York style", "thin crust"]
            }
        ]
        
        for summary_data in test_summaries:
            emb = embedding_service.embed(summary_data["text"])
            meta = {
                "conversation_id": summary_data["id"],
                "character_id": character_id,
                "title": summary_data["title"],
                "themes": summary_data["themes"],
                "key_topics": summary_data["key_topics"],
                "source": "web"
            }
            vector_store.add_summary(
                character_id=character_id,
                conversation_id=summary_data["id"],
                summary_text=summary_data["text"],
                embedding=emb,
                metadata=meta
            )
        print(f"✓ Added {len(test_summaries)} additional summaries")
        
        # Check count
        count = vector_store.get_collection_count(character_id)
        print(f"✓ Total summaries in collection: {count}")
        assert count == 4
        print()
        
        # Test 4: Semantic search
        print("Test 4: Semantic search")
        
        # Search for food-related conversations
        search_query = "What do we know about the user's food preferences?"
        search_embedding = embedding_service.embed(search_query)
        
        results = vector_store.search_conversations(
            character_id=character_id,
            query_embedding=search_embedding,
            n_results=3
        )
        
        print(f"Search query: '{search_query}'")
        print("Results:")
        
        if results['ids'] and results['ids'][0]:
            for i, (conv_id, distance, doc) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['documents'][0]
            )):
                print(f"  {i+1}. {conv_id} (distance: {distance:.4f})")
                print(f"     {doc[:80]}...")
        
        # Verify food-related conversations rank highest
        assert results['ids'][0][0] in ["test_conv_001", "test_conv_004"], \
            "Expected food-related conversation to rank first"
        print("✓ Search results are relevant to food preferences")
        print()
        
        # Test 5: Get specific summary
        print("Test 5: Get specific summary")
        summary = vector_store.get_summary(character_id, "test_conv_002")
        assert summary is not None
        assert summary['conversation_id'] == "test_conv_002"
        assert "machine learning" in summary['summary'].lower()
        print(f"✓ Retrieved summary: {summary['metadata'].get('title', 'No title')}")
        print()
        
        # Test 6: Upsert (update existing)
        print("Test 6: Upsert behavior")
        updated_text = "Updated: We had an extensive discussion about AI and machine learning. The user is particularly interested in large language models."
        updated_embedding = embedding_service.embed(updated_text)
        
        success = vector_store.add_summary(
            character_id=character_id,
            conversation_id="test_conv_002",
            summary_text=updated_text,
            embedding=updated_embedding,
            metadata={
                "conversation_id": "test_conv_002",
                "character_id": character_id,
                "title": "AI Discussion (Updated)",
                "key_topics": ["AI", "machine learning", "LLMs"]
            }
        )
        assert success
        
        # Verify count didn't increase (upsert)
        count_after = vector_store.get_collection_count(character_id)
        assert count_after == 4, "Count should remain 4 after upsert"
        
        # Verify content was updated
        updated_summary = vector_store.get_summary(character_id, "test_conv_002")
        assert "Updated:" in updated_summary['summary']
        print("✓ Upsert updated existing summary without increasing count")
        print()
        
        # Test 7: Delete summary
        print("Test 7: Delete summary")
        success = vector_store.delete_summary(character_id, "test_conv_003")
        assert success
        
        count_after_delete = vector_store.get_collection_count(character_id)
        assert count_after_delete == 3
        
        # Verify deleted summary is gone
        deleted_summary = vector_store.get_summary(character_id, "test_conv_003")
        assert deleted_summary is None
        print("✓ Summary deleted successfully")
        print()
        
        # Test 8: List collections
        print("Test 8: List collections")
        collections = vector_store.list_collections()
        assert f"conversation_summaries_{character_id}" in collections
        print(f"✓ Found collection in list: {collections}")
        print()
        
        # Test 9: Delete collection
        print("Test 9: Delete collection")
        success = vector_store.delete_collection(character_id)
        assert success
        
        count_after_collection_delete = vector_store.get_collection_count(character_id)
        assert count_after_collection_delete == 0
        print("✓ Collection deleted successfully")
        print()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    test_vector_store()
