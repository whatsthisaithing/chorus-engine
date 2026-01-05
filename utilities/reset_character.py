"""
Reset a character's conversations and memories.

This utility script completely removes all conversations, threads, messages, 
memories (SQL database), and vector store entries for a specific character.

Usage:
    python utilities/reset_character.py <character_id>
    
Example:
    python utilities/reset_character.py sarah_v1
    
Warning: This action cannot be undone!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.models.conversation import Conversation, Thread, Message, Memory


def reset_character(character_id: str, confirm: bool = False):
    """
    Reset all data for a character.
    
    Args:
        character_id: The character ID to reset
        confirm: If True, skip confirmation prompt
    """
    session = SessionLocal()
    vector_store = VectorStore(Path('data/vector_store'))
    
    try:
        print(f"\n{'='*60}")
        print(f"CHARACTER RESET: {character_id}")
        print(f"{'='*60}\n")
        
        # Count items to be deleted
        conversations = session.query(Conversation).filter(
            Conversation.character_id == character_id
        ).all()
        
        thread_count = 0
        message_count = 0
        for conv in conversations:
            threads = session.query(Thread).filter(Thread.conversation_id == conv.id).all()
            thread_count += len(threads)
            for thread in threads:
                messages = session.query(Message).filter(Message.thread_id == thread.id).count()
                message_count += messages
        
        memory_count = session.query(Memory).filter(
            Memory.character_id == character_id
        ).count()
        
        # Check vector store
        collection = vector_store.get_collection(character_id)
        vector_count = collection.count() if collection else 0
        
        print("Items to be deleted:")
        print(f"  • Conversations: {len(conversations)}")
        print(f"  • Threads: {thread_count}")
        print(f"  • Messages: {message_count}")
        print(f"  • Memories (SQL): {memory_count}")
        print(f"  • Vector Store Entries: {vector_count}")
        print()
        
        if not conversations and memory_count == 0 and vector_count == 0:
            print("✓ No data found for this character. Nothing to delete.")
            return
        
        # Confirmation
        if not confirm:
            print("⚠️  WARNING: This action cannot be undone!")
            response = input("Type the character ID to confirm deletion: ")
            if response != character_id:
                print("\n❌ Deletion cancelled.")
                return
            print()
        
        # Delete SQL data
        print("Deleting SQL database entries...")
        
        # Delete messages (cascade through threads -> conversations)
        for conv in conversations:
            threads = session.query(Thread).filter(Thread.conversation_id == conv.id).all()
            for thread in threads:
                deleted_messages = session.query(Message).filter(
                    Message.thread_id == thread.id
                ).delete()
                print(f"  ✓ Deleted {deleted_messages} messages from thread {thread.id[:8]}...")
            
            deleted_threads = session.query(Thread).filter(
                Thread.conversation_id == conv.id
            ).delete()
            print(f"  ✓ Deleted {deleted_threads} threads from conversation {conv.id[:8]}...")
        
        # Delete conversations
        deleted_conversations = session.query(Conversation).filter(
            Conversation.character_id == character_id
        ).delete()
        print(f"  ✓ Deleted {deleted_conversations} conversations")
        
        # Delete memories
        deleted_memories = session.query(Memory).filter(
            Memory.character_id == character_id
        ).delete()
        print(f"  ✓ Deleted {deleted_memories} memories")
        
        session.commit()
        print()
        
        # Delete vector store
        if vector_count > 0:
            print("Deleting vector store entries...")
            success = vector_store.delete_collection(character_id)
            if success:
                print(f"  ✓ Deleted {vector_count} vector store entries")
            else:
                print(f"  ⚠ Failed to delete vector store collection")
            print()
        
        print("="*60)
        print("✅ CHARACTER RESET COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        
    finally:
        session.close()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python utilities/reset_character.py <character_id>")
        print()
        print("Example:")
        print("  python utilities/reset_character.py sarah_v1")
        sys.exit(1)
    
    character_id = sys.argv[1]
    reset_character(character_id)


if __name__ == "__main__":
    main()
