"""
Backfill conversation summaries to the vector store.

This utility migrates existing ConversationSummary records from the SQL database
to the ConversationSummaryVectorStore, enabling semantic search across past
conversations.

Run this once after updating to the new conversation memory enrichment system.

Usage:
    python utilities/backfill_conversation_summaries.py
    python utilities/backfill_conversation_summaries.py --character nova
    python utilities/backfill_conversation_summaries.py --dry-run
    
Options:
    --character <id>  Only backfill summaries for a specific character
    --dry-run         Show what would be done without making changes
    --verbose         Show detailed output for each summary
    
Note: Existing vector entries will be updated (upsert behavior).
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.models.conversation import ConversationSummary, Conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_summaries(
    character_id: str = None,
    dry_run: bool = False,
    verbose: bool = False
):
    """
    Backfill conversation summaries to vector store.
    
    Args:
        character_id: Optional character ID to filter by
        dry_run: If True, only show what would be done
        verbose: If True, show details for each summary
    """
    session = SessionLocal()
    vector_store = ConversationSummaryVectorStore(Path("data/vector_store"))
    embedding_service = EmbeddingService()
    
    try:
        print(f"\n{'='*60}")
        print("CONVERSATION SUMMARY BACKFILL")
        print(f"{'='*60}\n")
        
        # Build query
        query = session.query(ConversationSummary)
        
        # Get summaries to process
        summaries = query.all()
        
        if not summaries:
            print("✓ No conversation summaries found. Nothing to backfill.")
            return
        
        # Filter by character if specified
        if character_id:
            # We need to join with Conversation to get character_id
            summaries = session.query(ConversationSummary).join(
                Conversation,
                ConversationSummary.conversation_id == Conversation.id
            ).filter(
                Conversation.character_id == character_id
            ).all()
            
            if not summaries:
                print(f"✓ No summaries found for character '{character_id}'.")
                return
        
        print(f"Found {len(summaries)} summaries to backfill")
        if character_id:
            print(f"  Filtering by character: {character_id}")
        if dry_run:
            print("  Mode: DRY RUN (no changes will be made)")
        print()
        
        # Statistics
        processed = 0
        skipped = 0
        errors = 0
        
        # Process each summary
        for summary in summaries:
            try:
                # Get the associated conversation
                conversation = session.query(Conversation).filter(
                    Conversation.id == summary.conversation_id
                ).first()
                
                if not conversation:
                    logger.warning(f"Conversation {summary.conversation_id} not found, skipping")
                    skipped += 1
                    continue
                
                char_id = conversation.character_id
                
                if verbose:
                    print(f"\nProcessing: {summary.conversation_id[:8]}...")
                    print(f"  Character: {char_id}")
                    print(f"  Title: {conversation.title}")
                    print(f"  Messages: {summary.message_count}")
                    print(f"  Summary: {summary.summary[:100]}..." if len(summary.summary) > 100 else f"  Summary: {summary.summary}")
                
                if dry_run:
                    processed += 1
                    continue
                
                # Build searchable text
                searchable_text = summary.summary
                if summary.key_topics:
                    topics = summary.key_topics if isinstance(summary.key_topics, list) else []
                    if topics:
                        searchable_text += f"\nTopics: {', '.join(topics)}"
                
                # Generate embedding
                embedding = embedding_service.embed(searchable_text)
                
                # Build metadata
                metadata = {
                    "conversation_id": summary.conversation_id,
                    "character_id": char_id,
                    "title": conversation.title or "Untitled",
                    "created_at": conversation.created_at.isoformat() if conversation.created_at else "",
                    "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else "",
                    "message_count": summary.message_count or 0,
                    "themes": [],  # Not stored in old summaries
                    "tone": summary.tone or "",
                    "emotional_arc": summary.emotional_arc or "",
                    "key_topics": summary.key_topics if isinstance(summary.key_topics, list) else [],
                    "participants": summary.participants if isinstance(summary.participants, list) else [],
                    "source": conversation.source if hasattr(conversation, 'source') and conversation.source else "web",
                    "analyzed_at": summary.created_at.isoformat() if summary.created_at else datetime.utcnow().isoformat(),
                    "manual_analysis": summary.manual == "true" if summary.manual else False
                }
                
                # Save to vector store
                success = vector_store.add_summary(
                    character_id=char_id,
                    conversation_id=summary.conversation_id,
                    summary_text=searchable_text,
                    embedding=embedding,
                    metadata=metadata
                )
                
                if success:
                    processed += 1
                    if verbose:
                        print(f"  ✓ Saved to vector store")
                else:
                    errors += 1
                    logger.error(f"Failed to save summary for {summary.conversation_id}")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error processing summary {summary.conversation_id}: {e}")
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print("BACKFILL COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors: {errors}")
        
        if dry_run:
            print("\n  ℹ️  This was a dry run. Run without --dry-run to apply changes.")
        else:
            # Show collection stats
            collections = vector_store.list_collections()
            print(f"\nVector store collections: {len(collections)}")
            for coll_name in collections:
                # Extract character_id from collection name
                char = coll_name.replace("conversation_summaries_", "")
                count = vector_store.get_collection_count(char)
                print(f"  • {coll_name}: {count} summaries")
        
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill conversation summaries to vector store"
    )
    parser.add_argument(
        "--character",
        type=str,
        help="Only backfill summaries for a specific character"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each summary"
    )
    
    args = parser.parse_args()
    
    backfill_summaries(
        character_id=args.character,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
