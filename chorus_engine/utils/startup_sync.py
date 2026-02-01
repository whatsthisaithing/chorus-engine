"""
Startup synchronization utilities.

This module handles automatic synchronization tasks that run on server startup,
such as ensuring vector stores are in sync with SQL data.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Set

from sqlalchemy.orm import Session

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.models.conversation import ConversationSummary, Conversation

logger = logging.getLogger(__name__)


async def sync_conversation_summary_vectors(
    db_session: Session,
    summary_vector_store: ConversationSummaryVectorStore,
    embedding_service: EmbeddingService,
    character_id: Optional[str] = None
) -> dict:
    """
    Sync conversation summaries from SQL to vector store.
    
    Finds summaries that exist in SQL but not in the vector store and adds them.
    This provides self-healing after restore, migration, or corruption.
    
    Args:
        db_session: SQLAlchemy database session
        summary_vector_store: ConversationSummaryVectorStore instance
        embedding_service: EmbeddingService for generating embeddings
        character_id: Optional - sync only this character (all if None)
        
    Returns:
        Dict with sync statistics: {synced: int, skipped: int, errors: int, characters: list}
    """
    stats = {
        "synced": 0,
        "skipped": 0,
        "errors": 0,
        "characters": []
    }
    
    try:
        # Get all summaries from SQL, joined with Conversation for character_id
        query = db_session.query(ConversationSummary).join(
            Conversation,
            ConversationSummary.conversation_id == Conversation.id
        )
        
        if character_id:
            query = query.filter(Conversation.character_id == character_id)
        
        summaries = query.all()
        
        if not summaries:
            logger.debug("No conversation summaries in database - nothing to sync")
            return stats
        
        # Group summaries by character
        summaries_by_character: dict[str, list] = {}
        for summary in summaries:
            # Get character_id from the joined conversation
            conv = db_session.query(Conversation).filter(
                Conversation.id == summary.conversation_id
            ).first()
            
            if conv:
                char_id = conv.character_id
                if char_id not in summaries_by_character:
                    summaries_by_character[char_id] = []
                summaries_by_character[char_id].append((summary, conv))
        
        # Process each character's summaries
        for char_id, char_summaries in summaries_by_character.items():
            # Get existing vector IDs for this character
            existing_ids = _get_existing_vector_ids(summary_vector_store, char_id)
            
            # Find missing summaries
            missing = [
                (summary, conv) for summary, conv in char_summaries
                if summary.conversation_id not in existing_ids
            ]
            
            if not missing:
                logger.debug(f"Character '{char_id}': all {len(char_summaries)} summaries already in vector store")
                stats["skipped"] += len(char_summaries)
                continue
            
            logger.info(f"Syncing {len(missing)} missing summaries for character '{char_id}'...")
            stats["characters"].append(char_id)
            
            # Process missing summaries
            for summary, conv in missing:
                try:
                    success = _sync_single_summary(
                        summary_vector_store=summary_vector_store,
                        embedding_service=embedding_service,
                        summary=summary,
                        conversation=conv,
                        character_id=char_id
                    )
                    
                    if success:
                        stats["synced"] += 1
                    else:
                        stats["errors"] += 1
                        
                except Exception as e:
                    logger.error(f"Error syncing summary {summary.conversation_id[:8]}...: {e}")
                    stats["errors"] += 1
            
            # Count already-synced as skipped
            stats["skipped"] += len(char_summaries) - len(missing)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to sync conversation summary vectors: {e}")
        stats["errors"] += 1
        return stats


def _get_existing_vector_ids(
    summary_vector_store: ConversationSummaryVectorStore,
    character_id: str
) -> Set[str]:
    """
    Get set of conversation IDs that already have vectors.
    
    Args:
        summary_vector_store: Vector store instance
        character_id: Character to check
        
    Returns:
        Set of conversation_id strings
    """
    try:
        collection = summary_vector_store.get_collection(character_id)
        if collection is None:
            return set()
        
        # Get all IDs from the collection
        # ChromaDB's get() with no filters returns all
        results = collection.get(include=[])  # Only get IDs, no embeddings/docs
        
        if results and results.get('ids'):
            return set(results['ids'])
        
        return set()
        
    except Exception as e:
        logger.warning(f"Could not get existing vector IDs for '{character_id}': {e}")
        return set()


def _sync_single_summary(
    summary_vector_store: ConversationSummaryVectorStore,
    embedding_service: EmbeddingService,
    summary: ConversationSummary,
    conversation: Conversation,
    character_id: str
) -> bool:
    """
    Sync a single summary to the vector store.
    
    Args:
        summary_vector_store: Vector store instance
        embedding_service: Embedding service
        summary: ConversationSummary record
        conversation: Parent Conversation record
        character_id: Character ID
        
    Returns:
        True if successful
    """
    try:
        # Build searchable text (same logic as backfill utility)
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
            "character_id": character_id,
            "title": conversation.title or "Untitled",
            "created_at": conversation.created_at.isoformat() if conversation.created_at else "",
            "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else "",
            "message_count": summary.message_count or 0,
            "themes": [],  # May not exist in older summaries
            "tone": summary.tone or "",
            "emotional_arc": summary.emotional_arc or "",
            "key_topics": summary.key_topics if isinstance(summary.key_topics, list) else [],
            "participants": summary.participants if isinstance(summary.participants, list) else [],
            "source": conversation.source if hasattr(conversation, 'source') and conversation.source else "web",
            "analyzed_at": summary.created_at.isoformat() if summary.created_at else datetime.utcnow().isoformat(),
            "manual_analysis": summary.manual == "true" if summary.manual else False
        }
        
        # Add to vector store
        success = summary_vector_store.add_summary(
            character_id=character_id,
            conversation_id=summary.conversation_id,
            summary_text=searchable_text,
            embedding=embedding,
            metadata=metadata
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to sync summary {summary.conversation_id[:8]}...: {e}")
        return False
