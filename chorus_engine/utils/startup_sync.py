"""
Startup synchronization utilities.

This module handles automatic synchronization tasks that run on server startup,
such as ensuring vector stores are in sync with SQL data.
"""

import logging
from datetime import datetime
from typing import Optional, Set, List
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.models.conversation import ConversationSummary, Conversation, Memory

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
        "deleted_orphans": 0,
        "errors": 0,
        "characters": []
    }
    
    try:
        # Subquery to find latest summary per conversation
        latest_subquery = (
            db_session.query(
                ConversationSummary.conversation_id.label("conversation_id"),
                func.max(ConversationSummary.created_at).label("max_created_at")
            )
            .group_by(ConversationSummary.conversation_id)
        )
        
        if character_id:
            latest_subquery = latest_subquery.join(
                Conversation,
                ConversationSummary.conversation_id == Conversation.id
            ).filter(Conversation.character_id == character_id)
        
        latest_subquery = latest_subquery.subquery()
        
        latest_rows = (
            db_session.query(ConversationSummary, Conversation)
            .join(
                latest_subquery,
                and_(
                    ConversationSummary.conversation_id == latest_subquery.c.conversation_id,
                    ConversationSummary.created_at == latest_subquery.c.max_created_at
                )
            )
            .join(Conversation, ConversationSummary.conversation_id == Conversation.id)
            .all()
        )
        
        if not latest_rows:
            logger.debug("No conversation summaries in database - nothing to sync")
            return stats
        
        # Group latest summaries by character
        summaries_by_character: dict[str, list] = {}
        for summary, conv in latest_rows:
            char_id = conv.character_id
            summaries_by_character.setdefault(char_id, [])
            summaries_by_character[char_id].append((summary, conv))
        
        # Include characters with existing summary collections (to handle orphans)
        existing_collections = summary_vector_store.list_collections()
        existing_char_ids = {
            name.replace("conversation_summaries_", "", 1)
            for name in existing_collections
            if name.startswith("conversation_summaries_")
        }
        
        all_char_ids = set(summaries_by_character.keys()) | existing_char_ids
        
        # Process each character's summaries
        for char_id in all_char_ids:
            char_summaries = summaries_by_character.get(char_id, [])
            sql_conversation_ids = {summary.conversation_id for summary, _ in char_summaries}
            
            # Get existing vector IDs for this character
            existing_ids = _get_existing_vector_ids(summary_vector_store, char_id)
            
            # Delete orphaned vectors (present in vector store but not in SQL)
            orphan_ids = existing_ids - sql_conversation_ids
            if orphan_ids:
                logger.info(f"Deleting {len(orphan_ids)} orphaned summary vectors for character '{char_id}'...")
                _delete_summary_vectors(summary_vector_store, char_id, orphan_ids)
                stats["deleted_orphans"] = stats.get("deleted_orphans", 0) + len(orphan_ids)
                if char_id not in stats["characters"]:
                    stats["characters"].append(char_id)
            
            if not char_summaries:
                continue
            
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


async def sync_memory_vectors(
    db_session: Session,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    character_id: Optional[str] = None
) -> dict:
    """
    Sync memory vectors from SQL to vector store.
    
    Deletes orphaned vectors and regenerates missing ones for eligible memories.
    
    Eligibility:
    - status in ["approved", "auto_approved"]
    - durability != "ephemeral"
    
    Args:
        db_session: SQLAlchemy database session
        vector_store: VectorStore instance
        embedding_service: EmbeddingService for generating embeddings
        character_id: Optional - sync only this character (all if None)
        
    Returns:
        Dict with sync statistics: {synced: int, deleted_orphans: int, updated_ids: int, errors: int, characters: list}
    """
    stats = {
        "synced": 0,
        "deleted_orphans": 0,
        "updated_ids": 0,
        "errors": 0,
        "characters": []
    }
    
    try:
        eligible_query = db_session.query(Memory).filter(
            Memory.status.in_(["approved", "auto_approved"]),
            or_(Memory.durability.is_(None), Memory.durability != "ephemeral")
        )
        
        if character_id:
            eligible_query = eligible_query.filter(Memory.character_id == character_id)
        
        eligible_memories = eligible_query.all()
        
        # Group memories by character
        memories_by_character: dict[str, list] = {}
        for memory in eligible_memories:
            memories_by_character.setdefault(memory.character_id, [])
            memories_by_character[memory.character_id].append(memory)
        
        # Include characters with existing memory collections
        existing_collections = vector_store.list_collections()
        existing_char_ids = {
            name.replace("character_", "", 1)
            for name in existing_collections
            if name.startswith("character_")
        }
        
        all_char_ids = set(memories_by_character.keys()) | existing_char_ids
        
        for char_id in all_char_ids:
            char_memories = memories_by_character.get(char_id, [])
            existing_ids = _get_existing_memory_vector_ids(vector_store, char_id)
            
            if not char_memories:
                if existing_ids:
                    logger.info(f"Deleting {len(existing_ids)} orphaned memory vectors for character '{char_id}'...")
                    _delete_memory_vectors(vector_store, char_id, existing_ids)
                    stats["deleted_orphans"] += len(existing_ids)
                    if char_id not in stats["characters"]:
                        stats["characters"].append(char_id)
                continue
            
            # Build SQL vector ID set for this character
            sql_vector_ids = {m.vector_id for m in char_memories if m.vector_id}
            
            # Delete orphaned vectors
            orphan_ids = existing_ids - sql_vector_ids
            if orphan_ids:
                logger.info(f"Deleting {len(orphan_ids)} orphaned memory vectors for character '{char_id}'...")
                _delete_memory_vectors(vector_store, char_id, orphan_ids)
                stats["deleted_orphans"] += len(orphan_ids)
            
            # Find missing vectors (SQL has vector_id but vector store does not)
            missing_ids = sql_vector_ids - existing_ids
            missing_memories = {m.vector_id: m for m in char_memories if m.vector_id in missing_ids}
            
            # Add missing vectors
            if missing_memories:
                logger.info(f"Regenerating {len(missing_memories)} missing memory vectors for character '{char_id}'...")
                _add_memory_vectors(
                    vector_store=vector_store,
                    embedding_service=embedding_service,
                    memories=list(missing_memories.values())
                )
                stats["synced"] += len(missing_memories)
            
            # Add vectors for eligible memories without vector_id
            no_vector_memories = [m for m in char_memories if not m.vector_id]
            if no_vector_memories:
                logger.info(f"Assigning vectors to {len(no_vector_memories)} eligible memories without vector_id for '{char_id}'...")
                for memory in no_vector_memories:
                    memory.vector_id = str(uuid.uuid4())
                    stats["updated_ids"] += 1
                
                _add_memory_vectors(
                    vector_store=vector_store,
                    embedding_service=embedding_service,
                    memories=no_vector_memories
                )
                
                stats["synced"] += len(no_vector_memories)
                db_session.commit()
            
            stats["characters"].append(char_id)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to sync memory vectors: {e}")
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
        # Build searchable text (summary + open questions)
        searchable_text = summary.summary
        if summary.open_questions:
            questions = summary.open_questions if isinstance(summary.open_questions, list) else []
            if questions:
                searchable_text += f"\nOpen Questions: {', '.join(questions)}"
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
            "open_questions": summary.open_questions if isinstance(summary.open_questions, list) else [],
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


def _get_existing_memory_vector_ids(
    vector_store: VectorStore,
    character_id: str
) -> Set[str]:
    """Get set of memory vector IDs for a character from vector store."""
    try:
        collection = vector_store.get_collection(character_id)
        if collection is None:
            return set()
        results = collection.get(include=[])
        if results and results.get("ids"):
            return set(results["ids"])
        return set()
    except Exception as e:
        logger.warning(f"Could not get memory vector IDs for '{character_id}': {e}")
        return set()


def _delete_summary_vectors(
    summary_vector_store: ConversationSummaryVectorStore,
    character_id: str,
    vector_ids: Set[str]
) -> None:
    """Delete summary vectors in batches."""
    collection = summary_vector_store.get_collection(character_id)
    if collection is None:
        return
    
    ids = list(vector_ids)
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            collection.delete(ids=batch)
        except Exception as e:
            logger.warning(f"Failed to delete summary vectors batch for '{character_id}': {e}")


def _delete_memory_vectors(
    vector_store: VectorStore,
    character_id: str,
    vector_ids: Set[str]
) -> None:
    """Delete memory vectors in batches."""
    collection = vector_store.get_collection(character_id)
    if collection is None:
        return
    
    ids = list(vector_ids)
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            collection.delete(ids=batch)
        except Exception as e:
            logger.warning(f"Failed to delete memory vectors batch for '{character_id}': {e}")


def _add_memory_vectors(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    memories: List[Memory]
) -> None:
    """Add memory vectors for the provided memories."""
    if not memories:
        return
    
    # Group by character for collection writes
    memories_by_character: dict[str, List[Memory]] = {}
    for memory in memories:
        memories_by_character.setdefault(memory.character_id, [])
        memories_by_character[memory.character_id].append(memory)
    
    for char_id, char_memories in memories_by_character.items():
        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[dict] = []
        
        for memory in char_memories:
            if not memory.vector_id:
                continue
            
            embedding = embedding_service.embed(memory.content)
            
            ids.append(memory.vector_id)
            embeddings.append(embedding)
            documents.append(memory.content)
            metadatas.append({
                "type": memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type),
                "category": memory.category or "",
                "confidence": memory.confidence or 0.0,
                "status": memory.status,
                "durability": memory.durability if hasattr(memory, "durability") else "situational",
                "pattern_eligible": bool(getattr(memory, "pattern_eligible", 0))
            })
        
        if ids:
            success = vector_store.add_memories(
                character_id=char_id,
                memory_ids=ids,
                contents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            if not success:
                logger.warning(f"Failed to add {len(ids)} memory vectors for '{char_id}'")
