"""
Conversation Analysis Task Handler for Heartbeat System.

Handles background analysis of stale conversations to generate
searchable summaries for the conversation context enrichment system.

Phase D.5: Conversation Analysis Task
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from chorus_engine.services.heartbeat_service import (
    BackgroundTaskHandler, BackgroundTask, TaskResult, TaskPriority
)

logger = logging.getLogger(__name__)


class ConversationAnalysisTaskHandler(BackgroundTaskHandler):
    """
    Task handler for analyzing conversations during idle time.
    
    Analyzes conversations that meet criteria:
    - At least N messages (configurable, default 10)
    - Inactive for at least N hours (configurable, default 24)
    - Not already analyzed or needs re-analysis
    
    Uses the existing ConversationAnalysisService for the actual analysis.
    """
    
    @property
    def task_type(self) -> str:
        return "conversation_analysis"
    
    async def execute(self, task: BackgroundTask, app_state: Dict[str, Any]) -> TaskResult:
        """
        Execute conversation analysis for a single conversation.
        
        Task data expected:
        - conversation_id: str - ID of conversation to analyze
        - character_id: str - Character ID for the conversation
        """
        start_time = datetime.utcnow()
        
        conversation_id = task.data.get("conversation_id")
        character_id = task.data.get("character_id")
        
        if not conversation_id or not character_id:
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=0,
                error="Missing conversation_id or character_id in task data"
            )
        
        try:
            # Get required services from app_state
            analysis_service = app_state.get("analysis_service")
            characters = app_state.get("characters", {})
            
            if not analysis_service:
                return TaskResult(
                    success=False,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=0,
                    error="ConversationAnalysisService not available"
                )
            
            character = characters.get(character_id)
            if not character:
                return TaskResult(
                    success=False,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=0,
                    error=f"Character '{character_id}' not found"
                )
            
            logger.info(
                f"[ANALYSIS TASK] Analyzing conversation {conversation_id[:8]}... "
                f"for character {character.name}"
            )
            
            # Run the analysis
            analysis = await analysis_service.analyze_conversation(
                conversation_id=conversation_id,
                character=character,
                manual=False  # Background analysis
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            if analysis:
                # SAVE the analysis to database - this was missing!
                save_success = await analysis_service.save_analysis(
                    conversation_id=conversation_id,
                    character_id=character_id,
                    analysis=analysis,
                    manual=False
                )
                
                if not save_success:
                    logger.error(
                        f"[ANALYSIS TASK] Failed to save analysis for {conversation_id[:8]}..."
                    )
                    return TaskResult(
                        success=False,
                        task_id=task.id,
                        task_type=self.task_type,
                        duration_seconds=duration,
                        error="Failed to save analysis to database"
                    )
                
                logger.info(
                    f"[ANALYSIS TASK] Completed analysis of {conversation_id[:8]}... "
                    f"in {duration:.1f}s - extracted {len(analysis.memories)} memories"
                )
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=duration,
                    data={
                        "conversation_id": conversation_id,
                        "memories_extracted": len(analysis.memories),
                        "summary_length": len(analysis.summary),
                        "open_questions_count": len(analysis.open_questions)
                    }
                )
            else:
                logger.warning(
                    f"[ANALYSIS TASK] Analysis returned None for {conversation_id[:8]}..."
                )
                return TaskResult(
                    success=False,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=duration,
                    error="Analysis returned None (conversation may be too short)"
                )
                
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                f"[ANALYSIS TASK] Error analyzing {conversation_id[:8]}...: {e}",
                exc_info=True
            )
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=duration,
                error=str(e)
            )
    
    def should_retry(self, task: BackgroundTask, error: Exception) -> bool:
        """
        Determine if analysis should be retried.
        
        Don't retry for:
        - Missing conversation/character (permanent failure)
        - Conversation too short (won't change without more messages)
        
        Do retry for:
        - LLM timeout/errors (transient)
        - Database connection issues (transient)
        """
        error_str = str(error).lower()
        
        # Permanent failures - don't retry
        if "not found" in error_str or "too short" in error_str:
            return False
        
        # Transient failures - retry
        if task.retry_count < task.max_retries:
            return True
        
        return False


class StaleConversationFinder:
    """
    Utility class to find conversations that need background analysis.
    
    Used by HeartbeatService to populate the task queue with
    conversations that meet the stale threshold criteria.
    """
    
    def __init__(
        self,
        stale_hours: int = 24,
        min_messages: int = 10
    ):
        """
        Initialize finder.
        
        Args:
            stale_hours: Hours of inactivity to consider conversation stale
            min_messages: Minimum messages required for analysis
        """
        self.stale_hours = stale_hours
        self.min_messages = min_messages
    
    def find_stale_conversations(
        self,
        db,
        character_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find conversations that need analysis.
        
        Args:
            db: Database session
            character_id: Optional - only check this character's conversations
            limit: Maximum number of conversations to return
            
        Returns:
            List of dicts with conversation_id, character_id, message_count
        """
        from chorus_engine.repositories.conversation_repository import ConversationRepository
        from chorus_engine.repositories.message_repository import MessageRepository
        from chorus_engine.models.conversation import Conversation, Message, Thread
        from sqlalchemy import func
        
        conv_repo = ConversationRepository(db)
        msg_repo = MessageRepository(db)
        
        candidates = []
        cutoff_time = datetime.utcnow() - timedelta(hours=self.stale_hours)
        
        try:
            # Build query for conversations
            query = db.query(Conversation)
            
            # Filter by character if specified
            if character_id:
                query = query.filter(Conversation.character_id == character_id)
            
            # Filter out private conversations
            query = query.filter(
                (Conversation.is_private != "true") | 
                (Conversation.is_private.is_(None))
            )
            
            # Include all stale conversations; message-based new-content check handled below
            
            # Also filter to only stale conversations (activity before cutoff)
            # Keep DB filter broad; message-based cutoff applied below.
            
            # Order by oldest activity first (process oldest conversations first)
            query = query.order_by(Conversation.updated_at.asc())
            
            # Get more conversations than needed since we filter by message count
            # Most conversations are short (testing), so we need extra buffer
            conversations = query.limit(limit * 10).all()  # Get extra for filtering
            
            logger.debug(
                f"[STALE FINDER] Query returned {len(conversations)} conversations "
                f"(cutoff: {cutoff_time.isoformat()}, limit requested: {limit})"
            )
            
            for conv in conversations:
                # Get message count from threads
                message_count = 0
                for thread in conv.threads:
                    message_count += msg_repo.count_thread_messages(thread.id)

                latest_message_at = (
                    db.query(func.max(Message.created_at))
                    .join(Thread, Message.thread_id == Thread.id)
                    .filter(Thread.conversation_id == conv.id)
                    .scalar()
                )
                
                logger.debug(
                    f"[STALE FINDER] Checking {conv.id[:8]}... "
                    f"msgs={message_count}, updated={conv.updated_at}, "
                    f"latest_message={latest_message_at}, analyzed={conv.last_analyzed_at}"
                )

                if not latest_message_at:
                    continue

                if latest_message_at >= cutoff_time:
                    continue

                if conv.last_analyzed_at and latest_message_at <= conv.last_analyzed_at:
                    continue
                
                if message_count >= self.min_messages:
                    # Conversation passes all filters
                    candidates.append({
                        "conversation_id": conv.id,
                        "character_id": conv.character_id,
                        "message_count": message_count,
                        "last_activity": latest_message_at.isoformat() if latest_message_at else None,
                        "title": conv.title
                    })
                    
                    if len(candidates) >= limit:
                        break
            
            logger.debug(
                f"[STALE FINDER] Found {len(candidates)} stale conversations "
                f"(threshold: {self.stale_hours}h, min_messages: {self.min_messages})"
            )
            
            return candidates
            
        except Exception as e:
            logger.error(f"[STALE FINDER] Error finding stale conversations: {e}", exc_info=True)
            return []
    
    def queue_stale_conversations(
        self,
        heartbeat_service,
        db,
        character_id: Optional[str] = None,
        limit: int = 10,
        priority: TaskPriority = TaskPriority.LOW
    ) -> int:
        """
        Find stale conversations and queue them for analysis.
        
        Args:
            heartbeat_service: HeartbeatService instance
            db: Database session
            character_id: Optional - only check this character
            limit: Max conversations to queue
            priority: Task priority level
            
        Returns:
            Number of tasks queued
        """
        stale = self.find_stale_conversations(db, character_id, limit)
        
        queued = 0
        for conv in stale:
            task_id = f"analysis_{conv['conversation_id']}"
            
            heartbeat_service.queue_task(
                task_type="conversation_analysis",
                data={
                    "conversation_id": conv["conversation_id"],
                    "character_id": conv["character_id"]
                },
                priority=priority,
                task_id=task_id
            )
            queued += 1
        
        if queued > 0:
            logger.info(
                f"[STALE FINDER] Queued {queued} conversations for background analysis"
            )
        
        return queued
