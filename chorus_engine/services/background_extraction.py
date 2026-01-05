"""Background extraction manager for processing memory extraction asynchronously."""

import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass

from chorus_engine.models.conversation import Message
from chorus_engine.services.memory_extraction import MemoryExtractionService
from chorus_engine.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


@dataclass
class ExtractionTask:
    """Represents a queued extraction task."""
    conversation_id: str
    character_id: str
    messages: List[Message]
    model: Optional[str] = None  # Character's preferred model for extraction
    character_name: Optional[str] = None  # Character name for prompt context


class BackgroundExtractionManager:
    """
    Manages background extraction of implicit memories from conversations.
    
    Processes extraction tasks asynchronously without blocking conversation flow.
    """
    
    def __init__(
        self,
        extraction_service: MemoryExtractionService,
        conversation_repo: ConversationRepository,
        db_session_factory = None
    ):
        self.extraction_service = extraction_service
        self.conversation_repo = conversation_repo
        self.db_session_factory = db_session_factory
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.paused = False  # For pausing during VRAM-intensive operations
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background extraction worker."""
        if self.running:
            logger.warning("Background extraction manager already running")
            return
        
        self.running = True
        self._worker_task = asyncio.create_task(self._worker())
        print(">>> Background extraction worker STARTED <<<")
        logger.info("Background extraction manager started")
    
    async def stop(self):
        """Stop the background extraction worker."""
        if not self.running:
            return
        
        self.running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background extraction manager stopped")
    
    async def pause(self):
        """Pause extraction processing (e.g., during image generation to prevent VRAM conflicts)."""
        if not self.running:
            logger.warning("Cannot pause: extraction manager not running")
            return
        
        self.paused = True
        logger.info("[EXTRACTION PAUSE] Background extraction PAUSED for VRAM-intensive operation")
        print(">>> Background extraction PAUSED <<<")
    
    async def resume(self):
        """Resume extraction processing after VRAM-intensive operation completes."""
        if not self.running:
            logger.warning("Cannot resume: extraction manager not running")
            return
        
        self.paused = False
        logger.info("[EXTRACTION RESUME] Background extraction RESUMED")
        print(">>> Background extraction RESUMED <<<")
    
    async def queue_extraction(
        self,
        conversation_id: str,
        character_id: str,
        messages: List[Message],
        model: Optional[str] = None,
        character_name: Optional[str] = None
    ):
        """
        Queue messages for extraction.
        
        Args:
            conversation_id: Conversation ID
            character_id: Character ID
            messages: Messages to analyze for extraction (should already be filtered to exclude private messages)
            model: Model to use for extraction (character's preferred model)
            character_name: Name of the character (for prompt context)
        """
        # Note: Privacy filtering is now done at message level before calling this method
        # Messages passed here should already exclude private messages
        
        # Create and queue task
        task = ExtractionTask(
            conversation_id=conversation_id,
            character_id=character_id,
            messages=messages,
            model=model,
            character_name=character_name
        )
        
        await self.queue.put(task)
        logger.debug(f"Queued extraction task for conversation {conversation_id} with {len(messages)} messages")
    
    def _should_skip_extraction(self, conversation_id: str) -> bool:
        """
        Check if extraction should be skipped for this conversation.
        
        Args:
            conversation_id: Conversation ID to check
        
        Returns:
            True if extraction should be skipped
        """
        # Check if conversation is marked as private
        return self.conversation_repo.is_private(conversation_id)
    
    async def _worker(self):
        """
        Background worker that processes extraction queue.
        
        Runs continuously while manager is active.
        """
        print(">>> Background extraction worker RUNNING <<<")
        logger.info("Background extraction worker started")
        
        while self.running:
            try:
                # Check if paused (e.g., during image generation)
                if self.paused:
                    await asyncio.sleep(0.5)  # Check every 500ms
                    continue
                
                # Wait for task with timeout to allow checking running/paused flags
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Double-check pause state before processing
                # (in case pause happened between queue.get() and here)
                if self.paused:
                    # Put task back in queue and wait
                    await self.queue.put(task)
                    await asyncio.sleep(0.5)
                    continue
                
                # Process extraction
                await self._process_task(task)
                
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                # No task available, continue waiting
                continue
            except asyncio.CancelledError:
                # Worker is being stopped
                logger.info("Background extraction worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in extraction worker: {e}", exc_info=True)
                continue
        
        logger.info("Background extraction worker stopped")
    
    async def _process_task(self, task: ExtractionTask):
        """
        Process a single extraction task.
        
        Args:
            task: Extraction task to process
        """
        # Create a fresh database session for this task
        if self.db_session_factory:
            db_session = next(self.db_session_factory())
        else:
            # Fallback to existing service (for backwards compatibility)
            db_session = None
        
        try:
            print(f">>> Processing extraction for conversation {task.conversation_id} <<<")
            logger.debug(f"Processing extraction for conversation {task.conversation_id}")
            
            # Use existing service or create fresh one with new session
            if db_session:
                from chorus_engine.repositories import MemoryRepository
                from chorus_engine.services.memory_extraction import MemoryExtractionService
                
                memory_repo = MemoryRepository(db_session)
                extraction_service = MemoryExtractionService(
                    llm_client=self.extraction_service.llm,
                    memory_repository=memory_repo,
                    vector_store=self.extraction_service.vector_store,
                    embedding_service=self.extraction_service.embedding_service
                )
            else:
                extraction_service = self.extraction_service
            
            # Extract memories
            print(f">>> Calling extract_from_messages with {len(task.messages)} messages <<<")
            extracted_memories = await extraction_service.extract_from_messages(
                messages=task.messages,
                character_id=task.character_id,
                conversation_id=task.conversation_id,
                model=task.model,  # Pass character's model
                character_name=task.character_name  # Pass character name for context
            )
            print(f">>> extract_from_messages returned {len(extracted_memories) if extracted_memories else 0} memories <<<")
            
            if not extracted_memories:
                print(f">>> No memories extracted, returning <<<")
                logger.debug(f"No memories extracted from conversation {task.conversation_id}")
                return
            
            # Save each extracted memory
            saved_count = 0
            print(f">>> Saving {len(extracted_memories)} extracted memories <<<")
            for extracted in extracted_memories:
                memory = await extraction_service.save_extracted_memory(
                    extracted=extracted,
                    character_id=task.character_id,
                    conversation_id=task.conversation_id
                )
                
                if memory:
                    saved_count += 1
                    print(f">>> Saved memory: {memory.content[:50]}... <<<")
            
            # Ensure changes are committed and visible
            if db_session:
                db_session.commit()
                print(f">>> Database session committed <<<")
            
            print(f">>> Extraction complete: {len(extracted_memories)} extracted, {saved_count} saved <<<")
            logger.info(
                f"Extraction complete for conversation {task.conversation_id}: "
                f"{len(extracted_memories)} extracted, {saved_count} saved"
            )
            
        except Exception as e:
            print(f">>> EXCEPTION in _process_task: {e} <<<")
            logger.error(f"Failed to process extraction task: {e}", exc_info=True)
            if db_session:
                db_session.rollback()
        finally:
            # Close the session
            if db_session:
                db_session.close()
                print(f">>> Database session closed <<<")
    
    async def wait_for_queue_empty(self, timeout: float = 10.0):
        """
        Wait for extraction queue to be empty.
        
        Useful for testing or graceful shutdown.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for extraction queue to empty")
    
    def get_queue_size(self) -> int:
        """Get current size of extraction queue."""
        return self.queue.qsize()
