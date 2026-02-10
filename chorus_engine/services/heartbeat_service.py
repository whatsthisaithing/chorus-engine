"""
HeartbeatService - Background processing manager for Chorus Engine.

This service runs a continuous background loop that processes tasks
during idle periods, ensuring it never interferes with active user
interactions or ongoing LLM/ComfyUI operations.

Phase D: Heartbeat System for Conversation Memory Enrichment
"""

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def get_gpu_utilization() -> Optional[int]:
    """
    Get current GPU compute utilization percentage using nvidia-smi.
    
    Returns:
        GPU utilization percentage (0-100), or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Handle multi-GPU: take the max utilization across all GPUs
            lines = result.stdout.strip().split("\n")
            utilizations = [int(line.strip()) for line in lines if line.strip().isdigit()]
            if utilizations:
                return max(utilizations)
    except FileNotFoundError:
        # nvidia-smi not available (no NVIDIA GPU or drivers)
        pass
    except subprocess.TimeoutExpired:
        logger.warning("[GPU CHECK] nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"[GPU CHECK] Error querying GPU utilization: {e}")
    return None


class TaskPriority(Enum):
    """Priority levels for background tasks."""
    CRITICAL = 1  # Must run ASAP (e.g., failed Discord messages)
    HIGH = 2      # Important but can wait for idle
    NORMAL = 3    # Standard background tasks
    LOW = 4       # Can be deferred indefinitely


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result from executing a background task."""
    success: bool
    task_id: str
    task_type: str
    duration_seconds: float
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class BackgroundTask:
    """Represents a task to be processed during idle time."""
    id: str
    task_type: str
    priority: TaskPriority
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: Optional[datetime] = None
    
    def __lt__(self, other):
        """Enable priority queue sorting."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class BackgroundTaskHandler(ABC):
    """Base class for background task handlers."""
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """The type of task this handler processes."""
        pass
    
    @abstractmethod
    async def execute(self, task: BackgroundTask, app_state: Dict[str, Any]) -> TaskResult:
        """Execute the task and return a result."""
        pass
    
    def should_retry(self, task: BackgroundTask, error: Exception) -> bool:
        """Determine if the task should be retried after an error."""
        return task.retry_count < task.max_retries


class HeartbeatService:
    """
    Background processing service that runs tasks during idle periods.
    
    The heartbeat continuously monitors system activity and processes
    queued tasks when no user interactions or heavy operations are ongoing.
    
    Key features:
    - Respects user activity (pauses when user is active)
    - Tracks LLM and ComfyUI operations (never interferes)
    - Priority-based task queue
    - Automatic pause/resume based on system state
    - Graceful shutdown support
    """
    
    def __init__(
        self,
        idle_detector,  # IdleDetector instance
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HeartbeatService.
        
        Args:
            idle_detector: IdleDetector instance for checking system idle state
            config: Optional configuration overrides (from HeartbeatConfig)
        """
        self.idle_detector = idle_detector
        
        # Configuration with defaults
        self._config = config or {}
        self.enabled = self._config.get("enabled", True)
        self.interval_seconds = self._config.get("interval_seconds", 60)
        self.idle_threshold_minutes = self._config.get("idle_threshold_minutes", 5)
        self.resume_grace_seconds = self._config.get("resume_grace_seconds", 2)
        self.batch_size = self._config.get("analysis_batch_size", 3)
        self.gpu_check_cooldown_seconds = self._config.get("gpu_check_cooldown_seconds", 0.5)
        
        # Task queue and handlers
        self._task_queue: List[BackgroundTask] = []
        self._handlers: Dict[str, BackgroundTaskHandler] = {}
        
        # Task finder callback (called when queue is empty to discover new tasks)
        self._task_finder: Optional[callable] = None
        
        # State tracking
        self._running = False
        self._paused = False
        self._current_task: Optional[BackgroundTask] = None
        self._task: Optional[asyncio.Task] = None
        self._app_state: Optional[Dict[str, Any]] = None
        self._last_task_completed_at: Optional[datetime] = None
        
        # Statistics
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_runtime_seconds": 0,
            "last_run": None,
            "last_task": None
        }
        
        logger.info(
            f"HeartbeatService initialized: enabled={self.enabled}, "
            f"interval={self.interval_seconds}s, idle_threshold={self.idle_threshold_minutes}min"
        )
    
    def register_handler(self, handler: BackgroundTaskHandler) -> None:
        """Register a task handler for a specific task type."""
        self._handlers[handler.task_type] = handler
        logger.info(f"[HEARTBEAT] Registered handler for task type: {handler.task_type}")
    
    def set_task_finder(self, finder: callable) -> None:
        """
        Set a callback function to discover tasks when queue is empty.
        
        The finder function should accept (heartbeat_service, app_state) and
        queue any discovered tasks directly.
        
        Args:
            finder: Callable that finds and queues tasks
        """
        self._task_finder = finder
        logger.info("[HEARTBEAT] Task finder registered")
    
    def queue_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None
    ) -> str:
        """
        Add a task to the processing queue.
        
        Args:
            task_type: Type of task (must have a registered handler)
            data: Task-specific data
            priority: Task priority level
            task_id: Optional custom task ID
            
        Returns:
            The task ID
        """
        import uuid
        
        if task_type not in self._handlers:
            logger.warning(f"[HEARTBEAT] No handler registered for task type: {task_type}")
        
        task = BackgroundTask(
            id=task_id or str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            data=data
        )
        
        self._task_queue.append(task)
        self._task_queue.sort()  # Maintain priority order
        
        logger.debug(f"[HEARTBEAT] Queued task {task.id} ({task_type}) with priority {priority.name}")
        return task.id
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        status = {
            "running": self._running,
            "paused": self._paused,
            "current_task": self._current_task.id if self._current_task else None,
            "current_task_type": self._current_task.task_type if self._current_task else None,
            "queue_length": len(self._task_queue),
            "pending_by_type": self._count_tasks_by_type(),
            "pending_by_priority": self._count_tasks_by_priority(),
            "stats": self._stats.copy(),
            "idle_status": self.idle_detector.get_status() if self.idle_detector else None
        }
        
        # Add GPU utilization if check is enabled
        if self._config.get("gpu_check_enabled", False):
            gpu_util = get_gpu_utilization()
            status["gpu_utilization"] = gpu_util
            status["gpu_max_threshold"] = self._config.get("gpu_max_utilization_percent", 15)
            status["gpu_check_enabled"] = True
        else:
            status["gpu_check_enabled"] = False
        
        return status
    
    def _count_tasks_by_type(self) -> Dict[str, int]:
        """Count pending tasks by type."""
        counts = {}
        for task in self._task_queue:
            if task.status == TaskStatus.PENDING:
                counts[task.task_type] = counts.get(task.task_type, 0) + 1
        return counts
    
    def _count_tasks_by_priority(self) -> Dict[str, int]:
        """Count pending tasks by priority."""
        counts = {}
        for task in self._task_queue:
            if task.status == TaskStatus.PENDING:
                name = task.priority.name
                counts[name] = counts.get(name, 0) + 1
        return counts
    
    async def start(self, app_state: Dict[str, Any]) -> None:
        """Start the heartbeat background loop."""
        if not self.enabled:
            logger.info("[HEARTBEAT] Service is disabled, not starting")
            return
        
        if self._running:
            logger.warning("[HEARTBEAT] Service already running")
            return
        
        self._app_state = app_state
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("[HEARTBEAT] Service started")
    
    async def stop(self) -> None:
        """Stop the heartbeat service gracefully."""
        if not self._running:
            return
        
        logger.info("[HEARTBEAT] Stopping service...")
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("[HEARTBEAT] Service stopped")
    
    def pause(self) -> None:
        """Pause background processing."""
        self._paused = True
        logger.info("[HEARTBEAT] Service paused")
    
    def resume(self) -> None:
        """Resume background processing."""
        self._paused = False
        logger.info("[HEARTBEAT] Service resumed")
    
    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop that processes tasks during idle periods."""
        logger.info("[HEARTBEAT] Heartbeat loop started")
        
        while self._running:
            try:
                # Wait for the configured interval
                await asyncio.sleep(self.interval_seconds)
                
                if not self._running:
                    break
                
                # Skip if paused
                if self._paused:
                    logger.debug("[HEARTBEAT] Paused, skipping cycle")
                    continue
                
                # Check if system is idle
                if not self._is_safe_to_process():
                    logger.debug("[HEARTBEAT] System not idle, skipping cycle")
                    continue
                
                # Process tasks if we have any
                if self._task_queue:
                    await self._process_batch()
                else:
                    # No tasks - try to discover new tasks
                    if self._task_finder:
                        try:
                            logger.debug("[HEARTBEAT] Queue empty, running task finder...")
                            self._task_finder(self, self._app_state)
                            
                            # If we found tasks, process them
                            if self._task_queue:
                                logger.info(f"[HEARTBEAT] Task finder queued {len(self._task_queue)} tasks")
                                await self._process_batch()
                            else:
                                logger.debug("[HEARTBEAT] Task finder found no tasks")
                        except Exception as e:
                            logger.error(f"[HEARTBEAT] Task finder error: {e}", exc_info=True)
                    else:
                        logger.debug("[HEARTBEAT] No tasks in queue and no task finder registered")
                    
            except asyncio.CancelledError:
                logger.info("[HEARTBEAT] Loop cancelled")
                break
            except Exception as e:
                logger.error(f"[HEARTBEAT] Error in heartbeat loop: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(5)
        
        logger.info("[HEARTBEAT] Heartbeat loop ended")
    
    def _is_safe_to_process(self) -> bool:
        """Check if it's safe to process background tasks."""
        if not self.idle_detector:
            return True
        
        # Check if system is idle (uses IdleDetector's configured thresholds)
        if not self.idle_detector.is_idle():
            return False
        
        # Double-check no active LLM or ComfyUI operations
        status = self.idle_detector.get_status()
        if status["active_llm_calls"] > 0:
            logger.debug(f"[HEARTBEAT] Active LLM calls: {status['active_llm_calls']}")
            return False
        
        if status["active_comfy_jobs"] > 0:
            logger.debug(f"[HEARTBEAT] Active ComfyUI jobs: {status['active_comfy_jobs']}")
            return False
        
        # Final check: GPU utilization (NVIDIA only, if enabled)
        if self._config.get("gpu_check_enabled", False):
            max_util = self._config.get("gpu_max_utilization_percent", 15)
            gpu_util = get_gpu_utilization()
            
            if gpu_util is not None and gpu_util > max_util:
                logger.info(
                    f"[HEARTBEAT] GPU utilization {gpu_util}% exceeds threshold {max_util}%, "
                    f"deferring background tasks (external GPU activity detected)"
                )
                return False
            elif gpu_util is not None:
                logger.debug(f"[HEARTBEAT] GPU utilization {gpu_util}% OK (threshold: {max_util}%)")
        
        return True
    
    async def _process_batch(self) -> None:
        """Process a batch of tasks."""
        start_time = datetime.utcnow()
        processed = 0
        
        logger.info(f"[HEARTBEAT] Starting batch processing ({len(self._task_queue)} tasks queued)")
        
        while processed < self.batch_size and self._task_queue and self._running and not self._paused:
            await self._apply_gpu_cooldown()
            # Check idle state before each task
            if not self._is_safe_to_process():
                logger.info(f"[HEARTBEAT] System became active, stopping batch after {processed} tasks")
                break
            
            # Get next pending task
            task = self._get_next_pending_task()
            if not task:
                break
            
            # Process the task
            result = await self._execute_task(task)
            processed += 1
            
            # Update statistics
            if result.success:
                self._stats["tasks_completed"] += 1
            else:
                self._stats["tasks_failed"] += 1
            
            self._stats["total_runtime_seconds"] += result.duration_seconds
            self._stats["last_task"] = task.task_type
            
            # Small delay between tasks
            await asyncio.sleep(0.5)
        
        self._stats["last_run"] = datetime.utcnow().isoformat()
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"[HEARTBEAT] Batch complete: processed {processed} tasks in {duration:.1f}s")
    
    def _get_next_pending_task(self) -> Optional[BackgroundTask]:
        """Get the next pending task from the queue."""
        for task in self._task_queue:
            if task.status == TaskStatus.PENDING:
                return task
        return None
    
    async def _execute_task(self, task: BackgroundTask) -> TaskResult:
        """Execute a single task."""
        start_time = datetime.utcnow()
        
        logger.info(f"[HEARTBEAT] Executing task {task.id} ({task.task_type})")
        
        # Mark task as running
        task.status = TaskStatus.RUNNING
        self._current_task = task
        
        try:
            # Get handler
            handler = self._handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            # Execute
            result = await handler.execute(task, self._app_state)
            
            # Mark complete
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Remove from queue
            self._task_queue.remove(task)
            
            logger.info(
                f"[HEARTBEAT] Task {task.id} completed in {result.duration_seconds:.1f}s"
            )
            self._last_task_completed_at = datetime.utcnow()
            
            return result
            
        except Exception as e:
            logger.error(f"[HEARTBEAT] Task {task.id} failed: {e}", exc_info=True)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Check if should retry
            handler = self._handlers.get(task.task_type)
            if handler and handler.should_retry(task, e):
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = str(e)
                logger.info(
                    f"[HEARTBEAT] Task {task.id} will retry (attempt {task.retry_count}/{task.max_retries})"
                )
            else:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.utcnow()
                # Remove failed task from queue
                self._task_queue.remove(task)
            
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=task.task_type,
                duration_seconds=duration,
                error=str(e)
            )
        
        finally:
            self._current_task = None

    async def _apply_gpu_cooldown(self) -> None:
        """Optional cooldown before checking GPU utilization after a task completes."""
        if not self._config.get("gpu_check_enabled", False):
            return
        if not self._last_task_completed_at:
            return
        cooldown = float(self.gpu_check_cooldown_seconds or 0)
        if cooldown <= 0:
            return
        elapsed = (datetime.utcnow() - self._last_task_completed_at).total_seconds()
        remaining = cooldown - elapsed
        if remaining > 0:
            logger.debug(f"[HEARTBEAT] GPU check cooldown: waiting {remaining:.2f}s")
            await asyncio.sleep(remaining)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        for task in self._task_queue:
            if task.id == task_id and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                self._task_queue.remove(task)
                logger.info(f"[HEARTBEAT] Task {task_id} cancelled")
                return True
        return False
    
    def clear_queue(self) -> int:
        """Clear all pending tasks from the queue."""
        count = len([t for t in self._task_queue if t.status == TaskStatus.PENDING])
        self._task_queue = [t for t in self._task_queue if t.status != TaskStatus.PENDING]
        logger.info(f"[HEARTBEAT] Cleared {count} pending tasks")
        return count
