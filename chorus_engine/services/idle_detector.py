"""
Idle Detector Service.

Tracks system activity to determine when the server is idle and safe
to run background processing tasks. Considers both user activity
(API requests) and active workloads (LLM calls, ComfyUI jobs).

Usage:
    idle_detector = IdleDetector(idle_threshold_minutes=5)
    
    # Record activity on each request (via middleware)
    idle_detector.record_activity()
    
    # Track active workloads
    with idle_detector.llm_call():
        # LLM generation here
        pass
    
    # Check if safe to run background tasks
    if idle_detector.is_idle():
        # Run background processing
        pass
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IdleDetectorConfig:
    """Configuration for IdleDetector."""
    idle_threshold_minutes: float = 5.0  # Minutes of inactivity before considered idle
    
    # Grace period after activity resumes before interrupting background tasks
    # This prevents the heartbeat from being constantly interrupted by polling
    resume_grace_seconds: float = 2.0


class IdleDetector:
    """
    Tracks system activity to determine when background processing is safe.
    
    Idle conditions (ALL must be true):
    1. No API requests for `idle_threshold_minutes`
    2. No active LLM calls
    3. No active ComfyUI jobs
    
    Thread-safe for concurrent access from multiple requests.
    """
    
    def __init__(
        self,
        idle_threshold_minutes: float = 5.0,
        resume_grace_seconds: float = 2.0
    ):
        """
        Initialize idle detector.
        
        Args:
            idle_threshold_minutes: Minutes of inactivity before considered idle
            resume_grace_seconds: Grace period after activity before interrupting tasks
        """
        self.config = IdleDetectorConfig(
            idle_threshold_minutes=idle_threshold_minutes,
            resume_grace_seconds=resume_grace_seconds
        )
        
        # Activity tracking
        self._last_activity: datetime = datetime.utcnow()
        self._activity_lock = Lock()
        
        # Workload tracking
        self._active_llm_calls: int = 0
        self._active_comfy_jobs: int = 0
        self._workload_lock = Lock()
        
        # Paths to exclude from activity tracking (polling/monitoring endpoints)
        self._excluded_paths = {
            "/health",
            "/heartbeat/status",
            "/heartbeat/pause",
            "/heartbeat/resume",
            "/heartbeat/queue-stale",
            "/heartbeat/queue",
            "/characters/{character_id}/memory-stats",
            "/characters/{character_id}/pending-memories",
        }
        
        logger.info(
            f"IdleDetector initialized: threshold={idle_threshold_minutes}min, "
            f"grace={resume_grace_seconds}s"
        )
    
    def record_activity(self, path: Optional[str] = None) -> None:
        """
        Record user activity. Called on each API request.
        
        Args:
            path: Optional request path for filtering (exclude polling endpoints)
        """
        # Skip excluded paths (polling, health checks)
        if path:
            # Normalize path for comparison
            normalized = self._normalize_path(path)
            if normalized in self._excluded_paths:
                return
        
        with self._activity_lock:
            self._last_activity = datetime.utcnow()
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path by replacing UUIDs/IDs with placeholders.
        
        Examples:
            /characters/nova_custom/memory-stats -> /characters/{character_id}/memory-stats
        """
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path
        )
        
        # Replace character IDs (alphanumeric with underscores)
        # Pattern: /characters/[word]/ where word contains letters/numbers/underscores
        path = re.sub(
            r'/characters/[a-zA-Z0-9_-]+/',
            '/characters/{character_id}/',
            path
        )
        
        return path
    
    def is_idle(self) -> bool:
        """
        Check if system is idle and safe for background processing.
        
        Returns:
            True if ALL conditions met:
            - No recent activity (threshold exceeded)
            - No active LLM calls
            - No active ComfyUI jobs
        """
        # Check active workloads first (fast, no time calculation)
        with self._workload_lock:
            if self._active_llm_calls > 0:
                return False
            if self._active_comfy_jobs > 0:
                return False
        
        # Check activity threshold
        with self._activity_lock:
            idle_duration = datetime.utcnow() - self._last_activity
            threshold = timedelta(minutes=self.config.idle_threshold_minutes)
            
            if idle_duration < threshold:
                return False
        
        return True
    
    def should_interrupt_background(self) -> bool:
        """
        Check if background processing should be interrupted.
        
        This is different from is_idle() - it checks if activity has resumed
        AFTER we started processing. Uses a shorter grace period to avoid
        interrupting for brief polling requests.
        
        Returns:
            True if user activity has resumed and grace period exceeded
        """
        with self._workload_lock:
            # Always interrupt if a user-initiated workload started
            if self._active_llm_calls > 0 or self._active_comfy_jobs > 0:
                return True
        
        with self._activity_lock:
            idle_duration = datetime.utcnow() - self._last_activity
            grace = timedelta(seconds=self.config.resume_grace_seconds)
            
            # If activity within grace period, interrupt
            return idle_duration < grace
    
    def get_idle_duration_seconds(self) -> float:
        """Get seconds since last activity."""
        with self._activity_lock:
            return (datetime.utcnow() - self._last_activity).total_seconds()
    
    # =========================================================================
    # Workload Tracking
    # =========================================================================
    
    def increment_llm_calls(self) -> None:
        """Record start of an LLM call."""
        with self._workload_lock:
            self._active_llm_calls += 1
            logger.debug(f"LLM call started (active: {self._active_llm_calls})")
    
    def decrement_llm_calls(self) -> None:
        """Record end of an LLM call."""
        with self._workload_lock:
            self._active_llm_calls = max(0, self._active_llm_calls - 1)
            logger.debug(f"LLM call ended (active: {self._active_llm_calls})")
    
    def increment_comfy_jobs(self) -> None:
        """Record start of a ComfyUI job."""
        with self._workload_lock:
            self._active_comfy_jobs += 1
            logger.debug(f"ComfyUI job started (active: {self._active_comfy_jobs})")
    
    def decrement_comfy_jobs(self) -> None:
        """Record end of a ComfyUI job."""
        with self._workload_lock:
            self._active_comfy_jobs = max(0, self._active_comfy_jobs - 1)
            logger.debug(f"ComfyUI job ended (active: {self._active_comfy_jobs})")
    
    @contextmanager
    def llm_call(self):
        """
        Context manager for tracking LLM calls.
        
        Usage:
            with idle_detector.llm_call():
                # LLM generation here
                pass
        """
        self.increment_llm_calls()
        try:
            yield
        finally:
            self.decrement_llm_calls()
    
    @contextmanager
    def comfy_job(self):
        """
        Context manager for tracking ComfyUI jobs.
        
        Usage:
            with idle_detector.comfy_job():
                # ComfyUI workflow here
                pass
        """
        self.increment_comfy_jobs()
        try:
            yield
        finally:
            self.decrement_comfy_jobs()
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_status(self) -> dict:
        """
        Get current idle detector status.
        
        Returns:
            Dict with status information for API/debugging
        """
        with self._activity_lock:
            idle_seconds = (datetime.utcnow() - self._last_activity).total_seconds()
            last_activity_iso = self._last_activity.isoformat()
        
        with self._workload_lock:
            llm_calls = self._active_llm_calls
            comfy_jobs = self._active_comfy_jobs
        
        return {
            "is_idle": self.is_idle(),
            "idle_seconds": round(idle_seconds, 1),
            "idle_threshold_minutes": self.config.idle_threshold_minutes,
            "last_activity": last_activity_iso,
            "active_llm_calls": llm_calls,
            "active_comfy_jobs": comfy_jobs,
        }
