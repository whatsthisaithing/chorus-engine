"""
Continuity bootstrap task handler for Heartbeat System.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from chorus_engine.services.heartbeat_service import (
    BackgroundTaskHandler, BackgroundTask, TaskResult
)

logger = logging.getLogger(__name__)


class ContinuityBootstrapTaskHandler(BackgroundTaskHandler):
    """Task handler for generating continuity bootstraps during idle time."""

    @property
    def task_type(self) -> str:
        return "continuity_bootstrap"

    async def execute(self, task: BackgroundTask, app_state: Dict[str, Any]) -> TaskResult:
        start_time = datetime.utcnow()
        character_id = task.data.get("character_id")

        if not character_id:
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=0,
                error="Missing character_id in task data"
            )

        continuity_service = app_state.get("continuity_service")
        characters = app_state.get("characters", {})
        if not continuity_service:
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=0,
                error="Continuity service not available"
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

        try:
            result = await continuity_service.generate_and_save(
                character=character,
                conversation_id=None,
                force=False
            )
            duration = (datetime.utcnow() - start_time).total_seconds()
            if result:
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=duration,
                    data={
                        "character_id": character_id,
                        "skipped": bool(result.get("skipped"))
                    }
                )
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=duration,
                error="Continuity generation returned no result"
            )
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"[CONTINUITY TASK] Error: {e}", exc_info=True)
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=duration,
                error=str(e)
            )
