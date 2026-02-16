"""
Continuity bootstrap task handler for Heartbeat System.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from chorus_engine.ens.models import SignalEnvelope
from chorus_engine.ens.runtime import ENSContext
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

        characters = app_state.get("characters", {})
        continuity_service = app_state.get("continuity_service")
        ens_runtime = app_state.get("ens_runtime")
        ens_cfg = getattr(app_state.get("system_config"), "ens", None)
        slice3_owned = bool(
            ens_cfg
            and getattr(ens_cfg, "enabled", False)
            and getattr(ens_cfg, "slice3_continuity_writes_ownership", False)
        )
        if not slice3_owned and not continuity_service:
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
            if slice3_owned:
                if not ens_runtime:
                    return TaskResult(
                        success=False,
                        task_id=task.id,
                        task_type=self.task_type,
                        duration_seconds=0,
                        error="ENS runtime not available",
                    )
                signal = SignalEnvelope(
                    type="continuity.bootstrap_requested",
                    scope="ASSISTANT",
                    source="external",
                    assistant_id=character_id,
                    payload={
                        "character_id": character_id,
                        "conversation_id": None,
                        "force": False,
                    },
                )
                outcome = await ens_runtime.ingest(
                    signal,
                    ENSContext(app_state=app_state, surface="web", source="web"),
                )
                output = dict(outcome.response_payload or {})
                result = {
                    "skipped": bool(output.get("skipped")),
                    "via_ens": True,
                }
            else:
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
                        "skipped": bool(result.get("skipped")),
                        "via_ens": bool(result.get("via_ens", False)),
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
