"""Heartbeat task handler for scheduled character backups."""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Conversation, Memory, Message, Thread
from chorus_engine.models.continuity import CharacterBackupState
from chorus_engine.services.backup_service import CharacterBackupService
from chorus_engine.services.heartbeat_service import (
    BackgroundTask,
    BackgroundTaskHandler,
    TaskResult,
    TaskPriority,
    TaskStatus,
)
from chorus_engine.db.database import SessionLocal

logger = logging.getLogger(__name__)


@dataclass
class BackupDecision:
    due: bool
    reason: str
    scheduled_for: Optional[datetime] = None


class CharacterBackupTaskHandler(BackgroundTaskHandler):
    """Run scheduled character backups for heartbeat-queued tasks."""

    @property
    def task_type(self) -> str:
        return "character_backup"

    async def execute(self, task: BackgroundTask, app_state: Dict[str, Any]) -> TaskResult:
        started = datetime.utcnow()
        character_id = task.data.get("character_id")
        if not character_id:
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=0.0,
                error="Missing character_id",
            )

        db: Session = SessionLocal()
        try:
            characters = app_state.get("characters", {})
            character = characters.get(character_id)
            if not character:
                return TaskResult(
                    success=False,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=0.0,
                    error=f"Character '{character_id}' not loaded",
                )

            system_config = app_state.get("system_config")
            heartbeat_backups = getattr(getattr(system_config, "heartbeat", None), "backups", None)
            if not heartbeat_backups or not heartbeat_backups.enabled:
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=0.0,
                    data={"skipped": True, "reason": "heartbeat.backups disabled"},
                )

            destination = character.backup.destination_override or heartbeat_backups.destination_dir
            if destination is None:
                raise RuntimeError("No destination configured for scheduled backups")

            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)

            state = db.query(CharacterBackupState).filter(CharacterBackupState.character_id == character_id).first()
            if not state:
                state = CharacterBackupState(character_id=character_id, last_status="never")
                db.add(state)
                db.flush()

            fingerprint = compute_character_backup_fingerprint(db, character_id)
            if heartbeat_backups.skip_if_unchanged and state.last_manifest_fingerprint == fingerprint:
                state.last_attempt_at = datetime.utcnow()
                state.last_status = "skipped_unchanged"
                db.commit()
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=(datetime.utcnow() - started).total_seconds(),
                    data={"skipped": True, "reason": "unchanged"},
                )

            backup_service = CharacterBackupService(db=db, backup_dir=destination)
            backup_path = backup_service.backup_character(
                character_id=character_id,
                include_workflows=character.backup.include_workflows,
                notes=character.backup.notes_template,
            )

            if heartbeat_backups.verify_archive:
                with zipfile.ZipFile(backup_path, "r") as zip_file:
                    bad_member = zip_file.testzip()
                    if bad_member:
                        raise RuntimeError(f"Archive integrity failed at member: {bad_member}")

            prune_backups(destination / character_id, heartbeat_backups.retention_days)

            now = datetime.utcnow()
            state.last_attempt_at = now
            state.last_success_at = now
            state.last_status = "success"
            state.last_error = None
            state.last_manifest_fingerprint = fingerprint
            db.commit()

            return TaskResult(
                success=True,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=(datetime.utcnow() - started).total_seconds(),
                data={"character_id": character_id, "backup_path": str(backup_path)},
            )
        except Exception as e:
            logger.error(f"[BACKUP TASK] Backup failed for '{character_id}': {e}", exc_info=True)
            try:
                state = db.query(CharacterBackupState).filter(CharacterBackupState.character_id == character_id).first()
                if state:
                    state.last_attempt_at = datetime.utcnow()
                    state.last_status = "failed"
                    state.last_error = str(e)
                    db.commit()
            except Exception:
                db.rollback()
            return TaskResult(
                success=False,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=(datetime.utcnow() - started).total_seconds(),
                error=str(e),
            )
        finally:
            db.close()


def compute_character_backup_fingerprint(db: Session, character_id: str) -> str:
    """Compute lightweight fingerprint used by skip-if-unchanged policy."""
    conversation_count = db.query(Conversation).filter(Conversation.character_id == character_id).count()
    memory_count = db.query(Memory).filter(Memory.character_id == character_id).count()

    latest_conversation = db.query(func.max(Conversation.updated_at)).filter(
        Conversation.character_id == character_id
    ).scalar()
    latest_memory = db.query(func.max(Memory.created_at)).filter(
        Memory.character_id == character_id
    ).scalar()
    latest_message = (
        db.query(func.max(Message.created_at))
        .join(Thread, Thread.id == Message.thread_id)
        .join(Conversation, Conversation.id == Thread.conversation_id)
        .filter(Conversation.character_id == character_id)
        .scalar()
    )

    raw = "|".join([
        character_id,
        str(conversation_count),
        str(memory_count),
        latest_conversation.isoformat() if latest_conversation else "",
        latest_memory.isoformat() if latest_memory else "",
        latest_message.isoformat() if latest_message else "",
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prune_backups(character_backup_dir: Path, retention_days: int) -> None:
    if not character_backup_dir.exists():
        return
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    for archive in character_backup_dir.glob("*.zip"):
        try:
            mtime = datetime.utcfromtimestamp(archive.stat().st_mtime)
            if mtime < cutoff:
                archive.unlink()
        except Exception:
            logger.debug(f"Could not prune backup archive {archive}")


def evaluate_backup_due(character, state: Optional[CharacterBackupState], now_utc: datetime) -> BackupDecision:
    if not getattr(character, "backup", None) or not character.backup.enabled:
        return BackupDecision(due=False, reason="character backup disabled")
    if character.backup.schedule != "daily":
        return BackupDecision(due=False, reason="unsupported schedule")

    hour_text, minute_text = character.backup.local_time.split(":")
    scheduled_today = now_utc.replace(hour=int(hour_text), minute=int(minute_text), second=0, microsecond=0)
    if now_utc < scheduled_today:
        return BackupDecision(due=False, reason="before scheduled time", scheduled_for=scheduled_today)
    if state and state.last_success_at and state.last_success_at >= scheduled_today:
        return BackupDecision(due=False, reason="already backed up today", scheduled_for=scheduled_today)
    return BackupDecision(due=True, reason="due", scheduled_for=scheduled_today)


def queue_due_character_backups(
    heartbeat_service,
    characters: Dict[str, Any],
    db: Session,
    max_backups_per_cycle: int,
    global_destination: Optional[Path] = None,
) -> int:
    """Queue due daily character backup tasks and return count."""
    now = datetime.utcnow()
    queued = 0
    for character_id, character in characters.items():
        if queued >= max_backups_per_cycle:
            break
        state = db.query(CharacterBackupState).filter(CharacterBackupState.character_id == character_id).first()
        decision = evaluate_backup_due(character, state, now)
        if not decision.due:
            continue
        effective_destination = character.backup.destination_override or global_destination
        if effective_destination is None:
            continue
        # Do not queue if there is already a pending/running backup task for this character.
        existing_tasks = getattr(heartbeat_service, "_task_queue", [])
        already_queued = any(
            getattr(task, "task_type", None) == "character_backup"
            and isinstance(getattr(task, "data", None), dict)
            and task.data.get("character_id") == character_id
            and getattr(task, "status", None) in {TaskStatus.PENDING, TaskStatus.RUNNING}
            for task in existing_tasks
        )
        if already_queued:
            continue
        heartbeat_service.queue_task(
            task_type="character_backup",
            data={"character_id": character_id},
            priority=TaskPriority.LOW,
            task_id=f"character_backup_{character_id}",
        )
        queued += 1
    return queued
