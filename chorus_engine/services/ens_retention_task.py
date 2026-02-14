"""Heartbeat maintenance task for ENS retention cleanup."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from chorus_engine.services.heartbeat_service import BackgroundTask, BackgroundTaskHandler, TaskResult
from chorus_engine.models.ens import ENSActionResult, ENSDecision

logger = logging.getLogger(__name__)


class ENSRetentionTaskHandler(BackgroundTaskHandler):
    """Performs SQL and JSONL retention cleanup with bounded work per run."""

    _mutex = asyncio.Lock()
    _mutex_key = "global:ens_retention"

    @property
    def task_type(self) -> str:
        return "ens_retention"

    async def execute(self, task: BackgroundTask, app_state: Dict[str, Any]) -> TaskResult:
        started = datetime.utcnow()
        config = task.data or {}
        max_rows = int(config.get("max_rows_per_run", 500))
        max_seconds = float(config.get("max_seconds_per_run", 2.5))
        sql_days = int(config.get("sql_retention_days", 30))
        jsonl_days = int(config.get("jsonl_retention_days", 14))
        compress_on_rotation = bool(config.get("compress_on_rotation", True))
        max_jsonl_size_mb = int(config.get("max_jsonl_size_mb", 25))

        if self._mutex.locked():
            logger.info("ens.retention.skipped reason=mutex_locked mutex=%s", self._mutex_key)
            return TaskResult(
                success=True,
                task_id=task.id,
                task_type=self.task_type,
                duration_seconds=0.0,
                data={"skipped": True, "reason": "mutex_locked"},
            )

        async with self._mutex:
            db = app_state.get("db_session")
            if db is None:
                logger.info("ens.retention.skipped reason=no_db_session")
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=0.0,
                    data={"skipped": True, "reason": "no_db_session"},
                )

            idle_detector = app_state.get("idle_detector")
            if idle_detector and not idle_detector.is_idle():
                logger.info("ens.retention.skipped reason=not_idle")
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=0.0,
                    data={"skipped": True, "reason": "not_idle"},
                )

            run_started = time.monotonic()
            deleted = {"ens_decisions": 0, "ens_action_results": 0, "jsonl_deleted": 0}

            try:
                sql_cutoff = datetime.utcnow() - timedelta(days=sql_days)
                deleted["ens_action_results"] += self._delete_bounded(
                    db, ENSActionResult, ENSActionResult.created_at, sql_cutoff, max_rows, run_started, max_seconds
                )
                deleted["ens_decisions"] += self._delete_bounded(
                    db, ENSDecision, ENSDecision.created_at, sql_cutoff, max_rows, run_started, max_seconds
                )
                deleted["jsonl_deleted"] += self._cleanup_jsonl(
                    retention_days=jsonl_days,
                    compress_on_rotation=compress_on_rotation,
                    max_jsonl_size_mb=max_jsonl_size_mb,
                )

                duration = (datetime.utcnow() - started).total_seconds()
                logger.info(
                    "ens.retention.run deleted_counts=%s mutex=%s duration=%.2fs",
                    json.dumps(deleted),
                    self._mutex_key,
                    duration,
                )
                logger.info("ens.retention.deleted_counts %s", json.dumps(deleted))
                return TaskResult(
                    success=True,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=duration,
                    data={"deleted_counts": deleted},
                )
            except Exception as e:
                db.rollback()
                logger.info("ens.retention.skipped reason=error error=%s", e)
                return TaskResult(
                    success=False,
                    task_id=task.id,
                    task_type=self.task_type,
                    duration_seconds=(datetime.utcnow() - started).total_seconds(),
                    error=str(e),
                )

    def _delete_bounded(self, db, model, ts_col, cutoff: datetime, max_rows: int, run_started: float, max_seconds: float) -> int:
        deleted = 0
        while deleted < max_rows and (time.monotonic() - run_started) < max_seconds:
            ids = (
                db.query(model)
                .filter(ts_col < cutoff)
                .order_by(ts_col.asc())
                .limit(min(100, max_rows - deleted))
                .all()
            )
            if not ids:
                break
            for row in ids:
                db.delete(row)
                deleted += 1
            db.commit()
        return deleted

    def _cleanup_jsonl(self, retention_days: int, compress_on_rotation: bool, max_jsonl_size_mb: int) -> int:
        root = Path("data/debug_logs/ens")
        if not root.exists():
            return 0
        active_files = {
            root / "decisions.jsonl",
            root / "action_results.jsonl",
        }
        now = datetime.utcnow()
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        deleted = 0

        for active_path in active_files:
            self._rotate_active_if_needed(
                active_path=active_path,
                now=now,
                compress_on_rotation=compress_on_rotation,
                max_jsonl_size_mb=max_jsonl_size_mb,
            )

        for path in root.glob("*"):
            if path in active_files:
                continue
            if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
                continue
            mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
            if mtime < cutoff:
                path.unlink(missing_ok=True)
                deleted += 1
        return deleted

    def _rotate_active_if_needed(
        self,
        *,
        active_path: Path,
        now: datetime,
        compress_on_rotation: bool,
        max_jsonl_size_mb: int,
    ) -> None:
        if not active_path.exists():
            return
        stat = active_path.stat()
        if stat.st_size <= 0:
            return

        current_day = now.strftime("%Y%m%d")
        file_day = datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y%m%d")
        size_limit_bytes = max_jsonl_size_mb * 1024 * 1024
        should_rotate = file_day != current_day or stat.st_size >= size_limit_bytes
        if not should_rotate:
            return

        rotation_suffix = now.strftime("%Y%m%d_%H%M%S")
        rotated = active_path.parent / f"{active_path.stem}.{rotation_suffix}.jsonl"
        shutil.move(str(active_path), str(rotated))
        active_path.touch()

        if compress_on_rotation:
            gz_path = Path(f"{rotated}.gz")
            with rotated.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            rotated.unlink(missing_ok=True)
