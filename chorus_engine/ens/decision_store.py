"""Decision and action-result persistence for ENS."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from chorus_engine.ens.loop_memory_compression import canonical_hash, canonical_json
from chorus_engine.models.ens import (
    ENSActionResult,
    ENSDecision,
    ENSLoopCompressionArtifact,
    ENSLoopSession,
    ENSLoopStepEvent,
    ENSSchedulerTick,
)

logger = logging.getLogger(__name__)


@dataclass(eq=True)
class RunSignature:
    selected_sequence: List[str]
    reason_trace_hashes: List[str]
    loop_terminals: Dict[str, Dict[str, Any]]
    step_event_counts: Dict[str, int]
    compression_artifacts: Dict[str, List[Dict[str, Any]]]


class ENSDecisionStore:
    """Persists ENS observability artifacts to JSONL and SQL."""

    def __init__(self) -> None:
        self._decisions_log = Path("data/debug_logs/ens/decisions.jsonl")
        self._actions_log = Path("data/debug_logs/ens/action_results.jsonl")
        self._decisions_log.parent.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        db: Session,
        decision_doc: Dict[str, Any],
        action_result_docs: List[Dict[str, Any]],
        *,
        sql_action_results: List[Dict[str, Any]] | None = None,
    ) -> None:
        self._write_jsonl(decision_doc, action_result_docs)
        self._write_sql(db, decision_doc, sql_action_results or action_result_docs)

    def _write_jsonl(self, decision_doc: Dict[str, Any], action_result_docs: List[Dict[str, Any]]) -> None:
        try:
            with self._decisions_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(decision_doc, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("ens.persistence.partial.jsonl_failed decision=%s error=%s", decision_doc.get("decision_id"), e)

        for item in action_result_docs:
            try:
                with self._actions_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error("ens.persistence.partial.jsonl_failed action=%s error=%s", item.get("action_id"), e)

    def _write_sql(self, db: Session, decision_doc: Dict[str, Any], action_result_docs: List[Dict[str, Any]]) -> None:
        try:
            decision_row = ENSDecision(
                decision_id=decision_doc["decision_id"],
                trace_id=decision_doc.get("trace_id"),
                signal_id=decision_doc["signal_id"],
                session_id=decision_doc.get("session_id"),
                assistant_id=decision_doc.get("assistant_id"),
                user_id=decision_doc.get("user_id"),
                scope=decision_doc.get("scope", "SESSION"),
                signal_type=decision_doc.get("signal_type", "unknown"),
                appraisal_json=decision_doc.get("appraisal"),
                constraints_json=decision_doc.get("constraints", []),
                intent_proposals_json=decision_doc.get("intent_proposals", []),
                arbitration_json=decision_doc.get("arbitration"),
                actions_json=decision_doc.get("actions", []),
                explanation=decision_doc.get("explanation"),
                created_at=self._parse_ts(decision_doc.get("timestamp")),
            )
            db.add(decision_row)

            for item in action_result_docs:
                db.add(
                    ENSActionResult(
                        action_result_id=item["action_result_id"],
                        decision_id=item["decision_id"],
                        action_id=item["action_id"],
                        idempotency_key=item.get("idempotency_key"),
                        kind=item["kind"],
                        execution_class=item.get("execution_class", "user_facing"),
                        status=item["status"],
                        error_code=item.get("error_code"),
                        error_message=item.get("error_message"),
                        metrics_json=item.get("metrics"),
                        output_json=item.get("output"),
                        created_at=self._parse_ts(item.get("timestamp")),
                    )
                )
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("ens.persistence.partial.sql_failed decision=%s error=%s", decision_doc.get("decision_id"), e)

    @staticmethod
    def _parse_ts(value: str | None) -> datetime:
        if not value:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.utcnow()


def normalize_reason_trace(reason_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize reason trace for deterministic signature hashing."""
    doc = dict(reason_trace or {})
    stable_keys = {
        "selection",
        "phase",
        "priority_tier",
        "selected_priority_tier",
        "status",
        "stop_reason",
        "tier_counts",
        "total_candidates",
        "non_user_surface_counts",
        "fairness_rotation",
        "selected_signal_id",
    }
    filtered = {k: v for k, v in doc.items() if k in stable_keys}
    return json.loads(canonical_json(filtered))


def extract_run_signature_from_session(db: Session) -> RunSignature:
    ticks = (
        db.query(ENSSchedulerTick)
        .filter(ENSSchedulerTick.selected_signal_id.isnot(None))
        .order_by(ENSSchedulerTick.created_at_us.asc(), ENSSchedulerTick.tick_id.asc())
        .all()
    )
    selected_sequence = [str(row.selected_signal_id) for row in ticks if str(row.selected_signal_id or "").strip()]
    reason_trace_hashes = [canonical_hash(normalize_reason_trace(dict(row.reason_trace_json or {}))) for row in ticks]

    loops = db.query(ENSLoopSession).order_by(ENSLoopSession.loop_id.asc()).all()
    loop_terminals: Dict[str, Dict[str, Any]] = {
        str(row.loop_id): {
            "state": str(row.state or ""),
            "stop_reason": str(row.stop_reason or ""),
            "step_count": int(row.step_count or 0),
        }
        for row in loops
    }

    step_events = db.query(ENSLoopStepEvent.loop_id).all()
    step_event_counts: Dict[str, int] = {}
    for (loop_id,) in step_events:
        key = str(loop_id or "")
        if not key:
            continue
        step_event_counts[key] = step_event_counts.get(key, 0) + 1

    artifacts = (
        db.query(ENSLoopCompressionArtifact)
        .order_by(
            ENSLoopCompressionArtifact.loop_id.asc(),
            ENSLoopCompressionArtifact.to_step_index.asc(),
            ENSLoopCompressionArtifact.created_at_us.asc(),
        )
        .all()
    )
    compression_artifacts: Dict[str, List[Dict[str, Any]]] = {}
    for row in artifacts:
        loop_id = str(row.loop_id or "")
        if not loop_id:
            continue
        compression_artifacts.setdefault(loop_id, []).append(
            {
                "from": int(row.from_step_index),
                "to": int(row.to_step_index),
                "input_hash": str(row.input_hash),
                "output_hash": str(row.output_hash),
                "config_hash": str(row.config_hash),
            }
        )

    return RunSignature(
        selected_sequence=selected_sequence,
        reason_trace_hashes=reason_trace_hashes,
        loop_terminals=loop_terminals,
        step_event_counts=step_event_counts,
        compression_artifacts=compression_artifacts,
    )


def extract_run_signature(db_path: str) -> RunSignature:
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False, "timeout": 30}, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        return extract_run_signature_from_session(db)
    finally:
        db.close()
        engine.dispose()
