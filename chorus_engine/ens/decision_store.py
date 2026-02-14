"""Decision and action-result persistence for ENS."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from chorus_engine.models.ens import ENSDecision, ENSActionResult

logger = logging.getLogger(__name__)


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
