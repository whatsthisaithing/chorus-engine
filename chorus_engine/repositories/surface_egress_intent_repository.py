"""Repository for ENS surface egress outbox intents."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chorus_engine.models.ens import SurfaceEgressIntent


TERMINAL_STATUSES = {"delivered", "failed", "canceled"}


class SurfaceEgressIntentRepository:
    """CRUD/status helpers for surface outbox intents."""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def normalize_surface_instance_id(surface_instance_id: Optional[str]) -> str:
        return str(surface_instance_id or "").strip()

    def get_by_id(self, intent_id: str) -> Optional[SurfaceEgressIntent]:
        return self.db.query(SurfaceEgressIntent).filter(SurfaceEgressIntent.id == intent_id).first()

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[SurfaceEgressIntent]:
        return (
            self.db.query(SurfaceEgressIntent)
            .filter(SurfaceEgressIntent.idempotency_key == idempotency_key)
            .first()
        )

    def create_or_replay(
        self,
        *,
        surface_id: str,
        surface_instance_id: Optional[str],
        external_thread_id: str,
        relationship_id: Optional[str],
        conversation_id: Optional[str],
        thread_id: Optional[str],
        in_reply_to_message_id: Optional[str],
        payload_json: Dict[str, Any],
        idempotency_key: str,
        trace_json: Optional[Dict[str, Any]],
    ) -> Tuple[SurfaceEgressIntent, bool]:
        """Create new intent or replay existing one by idempotency key."""
        normalized_instance_id = self.normalize_surface_instance_id(surface_instance_id)
        row = SurfaceEgressIntent(
            surface_id=surface_id,
            surface_instance_id=normalized_instance_id,
            external_thread_id=external_thread_id,
            relationship_id=relationship_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            in_reply_to_message_id=in_reply_to_message_id,
            payload_json=payload_json or {},
            status="pending",
            attempt_count=0,
            idempotency_key=idempotency_key,
            trace_json=trace_json,
        )
        self.db.add(row)
        try:
            self.db.commit()
            self.db.refresh(row)
            return row, True
        except IntegrityError:
            self.db.rollback()
            existing = self.get_by_idempotency_key(idempotency_key)
            if not existing:
                raise
            return existing, False

    def list_intents(
        self,
        *,
        surface_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[SurfaceEgressIntent]:
        q = self.db.query(SurfaceEgressIntent)
        if surface_id:
            q = q.filter(SurfaceEgressIntent.surface_id == surface_id)
        if status:
            q = q.filter(SurfaceEgressIntent.status == status)
        return q.order_by(SurfaceEgressIntent.created_at.desc()).limit(max(1, min(limit, 500))).all()

    def ack_delivered(self, intent_id: str, *, metadata: Optional[Dict[str, Any]] = None) -> Optional[SurfaceEgressIntent]:
        row = self.get_by_id(intent_id)
        if not row:
            return None
        if row.status == "delivered":
            return row
        if row.status in TERMINAL_STATUSES:
            return row
        row.status = "delivered"
        row.updated_at = datetime.utcnow()
        trace = dict(row.trace_json or {})
        if metadata:
            trace["delivery"] = metadata
        row.trace_json = trace
        self.db.commit()
        self.db.refresh(row)
        return row

    def mark_failed(
        self,
        intent_id: str,
        *,
        error: str,
        retry_at: Optional[datetime] = None,
    ) -> Optional[SurfaceEgressIntent]:
        row = self.get_by_id(intent_id)
        if not row:
            return None
        if row.status in TERMINAL_STATUSES:
            return row
        row.status = "failed"
        row.attempt_count = int(row.attempt_count or 0) + 1
        row.last_error = str(error or "unknown_error")
        row.next_attempt_at = retry_at
        row.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(row)
        return row
