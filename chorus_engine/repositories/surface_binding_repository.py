"""Repository for surface binding routing records."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chorus_engine.models.ens import SurfaceBinding


class SurfaceBindingRepository:
    """CRUD helpers for canonical surface bindings."""

    OWNER_USER_ID = "user:local:owner"

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _normalize_surface_instance_id(surface_instance_id: Optional[str]) -> str:
        return str(surface_instance_id or "").strip()

    def get_by_lookup(
        self,
        *,
        surface_id: str,
        external_thread_id: str,
        surface_instance_id: Optional[str] = None,
    ) -> Optional[SurfaceBinding]:
        normalized_instance_id = self._normalize_surface_instance_id(surface_instance_id)
        return (
            self.db.query(SurfaceBinding)
            .filter(
                SurfaceBinding.surface_id == surface_id,
                SurfaceBinding.external_thread_id == external_thread_id,
                SurfaceBinding.surface_instance_id == normalized_instance_id,
            )
            .first()
        )

    def touch(self, binding: SurfaceBinding) -> SurfaceBinding:
        binding.last_seen_at = datetime.utcnow()
        binding.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(binding)
        return binding

    def create_with_retry(
        self,
        *,
        surface_id: str,
        external_thread_id: str,
        conversation_id: str,
        thread_id: str,
        surface_instance_id: Optional[str] = None,
        relationship_id: Optional[str] = None,
        owner_user_id: Optional[str] = None,
    ) -> SurfaceBinding:
        """Insert binding with unique-conflict retry for concurrent creators."""
        normalized_instance_id = self._normalize_surface_instance_id(surface_instance_id)
        binding = SurfaceBinding(
            surface_id=surface_id,
            surface_instance_id=normalized_instance_id,
            external_thread_id=external_thread_id,
            relationship_id=relationship_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            owner_user_id=owner_user_id or self.OWNER_USER_ID,
            last_seen_at=datetime.utcnow(),
        )
        self.db.add(binding)
        try:
            self.db.commit()
            self.db.refresh(binding)
            return binding
        except IntegrityError:
            self.db.rollback()
            existing = self.get_by_lookup(
                surface_id=surface_id,
                external_thread_id=external_thread_id,
                surface_instance_id=normalized_instance_id,
            )
            if existing:
                return self.touch(existing)
            raise
