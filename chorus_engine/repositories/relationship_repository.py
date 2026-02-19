"""Repository for relationship and relationship-surface records."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chorus_engine.models.relationship import Relationship, RelationshipSurface


class RelationshipRepository:
    """CRUD helpers for relationship-first entities."""

    OWNER_USER_ID = "user:local:owner"

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def normalize_surface_instance_id(surface_instance_id: Optional[str]) -> str:
        return str(surface_instance_id or "").strip()

    def get_relationship(self, owner_user_id: str, character_id: str) -> Optional[Relationship]:
        return (
            self.db.query(Relationship)
            .filter(
                Relationship.owner_user_id == owner_user_id,
                Relationship.character_id == character_id,
            )
            .first()
        )

    def get_or_create_relationship(self, owner_user_id: str, character_id: str) -> Relationship:
        existing = self.get_relationship(owner_user_id, character_id)
        if existing:
            return existing

        row = Relationship(owner_user_id=owner_user_id, character_id=character_id)
        self.db.add(row)
        try:
            self.db.commit()
            self.db.refresh(row)
            return row
        except IntegrityError:
            self.db.rollback()
            existing = self.get_relationship(owner_user_id, character_id)
            if existing:
                return existing
            raise

    def get_surface(
        self,
        *,
        relationship_id: str,
        surface_id: str,
        surface_instance_id: Optional[str] = None,
    ) -> Optional[RelationshipSurface]:
        normalized_instance = self.normalize_surface_instance_id(surface_instance_id)
        return (
            self.db.query(RelationshipSurface)
            .filter(
                RelationshipSurface.relationship_id == relationship_id,
                RelationshipSurface.surface_id == surface_id,
                RelationshipSurface.surface_instance_id == normalized_instance,
            )
            .first()
        )

    def get_or_create_surface(
        self,
        *,
        relationship_id: str,
        surface_id: str,
        surface_instance_id: Optional[str] = None,
    ) -> RelationshipSurface:
        existing = self.get_surface(
            relationship_id=relationship_id,
            surface_id=surface_id,
            surface_instance_id=surface_instance_id,
        )
        if existing:
            return existing

        normalized_instance = self.normalize_surface_instance_id(surface_instance_id)
        row = RelationshipSurface(
            relationship_id=relationship_id,
            surface_id=surface_id,
            surface_instance_id=normalized_instance,
        )
        self.db.add(row)
        try:
            self.db.commit()
            self.db.refresh(row)
            return row
        except IntegrityError:
            self.db.rollback()
            existing = self.get_surface(
                relationship_id=relationship_id,
                surface_id=surface_id,
                surface_instance_id=normalized_instance,
            )
            if existing:
                return existing
            raise

    def set_general_conversation_id(self, relationship_surface_id: str, conversation_id: str) -> Optional[RelationshipSurface]:
        row = (
            self.db.query(RelationshipSurface)
            .filter(RelationshipSurface.id == relationship_surface_id)
            .first()
        )
        if not row:
            return None
        row.general_conversation_id = conversation_id
        row.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(row)
        return row

    def touch_interaction(
        self,
        *,
        relationship_id: str,
        surface_id: str,
        surface_instance_id: Optional[str] = None,
    ) -> Tuple[Optional[Relationship], Optional[RelationshipSurface]]:
        relationship = (
            self.db.query(Relationship)
            .filter(Relationship.id == relationship_id)
            .first()
        )
        surface = self.get_surface(
            relationship_id=relationship_id,
            surface_id=surface_id,
            surface_instance_id=surface_instance_id,
        )
        now = datetime.utcnow()
        if relationship:
            relationship.last_interaction_at = now
            relationship.updated_at = now
        if surface:
            surface.last_interaction_at = now
            surface.updated_at = now
        if relationship or surface:
            self.db.commit()
        if relationship:
            self.db.refresh(relationship)
        if surface:
            self.db.refresh(surface)
        return relationship, surface

    def mark_bootstrap_seen_fingerprint(
        self,
        *,
        relationship_id: str,
        surface_id: str,
        surface_instance_id: Optional[str],
        fingerprint: Optional[str],
    ) -> Optional[RelationshipSurface]:
        row = self.get_surface(
            relationship_id=relationship_id,
            surface_id=surface_id,
            surface_instance_id=surface_instance_id,
        )
        if not row:
            return None
        row.last_bootstrap_seen_fingerprint = fingerprint
        row.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_surface_by_general_conversation(self, conversation_id: str) -> Optional[RelationshipSurface]:
        return (
            self.db.query(RelationshipSurface)
            .filter(RelationshipSurface.general_conversation_id == conversation_id)
            .first()
        )
