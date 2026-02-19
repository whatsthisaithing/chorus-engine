"""Database models for relationship-first identity anchors."""

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Index

from chorus_engine.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Relationship(Base):
    """Durable identity anchor for (owner_user_id, character_id)."""

    __tablename__ = "relationships"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_user_id = Column(String(200), nullable=False, index=True)
    character_id = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_interaction_at = Column(DateTime, nullable=True, default=None)
    state_version = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("uq_relationships_owner_character", "owner_user_id", "character_id", unique=True),
    )


class RelationshipSurface(Base):
    """Per-surface mapping for relationship-scoped general chat."""

    __tablename__ = "relationship_surfaces"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    relationship_id = Column(String(36), ForeignKey("relationships.id"), nullable=False, index=True)
    surface_id = Column(String(20), nullable=False, index=True)
    surface_instance_id = Column(String(100), nullable=False, default="", index=True)
    general_conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    last_bootstrap_seen_fingerprint = Column(String(128), nullable=True, default=None)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_interaction_at = Column(DateTime, nullable=True, default=None)

    __table_args__ = (
        Index(
            "uq_relationship_surfaces_lookup",
            "relationship_id",
            "surface_id",
            "surface_instance_id",
            unique=True,
        ),
    )
