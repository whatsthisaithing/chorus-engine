"""Database models for conversation continuity bootstrapping."""

from datetime import datetime
import uuid
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer

from chorus_engine.db.database import Base


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class ContinuityRelationshipState(Base):
    """Relationship state per character."""

    __tablename__ = "continuity_relationship_states"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    character_id = Column(String(50), nullable=False, index=True)
    familiarity_level = Column(String(20), nullable=False, default="new")
    tone_baseline = Column(JSON, nullable=False, default=list)
    interaction_contract = Column(JSON, nullable=False, default=list)
    boundaries = Column(JSON, nullable=False, default=list)
    assistant_role_frame = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ContinuityArc(Base):
    """Active continuity arc for a character."""

    __tablename__ = "continuity_arcs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    character_id = Column(String(50), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    kind = Column(String(30), nullable=False, default="theme")
    summary = Column(Text, nullable=False, default="")
    status = Column(String(20), nullable=False, default="active")
    confidence = Column(String(10), nullable=False, default="medium")
    stickiness = Column(String(10), nullable=False, default="normal")
    last_touched_conversation_id = Column(String(36), nullable=True)
    last_touched_conversation_at = Column(DateTime, nullable=True)
    frequency_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ContinuityBootstrapCache(Base):
    """Cached continuity bootstrap packets for a character."""

    __tablename__ = "continuity_bootstrap_cache"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    character_id = Column(String(50), nullable=False, index=True)
    bootstrap_packet_internal = Column(Text, nullable=False, default="")
    bootstrap_packet_user_preview = Column(Text, nullable=False, default="")
    bootstrap_generated_at = Column(DateTime, nullable=True)
    bootstrap_inputs_fingerprint = Column(String(128), nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ContinuityPreference(Base):
    """Continuity preference per character."""

    __tablename__ = "continuity_preferences"

    character_id = Column(String(50), primary_key=True)
    default_mode = Column(String(10), nullable=False, default="ask")
    skip_preview = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
