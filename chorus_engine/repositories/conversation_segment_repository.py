"""Repository for conversation segment operations."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import ConversationSegment, Message


class ConversationSegmentRepository:
    """Database helpers for episodic conversation segments."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, segment_id: str) -> Optional[ConversationSegment]:
        return self.db.query(ConversationSegment).filter(ConversationSegment.id == segment_id).first()

    def get_open_segment(self, conversation_id: str) -> Optional[ConversationSegment]:
        return (
            self.db.query(ConversationSegment)
            .filter(
                ConversationSegment.conversation_id == conversation_id,
                ConversationSegment.state == "open",
            )
            .order_by(ConversationSegment.started_at.desc())
            .first()
        )

    def list_segments(self, conversation_id: str, skip: int = 0, limit: int = 500) -> List[ConversationSegment]:
        return (
            self.db.query(ConversationSegment)
            .filter(ConversationSegment.conversation_id == conversation_id)
            .order_by(ConversationSegment.started_at.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create_open_segment(
        self,
        *,
        conversation_id: str,
        relationship_id: Optional[str],
        surface_id: Optional[str],
        surface_instance_id: Optional[str],
        segment_kind: str,
        started_at: datetime,
        start_message_id: Optional[str] = None,
        resume_source_segment_id: Optional[str] = None,
    ) -> ConversationSegment:
        segment = ConversationSegment(
            conversation_id=conversation_id,
            relationship_id=relationship_id,
            surface_id=surface_id,
            surface_instance_id=surface_instance_id or "",
            segment_kind=segment_kind,
            state="open",
            start_message_id=start_message_id,
            started_at=started_at,
            resume_source_segment_id=resume_source_segment_id,
        )
        self.db.add(segment)
        self.db.commit()
        self.db.refresh(segment)
        return segment

    def close_segment(
        self,
        segment_id: str,
        *,
        ended_at: datetime,
        end_message_id: Optional[str] = None,
        segment_kind_override: Optional[str] = None,
    ) -> Optional[ConversationSegment]:
        segment = self.get_by_id(segment_id)
        if not segment:
            return None
        segment.state = "closed"
        if segment_kind_override:
            segment.segment_kind = segment_kind_override
        segment.ended_at = ended_at
        if end_message_id:
            segment.end_message_id = end_message_id
        segment.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(segment)
        return segment

    def upsert_segment_summary(
        self,
        segment_id: str,
        *,
        summary_text: str,
        usefulness: str,
        key_events: Optional[list],
        open_threads: Optional[list],
        participants: Optional[list],
        summary_model: Optional[str],
        summary_prompt_version: Optional[str],
        summary_input_hash: Optional[str],
        summary_created_at: datetime,
        summary_vector_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Optional[ConversationSegment]:
        segment = self.get_by_id(segment_id)
        if not segment:
            return None
        if summary_input_hash and segment.summary_input_hash == summary_input_hash and segment.summary_text:
            return segment
        segment.summary_text = summary_text
        segment.usefulness = usefulness or "unknown"
        segment.key_events = key_events or []
        segment.open_threads = open_threads or []
        segment.participants = participants or []
        segment.summary_model = summary_model
        segment.summary_prompt_version = summary_prompt_version
        segment.summary_input_hash = summary_input_hash
        segment.summary_created_at = summary_created_at
        if summary_vector_id:
            segment.summary_vector_id = summary_vector_id
        if embedding_model:
            segment.embedding_model = embedding_model
        segment.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(segment)
        return segment

    def mark_resume_recap_injected(self, segment_id: str, at: Optional[datetime] = None) -> Optional[ConversationSegment]:
        segment = self.get_by_id(segment_id)
        if not segment:
            return None
        segment.resume_recap_injected_at = at or datetime.utcnow()
        segment.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(segment)
        return segment

    def set_resume_source_segment(
        self,
        segment_id: str,
        resume_source_segment_id: Optional[str],
    ) -> Optional[ConversationSegment]:
        segment = self.get_by_id(segment_id)
        if not segment:
            return None
        segment.resume_source_segment_id = resume_source_segment_id
        segment.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(segment)
        return segment

    def find_latest_useful_segment(
        self,
        *,
        conversation_id: str,
        before_started_at: datetime,
        max_age_hours: int,
        limit_recent_segments: int,
    ) -> Optional[ConversationSegment]:
        candidates = (
            self.db.query(ConversationSegment)
            .filter(
                ConversationSegment.conversation_id == conversation_id,
                ConversationSegment.state == "closed",
                ConversationSegment.usefulness == "useful",
                ConversationSegment.summary_text.isnot(None),
                ConversationSegment.ended_at.isnot(None),
                ConversationSegment.ended_at < before_started_at,
            )
            .order_by(ConversationSegment.ended_at.desc())
            .limit(max(1, limit_recent_segments))
            .all()
        )
        if not candidates:
            return None
        if max_age_hours <= 0:
            return candidates[0]
        cutoff = before_started_at.timestamp() - (max_age_hours * 3600)
        for candidate in candidates:
            if candidate.ended_at and candidate.ended_at.timestamp() >= cutoff:
                return candidate
        return None

    def list_by_ids(self, segment_ids: List[str]) -> List[ConversationSegment]:
        if not segment_ids:
            return []
        return (
            self.db.query(ConversationSegment)
            .filter(ConversationSegment.id.in_(segment_ids))
            .all()
        )

    def resolve_segment_id_for_message(
        self,
        *,
        conversation_id: str,
        message: Message,
    ) -> Optional[str]:
        """
        Resolve a message to a segment boundary by timestamp ranges.

        Returns None if no deterministic mapping exists.
        """
        if not message:
            return None

        closed = (
            self.db.query(ConversationSegment)
            .filter(
                ConversationSegment.conversation_id == conversation_id,
                ConversationSegment.state == "closed",
                ConversationSegment.started_at <= message.created_at,
                ConversationSegment.ended_at.isnot(None),
                ConversationSegment.ended_at >= message.created_at,
            )
            .order_by(ConversationSegment.started_at.desc(), ConversationSegment.id.desc())
            .first()
        )
        if closed:
            return closed.id

        open_segment = self.get_open_segment(conversation_id)
        if open_segment and open_segment.started_at <= message.created_at:
            return open_segment.id
        return None
