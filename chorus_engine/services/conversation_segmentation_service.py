"""General-chat episodic segmentation lifecycle service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Conversation, ConversationSegment, Message
from chorus_engine.repositories.conversation_segment_repository import ConversationSegmentRepository


@dataclass
class SegmentTransitionResult:
    segment_id: Optional[str]
    transitioned: bool
    transition_reason: str
    resume_source_segment_id: Optional[str]
    resume_recap_pending: bool
    closed_segment_id: Optional[str] = None
    should_summarize_inline: bool = False


class ConversationSegmentationService:
    """Lifecycle management for v1 segment boundaries."""

    def __init__(self, db: Session, cfg):
        self.db = db
        self.repo = ConversationSegmentRepository(db)
        self.cfg = cfg

    def ensure_segment_for_turn(
        self,
        *,
        conversation: Conversation,
        thread_id: str,
        user_message_id: str,
        surface_id: Optional[str],
        surface_instance_id: Optional[str],
    ) -> SegmentTransitionResult:
        if getattr(conversation, "conversation_kind", "standard") != "general_chat":
            return SegmentTransitionResult(
                segment_id=None,
                transitioned=False,
                transition_reason="none",
                resume_source_segment_id=None,
                resume_recap_pending=False,
            )
        if not getattr(self.cfg, "enabled", True):
            return SegmentTransitionResult(
                segment_id=None,
                transitioned=False,
                transition_reason="none",
                resume_source_segment_id=None,
                resume_recap_pending=False,
            )

        current_message = self.db.query(Message).filter(Message.id == user_message_id).first()
        if not current_message:
            return SegmentTransitionResult(
                segment_id=None,
                transitioned=False,
                transition_reason="none",
                resume_source_segment_id=None,
                resume_recap_pending=False,
            )

        open_segment = self.repo.get_open_segment(conversation.id)
        if open_segment is None:
            segment = self.repo.create_open_segment(
                conversation_id=conversation.id,
                relationship_id=getattr(conversation, "relationship_id", None),
                surface_id=surface_id or conversation.source,
                surface_instance_id=surface_instance_id,
                segment_kind="manual_break",
                started_at=current_message.created_at,
                start_message_id=user_message_id,
                resume_source_segment_id=None,
            )
            return SegmentTransitionResult(
                segment_id=segment.id,
                transitioned=True,
                transition_reason="manual_break",
                resume_source_segment_id=None,
                resume_recap_pending=False,
                closed_segment_id=None,
                should_summarize_inline=False,
            )

        prior_messages = (
            self.db.query(Message)
            .filter(
                Message.thread_id == thread_id,
                Message.id != user_message_id,
                Message.created_at < current_message.created_at,
                Message.deleted_at.is_(None),
                Message.created_at >= open_segment.started_at,
            )
            .order_by(Message.created_at.desc())
            .limit(max(int(self.cfg.density_window_messages), int(self.cfg.pending_question_tail_messages)) + 2)
            .all()
        )
        split_reason = self._split_reason(
            current_ts=current_message.created_at,
            prior_messages=prior_messages,
        )
        if not split_reason:
            return SegmentTransitionResult(
                segment_id=open_segment.id,
                transitioned=False,
                transition_reason="none",
                resume_source_segment_id=open_segment.resume_source_segment_id,
                resume_recap_pending=bool(
                    open_segment.resume_source_segment_id and open_segment.resume_recap_injected_at is None
                ),
            )

        prior_msg = prior_messages[0] if prior_messages else None
        closed = self.repo.close_segment(
            open_segment.id,
            ended_at=prior_msg.created_at if prior_msg else current_message.created_at,
            end_message_id=prior_msg.id if prior_msg else None,
            segment_kind_override=split_reason,
        )
        resume_source = self.repo.find_latest_useful_segment(
            conversation_id=conversation.id,
            before_started_at=current_message.created_at,
            max_age_hours=int(self.cfg.resume_max_age_hours),
            limit_recent_segments=int(self.cfg.resume_recent_useful_segments),
        )
        new_segment = self.repo.create_open_segment(
            conversation_id=conversation.id,
            relationship_id=getattr(conversation, "relationship_id", None),
            surface_id=surface_id or conversation.source,
            surface_instance_id=surface_instance_id,
            segment_kind=split_reason,
            started_at=current_message.created_at,
            start_message_id=user_message_id,
            resume_source_segment_id=resume_source.id if resume_source else None,
        )
        return SegmentTransitionResult(
            segment_id=new_segment.id,
            transitioned=True,
            transition_reason=split_reason,
            resume_source_segment_id=resume_source.id if resume_source else None,
            resume_recap_pending=bool(resume_source),
            closed_segment_id=closed.id if closed else None,
            should_summarize_inline=(split_reason == "idle_break" and bool(resume_source)),
        )

    def list_boundaries_for_thread(self, *, conversation_id: str, thread_id: Optional[str] = None) -> List[ConversationSegment]:
        del thread_id
        return self.repo.list_segments(conversation_id=conversation_id)

    def mark_resume_recap_injected(self, segment_id: Optional[str]) -> None:
        if not segment_id:
            return
        self.repo.mark_resume_recap_injected(segment_id)

    def _split_reason(self, *, current_ts: datetime, prior_messages: List[Message]) -> Optional[str]:
        if not prior_messages:
            return None
        prior = prior_messages[0]
        idle_minutes = max(0.0, (current_ts - prior.created_at).total_seconds() / 60.0)

        if idle_minutes >= float(self.cfg.sleep_break_minutes):
            return "idle_break"
        if idle_minutes >= float(self.cfg.idle_hard_minutes):
            return "idle_break"
        if idle_minutes < float(self.cfg.idle_soft_minutes):
            return None
        if self._has_pending_question(prior_messages):
            return None
        if self._is_low_density(prior_messages):
            return "idle_break"
        return None

    def _is_low_density(self, prior_messages: List[Message]) -> bool:
        window = prior_messages[: max(1, int(self.cfg.density_window_messages))]
        turn_count = len(window)
        avg_chars = sum(len((msg.content or "").strip()) for msg in window) / max(1, turn_count)
        return (
            turn_count <= int(self.cfg.density_low_turn_count_max)
            and avg_chars <= float(self.cfg.density_low_avg_chars_max)
        )

    def _has_pending_question(self, prior_messages: List[Message]) -> bool:
        tail = prior_messages[: max(0, int(self.cfg.pending_question_tail_messages))]
        for msg in tail:
            if str(getattr(msg.role, "value", msg.role)).lower() != "assistant":
                continue
            text = (msg.content or "").strip()
            if text.endswith("?"):
                return True
        return False
