"""Branching service for explicit general-chat -> standard conversation flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import Conversation, Message, MessageRole, Thread
from chorus_engine.repositories.conversation_repository import ConversationRepository
from chorus_engine.repositories.conversation_segment_repository import ConversationSegmentRepository
from chorus_engine.repositories.message_repository import MessageRepository
from chorus_engine.repositories.thread_repository import ThreadRepository
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.db.conversation_segment_vector_store import ConversationSegmentVectorStore


@dataclass
class BranchFromGeneralChatResult:
    new_conversation_id: str
    new_thread_id: str
    origin_mode: str
    closed_segment_id: Optional[str]
    selected_segment_ids: List[str]
    recap_source_segment_id: Optional[str]
    imported_message_count: int
    segment_summary_generated: bool
    segment_summary_usefulness: Optional[str]


class ConversationBranchingService:
    """Implements present/archival branch behavior for general chat selections."""

    def __init__(self, db: Session, app_state: Dict[str, Any]):
        self.db = db
        self.app_state = app_state
        self.conv_repo = ConversationRepository(db)
        self.msg_repo = MessageRepository(db)
        self.thread_repo = ThreadRepository(db)
        self.segment_repo = ConversationSegmentRepository(db)

    async def branch_from_general_chat(
        self,
        *,
        source_conversation_id: str,
        selected_message_ids: List[str],
        user_id: Optional[str] = None,
        character_id_hint: Optional[str] = None,
    ) -> BranchFromGeneralChatResult:
        source_conversation = self.conv_repo.get_by_id(source_conversation_id)
        if not source_conversation:
            raise ValueError("Conversation not found")
        if source_conversation.conversation_kind != "general_chat":
            raise ValueError("Branching is only supported from general chat conversations")
        if character_id_hint and character_id_hint != source_conversation.character_id:
            raise ValueError("Character mismatch for source conversation")

        requested_ids = [str(mid) for mid in (selected_message_ids or []) if str(mid).strip()]
        if not requested_ids:
            raise ValueError("selected_message_ids is required")
        unique_requested_ids = sorted(set(requested_ids))

        selected_messages = self.msg_repo.list_selected_for_conversation(
            conversation_id=source_conversation_id,
            selected_message_ids=unique_requested_ids,
            include_deleted=False,
        )
        selected_ids_found = {str(m.id) for m in selected_messages}
        missing = [mid for mid in unique_requested_ids if mid not in selected_ids_found]
        if missing:
            raise ValueError(f"Selected messages not found or deleted: {', '.join(missing[:5])}")
        if any(m.role not in (MessageRole.USER, MessageRole.ASSISTANT) for m in selected_messages):
            raise ValueError("Only user/assistant transcript messages can be branched")

        open_segment = self.segment_repo.get_open_segment(source_conversation_id)
        mapped_segment_ids: List[str] = []
        for msg in selected_messages:
            segment_id = self.segment_repo.resolve_segment_id_for_message(
                conversation_id=source_conversation_id,
                message=msg,
            )
            if segment_id:
                mapped_segment_ids.append(segment_id)
        selected_segment_ids = list(dict.fromkeys(mapped_segment_ids))
        selection_includes_open_segment = bool(
            open_segment and any(seg_id == open_segment.id for seg_id in selected_segment_ids)
        )
        origin_mode = "present" if selection_includes_open_segment else "archival"

        closed_segment_id: Optional[str] = None
        summary_generated = False
        summary_usefulness: Optional[str] = None
        recap_source_segment_id: Optional[str] = None

        if origin_mode == "present":
            closed_segment = await self._close_present_segment_with_nonfatal_summary(
                source_conversation=source_conversation,
                open_segment=open_segment,
            )
            closed_segment_id = closed_segment.id if closed_segment else None
            recap_source_segment_id = closed_segment_id
            if closed_segment and closed_segment.summary_text and closed_segment.usefulness == "useful":
                summary_generated = True
                summary_usefulness = closed_segment.usefulness
            elif closed_segment:
                summary_usefulness = closed_segment.usefulness
        else:
            recap_source_segment_id = self._select_archival_recap_source_segment(
                selected_segment_ids=selected_segment_ids,
            )

        now = datetime.utcnow()
        character_obj = self.app_state.get("characters", {}).get(source_conversation.character_id)
        character_name = getattr(character_obj, "name", None) or source_conversation.character_id
        destination = self.conv_repo.create(
            character_id=source_conversation.character_id,
            title=f"Conversation with {character_name}",
            source=source_conversation.source or "web",
            primary_user=source_conversation.primary_user,
            continuity_mode="ask",
            relationship_id=source_conversation.relationship_id,
            conversation_kind="standard",
            origin_conversation_id=source_conversation.id,
            origin_mode=origin_mode,
            origin_segment_id=recap_source_segment_id,
            origin_segment_ids_json=selected_segment_ids or None,
            branch_created_at=now,
        )
        destination_thread = self.thread_repo.create(conversation_id=destination.id, title="Main Thread")

        self._import_selected_messages(
            destination_thread_id=destination_thread.id,
            source_messages=selected_messages,
            imported_from_conversation_id=source_conversation.id,
            imported_at=now,
        )

        if origin_mode == "present" and closed_segment_id and summary_usefulness is None:
            closed = self.segment_repo.get_by_id(closed_segment_id)
            if closed:
                summary_usefulness = closed.usefulness
                if closed.summary_text and closed.usefulness == "useful":
                    summary_generated = True

        return BranchFromGeneralChatResult(
            new_conversation_id=destination.id,
            new_thread_id=destination_thread.id,
            origin_mode=origin_mode,
            closed_segment_id=closed_segment_id,
            selected_segment_ids=selected_segment_ids,
            recap_source_segment_id=recap_source_segment_id,
            imported_message_count=len(selected_messages),
            segment_summary_generated=summary_generated,
            segment_summary_usefulness=summary_usefulness,
        )

    async def _close_present_segment_with_nonfatal_summary(
        self,
        *,
        source_conversation: Conversation,
        open_segment,
    ):
        if open_segment is None:
            latest = self._latest_message_for_conversation(source_conversation.id)
            if latest is None:
                return None
            open_segment = self.segment_repo.create_open_segment(
                conversation_id=source_conversation.id,
                relationship_id=source_conversation.relationship_id,
                surface_id=source_conversation.source,
                surface_instance_id="",
                segment_kind="manual_break",
                started_at=latest.created_at,
                start_message_id=latest.id,
                resume_source_segment_id=None,
            )

        latest_msg = self._latest_message_for_conversation(source_conversation.id)
        ended_at = latest_msg.created_at if latest_msg else datetime.utcnow()
        end_message_id = latest_msg.id if latest_msg else None
        closed = self.segment_repo.close_segment(
            open_segment.id,
            ended_at=ended_at,
            end_message_id=end_message_id,
            segment_kind_override="branch_break",
        )
        if not closed:
            return None

        cfg = getattr(self.app_state.get("system_config"), "general_chat_segmentation", None)
        max_tokens = int(getattr(cfg, "summary_max_tokens", 1200) or 1200)
        model_override = getattr(cfg, "summary_model_override", None) if cfg else None
        try:
            await self._summarize_closed_segment(
                source_conversation=source_conversation,
                segment_id=closed.id,
                max_tokens=max_tokens,
                model_override=model_override,
            )
        except Exception:
            # Non-fatal by design: branch creation should proceed even on summary failures.
            pass
        return self.segment_repo.get_by_id(closed.id)

    def _latest_message_for_conversation(self, conversation_id: str) -> Optional[Message]:
        rows = (
            self.db.query(Message)
            .join(Thread, Message.thread_id == Thread.id)
            .filter(
                Thread.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(1)
            .all()
        )
        return rows[0] if rows else None

    def _select_archival_recap_source_segment(self, *, selected_segment_ids: List[str]) -> Optional[str]:
        if not selected_segment_ids:
            return None
        segments = self.segment_repo.list_by_ids(selected_segment_ids)
        useful = [
            seg
            for seg in segments
            if seg.state == "closed" and seg.summary_text and seg.usefulness == "useful"
        ]
        if not useful:
            return None
        useful.sort(key=lambda seg: (seg.ended_at or datetime.min, seg.id))
        return useful[-1].id

    def _import_selected_messages(
        self,
        *,
        destination_thread_id: str,
        source_messages: List[Message],
        imported_from_conversation_id: str,
        imported_at: datetime,
    ) -> None:
        to_insert: List[Message] = []
        for idx, msg in enumerate(source_messages):
            metadata = dict(msg.meta_data or {})
            metadata["imported"] = True
            metadata["imported_at"] = imported_at.isoformat()
            to_insert.append(
                Message(
                    thread_id=destination_thread_id,
                    role=msg.role,
                    content=msg.content,
                    meta_data=metadata,
                    is_private=msg.is_private,
                    created_at=imported_at + timedelta(microseconds=idx),
                    imported_from_conversation_id=imported_from_conversation_id,
                    imported_from_message_id=msg.id,
                    imported_from_segment_id=self.segment_repo.resolve_segment_id_for_message(
                        conversation_id=imported_from_conversation_id,
                        message=msg,
                    ),
                )
            )
        if to_insert:
            self.db.add_all(to_insert)
            self.db.commit()

    async def _summarize_closed_segment(
        self,
        *,
        source_conversation: Conversation,
        segment_id: str,
        max_tokens: int,
        model_override: Optional[str],
    ) -> str:
        segment = self.segment_repo.get_by_id(segment_id)
        if not segment or segment.summary_text:
            return "already_present"
        if not segment.start_message_id or not segment.end_message_id:
            return "range_missing"

        start_msg = self.db.query(Message).filter(Message.id == segment.start_message_id).first()
        end_msg = self.db.query(Message).filter(Message.id == segment.end_message_id).first()
        if not start_msg or not end_msg:
            return "range_messages_missing"

        messages = (
            self.db.query(Message)
            .filter(
                Message.thread_id == start_msg.thread_id,
                Message.deleted_at.is_(None),
                Message.created_at >= start_msg.created_at,
                Message.created_at <= end_msg.created_at,
            )
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )
        transcript_payload = [
            {
                "role": (m.role.value if hasattr(m.role, "value") else str(m.role)),
                "content": m.content,
            }
            for m in messages
        ]
        transcript_json = json.dumps(transcript_payload, ensure_ascii=False)

        analysis_service: ConversationAnalysisService = self.app_state.get("analysis_service")
        if not analysis_service:
            return "analysis_service_missing"
        character = self.app_state.get("characters", {}).get(source_conversation.character_id)
        if not character:
            return "character_missing"
        token_count = analysis_service.token_counter.count_tokens(transcript_json)
        analysis = await analysis_service.analyze_segment_summary_only(
            conversation_id=source_conversation.id,
            character=character,
            transcript_json=transcript_json,
            token_count=token_count,
            summary_model=model_override,
            max_tokens=max_tokens,
        )
        if not analysis:
            return "analysis_failed"

        summary_vector_id = None
        try:
            embedder = self.app_state.get("embedding_service")
            if embedder is None:
                from chorus_engine.services.embedding_service import EmbeddingService

                embedder = EmbeddingService()
            vector_store = self.app_state.get("segment_summary_vector_store")
            if vector_store is None:
                vector_store = ConversationSegmentVectorStore(Path("data/vector_store"))
            embedding = embedder.embed(analysis.summary)
            if vector_store.upsert_segment_summary(
                character_id=source_conversation.character_id,
                segment_id=segment_id,
                summary_text=analysis.summary,
                embedding=embedding,
                metadata={
                    "conversation_id": source_conversation.id,
                    "segment_kind": segment.segment_kind,
                    "usefulness": analysis.usefulness,
                },
            ):
                summary_vector_id = segment_id
        except Exception:
            pass

        self.segment_repo.upsert_segment_summary(
            segment_id,
            summary_text=analysis.summary,
            usefulness=analysis.usefulness,
            key_events=analysis.key_events,
            open_threads=analysis.open_threads,
            participants=analysis.participants,
            summary_model=(
                model_override
                or analysis_service.archivist_model
                or getattr(getattr(self.app_state.get("system_config"), "llm", None), "model", None)
            ),
            summary_prompt_version=analysis.summary_prompt_version,
            summary_input_hash=analysis.summary_input_hash,
            summary_created_at=datetime.utcnow(),
            summary_vector_id=summary_vector_id,
            embedding_model=getattr(getattr(self.app_state.get("system_config"), "llm", None), "embedding_model", None),
        )
        return "generated"
