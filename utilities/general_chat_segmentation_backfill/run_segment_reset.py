from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from chorus_engine.config import ConfigLoader
from chorus_engine.db.conversation_segment_vector_store import ConversationSegmentVectorStore
from chorus_engine.db.database import SessionLocal
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.llm import create_llm_client
from chorus_engine.models.conversation import Conversation, ConversationSegment, Message, MessageRole
from chorus_engine.repositories.thread_repository import ThreadRepository
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.services.conversation_segmentation_service import ConversationSegmentationService
from chorus_engine.services.embedding_service import EmbeddingService
from utilities.general_chat_segmentation_backfill.run_backfill import _summarize_missing_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset general-chat segments (DB + vectors), optionally rebuild via v1 segmentation."
    )
    parser.add_argument("--apply", action="store_true", help="Apply destructive changes. Default is dry-run.")
    parser.add_argument("--character-id", default=None, help="Restrict to one character.")
    parser.add_argument("--conversation-id", default=None, help="Restrict to one conversation.")
    parser.add_argument("--limit", type=int, default=100, help="Limit conversations processed.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="After reset, rebuild segments/summaries with current v1 logic.",
    )
    return parser.parse_args()


def _fetch_target_conversations(db, *, character_id: Optional[str], conversation_id: Optional[str], limit: int) -> List[Conversation]:
    query = db.query(Conversation).filter(Conversation.conversation_kind == "general_chat")
    if character_id:
        query = query.filter(Conversation.character_id == character_id)
    if conversation_id:
        query = query.filter(Conversation.id == conversation_id)
    return query.order_by(Conversation.updated_at.desc()).limit(max(1, limit)).all()


def _collect_segments(db, conversations: List[Conversation]) -> List[ConversationSegment]:
    if not conversations:
        return []
    conversation_ids = [c.id for c in conversations]
    return (
        db.query(ConversationSegment)
        .filter(ConversationSegment.conversation_id.in_(conversation_ids))
        .order_by(ConversationSegment.conversation_id.asc(), ConversationSegment.started_at.asc())
        .all()
    )


def _group_segment_ids_by_character(
    conversations: List[Conversation], segments: List[ConversationSegment]
) -> Dict[str, List[str]]:
    by_conv = {c.id: c.character_id for c in conversations}
    grouped: Dict[str, List[str]] = defaultdict(list)
    for seg in segments:
        character_id = by_conv.get(seg.conversation_id)
        if character_id:
            grouped[character_id].append(seg.id)
    return grouped


def _rebuild_segments_for_conversation(
    *,
    db,
    conversation: Conversation,
    config_loader: ConfigLoader,
    segmentation_service: ConversationSegmentationService,
    analysis_service: ConversationAnalysisService,
    segment_vector_store: ConversationSegmentVectorStore,
    embedding_service: EmbeddingService,
    run_async,
) -> Dict[str, int]:
    report = {"created_segments": 0, "summaries_generated": 0, "summaries_failed": 0}
    thread_repo = ThreadRepository(db)
    threads = thread_repo.list_by_conversation(conversation.id)
    if not threads:
        return report
    thread = threads[0]
    user_messages = (
        db.query(Message)
        .filter(
            Message.thread_id == thread.id,
            Message.deleted_at.is_(None),
            Message.role == MessageRole.USER,
        )
        .order_by(Message.created_at.asc())
        .all()
    )
    for msg in user_messages:
        result = segmentation_service.ensure_segment_for_turn(
            conversation=conversation,
            thread_id=thread.id,
            user_message_id=msg.id,
            surface_id=conversation.source,
            surface_instance_id="",
        )
        if result.transitioned:
            report["created_segments"] += 1

    system_config = config_loader.load_system_config()
    summary_report = _summarize_missing_segments(
        db=db,
        analysis_service=analysis_service,
        segment_store=segment_vector_store,
        embedder=embedding_service,
        character_id=conversation.character_id,
        conversation_id=conversation.id,
        max_tokens=system_config.general_chat_segmentation.summary_max_tokens,
        model_override=system_config.general_chat_segmentation.summary_model_override,
        run_async=run_async,
    )
    report["summaries_generated"] += int(summary_report.get("generated", 0))
    report["summaries_failed"] += int(summary_report.get("failed", 0))
    return report


def main() -> int:
    args = parse_args()
    loader = ConfigLoader()
    db = SessionLocal()
    loop = None
    llm_client = None

    report = {
        "dry_run": not args.apply,
        "rebuild_requested": bool(args.rebuild),
        "processed_conversations": 0,
        "found_segments": 0,
        "deleted_segments": 0,
        "deleted_vectors": 0,
        "rebuild_created_segments": 0,
        "rebuild_summaries_generated": 0,
        "rebuild_summaries_failed": 0,
    }

    try:
        conversations = _fetch_target_conversations(
            db,
            character_id=args.character_id,
            conversation_id=args.conversation_id,
            limit=args.limit,
        )
        report["processed_conversations"] = len(conversations)

        segments = _collect_segments(db, conversations)
        report["found_segments"] = len(segments)
        segment_ids_by_character = _group_segment_ids_by_character(conversations, segments)

        if args.apply and segments:
            segment_store = ConversationSegmentVectorStore(Path("data/vector_store"))
            for character_id, segment_ids in segment_ids_by_character.items():
                report["deleted_vectors"] += segment_store.delete_segment_summaries(
                    character_id=character_id,
                    segment_ids=segment_ids,
                )

            deleted = (
                db.query(ConversationSegment)
                .filter(ConversationSegment.id.in_([s.id for s in segments]))
                .delete(synchronize_session=False)
            )
            db.commit()
            report["deleted_segments"] = int(deleted or 0)

        if args.apply and args.rebuild and conversations:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            run_async = loop.run_until_complete
            system_config = loader.load_system_config()
            llm_client = create_llm_client(system_config.llm)
            vector_store = VectorStore(Path("data/vector_store"))
            embedding_service = EmbeddingService()
            analysis_service = ConversationAnalysisService(
                db=db,
                llm_client=llm_client,
                vector_store=vector_store,
                embedding_service=embedding_service,
                archivist_model=system_config.llm.archivist_model,
            )
            segment_vector_store = ConversationSegmentVectorStore(Path("data/vector_store"))
            segmentation_service = ConversationSegmentationService(db, system_config.general_chat_segmentation)
            for conversation in conversations:
                rebuild = _rebuild_segments_for_conversation(
                    db=db,
                    conversation=conversation,
                    config_loader=loader,
                    segmentation_service=segmentation_service,
                    analysis_service=analysis_service,
                    segment_vector_store=segment_vector_store,
                    embedding_service=embedding_service,
                    run_async=run_async,
                )
                report["rebuild_created_segments"] += rebuild["created_segments"]
                report["rebuild_summaries_generated"] += rebuild["summaries_generated"]
                report["rebuild_summaries_failed"] += rebuild["summaries_failed"]

        out_dir = Path("data/segment_backfill_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"segment_reset_{stamp}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Report: {out_path}")
        return 0
    finally:
        if loop is not None:
            try:
                if llm_client is not None:
                    loop.run_until_complete(llm_client.close())
            except Exception:
                pass
            loop.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
