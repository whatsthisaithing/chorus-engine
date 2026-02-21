from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from chorus_engine.config import ConfigLoader
from chorus_engine.db.database import SessionLocal
from chorus_engine.db.conversation_segment_vector_store import ConversationSegmentVectorStore
from chorus_engine.llm import create_llm_client
from chorus_engine.models.conversation import Conversation, Message, MessageRole
from chorus_engine.repositories.conversation_segment_repository import ConversationSegmentRepository
from chorus_engine.repositories.thread_repository import ThreadRepository
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.services.conversation_segmentation_service import ConversationSegmentationService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.db.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill general-chat segments and segment summaries.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    parser.add_argument("--character-id", default=None)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def _summarize_missing_segments(
    *,
    db,
    analysis_service: ConversationAnalysisService,
    segment_store: ConversationSegmentVectorStore,
    embedder: EmbeddingService,
    character_id: str,
    conversation_id: str,
    max_tokens: int,
    model_override: Optional[str],
    run_async=None,
) -> dict:
    seg_repo = ConversationSegmentRepository(db)
    segments = seg_repo.list_segments(conversation_id=conversation_id)
    generated = 0
    failed = 0
    failed_segment_ids = []
    character = ConfigLoader().load_character(character_id)
    system_config = ConfigLoader().load_system_config()

    def _build_transcript(messages_subset) -> tuple[str, int]:
        payload = [
            {"role": (m.role.value if hasattr(m.role, "value") else str(m.role)), "content": m.content}
            for m in messages_subset
        ]
        txt = json.dumps(payload, ensure_ascii=False)
        return txt, analysis_service.token_counter.count_tokens(txt)

    for segment in segments:
        if segment.state != "closed" or segment.summary_text:
            continue
        if not segment.start_message_id or not segment.end_message_id:
            continue
        start_msg = db.query(Message).filter(Message.id == segment.start_message_id).first()
        end_msg = db.query(Message).filter(Message.id == segment.end_message_id).first()
        if not start_msg or not end_msg:
            continue
        messages = (
            db.query(Message)
            .filter(
                Message.thread_id == start_msg.thread_id,
                Message.deleted_at.is_(None),
                Message.created_at >= start_msg.created_at,
                Message.created_at <= end_msg.created_at,
            )
            .order_by(Message.created_at.asc())
            .all()
        )
        transcript_json, token_count = _build_transcript(messages)
        analysis = None
        for attempt in range(1, 4):
            analysis = analysis_service.analyze_segment_summary_only(
                conversation_id=conversation_id,
                character=character,
                transcript_json=transcript_json,
                token_count=token_count,
                summary_model=model_override,
                max_tokens=max_tokens,
            )
            if hasattr(analysis, "__await__"):
                if run_async is None:
                    raise RuntimeError("run_async callback required for async segment summary analysis")
                analysis = run_async(analysis)
            if analysis:
                break
            if attempt < 3:
                time.sleep(0.75 * attempt)
        if not analysis:
            failed += 1
            failed_segment_ids.append(str(segment.id))
            print(
                f"[WARN] Segment summary generation failed: conversation={conversation_id} "
                f"segment={segment.id}"
            )
            continue
        summary_vector_id = None
        embedding = embedder.embed(analysis.summary)
        if segment_store.upsert_segment_summary(
            character_id=character_id,
            segment_id=segment.id,
            summary_text=analysis.summary,
            embedding=embedding,
            metadata={
                "conversation_id": conversation_id,
                "segment_kind": segment.segment_kind,
                "usefulness": analysis.usefulness,
            },
        ):
            summary_vector_id = segment.id
        seg_repo.upsert_segment_summary(
            segment.id,
            summary_text=analysis.summary,
            usefulness=analysis.usefulness,
            key_events=analysis.key_events,
            open_threads=analysis.open_threads,
            participants=analysis.participants,
            summary_model=model_override,
            summary_prompt_version=analysis.summary_prompt_version,
            summary_input_hash=analysis.summary_input_hash,
            summary_created_at=datetime.utcnow(),
            summary_vector_id=summary_vector_id,
            embedding_model=getattr(getattr(system_config, "llm", None), "embedding_model", None),
        )
        generated += 1
    return {
        "generated": generated,
        "failed": failed,
        "failed_segment_ids": failed_segment_ids,
    }


def main() -> int:
    args = parse_args()
    loader = ConfigLoader()
    system_config = loader.load_system_config()
    db = SessionLocal()
    loop = None
    run_async = None
    report = {
        "dry_run": not args.apply,
        "processed_conversations": 0,
        "created_segments": 0,
        "summaries_generated": 0,
        "summaries_failed": 0,
        "summary_failed_segments": [],
    }
    try:
        conv_query = db.query(Conversation).filter(Conversation.conversation_kind == "general_chat")
        if args.character_id:
            conv_query = conv_query.filter(Conversation.character_id == args.character_id)
        if args.conversation_id:
            conv_query = conv_query.filter(Conversation.id == args.conversation_id)
        conversations = conv_query.order_by(Conversation.updated_at.desc()).limit(max(1, args.limit)).all()
        if args.apply:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            run_async = loop.run_until_complete
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
        else:
            llm_client = None
            vector_store = None
            embedding_service = None
            analysis_service = None
            segment_vector_store = None

        for conversation in conversations:
            report["processed_conversations"] += 1
            thread_repo = ThreadRepository(db)
            threads = thread_repo.list_by_conversation(conversation.id)
            if not threads:
                continue
            thread = threads[0]
            if args.apply:
                seg_service = ConversationSegmentationService(db, system_config.general_chat_segmentation)
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
                    result = seg_service.ensure_segment_for_turn(
                        conversation=conversation,
                        thread_id=thread.id,
                        user_message_id=msg.id,
                        surface_id=conversation.source,
                        surface_instance_id="",
                    )
                    if result.transitioned:
                        report["created_segments"] += 1
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
                report["summary_failed_segments"].extend(summary_report.get("failed_segment_ids", []))

        out_dir = Path("data/segment_backfill_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"segment_backfill_{stamp}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Report: {out_path}")
        return 0
    finally:
        if args.apply and loop is not None:
            try:
                if llm_client is not None:
                    loop.run_until_complete(llm_client.close())
            except Exception:
                pass
            loop.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
