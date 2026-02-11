"""
Run continuity bootstrap generation without persisting results.

Usage:
    python utilities/continuity_bootstrap_runner/run_bootstrap.py <character_id>
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.config.loader import ConfigLoader
from chorus_engine.llm.client import create_llm_client
from chorus_engine.services.continuity_bootstrap_service import ContinuityBootstrapService
from chorus_engine.models.conversation import Conversation, ConversationSummary, Memory, MemoryType


async def run_bootstrap(
    character_id: str,
    output_path: str | None = None,
    include_prompts: bool = False,
    include_raw: bool = False
) -> Path | None:
    db = SessionLocal()
    try:
        config_loader = ConfigLoader(project_root)
        system_config = config_loader.load_system_config()
        character = config_loader.load_character(character_id)

        llm_client = create_llm_client(system_config.llm)
        service = ContinuityBootstrapService(
            db=db,
            llm_client=llm_client,
            llm_usage_lock=asyncio.Lock(),
            max_tokens=1024
        )

        result = await service.generate_preview(character=character, conversation_id=None)
        if not result:
            print("Continuity generation failed.")
            return None

        summaries = (
            db.query(ConversationSummary, Conversation)
            .join(Conversation, ConversationSummary.conversation_id == Conversation.id)
            .filter(Conversation.character_id == character_id)
            .order_by(ConversationSummary.created_at.desc())
            .limit(5)
            .all()
        )

        memories = (
            db.query(Memory, Conversation)
            .outerjoin(Conversation, Memory.conversation_id == Conversation.id)
            .filter(
                Memory.character_id == character_id,
                Memory.status.in_(["approved", "auto_approved"]),
                Memory.memory_type.in_([
                    MemoryType.FACT,
                    MemoryType.PROJECT,
                    MemoryType.EXPERIENCE,
                    MemoryType.STORY,
                    MemoryType.RELATIONSHIP
                ])
            )
            .order_by(Memory.created_at.desc())
            .limit(50)
            .all()
        )

        data = {
            "character_id": character_id,
            "generated_at": datetime.utcnow().isoformat(),
            "input_summaries": [
                {
                    "conversation_id": conv.id,
                    "conversation_title": conv.title,
                    "summary_id": summary.id,
                    "summary_type": summary.summary_type,
                    "created_at": summary.created_at.isoformat() if summary.created_at else None,
                    "message_range_start": summary.message_range_start,
                    "message_range_end": summary.message_range_end,
                    "message_count": summary.message_count,
                    "summary": summary.summary
                }
                for summary, conv in summaries
            ],
            "input_memories": [
                {
                    "memory_id": mem.id,
                    "conversation_id": mem.conversation_id,
                    "conversation_title": conv.title if conv else None,
                    "memory_type": mem.memory_type,
                    "status": mem.status,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None,
                    "content": mem.content
                }
                for mem, conv in memories
            ],
            "relationship_state": result.get("relationship_state").__dict__ if result.get("relationship_state") else {},
            "arc_candidates": [
                {
                    "title": c.title,
                    "kind": c.kind,
                    "summary": c.summary,
                    "confidence": c.confidence
                }
                for c in (result.get("candidates") or [])
            ],
            "merged_arcs": [
                {"title": a.title, "kind": a.kind, "summary": a.summary, "confidence": a.confidence}
                for a in (result.get("merged_arcs") or [])
            ],
            "normalized_arcs": [
                {"title": a.title, "kind": a.kind, "summary": a.summary, "confidence": a.confidence}
                for a in (result.get("normalized_arcs") or [])
            ],
            "arc_scores": result.get("score_map") or {},
            "core_arcs": [
                {"title": a.title, "kind": a.kind, "summary": a.summary, "confidence": a.confidence}
                for a in (result.get("core_arcs") or [])
            ],
            "active_arcs": [
                {"title": a.title, "kind": a.kind, "summary": a.summary, "confidence": a.confidence}
                for a in (result.get("active_arcs") or [])
            ],
            "internal_packet": result.get("internal_packet", ""),
            "user_preview": result.get("user_preview", ""),
            "fingerprint": result.get("fingerprint", "")
        }

        if include_prompts:
            data["prompts"] = {"note": "Prompts are defined in ContinuityBootstrapService."}
        if include_raw:
            data["raw"] = {"note": "Raw responses not captured in preview mode."}

        json_output = json.dumps(data, indent=2, ensure_ascii=False)

        if output_path:
            output_file = Path(output_path)
        else:
            output_dir = Path("data/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"continuity_bootstrap_preview_{character_id}_{timestamp}.json"

        output_file.write_text(json_output, encoding="utf-8")
        print(f"✓ Saved to: {output_file}")
        return output_file
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run continuity bootstrap generation without persisting results."
    )
    parser.add_argument("character_id", type=str, help="Character ID to analyze")
    parser.add_argument("--output", type=str, help="Optional output file path")
    parser.add_argument("--include-prompts", action="store_true", help="Include prompt note in output")
    parser.add_argument("--include-raw", action="store_true", help="Include raw note in output")

    args = parser.parse_args()
    asyncio.run(
        run_bootstrap(
            args.character_id,
            args.output,
            include_prompts=args.include_prompts,
            include_raw=args.include_raw
        )
    )


if __name__ == "__main__":
    main()
