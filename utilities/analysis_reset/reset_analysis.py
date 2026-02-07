"""
Reset conversation summaries and extracted memories.

Usage:
    python utilities/analysis_reset/reset_analysis.py [--character-id <id>] [--dry-run]
"""

import sys
import argparse
from pathlib import Path
from sqlalchemy import or_

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Conversation, ConversationSummary, Memory, MemoryType


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset conversation summaries and extracted memories."
    )
    parser.add_argument(
        "--character-id",
        type=str,
        help="Optional character ID to scope the reset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without deleting anything"
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        conv_query = db.query(Conversation).filter(
            or_(Conversation.is_private != "true", Conversation.is_private.is_(None))
        )
        if args.character_id:
            conv_query = conv_query.filter(Conversation.character_id == args.character_id)

        conversations = conv_query.all()
        conv_ids = [conv.id for conv in conversations]

        if not conv_ids:
            print("No matching conversations found.")
            return 0

        summaries_query = db.query(ConversationSummary).filter(
            ConversationSummary.conversation_id.in_(conv_ids)
        )

        memories_query = db.query(Memory).filter(
            Memory.conversation_id.in_(conv_ids),
            Memory.memory_type.notin_([MemoryType.CORE, MemoryType.EXPLICIT])
        )
        if args.character_id:
            memories_query = memories_query.filter(Memory.character_id == args.character_id)

        summaries_count = summaries_query.count()
        memories_count = memories_query.count()

        print("Reset scope:")
        print(f"- Conversations affected: {len(conv_ids)}")
        print(f"- Summaries: {summaries_count}")
        print(f"- Extracted memories (non-core/explicit): {memories_count}")
        if args.character_id:
            print(f"- Character: {args.character_id}")
        else:
            print("- Character: ALL")

        if args.dry_run:
            print("Dry run: no changes applied.")
            return 0

        if args.character_id:
            confirm_target = f"character '{args.character_id}'"
        else:
            confirm_target = "ALL characters"

        confirm = input(
            f"Type 'YES' to confirm reset for {confirm_target}: "
        ).strip()
        if confirm != "YES":
            print("Confirmation not received. No changes applied.")
            return 0

        summaries_deleted = summaries_query.delete(synchronize_session=False)
        memories_deleted = memories_query.delete(synchronize_session=False)

        update_values = {
            "last_analyzed_at": None
        }
        if hasattr(Conversation, "last_summary_analyzed_at"):
            update_values["last_summary_analyzed_at"] = None
        if hasattr(Conversation, "last_memories_analyzed_at"):
            update_values["last_memories_analyzed_at"] = None

        db.query(Conversation).filter(
            Conversation.id.in_(conv_ids)
        ).update(update_values, synchronize_session=False)

        db.commit()

        print("Reset complete:")
        print(f"- Summaries deleted: {summaries_deleted}")
        print(f"- Memories deleted: {memories_deleted}")
        print(f"- Conversations updated: {len(conv_ids)}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
