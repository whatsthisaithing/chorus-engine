from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.db.database import SessionLocal
from chorus_engine.db.moment_pin_vector_store import MomentPinVectorStore
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.models.conversation import Conversation, ConversationSummary, Memory, MomentPin


def _collection_ids(collection) -> Set[str]:
    if collection is None:
        return set()
    try:
        results = collection.get(include=[])
        return set(results.get("ids") or [])
    except Exception:
        return set()


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _iter_memory_issues(memories: Iterable[Memory], vector_ids: Set[str]) -> tuple[int, int, int]:
    expected = 0
    present = 0
    missing = 0

    for mem in memories:
        expected_vector = bool(mem.vector_id)
        if expected_vector:
            expected += 1
            if mem.vector_id in vector_ids:
                present += 1
            else:
                missing += 1
                print(
                    f"  MISSING memory_vector memory_id={mem.id} vector_id={mem.vector_id} "
                    f"type={mem.memory_type} status={mem.status}"
                )
    return expected, present, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify vector presence for summary, memories, and moment pins for one conversation."
    )
    parser.add_argument("conversation_id", help="Conversation UUID to inspect")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any expected vector is missing.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(Conversation.id == args.conversation_id).first()
        if not conversation:
            print(f"Conversation not found: {args.conversation_id}")
            return 1

        character_id = conversation.character_id
        print(f"conversation_id={conversation.id}")
        print(f"character_id={character_id}")

        vector_root = Path("data/vector_store")
        summary_store = ConversationSummaryVectorStore(vector_root)
        memory_store = VectorStore(vector_root)
        pin_store = MomentPinVectorStore(vector_root)

        # Summary check
        _print_header("Summary")
        current_summary = None
        if conversation.current_summary_id:
            current_summary = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.id == conversation.current_summary_id)
                .first()
            )
        if current_summary is None:
            current_summary = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.conversation_id == conversation.id)
                .order_by(ConversationSummary.created_at.desc())
                .first()
            )

        summary_vector = summary_store.get_summary(character_id, conversation.id)
        print(f"  sql_summary_exists={bool(current_summary)}")
        print(f"  current_summary_id={conversation.current_summary_id}")
        print(f"  vector_summary_exists={bool(summary_vector)}")

        # Memory check
        _print_header("Memories")
        memory_rows = (
            db.query(Memory)
            .filter(Memory.conversation_id == conversation.id)
            .order_by(Memory.created_at.desc())
            .all()
        )
        memory_collection = memory_store.get_collection(character_id)
        memory_vector_ids = _collection_ids(memory_collection)
        expected, present, missing = _iter_memory_issues(memory_rows, memory_vector_ids)
        print(f"  memories_sql={len(memory_rows)}")
        print(f"  memories_expected_vectors={expected}")
        print(f"  memories_vectors_present={present}")
        print(f"  memories_vectors_missing={missing}")

        # Moment pin check
        _print_header("Moment Pins")
        pin_rows = (
            db.query(MomentPin)
            .filter(MomentPin.conversation_id == conversation.id)
            .order_by(MomentPin.created_at.desc())
            .all()
        )
        pin_collection = pin_store.get_collection(character_id)
        pin_vector_ids = _collection_ids(pin_collection)
        pin_expected = len(pin_rows)
        pin_present = sum(1 for p in pin_rows if p.id in pin_vector_ids)
        pin_missing = pin_expected - pin_present
        for pin in pin_rows:
            if pin.id not in pin_vector_ids:
                print(f"  MISSING pin_vector pin_id={pin.id} archived={pin.archived}")
        print(f"  pins_sql={pin_expected}")
        print(f"  pins_vectors_present={pin_present}")
        print(f"  pins_vectors_missing={pin_missing}")

        summary_missing = 1 if current_summary and not summary_vector else 0
        total_missing = summary_missing + missing + pin_missing

        _print_header("Result")
        print(f"  total_missing_expected_vectors={total_missing}")
        if total_missing == 0:
            print("  status=OK")
            return 0

        print("  status=MISSING_VECTORS")
        return 2 if args.strict else 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
