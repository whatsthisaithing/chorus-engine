"""
Clear continuity bootstrap data from the database.

By default, removes relationship state, arcs, and cached packets.
Continuity preferences are preserved.

Usage:
    python utilities/continuity_bootstrap_runner/clear_bootstrap.py --character-id <id>
    python utilities/continuity_bootstrap_runner/clear_bootstrap.py --all --yes
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.continuity import (
    ContinuityRelationshipState,
    ContinuityArc,
    ContinuityBootstrapCache
)


def clear_for_character(db, character_id: str, dry_run: bool) -> dict:
    filters = (character_id,)
    counts = {}

    query_state = db.query(ContinuityRelationshipState).filter(
        ContinuityRelationshipState.character_id == character_id
    )
    query_arcs = db.query(ContinuityArc).filter(
        ContinuityArc.character_id == character_id
    )
    query_cache = db.query(ContinuityBootstrapCache).filter(
        ContinuityBootstrapCache.character_id == character_id
    )

    counts["relationship_states"] = query_state.count()
    counts["arcs"] = query_arcs.count()
    counts["cache"] = query_cache.count()

    if not dry_run:
        query_state.delete(synchronize_session=False)
        query_arcs.delete(synchronize_session=False)
        query_cache.delete(synchronize_session=False)
        db.commit()

    return counts


def clear_all(db, dry_run: bool) -> dict:
    counts = {
        "relationship_states": db.query(ContinuityRelationshipState).count(),
        "arcs": db.query(ContinuityArc).count(),
        "cache": db.query(ContinuityBootstrapCache).count()
    }

    if not dry_run:
        db.query(ContinuityRelationshipState).delete(synchronize_session=False)
        db.query(ContinuityArc).delete(synchronize_session=False)
        db.query(ContinuityBootstrapCache).delete(synchronize_session=False)
        db.commit()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear continuity bootstrap data (preferences are preserved)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--character-id", type=str, help="Character ID to clear")
    group.add_argument("--all", action="store_true", help="Clear all characters")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without modifying the database"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for --all to confirm deletion"
    )

    args = parser.parse_args()

    if args.all and not args.yes:
        print("Refusing to delete all continuity bootstrap data without --yes.")
        return

    db = SessionLocal()
    try:
        if args.character_id:
            counts = clear_for_character(db, args.character_id, args.dry_run)
            scope = f"character '{args.character_id}'"
        else:
            counts = clear_all(db, args.dry_run)
            scope = "all characters"

        action = "Would delete" if args.dry_run else "Deleted"
        print(f"{action} continuity bootstrap data for {scope}.")
        print(f"- Relationship states: {counts['relationship_states']}")
        print(f"- Arcs: {counts['arcs']}")
        print(f"- Cache: {counts['cache']}")
        print("Preferences preserved.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
