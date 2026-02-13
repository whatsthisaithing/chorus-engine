"""
Read-only vector health report.

Usage:
  python utilities/vector_health_check.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.services.document_vector_store import DocumentVectorStore
from chorus_engine.utils.startup_sync import run_vector_health_checks


def main() -> int:
    db = SessionLocal()
    try:
        vector_store = VectorStore(Path("data/vector_store"))
        summary_store = ConversationSummaryVectorStore(Path("data/vector_store"))
        document_store = DocumentVectorStore("data/vector_store")

        report = run_vector_health_checks(
            db_session=db,
            vector_store=vector_store,
            summary_vector_store=summary_store,
            document_vector_store=document_store
        )

        # Additional drift counters
        drift = {}
        try:
            drift["document_vector_count"] = document_store.collection.count()
        except Exception as e:
            drift["document_vector_count_error"] = str(e)
        report["drift"] = drift

        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("ok") else 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
