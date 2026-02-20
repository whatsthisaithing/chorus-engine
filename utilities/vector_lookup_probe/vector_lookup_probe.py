from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.db.database import SessionLocal
from chorus_engine.db.moment_pin_vector_store import MomentPinVectorStore
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.models.conversation import Conversation, Thread
from chorus_engine.services.conversation_context_retrieval import ConversationContextRetrievalService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.memory_retrieval import MemoryRetrievalService
from chorus_engine.services.moment_pin_retrieval_service import MomentPinRetrievalService


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(text: Optional[str], max_len: int) -> str:
    if not text:
        return ""
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


@dataclass
class ProbeScope:
    character_id: str
    user_id: str
    conversation_id: Optional[str]
    thread_id: Optional[str]
    conversation_source: Optional[str]


class SessionWriter:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.human_path = output_dir / f"vector_lookup_probe_{stamp}.log"
        self.jsonl_path = output_dir / f"vector_lookup_probe_{stamp}.jsonl"
        self._human = self.human_path.open("w", encoding="utf-8")
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._human.close()
        self._jsonl.close()

    def line(self, text: str = "") -> None:
        print(text)
        self._human.write(text + "\n")
        self._human.flush()

    def event(self, payload: Dict[str, Any]) -> None:
        self._jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._jsonl.flush()


def _resolve_scope(db, character_id: str, user_id: str, conversation_id: Optional[str]) -> ProbeScope:
    if not conversation_id:
        return ProbeScope(
            character_id=character_id,
            user_id=user_id,
            conversation_id=None,
            thread_id=None,
            conversation_source=None,
        )

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise ValueError(f"Conversation not found: {conversation_id}")
    if conversation.character_id != character_id:
        raise ValueError(
            f"Conversation '{conversation_id}' belongs to character '{conversation.character_id}', "
            f"not '{character_id}'."
        )
    thread = (
        db.query(Thread)
        .filter(Thread.conversation_id == conversation_id)
        .order_by(Thread.updated_at.desc())
        .first()
    )
    return ProbeScope(
        character_id=character_id,
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id=thread.id if thread else None,
        conversation_source=getattr(conversation, "source", None),
    )


def _report_scope(writer: SessionWriter, scope: ProbeScope, top_n: int, preview_chars: int) -> None:
    writer.line("=== Vector Lookup Probe Session ===")
    writer.line(f"started_utc: {_now_iso()}")
    writer.line(f"character_id: {scope.character_id}")
    writer.line(f"user_id: {scope.user_id}")
    writer.line(f"conversation_id: {scope.conversation_id or '(none)'}")
    writer.line(f"thread_id: {scope.thread_id or '(none)'}")
    writer.line(f"conversation_source: {scope.conversation_source or '(none)'}")
    writer.line(f"top_n: {top_n}")
    writer.line(f"preview_chars: {preview_chars}")
    writer.line("")


def _run_summary_lookup(
    context_service: ConversationContextRetrievalService,
    embedder: EmbeddingService,
    summary_store: ConversationSummaryVectorStore,
    scope: ProbeScope,
    query: str,
    top_n: int,
    preview_chars: int,
) -> Dict[str, Any]:
    selected, _ = context_service.retrieve_relevant_summaries(
        character_id=scope.character_id,
        user_message=query,
        current_conversation_id=scope.conversation_id,
        max_summaries=top_n,
        token_budget=None,
    )
    query_embedding = embedder.embed(query)
    raw = summary_store.search_conversations(
        character_id=scope.character_id,
        query_embedding=query_embedding,
        n_results=top_n,
        transient_retry_attempts=0,
    )
    raw_ids = (raw.get("ids") or [[]])[0]
    raw_distances = (raw.get("distances") or [[]])[0]
    raw_docs = (raw.get("documents") or [[]])[0]
    raw_meta = (raw.get("metadatas") or [[]])[0]

    raw_rows: List[Dict[str, Any]] = []
    for idx, conv_id in enumerate(raw_ids):
        distance = float(raw_distances[idx]) if idx < len(raw_distances) else None
        similarity = (1.0 - (distance / 2.0)) if distance is not None else None
        meta = raw_meta[idx] if idx < len(raw_meta) else {}
        raw_rows.append(
            {
                "rank": idx + 1,
                "conversation_id": conv_id,
                "distance": distance,
                "similarity": similarity,
                "title": meta.get("title") if isinstance(meta, dict) else None,
                "preview": _preview(raw_docs[idx] if idx < len(raw_docs) else "", preview_chars),
            }
        )

    selected_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected):
        selected_rows.append(
            {
                "rank": idx + 1,
                "conversation_id": row.conversation_id,
                "similarity": row.similarity,
                "title": row.title,
                "preview": _preview(row.summary, preview_chars),
                "tone": row.tone,
                "message_count": row.message_count,
                "created_at": row.created_at,
            }
        )

    return {
        "ok": True,
        "selected_count": len(selected_rows),
        "selected": selected_rows,
        "raw_count": len(raw_rows),
        "raw": raw_rows,
    }


def _run_memory_lookup(
    memory_service: MemoryRetrievalService,
    scope: ProbeScope,
    query: str,
    top_n: int,
    preview_chars: int,
) -> Dict[str, Any]:
    retrieved = memory_service.retrieve_memories(
        query=query,
        character_id=scope.character_id,
        conversation_id=scope.conversation_id,
        thread_id=scope.thread_id,
        token_budget=10000,
        max_memories=top_n,
        conversation_source=scope.conversation_source,
    )

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(retrieved):
        mem = item.memory
        rows.append(
            {
                "rank": idx + 1,
                "memory_id": mem.id,
                "memory_type": str(mem.memory_type),
                "similarity": item.similarity,
                "rank_score": item.rank_score,
                "priority": mem.priority,
                "status": mem.status,
                "created_at": str(mem.created_at) if mem.created_at else None,
                "conversation_id": mem.conversation_id,
                "thread_id": mem.thread_id,
                "preview": _preview(mem.content, preview_chars),
            }
        )
    return {"ok": True, "count": len(rows), "rows": rows}


def _run_pin_lookup(
    pin_service: MomentPinRetrievalService,
    embedder: EmbeddingService,
    pin_store: MomentPinVectorStore,
    scope: ProbeScope,
    query: str,
    top_n: int,
    inject_k: int,
    preview_chars: int,
) -> Dict[str, Any]:
    retrieved = pin_service.retrieve(
        user_id=scope.user_id,
        character_id=scope.character_id,
        query=query,
        top_n=top_n,
        inject_k=inject_k,
        recent_pin_ids=None,
    )

    query_embedding = embedder.embed(query)
    raw = pin_store.query_pins(
        character_id=scope.character_id,
        query_embedding=query_embedding,
        n_results=top_n,
    )
    raw_ids = (raw.get("ids") or [[]])[0]
    raw_distances = (raw.get("distances") or [[]])[0]
    raw_docs = (raw.get("documents") or [[]])[0]

    raw_rows: List[Dict[str, Any]] = []
    for idx, pin_id in enumerate(raw_ids):
        distance = float(raw_distances[idx]) if idx < len(raw_distances) else None
        similarity = (1.0 - (distance / 2.0)) if distance is not None else None
        raw_rows.append(
            {
                "rank": idx + 1,
                "pin_id": pin_id,
                "distance": distance,
                "similarity": similarity,
                "preview": _preview(raw_docs[idx] if idx < len(raw_docs) else "", preview_chars),
            }
        )

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(retrieved):
        pin = item.pin
        rows.append(
            {
                "rank": idx + 1,
                "pin_id": pin.id,
                "similarity": item.similarity,
                "score": item.score,
                "reinforcement_score": pin.reinforcement_score,
                "turns_since_reinforcement": pin.turns_since_reinforcement,
                "conversation_id": pin.conversation_id,
                "created_at": str(pin.created_at) if pin.created_at else None,
                "what_happened_preview": _preview(pin.what_happened, preview_chars),
                "quote_preview": _preview(pin.quote_snippet, preview_chars),
            }
        )
    return {
        "ok": True,
        "selected_count": len(rows),
        "selected": rows,
        "raw_count": len(raw_rows),
        "raw": raw_rows,
    }


def _print_rows(writer: SessionWriter, title: str, rows: List[Dict[str, Any]]) -> None:
    writer.line(title)
    if not rows:
        writer.line("  (none)")
        return
    for row in rows:
        writer.line(f"  {json.dumps(row, ensure_ascii=False)}")


def _run_query(
    writer: SessionWriter,
    context_service: ConversationContextRetrievalService,
    memory_service: MemoryRetrievalService,
    pin_service: MomentPinRetrievalService,
    embedder: EmbeddingService,
    summary_store: ConversationSummaryVectorStore,
    pin_store: MomentPinVectorStore,
    scope: ProbeScope,
    query: str,
    top_n: int,
    inject_k: int,
    preview_chars: int,
) -> None:
    writer.line("")
    writer.line(f"=== Query @ {_now_iso()} ===")
    writer.line(f"query: {query}")

    event: Dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "query": query,
        "scope": asdict(scope),
        "summary_lookup": None,
        "memory_lookup": None,
        "moment_pin_lookup": None,
    }

    try:
        summaries = _run_summary_lookup(
            context_service=context_service,
            embedder=embedder,
            summary_store=summary_store,
            scope=scope,
            query=query,
            top_n=top_n,
            preview_chars=preview_chars,
        )
    except Exception as exc:
        summaries = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        memories = _run_memory_lookup(
            memory_service=memory_service,
            scope=scope,
            query=query,
            top_n=top_n,
            preview_chars=preview_chars,
        )
    except Exception as exc:
        memories = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        pins = _run_pin_lookup(
            pin_service=pin_service,
            embedder=embedder,
            pin_store=pin_store,
            scope=scope,
            query=query,
            top_n=top_n,
            inject_k=inject_k,
            preview_chars=preview_chars,
        )
    except Exception as exc:
        pins = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    event["summary_lookup"] = summaries
    event["memory_lookup"] = memories
    event["moment_pin_lookup"] = pins
    writer.event(event)

    writer.line("--- Summary Lookup ---")
    if summaries.get("ok"):
        writer.line(f"selected_count: {summaries['selected_count']}")
        _print_rows(writer, "selected:", summaries["selected"])
        writer.line(f"raw_count: {summaries['raw_count']}")
        _print_rows(writer, "raw_candidates:", summaries["raw"])
    else:
        writer.line(f"[FAILED] {summaries.get('error')}")

    writer.line("--- Memory Lookup ---")
    if memories.get("ok"):
        writer.line(f"count: {memories['count']}")
        _print_rows(writer, "selected:", memories["rows"])
    else:
        writer.line(f"[FAILED] {memories.get('error')}")

    writer.line("--- Moment Pin Lookup ---")
    if pins.get("ok"):
        writer.line(f"selected_count: {pins['selected_count']}")
        _print_rows(writer, "selected:", pins["selected"])
        writer.line(f"raw_count: {pins['raw_count']}")
        _print_rows(writer, "raw_candidates:", pins["raw"])
    else:
        writer.line(f"[FAILED] {pins.get('error')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive probe for conversation summary, memory, and moment pin lookups. "
            "Writes terminal output plus timestamped session logs in data/vector_lookup_tests/."
        )
    )
    parser.add_argument("--character", required=True, help="Character ID (required).")
    parser.add_argument(
        "--user-id",
        default="user:local:owner",
        help="User identity scope. Default: user:local:owner",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Optional conversation ID to mirror conversation-scoped retrieval and source filtering.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N candidates/results per lookup. Default: 10",
    )
    parser.add_argument(
        "--inject-k",
        type=int,
        default=3,
        help="Moment pin selected count (inject_k). Default: 3",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=220,
        help="Preview length for content snippets. Default: 220",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Run non-interactively for one or more queries (can pass multiple times).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db = SessionLocal()
    writer = SessionWriter(Path("data/vector_lookup_tests"))
    try:
        scope = _resolve_scope(
            db=db,
            character_id=args.character.strip(),
            user_id=args.user_id.strip(),
            conversation_id=args.conversation_id.strip() if args.conversation_id else None,
        )

        vector_root = Path("data/vector_store")
        summary_store = ConversationSummaryVectorStore(vector_root)
        memory_store = VectorStore(vector_root)
        pin_store = MomentPinVectorStore(vector_root)
        embedder = EmbeddingService()

        context_service = ConversationContextRetrievalService(
            summary_vector_store=summary_store,
            embedding_service=embedder,
        )
        memory_service = MemoryRetrievalService(
            db=db,
            vector_store=memory_store,
            embedder=embedder,
        )
        pin_service = MomentPinRetrievalService(
            db=db,
            vector_store=pin_store,
            embedder=embedder,
        )

        _report_scope(
            writer=writer,
            scope=scope,
            top_n=args.top_n,
            preview_chars=args.preview_chars,
        )
        writer.line(f"human_log: {writer.human_path}")
        writer.line(f"jsonl_log: {writer.jsonl_path}")

        if args.query:
            for raw_query in args.query:
                query = raw_query.strip()
                if not query:
                    continue
                _run_query(
                    writer=writer,
                    context_service=context_service,
                    memory_service=memory_service,
                    pin_service=pin_service,
                    embedder=embedder,
                    summary_store=summary_store,
                    pin_store=pin_store,
                    scope=scope,
                    query=query,
                    top_n=args.top_n,
                    inject_k=args.inject_k,
                    preview_chars=args.preview_chars,
                )
            return 0

        writer.line("")
        writer.line("Enter a query and press Enter. Type 'exit' or ':q' to stop.")
        while True:
            try:
                raw = input("query> ")
            except EOFError:
                writer.line("\nEOF received, ending session.")
                break
            query = raw.strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                writer.line("Exiting.")
                break
            _run_query(
                writer=writer,
                context_service=context_service,
                memory_service=memory_service,
                pin_service=pin_service,
                embedder=embedder,
                summary_store=summary_store,
                pin_store=pin_store,
                scope=scope,
                query=query,
                top_n=args.top_n,
                inject_k=args.inject_k,
                preview_chars=args.preview_chars,
            )
        return 0
    finally:
        writer.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
