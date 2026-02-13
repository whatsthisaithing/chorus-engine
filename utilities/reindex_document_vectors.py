"""
Reindex document vectors from SQL document chunks.

Usage:
  python utilities/reindex_document_vectors.py --all --yes
  python utilities/reindex_document_vectors.py --character-id nova_custom --yes
  python utilities/reindex_document_vectors.py --document-id 42 --yes
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from sqlalchemy.orm import Session

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.document import Document, DocumentChunk
from chorus_engine.services.document_vector_store import DocumentVectorStore


def _build_metadata(doc: Document, chunk: DocumentChunk) -> dict:
    metadata = {
        "document_id": doc.id,
        "document_title": doc.title or doc.filename,
        "chunk_index": chunk.chunk_index,
        "chunk_method": chunk.chunk_method,
        "document_scope": doc.document_scope or "conversation",
    }
    if doc.character_id:
        metadata["character_id"] = doc.character_id
    if doc.conversation_id:
        metadata["conversation_id"] = doc.conversation_id
    if isinstance(chunk.metadata_json, dict):
        for key, value in chunk.metadata_json.items():
            if value is not None:
                metadata[key] = value
    return metadata


def _load_target_chunks(
    db: Session,
    character_id: str | None,
    document_id: int | None
) -> List[Tuple[DocumentChunk, Document]]:
    query = (
        db.query(DocumentChunk, Document)
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(Document.processing_status == "completed")
    )
    if character_id:
        query = query.filter(Document.character_id == character_id)
    if document_id is not None:
        query = query.filter(Document.id == document_id)
    return query.order_by(Document.id.asc(), DocumentChunk.chunk_index.asc()).all()


def main() -> int:
    parser = argparse.ArgumentParser(description="Reindex document vectors from SQL chunks.")
    parser.add_argument("--all", action="store_true", help="Reindex all documents.")
    parser.add_argument("--character-id", type=str, help="Reindex only documents for character.")
    parser.add_argument("--document-id", type=int, help="Reindex only one document ID.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without changes.")
    args = parser.parse_args()

    if not args.all and not args.character_id and args.document_id is None:
        print("Specify one scope: --all, --character-id, or --document-id.")
        return 1

    db = SessionLocal()
    vector_store = DocumentVectorStore("data/vector_store")
    try:
        rows = _load_target_chunks(db, args.character_id, args.document_id)
        if not rows:
            print("No completed document chunks found for the selected scope.")
            return 0

        target_document_ids = sorted({doc.id for _, doc in rows})
        print(f"Target documents: {len(target_document_ids)}")
        print(f"Target chunks: {len(rows)}")

        if args.dry_run:
            print("Dry run complete. No changes applied.")
            return 0

        if not args.yes:
            confirm = input("Type YES to proceed with reindex: ").strip()
            if confirm != "YES":
                print("Aborted.")
                return 0

        # Reset scope in vector collection.
        if args.all and not args.character_id and args.document_id is None:
            vector_store.clear_collection()
            print("Cleared full document vector collection.")
        else:
            deleted = 0
            for doc_id in target_document_ids:
                deleted += vector_store.delete_chunks_by_document(doc_id)
            print(f"Deleted {deleted} existing vectors in target scope.")

        chunk_ids: List[str] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        for chunk, doc in rows:
            chunk_ids.append(chunk.chunk_id)
            texts.append(chunk.content)
            metadatas.append(_build_metadata(doc, chunk))

        batch_size = 200
        added = 0
        for i in range(0, len(chunk_ids), batch_size):
            ids_batch = chunk_ids[i:i + batch_size]
            texts_batch = texts[i:i + batch_size]
            metas_batch = metadatas[i:i + batch_size]
            vector_store.add_chunks(
                chunk_ids=ids_batch,
                texts=texts_batch,
                metadatas=metas_batch
            )
            added += len(ids_batch)

        print(f"Reindex complete: added {added} document chunk vectors.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
