"""Vector store for episodic conversation segment summaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from chorus_engine.db.chroma_config_fix import normalize_collection_configs

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


class ConversationSegmentVectorStore:
    """Chroma-backed storage for segment summaries (v1 write-only in prompt path)."""

    def __init__(self, persist_directory: Path):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        normalize_collection_configs(self.persist_directory)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    def _collection_name(self, character_id: str) -> str:
        return f"segment_summaries_{character_id}"

    def get_collection(self, character_id: str):
        try:
            return self.client.get_collection(self._collection_name(character_id))
        except Exception:
            return None

    def get_or_create_collection(self, character_id: str):
        metadata = {
            "hnsw:space": "cosine",
            "character_id": character_id,
            "type": "segment_summaries",
        }
        collection = self.get_collection(character_id)
        if collection is not None:
            return collection
        return self.client.create_collection(name=self._collection_name(character_id), metadata=metadata)

    def upsert_segment_summary(
        self,
        *,
        character_id: str,
        segment_id: str,
        summary_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        collection = self.get_or_create_collection(character_id)
        try:
            collection.upsert(
                ids=[segment_id],
                documents=[summary_text],
                embeddings=[embedding],
                metadatas=[metadata or {}],
            )
            return True
        except Exception as exc:
            logger.error("Failed to upsert segment summary vector %s: %s", segment_id, exc)
            return False

    def delete_segment_summaries(self, *, character_id: str, segment_ids: Iterable[str]) -> int:
        collection = self.get_collection(character_id)
        if collection is None:
            return 0
        ids = [str(sid) for sid in segment_ids if sid]
        if not ids:
            return 0
        deleted = 0
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            try:
                collection.delete(ids=chunk)
                deleted += len(chunk)
            except Exception as exc:
                logger.warning("Failed deleting segment summary chunk for %s: %s", character_id, exc)
        return deleted
