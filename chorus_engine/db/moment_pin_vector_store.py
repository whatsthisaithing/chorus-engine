"""Vector store for moment pin hot-layer retrieval."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from chorus_engine.db.chroma_config_fix import normalize_collection_configs

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


class MomentPinVectorStore:
    """Character-scoped vector store for moment pin hot summaries."""

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
        return f"moment_pins_{character_id}"

    def get_or_create_collection(self, character_id: str):
        name = self._collection_name(character_id)
        metadata = {
            "hnsw:space": "cosine",
            "character_id": character_id,
            "type": "moment_pins",
        }
        try:
            try:
                return self.client.get_collection(name=name)
            except Exception:
                pass
            return self.client.create_collection(name=name, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to get/create moment pin collection '{name}': {e}")
            raise

    def get_collection(self, character_id: str):
        name = self._collection_name(character_id)
        try:
            return self.client.get_collection(name=name)
        except Exception:
            return None

    def upsert_pin(
        self,
        character_id: str,
        pin_id: str,
        hot_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        collection = self.get_or_create_collection(character_id)
        try:
            collection.upsert(
                ids=[pin_id],
                documents=[hot_text],
                embeddings=[embedding],
                metadatas=[metadata or {}],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert moment pin vector {pin_id}: {e}")
            return False

    def delete_pin(self, character_id: str, pin_id: str) -> bool:
        collection = self.get_collection(character_id)
        if collection is None:
            return True
        try:
            collection.delete(ids=[pin_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete moment pin vector {pin_id}: {e}")
            return False

    def query_pins(
        self,
        character_id: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        collection = self.get_collection(character_id)
        if collection is None:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        try:
            return collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)
        except Exception as e:
            logger.error(f"Failed to query moment pin vectors for '{character_id}': {e}")
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections if col.name.startswith("moment_pins_")]
        except Exception as e:
            if "_type" in str(e):
                try:
                    db_path = self.persist_directory / "chroma.sqlite3"
                    conn = sqlite3.connect(str(db_path))
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT name FROM collections WHERE name LIKE 'moment_pins_%' ORDER BY name"
                    )
                    names = [row[0] for row in cur.fetchall()]
                    conn.close()
                    return names
                except Exception as fallback_error:
                    logger.error(f"Failed fallback moment pin list_collections: {fallback_error}")
            logger.error(f"Failed to list moment pin collections: {e}")
            return []
