"""
Core Memory Loader Service

Loads immutable character backstory memories from character YAML files
and stores them in the database with vector embeddings.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from chorus_engine.config.loader import ConfigLoader
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.models.conversation import Memory, MemoryType
from chorus_engine.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class CoreMemoryLoader:
    """Loads and reconciles character core memories from YAML to DB/vector store."""

    def __init__(
        self,
        db: Session,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingService] = None,
        persist_directory: Path = Path("data/vector_store"),
    ):
        self.db = db
        self.vector_store = vector_store or VectorStore(persist_directory=persist_directory)
        self.embedder = embedder or EmbeddingService()
        self.config_loader = ConfigLoader()

    @staticmethod
    def _priority_map_to_int(priority: str) -> int:
        mapping = {"low": 60, "medium": 80, "high": 95}
        return mapping.get(priority or "medium", 80)

    @staticmethod
    def _priority_map_to_label(priority: Optional[int]) -> str:
        value = int(priority or 80)
        if value >= 90:
            return "high"
        if value <= 70:
            return "low"
        return "medium"

    def _normalize_yaml_payload(self, yaml_core_memories) -> list[dict]:
        normalized = []
        for item in yaml_core_memories or []:
            content = (item.content or "").strip()
            if not content:
                continue
            tags = [str(tag).strip() for tag in (item.tags or []) if str(tag).strip()]
            normalized.append(
                {
                    "content": content,
                    "tags": tags,
                    "embedding_priority": item.embedding_priority or "medium",
                }
            )
        return normalized

    def _normalize_db_payload(self, db_core_memories: list[Memory]) -> list[dict]:
        def sort_key(memory: Memory) -> tuple:
            meta = memory.meta_data if isinstance(memory.meta_data, dict) else {}
            yaml_index = meta.get("yaml_index")
            if isinstance(yaml_index, int):
                return (0, yaml_index)
            created = memory.created_at.isoformat() if memory.created_at else ""
            return (1, created)

        normalized = []
        for memory in sorted(db_core_memories, key=sort_key):
            content = (memory.content or "").strip()
            if not content:
                continue
            tags = [str(tag).strip() for tag in (memory.tags or []) if str(tag).strip()]
            normalized.append(
                {
                    "content": content,
                    "tags": tags,
                    "embedding_priority": self._priority_map_to_label(memory.priority),
                }
            )
        return normalized

    def _insert_core_memories(self, character_id: str, yaml_core_memories) -> int:
        core_memories = list(yaml_core_memories or [])
        if not core_memories:
            return 0

        memory_contents = []
        memory_metadata_list = []
        core_input = []

        for idx, core_mem in enumerate(core_memories):
            content = (core_mem.content or "").strip()
            if not content:
                continue
            memory_contents.append(content)
            core_input.append(core_mem)
            priority = self._priority_map_to_int(core_mem.embedding_priority)
            tags_str = ",".join(core_mem.tags) if core_mem.tags else ""
            memory_metadata_list.append(
                {
                    "tags": tags_str,
                    "priority": priority,
                    "source": "character_yaml",
                    "index": idx,
                }
            )

        if not memory_contents:
            return 0

        logger.info("Generating embeddings for core memories...")
        embeddings = self.embedder.embed_batch(memory_contents)
        vector_ids = [str(uuid.uuid4()) for _ in memory_contents]

        logger.info("Adding core memories to vector store...")
        success = self.vector_store.add_memories(
            character_id=character_id,
            memory_ids=vector_ids,
            contents=memory_contents,
            embeddings=embeddings,
            metadatas=memory_metadata_list,
        )
        if not success:
            raise RuntimeError(f"Failed to add memories to vector store for {character_id}")

        logger.info("Storing core memories in database...")
        new_memories = []
        for idx, (content, vector_id, metadata, core_mem) in enumerate(
            zip(memory_contents, vector_ids, memory_metadata_list, core_input)
        ):
            new_memories.append(
                Memory(
                    character_id=character_id,
                    memory_type=MemoryType.CORE,
                    content=content,
                    vector_id=vector_id,
                    embedding_model=self.embedder.model_name,
                    priority=metadata["priority"],
                    tags=core_mem.tags,
                    meta_data={"source": "character_yaml", "yaml_index": idx},
                    source_kind="character_yaml_core",
                )
            )

        self.db.add_all(new_memories)
        self.db.commit()
        return len(new_memories)

    def reconcile_character_core_memories(self, character_id: str) -> dict:
        """
        Reconcile DB/vector core memories against character YAML (authoritative).

        Returns:
            dict with: in_sync, loaded, deleted, yaml_count, db_count
        """
        character = self.config_loader.load_character(character_id)
        if not character:
            raise ValueError(f"Character not found: {character_id}")

        yaml_core_memories = character.core_memories or []
        existing_core_memories = self.get_core_memories(character_id)

        yaml_payload = self._normalize_yaml_payload(yaml_core_memories)
        db_payload = self._normalize_db_payload(existing_core_memories)

        if yaml_payload == db_payload:
            logger.info(
                "Core memories already in sync for %s (yaml=%s, db=%s)",
                character_id,
                len(yaml_payload),
                len(db_payload),
            )
            return {
                "in_sync": True,
                "loaded": 0,
                "deleted": 0,
                "yaml_count": len(yaml_payload),
                "db_count": len(db_payload),
            }

        deleted = self.delete_core_memories(character_id)
        loaded = self._insert_core_memories(character_id, yaml_core_memories)
        logger.info(
            "Core memory reconcile complete for %s: deleted=%s, loaded=%s, yaml=%s",
            character_id,
            deleted,
            loaded,
            len(yaml_payload),
        )
        return {
            "in_sync": False,
            "loaded": loaded,
            "deleted": deleted,
            "yaml_count": len(yaml_payload),
            "db_count": len(db_payload),
        }

    def load_character_core_memories(self, character_id: str) -> int:
        """Backwards-compatible wrapper that now performs authoritative reconcile."""
        logger.info(f"Loading core memories for character: {character_id}")
        result = self.reconcile_character_core_memories(character_id)
        return int(result.get("loaded", 0))

    def get_core_memories(self, character_id: str) -> List[Memory]:
        """Get all core memories for a character."""
        return (
            self.db.query(Memory)
            .filter(
                Memory.character_id == character_id,
                Memory.memory_type == MemoryType.CORE,
            )
            .order_by(Memory.priority.desc())
            .all()
        )

    def delete_core_memories(self, character_id: str) -> int:
        """Delete all core memories for a character in DB and vector store."""
        logger.warning(f"Deleting core memories for character: {character_id}")
        memories = self.get_core_memories(character_id)
        if not memories:
            return 0

        vector_ids = [m.vector_id for m in memories if m.vector_id]
        if vector_ids:
            self.vector_store.delete_memories(character_id, vector_ids)
            logger.info(f"Deleted {len(vector_ids)} vectors from vector store")

        count = (
            self.db.query(Memory)
            .filter(
                Memory.character_id == character_id,
                Memory.memory_type == MemoryType.CORE,
            )
            .delete()
        )
        self.db.commit()
        logger.info(f"Deleted {count} core memories for {character_id}")
        return count

    def reload_core_memories(self, character_id: str) -> int:
        """Reload core memories from YAML (authoritative reconcile)."""
        logger.info(f"Reloading core memories for character: {character_id}")
        result = self.reconcile_character_core_memories(character_id)
        return int(result.get("loaded", 0))
