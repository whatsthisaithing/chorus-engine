"""Chroma collection config normalization helpers."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_collection_config(space: str = "l2") -> str:
    payload = {
        "hnsw_configuration": {
            "space": space,
            "ef_construction": 100,
            "ef_search": 10,
            "num_threads": 20,
            "M": 16,
            "resize_factor": 1.2,
            "batch_size": 100,
            "sync_threshold": 1000,
            "_type": "HNSWConfigurationInternal",
        },
        "_type": "CollectionConfigurationInternal",
    }
    return json.dumps(payload)


def normalize_collection_configs(persist_directory: Path) -> int:
    """
    Normalize malformed collection config JSON rows in Chroma sqlite sysdb.

    Returns number of rows updated.
    """
    db_path = persist_directory / "chroma.sqlite3"
    if not db_path.exists():
        return 0

    updated = 0
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT name, config_json_str FROM collections"
            )
        except sqlite3.Error:
            # Chroma schema not initialized yet (or database is mid-migration).
            return 0
        rows = cur.fetchall()
        for name, config_json_str in rows:
            needs_fix = False
            if not config_json_str or config_json_str.strip() == "{}":
                needs_fix = True
            else:
                try:
                    parsed = json.loads(config_json_str)
                    if not isinstance(parsed, dict) or "_type" not in parsed:
                        needs_fix = True
                except Exception:
                    needs_fix = True

            if not needs_fix:
                continue

            # Preserve space hint where possible.
            space = "l2"
            try:
                cur.execute(
                    "SELECT str_value FROM collection_metadata WHERE collection_id="
                    "(SELECT id FROM collections WHERE name=?) AND key='hnsw:space'",
                    (name,),
                )
                meta_row = cur.fetchone()
                if meta_row and meta_row[0] in ("l2", "cosine", "ip"):
                    space = meta_row[0]
            except Exception:
                pass

            cur.execute(
                "UPDATE collections SET config_json_str=? WHERE name=?",
                (_default_collection_config(space), name),
            )
            updated += 1

        if updated > 0:
            conn.commit()
            logger.warning(
                f"[VECTOR_HEALTH] Normalized malformed Chroma collection configs: {updated}"
            )
    finally:
        conn.close()

    return updated
