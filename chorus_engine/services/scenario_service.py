"""Scenario storage and validation service."""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


MAX_SCENARIO_TEXT_LEN = 6000


class ScenarioRecord(BaseModel):
    """On-disk scenario schema for v1."""

    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    scenario_text: str = Field(min_length=1, max_length=MAX_SCENARIO_TEXT_LEN)
    tags: List[str] = Field(default_factory=list)
    version: Optional[Any] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    image_ref: Optional[str] = None

    # Reserved for future versions; v1 stores but ignores behaviorally.
    facts: Optional[List[Dict[str, Any]]] = None
    injection: Optional[Dict[str, Any]] = None

    @field_validator("title", "description", "scenario_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("Scenario id is required")
        return value


class ScenarioService:
    """CRUD and storage operations for character-bound scenarios."""

    def __init__(self, base_dir: Path = Path("data/scenarios")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _char_dir(self, character_id: str) -> Path:
        safe_id = str(character_id or "").strip()
        if not safe_id:
            raise ValueError("character_id is required")
        target = self.base_dir / safe_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _scenario_path(self, character_id: str, scenario_id: str) -> Path:
        return self._char_dir(character_id) / f"scn_{scenario_id}.yaml"

    def _write_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            yaml.safe_dump(data, tmp, sort_keys=False, allow_unicode=True)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def _load_record(self, path: Path) -> ScenarioRecord:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return ScenarioRecord(**payload)

    def list_scenarios(self, character_id: str) -> List[ScenarioRecord]:
        items: List[ScenarioRecord] = []
        char_dir = self._char_dir(character_id)
        for path in sorted(char_dir.glob("scn_*.yaml")):
            try:
                items.append(self._load_record(path))
            except Exception:
                continue
        items.sort(key=lambda row: (row.updated_at or row.created_at or datetime.min), reverse=True)
        return items

    def get_scenario(self, character_id: str, scenario_id: str) -> Optional[ScenarioRecord]:
        path = self._scenario_path(character_id, scenario_id)
        if not path.exists():
            return None
        return self._load_record(path)

    def create_scenario(self, character_id: str, payload: Dict[str, Any]) -> ScenarioRecord:
        now = datetime.utcnow()
        scenario_id = str(payload.get("id") or uuid.uuid4())
        record = ScenarioRecord(
            id=scenario_id,
            title=payload.get("title", ""),
            description=payload.get("description", ""),
            scenario_text=payload.get("scenario_text", ""),
            tags=payload.get("tags") or [],
            version=payload.get("version"),
            created_at=payload.get("created_at") or now,
            updated_at=payload.get("updated_at") or now,
            image_ref=payload.get("image_ref"),
            facts=payload.get("facts"),
            injection=payload.get("injection"),
        )
        path = self._scenario_path(character_id, record.id)
        if path.exists():
            raise ValueError(f"Scenario '{record.id}' already exists")
        self._write_atomic(path, record.model_dump(mode="json", exclude_none=True))
        return record

    def update_scenario(self, character_id: str, scenario_id: str, updates: Dict[str, Any]) -> ScenarioRecord:
        existing = self.get_scenario(character_id, scenario_id)
        if not existing:
            raise ValueError("Scenario not found")
        merged = existing.model_dump(mode="json")
        merged.update(dict(updates or {}))
        merged["id"] = scenario_id
        merged["created_at"] = existing.created_at
        merged["updated_at"] = datetime.utcnow()
        updated = ScenarioRecord(**merged)
        self._write_atomic(
            self._scenario_path(character_id, scenario_id),
            updated.model_dump(mode="json", exclude_none=True),
        )
        return updated

    def delete_scenario(self, character_id: str, scenario_id: str) -> bool:
        path = self._scenario_path(character_id, scenario_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def duplicate_scenario(self, character_id: str, scenario_id: str) -> ScenarioRecord:
        existing = self.get_scenario(character_id, scenario_id)
        if not existing:
            raise ValueError("Scenario not found")
        data = existing.model_dump(mode="json", exclude_none=True)
        data["id"] = str(uuid.uuid4())
        data["title"] = f"{existing.title} (Copy)"
        now = datetime.utcnow()
        data["created_at"] = now
        data["updated_at"] = now
        return self.create_scenario(character_id, data)

    def ensure_exists(self, character_id: str, scenario_id: str) -> ScenarioRecord:
        item = self.get_scenario(character_id, scenario_id)
        if not item:
            raise ValueError("Scenario not found")
        return item

    def parse_scenario_payload(self, payload: Dict[str, Any]) -> ScenarioRecord:
        try:
            return ScenarioRecord(**(payload or {}))
        except ValidationError as e:
            raise ValueError(str(e)) from e
