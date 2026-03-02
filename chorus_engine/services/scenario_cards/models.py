"""Pydantic models for scenario card metadata payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScenarioCardData(BaseModel):
    id: str
    title: str
    description: str = ""
    scenario_text: str
    tags: List[str] = Field(default_factory=list)
    version: Optional[Any] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    image_ref: Optional[str] = None
    facts: Optional[List[Dict[str, Any]]] = None
    injection: Optional[Dict[str, Any]] = None


class ScenarioCard(BaseModel):
    spec: str = "chorus_scenario_card_v1"
    spec_version: str = "1.0"
    exported_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    data: ScenarioCardData
