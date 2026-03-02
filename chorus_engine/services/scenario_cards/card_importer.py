"""Scenario card importer (extract embedded metadata from PNG)."""

from __future__ import annotations

import uuid
from typing import Dict, Tuple

import yaml

from chorus_engine.services.character_cards.metadata_handler import PNGMetadataHandler
from chorus_engine.services.scenario_cards.models import ScenarioCard


class ScenarioCardImporter:
    KEYWORD = "chorus_scenario_card"

    def import_card(self, png_data: bytes) -> Tuple[Dict, list[str]]:
        payload = PNGMetadataHandler.read_text_chunk(png_data, self.KEYWORD)
        if not payload:
            raise ValueError("No scenario metadata found in image")
        parsed = yaml.safe_load(payload) or {}
        card = ScenarioCard(**parsed)
        data = card.data.model_dump(mode="json", exclude_none=True)
        if not data.get("id"):
            data["id"] = str(uuid.uuid4())
        return data, []
