"""Scenario card exporter (PNG + embedded metadata)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

from chorus_engine.services.character_cards.metadata_handler import PNGMetadataHandler
from chorus_engine.services.scenario_cards.models import ScenarioCard, ScenarioCardData


class ScenarioCardExporter:
    KEYWORD = "chorus_scenario_card"

    def __init__(
        self,
        images_dir: Path = Path("data/scenario_images"),
        default_avatar_path: Path = Path("data/character_images/default.png"),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.default_avatar_path = Path(default_avatar_path)

    def _blank_png(self) -> bytes:
        img = Image.new("RGB", (512, 512), color=(90, 90, 90))
        output = BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    def _resolve_image(self, image_ref: str | None) -> bytes:
        if image_ref:
            path = self.images_dir / Path(str(image_ref)).name
            if path.exists():
                return PNGMetadataHandler.extract_image(str(path))
        if self.default_avatar_path.exists():
            return PNGMetadataHandler.extract_image(str(self.default_avatar_path))
        return self._blank_png()

    def export_card(self, scenario: dict) -> bytes:
        card = ScenarioCard(
            data=ScenarioCardData(
                id=str(scenario.get("id") or ""),
                title=str(scenario.get("title") or ""),
                description=str(scenario.get("description") or ""),
                scenario_text=str(scenario.get("scenario_text") or ""),
                tags=list(scenario.get("tags") or []),
                version=scenario.get("version"),
                created_at=(scenario.get("created_at").isoformat() if hasattr(scenario.get("created_at"), "isoformat") else scenario.get("created_at")),
                updated_at=(scenario.get("updated_at").isoformat() if hasattr(scenario.get("updated_at"), "isoformat") else scenario.get("updated_at")),
                image_ref=scenario.get("image_ref"),
                facts=scenario.get("facts"),
                injection=scenario.get("injection"),
            )
        )
        payload = yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        image = self._resolve_image(scenario.get("image_ref"))
        return PNGMetadataHandler.write_text_chunk(image, self.KEYWORD, payload)
