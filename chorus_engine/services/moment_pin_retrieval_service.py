"""Moment pin retrieval and ranking for hot-layer injection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from chorus_engine.models.conversation import MomentPin
from chorus_engine.repositories.moment_pin_repository import MomentPinRepository
from chorus_engine.db.moment_pin_vector_store import MomentPinVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.tool_payload import MOMENT_PIN_COLD_RECALL_TOOL

logger = logging.getLogger(__name__)


@dataclass
class RetrievedMomentPin:
    pin: MomentPin
    similarity: float
    score: float


class MomentPinRetrievalService:
    """Retrieve and rank moment pins for prompt injection."""

    def __init__(
        self,
        db: Session,
        vector_store: MomentPinVectorStore,
        embedder: EmbeddingService,
    ):
        self.db = db
        self.repo = MomentPinRepository(db)
        self.vector_store = vector_store
        self.embedder = embedder

    def _normalized_reinforcement(self, pin: MomentPin) -> float:
        raw = float(pin.reinforcement_score or 1.0)
        return max(0.0, min(1.0, raw / 5.0))

    def retrieve(
        self,
        user_id: str,
        character_id: str,
        query: str,
        top_n: int = 10,
        inject_k: int = 3,
        recent_pin_ids: Optional[List[str]] = None,
    ) -> List[RetrievedMomentPin]:
        if not query.strip():
            return []

        query_embedding = self.embedder.embed(query)
        results = self.vector_store.query_pins(character_id=character_id, query_embedding=query_embedding, n_results=top_n)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not ids:
            return []

        # Build similarity map from vector results and load DB rows.
        similarity_by_id = {}
        for pin_id, distance in zip(ids, distances):
            similarity_by_id[pin_id] = 1 - (float(distance) / 2.0)

        pins = (
            self.db.query(MomentPin)
            .filter(
                MomentPin.id.in_(ids),
                MomentPin.user_id == user_id,
                MomentPin.character_id == character_id,
                MomentPin.archived == 0,
            )
            .all()
        )

        ranked: List[RetrievedMomentPin] = []
        for pin in pins:
            similarity = similarity_by_id.get(pin.id, 0.0)
            score = 0.8 * similarity + 0.2 * self._normalized_reinforcement(pin)
            ranked.append(RetrievedMomentPin(pin=pin, similarity=similarity, score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)

        forced_recent_item: Optional[RetrievedMomentPin] = None
        if recent_pin_ids:
            for recent_pin_id in recent_pin_ids:
                recent_pin = (
                    self.db.query(MomentPin)
                    .filter(
                        MomentPin.id == recent_pin_id,
                        MomentPin.user_id == user_id,
                        MomentPin.character_id == character_id,
                        MomentPin.archived == 0,
                    )
                    .first()
                )
                if not recent_pin:
                    continue

                ranked_match = next((item for item in ranked if item.pin.id == recent_pin.id), None)
                if ranked_match:
                    forced_recent_item = ranked_match
                else:
                    # Deterministic continuity carryover: include most recently used valid pin
                    # even if semantic retrieval would otherwise drop it.
                    forced_recent_item = RetrievedMomentPin(pin=recent_pin, similarity=0.0, score=0.0)
                break

        selected: List[RetrievedMomentPin] = []
        if forced_recent_item:
            selected.append(forced_recent_item)
        for item in ranked:
            if len(selected) >= inject_k:
                break
            if forced_recent_item and item.pin.id == forced_recent_item.pin.id:
                continue
            selected.append(item)

        if selected:
            self.repo.reinforce_on_injection([item.pin.id for item in selected])
        return selected

    @staticmethod
    def format_for_prompt(retrieved: List[RetrievedMomentPin]) -> str:
        if not retrieved:
            return ""
        lines = [
            "MOMENT PIN INSTRUCTIONS",
            "",
            "The following Moment Pins are summaries of past conversational events.",
            "They are NOT full transcripts.",
            "",
            "If transcript precision is required:",
            "1. Complete your <assistant_response> normally.",
            "2. After </assistant_response>, append a tool payload using the required sentinel format.",
            "3. Do NOT reference the payload in visible text.",
            "4. Do NOT include any text after the END sentinel.",
            "",
            "Use tool name: moment_pin.cold_recall",
            "Only use this tool if necessary.",
            "Do NOT guess exact quotes.",
            "Maximum one cold recall per turn.",
            "",
            "Tool payload template:",
            "---CHORUS_TOOL_PAYLOAD_BEGIN---",
            "{",
            '  "version": 1,',
            '  "tool_calls": [',
            "    {",
            '      "id": "unique_identifier",',
            f'      "tool": "{MOMENT_PIN_COLD_RECALL_TOOL}",',
            '      "requires_approval": false,',
            '      "args": {',
            '        "pin_id": "<one_of_the_injected_pin_ids>",',
            '        "reason": "brief explanation"',
            "      }",
            "    }",
            "  ]",
            "}",
            "---CHORUS_TOOL_PAYLOAD_END---",
            "",
            "Injected Moment Pins:",
        ]
        for idx, item in enumerate(retrieved, 1):
            pin = item.pin
            why_text = (pin.why_user or pin.why_model or "").strip()
            tags = ", ".join(pin.tags or [])
            quote = (pin.quote_snippet or "").strip()
            lines.extend(
                [
                    f"{idx}. pin_id={pin.id}",
                    f"   what_happened: {pin.what_happened}",
                    f"   why: {why_text}",
                    f"   quote_snippet: {quote}" if quote else "   quote_snippet: (none)",
                    f"   tags: {tags}" if tags else "   tags: (none)",
                ]
            )
        return "\n".join(lines)
