"""Repository for continuity bootstrapping data."""

from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.models.continuity import (
    ContinuityRelationshipState,
    ContinuityArc,
    ContinuityBootstrapCache,
    ContinuityPreference,
)


class ContinuityRepository:
    """Handle database operations for continuity artifacts."""

    def __init__(self, db: Session):
        self.db = db

    def get_relationship_state(self, character_id: str) -> Optional[ContinuityRelationshipState]:
        return (
            self.db.query(ContinuityRelationshipState)
            .filter(ContinuityRelationshipState.character_id == character_id)
            .first()
        )

    def upsert_relationship_state(
        self,
        character_id: str,
        familiarity_level: str,
        tone_baseline: list,
        interaction_contract: list,
        boundaries: list,
        assistant_role_frame: str
    ) -> ContinuityRelationshipState:
        state = self.get_relationship_state(character_id)
        if not state:
            state = ContinuityRelationshipState(
                character_id=character_id,
                familiarity_level=familiarity_level,
                tone_baseline=tone_baseline,
                interaction_contract=interaction_contract,
                boundaries=boundaries,
                assistant_role_frame=assistant_role_frame
            )
            self.db.add(state)
        else:
            state.familiarity_level = familiarity_level
            state.tone_baseline = tone_baseline
            state.interaction_contract = interaction_contract
            state.boundaries = boundaries
            state.assistant_role_frame = assistant_role_frame
            state.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(state)
        return state

    def list_arcs(self, character_id: str) -> List[ContinuityArc]:
        return (
            self.db.query(ContinuityArc)
            .filter(ContinuityArc.character_id == character_id)
            .order_by(ContinuityArc.updated_at.desc())
            .all()
        )

    def upsert_arc(self, arc: ContinuityArc) -> ContinuityArc:
        self.db.add(arc)
        self.db.commit()
        self.db.refresh(arc)
        return arc

    def delete_arc(self, arc_id: str) -> None:
        arc = self.db.query(ContinuityArc).filter(ContinuityArc.id == arc_id).first()
        if arc:
            self.db.delete(arc)
            self.db.commit()

    def get_cache(self, character_id: str) -> Optional[ContinuityBootstrapCache]:
        return (
            self.db.query(ContinuityBootstrapCache)
            .filter(ContinuityBootstrapCache.character_id == character_id)
            .first()
        )

    def upsert_cache(
        self,
        character_id: str,
        internal_packet: str,
        user_preview: str,
        generated_at: Optional[datetime],
        fingerprint: Optional[str]
    ) -> ContinuityBootstrapCache:
        cache = self.get_cache(character_id)
        if not cache:
            cache = ContinuityBootstrapCache(
                character_id=character_id,
                bootstrap_packet_internal=internal_packet,
                bootstrap_packet_user_preview=user_preview,
                bootstrap_generated_at=generated_at,
                bootstrap_inputs_fingerprint=fingerprint
            )
            self.db.add(cache)
        else:
            cache.bootstrap_packet_internal = internal_packet
            cache.bootstrap_packet_user_preview = user_preview
            cache.bootstrap_generated_at = generated_at
            cache.bootstrap_inputs_fingerprint = fingerprint
            cache.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(cache)
        return cache

    def get_preferences(self, character_id: str) -> Optional[ContinuityPreference]:
        return (
            self.db.query(ContinuityPreference)
            .filter(ContinuityPreference.character_id == character_id)
            .first()
        )

    def upsert_preferences(
        self,
        character_id: str,
        default_mode: str,
        skip_preview: bool
    ) -> ContinuityPreference:
        pref = self.get_preferences(character_id)
        if not pref:
            pref = ContinuityPreference(
                character_id=character_id,
                default_mode=default_mode,
                skip_preview=1 if skip_preview else 0
            )
            self.db.add(pref)
        else:
            pref.default_mode = default_mode
            pref.skip_preview = 1 if skip_preview else 0
            pref.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(pref)
        return pref
