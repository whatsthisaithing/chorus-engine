"""Repository for moment pin operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import MomentPin


class MomentPinRepository:
    """Database operations for moment pins."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        user_id: str,
        character_id: str,
        conversation_id: Optional[str],
        selected_message_ids: List[str],
        transcript_snapshot: str,
        what_happened: str,
        why_model: str,
        why_user: Optional[str] = None,
        quote_snippet: Optional[str] = None,
        tags: Optional[List[str]] = None,
        telemetry_flags: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> MomentPin:
        pin = MomentPin(
            user_id=user_id,
            character_id=character_id,
            conversation_id=conversation_id,
            selected_message_ids=selected_message_ids,
            transcript_snapshot=transcript_snapshot,
            what_happened=what_happened,
            why_model=why_model,
            why_user=why_user,
            quote_snippet=quote_snippet,
            tags=tags or [],
            telemetry_flags=telemetry_flags
            or {
                "contains_roleplay": False,
                "contains_directives": False,
                "contains_sensitive_content": False,
            },
            vector_id=vector_id,
        )
        self.db.add(pin)
        self.db.commit()
        self.db.refresh(pin)
        return pin

    def get_by_id(self, pin_id: str) -> Optional[MomentPin]:
        return self.db.query(MomentPin).filter(MomentPin.id == pin_id).first()

    def list_by_conversation(self, conversation_id: str) -> List[MomentPin]:
        return (
            self.db.query(MomentPin)
            .filter(MomentPin.conversation_id == conversation_id)
            .order_by(MomentPin.created_at.desc())
            .all()
        )

    def list_by_character(
        self,
        character_id: str,
        conversation_id: Optional[str] = None,
        include_archived: bool = True,
    ) -> List[MomentPin]:
        query = self.db.query(MomentPin).filter(MomentPin.character_id == character_id)
        if conversation_id:
            query = query.filter(MomentPin.conversation_id == conversation_id)
        if not include_archived:
            query = query.filter(MomentPin.archived == 0)
        return query.order_by(MomentPin.created_at.desc()).all()

    def list_for_retrieval(
        self,
        user_id: str,
        character_id: str,
        archived: int = 0,
    ) -> List[MomentPin]:
        return (
            self.db.query(MomentPin)
            .filter(
                MomentPin.user_id == user_id,
                MomentPin.character_id == character_id,
                MomentPin.archived == archived,
            )
            .order_by(MomentPin.created_at.desc())
            .all()
        )

    def update_fields(
        self,
        pin_id: str,
        why_user: Optional[str] = None,
        tags: Optional[List[str]] = None,
        archived: Optional[bool] = None,
    ) -> Optional[MomentPin]:
        pin = self.get_by_id(pin_id)
        if not pin:
            return None

        if why_user is not None:
            pin.why_user = why_user
        if tags is not None:
            pin.tags = tags
        if archived is not None:
            pin.archived = 1 if archived else 0

        self.db.commit()
        self.db.refresh(pin)
        return pin

    def set_vector_id(self, pin_id: str, vector_id: Optional[str]) -> Optional[MomentPin]:
        pin = self.get_by_id(pin_id)
        if not pin:
            return None
        pin.vector_id = vector_id
        self.db.commit()
        self.db.refresh(pin)
        return pin

    def delete(self, pin_id: str) -> bool:
        pin = self.get_by_id(pin_id)
        if not pin:
            return False
        self.db.delete(pin)
        self.db.commit()
        return True

    def orphan_conversation_pins(self, conversation_id: str) -> int:
        count = (
            self.db.query(MomentPin)
            .filter(MomentPin.conversation_id == conversation_id)
            .update({"conversation_id": None})
        )
        self.db.commit()
        return count

    def delete_by_conversation(self, conversation_id: str) -> int:
        count = (
            self.db.query(MomentPin)
            .filter(MomentPin.conversation_id == conversation_id)
            .delete()
        )
        self.db.commit()
        return count

    def reinforce_on_injection(self, pin_ids: List[str]) -> None:
        if not pin_ids:
            return

        # Increment age for all active pins and reset selected pins with a small reinforcement bump.
        self.db.query(MomentPin).filter(MomentPin.archived == 0).update(
            {MomentPin.turns_since_reinforcement: MomentPin.turns_since_reinforcement + 1},
            synchronize_session=False,
        )
        self.db.query(MomentPin).filter(MomentPin.id.in_(pin_ids)).update(
            {
                MomentPin.turns_since_reinforcement: 0,
                MomentPin.reinforcement_score: MomentPin.reinforcement_score + 0.05,
            },
            synchronize_session=False,
        )
        self.db.commit()
