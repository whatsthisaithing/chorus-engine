"""Service for relationship-first general chat resolution."""

from __future__ import annotations

from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from chorus_engine.repositories import ConversationRepository, ThreadRepository
from chorus_engine.repositories.relationship_repository import RelationshipRepository
from chorus_engine.config import ConfigLoader


class RelationshipResolutionService:
    """Resolve and provision relationship-scoped general chat resources."""

    OWNER_USER_ID = "user:local:owner"

    def __init__(self, db: Session):
        self.db = db
        self.relationship_repo = RelationshipRepository(db)
        self.conversation_repo = ConversationRepository(db)
        self.thread_repo = ThreadRepository(db)
        self.config_loader = ConfigLoader()

    def _resolve_character_display_name(self, character_id: str) -> str:
        try:
            character = self.config_loader.load_character(character_id)
            if character and getattr(character, "name", None):
                return str(character.name).strip() or character_id
        except Exception:
            pass
        return character_id

    def resolve_general_chat(
        self,
        *,
        character_id: str,
        owner_user_id: Optional[str] = None,
        surface_id: str = "web",
        surface_instance_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        owner = owner_user_id or self.OWNER_USER_ID
        relationship = self.relationship_repo.get_or_create_relationship(owner, character_id)
        surface = self.relationship_repo.get_or_create_surface(
            relationship_id=relationship.id,
            surface_id=surface_id,
            surface_instance_id=surface_instance_id,
        )

        conversation = None
        if surface.general_conversation_id:
            conversation = self.conversation_repo.get_by_id(surface.general_conversation_id)

        if conversation is None:
            character_display_name = self._resolve_character_display_name(character_id)
            conversation = self.conversation_repo.create(
                character_id=character_id,
                title=f"General Chat with {character_display_name}",
                source=source or surface_id,
                relationship_id=relationship.id,
                conversation_kind="general_chat",
            )
            self.relationship_repo.set_general_conversation_id(surface.id, conversation.id)

        threads = self.thread_repo.list_by_conversation(conversation.id)
        if threads:
            thread = threads[0]
        else:
            thread = self.thread_repo.create(conversation_id=conversation.id, title="Main Thread")

        return {
            "relationship": relationship,
            "relationship_surface": surface,
            "conversation": conversation,
            "thread": thread,
        }
