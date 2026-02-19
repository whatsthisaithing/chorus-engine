"""Surface adapter routing resolver for ENS Slice 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from chorus_engine.ens.surface_identity import canonicalize_surface_id
from chorus_engine.repositories import ConversationRepository, SurfaceBindingRepository, ThreadRepository
from chorus_engine.services.relationship_resolution_service import RelationshipResolutionService


@dataclass
class ResolvedTarget:
    conversation_id: str
    thread_id: str
    relationship_id: Optional[str]
    binding_id: str
    ignored_target_hint: Optional[str] = None


class SurfaceRouter:
    """Resolves inbound adapter identifiers to canonical conversation/thread targets."""

    def __init__(self, db: Session):
        self.db = db
        self.binding_repo = SurfaceBindingRepository(db)
        self.conv_repo = ConversationRepository(db)
        self.thread_repo = ThreadRepository(db)

    def resolve(
        self,
        *,
        assistant_id: str,
        surface_id: Optional[str],
        external_thread_id: Optional[str],
        surface_instance_id: Optional[str] = None,
        relationship_hint: Optional[str] = None,
        target_hint: Optional[str] = None,
        conversation_id_hint: Optional[str] = None,
        thread_id_hint: Optional[str] = None,
    ) -> ResolvedTarget:
        surface = canonicalize_surface_id(surface_id)
        ext_thread = str(external_thread_id or thread_id_hint or "").strip()
        if not ext_thread:
            ext_thread = f"local:{surface}:{assistant_id}"

        ignored_target_hint: Optional[str] = None
        normalized_target_hint = (target_hint or "").strip().lower()
        if normalized_target_hint and normalized_target_hint != "general_chat":
            ignored_target_hint = normalized_target_hint

        existing = self.binding_repo.get_by_lookup(
            surface_id=surface,
            external_thread_id=ext_thread,
            surface_instance_id=surface_instance_id,
        )
        if existing:
            touched = self.binding_repo.touch(existing)
            return ResolvedTarget(
                conversation_id=touched.conversation_id,
                thread_id=touched.thread_id,
                relationship_id=touched.relationship_id,
                binding_id=touched.id,
                ignored_target_hint=ignored_target_hint,
            )

        if normalized_target_hint == "general_chat":
            resolver = RelationshipResolutionService(self.db)
            resolved_gc = resolver.resolve_general_chat(
                character_id=assistant_id,
                surface_id=surface,
                surface_instance_id=surface_instance_id,
                source=surface,
            )
            conversation_id = resolved_gc["conversation"].id
            thread_id = resolved_gc["thread"].id
            relationship_id = resolved_gc["relationship"].id
            binding = self.binding_repo.create_with_retry(
                surface_id=surface,
                surface_instance_id=surface_instance_id,
                external_thread_id=ext_thread,
                relationship_id=relationship_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
            )
            return ResolvedTarget(
                conversation_id=binding.conversation_id,
                thread_id=binding.thread_id,
                relationship_id=binding.relationship_id,
                binding_id=binding.id,
                ignored_target_hint=ignored_target_hint,
            )

        if thread_id_hint:
            thread = self.thread_repo.get_by_id(thread_id_hint)
            if thread:
                conversation_id = thread.conversation_id
                thread_id = thread.id
            else:
                conversation_id = None
                thread_id = None
        else:
            conversation_id = conversation_id_hint
            thread_id = None

        if not conversation_id:
            conversation = self.conv_repo.create(character_id=assistant_id, source=surface)
            conversation_id = conversation.id

        if not thread_id:
            if thread_id_hint:
                hinted = self.thread_repo.get_by_id(thread_id_hint)
                if hinted and hinted.conversation_id == conversation_id:
                    thread_id = hinted.id
            if not thread_id and conversation_id:
                existing_threads = self.thread_repo.list_by_conversation(conversation_id)
                if existing_threads:
                    thread_id = existing_threads[0].id
            if not thread_id:
                thread = self.thread_repo.create(conversation_id=conversation_id, title="Main Thread")
                thread_id = thread.id

        binding = self.binding_repo.create_with_retry(
            surface_id=surface,
            surface_instance_id=surface_instance_id,
            external_thread_id=ext_thread,
            relationship_id=relationship_hint,
            conversation_id=conversation_id,
            thread_id=thread_id,
        )
        return ResolvedTarget(
            conversation_id=binding.conversation_id,
            thread_id=binding.thread_id,
            relationship_id=binding.relationship_id,
            binding_id=binding.id,
            ignored_target_hint=ignored_target_hint,
        )
