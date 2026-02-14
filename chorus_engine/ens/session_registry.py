"""Session resolution for ENS."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from chorus_engine.models.ens import ENSSession


class ENSSessionRegistry:
    """Resolves canonical ENS sessions for ingress requests."""

    OWNER_USER_ID = "user:local:owner"

    def resolve_thread_session(
        self,
        db: Session,
        *,
        assistant_id: str,
        thread_id: str,
        conversation_id: Optional[str],
        surface: str,
        source: str,
        latency_sensitive: bool = False,
    ) -> ENSSession:
        session = (
            db.query(ENSSession)
            .filter(
                ENSSession.surface == surface,
                ENSSession.source == source,
                ENSSession.thread_id == thread_id,
            )
            .first()
        )
        if session:
            session.last_signal_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            session.conversation_id = conversation_id
            session.assistant_id = assistant_id
            session.latency_sensitive = 1 if latency_sensitive else 0
            db.commit()
            db.refresh(session)
            return session

        session = ENSSession(
            assistant_id=assistant_id,
            user_id=self.OWNER_USER_ID,
            conversation_id=conversation_id,
            thread_id=thread_id,
            surface=surface,
            source=source,
            latency_sensitive=1 if latency_sensitive else 0,
            last_signal_at=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
