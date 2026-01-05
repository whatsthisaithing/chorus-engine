"""
Greeting Context Service for Phase 8 Day 5.

Provides contextual greeting information based on conversation history
and temporal patterns. Uses TemporalWeightingService to determine the
appropriate greeting context.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.repositories.conversation_repository import ConversationRepository
from chorus_engine.repositories.memory_repository import MemoryRepository
from chorus_engine.services.temporal_weighting_service import TemporalWeightingService
from chorus_engine.models.conversation import MemoryType

logger = logging.getLogger(__name__)


class GreetingContextService:
    """
    Builds greeting context for conversations.
    
    Phase 8: Uses temporal weighting to determine conversation context
    and retrieves relevant memories (projects, recent experiences) to
    personalize greetings.
    """
    
    def __init__(
        self,
        db: Session,
        temporal_service: Optional[TemporalWeightingService] = None
    ):
        """
        Initialize greeting context service.
        
        Args:
            db: Database session
            temporal_service: Optional temporal weighting service (creates if None)
        """
        self.db = db
        self.conversation_repo = ConversationRepository(db)
        self.memory_repo = MemoryRepository(db)
        self.temporal_service = temporal_service or TemporalWeightingService()
    
    def build_greeting_context(
        self,
        character_id: str,
        conversation_id: Optional[str] = None,
        is_first_message: bool = True
    ) -> Dict[str, any]:
        """
        Build greeting context for a conversation.
        
        Args:
            character_id: Character ID
            conversation_id: Current conversation ID (if resuming)
            is_first_message: Whether this is the first message in conversation
            
        Returns:
            Dictionary with greeting context:
            {
                "time_gap_context": "continuing" | "recent" | "catching_up" | "welcoming_back" | "first_time",
                "last_conversation_date": Optional[datetime],
                "days_since_last": Optional[int],
                "ongoing_projects": List[str],
                "recent_experiences": List[str],
                "should_acknowledge_gap": bool,
                "greeting_tone": "casual" | "warm" | "enthusiastic"
            }
        """
        try:
            # Get last conversation for this character
            last_conversation = self._get_last_conversation(character_id, conversation_id)
            
            if not last_conversation:
                # No conversation history - first time
                return {
                    "time_gap_context": "first_time",
                    "last_conversation_date": None,
                    "days_since_last": None,
                    "ongoing_projects": [],
                    "recent_experiences": [],
                    "should_acknowledge_gap": False,
                    "greeting_tone": "warm"
                }
            
            # Get conversation context from temporal service
            context = self.temporal_service.get_conversation_context_summary(
                conversation=last_conversation,
                db=self.db
            )
            
            time_gap_context = context.get("time_gap_context", "first_time")
            
            # Determine if we should acknowledge the time gap
            should_acknowledge_gap = time_gap_context in ["catching_up", "welcoming_back"]
            
            # Determine greeting tone based on time gap
            greeting_tone = self._determine_greeting_tone(time_gap_context)
            
            # Get ongoing projects (for context in greeting)
            ongoing_projects = self._get_ongoing_projects(character_id, limit=3)
            
            # Get recent experiences (for context in greeting)
            recent_experiences = self._get_recent_experiences(character_id, limit=2)
            
            # Build greeting context
            greeting_context = {
                "time_gap_context": time_gap_context,
                "last_conversation_date": last_conversation.updated_at,
                "days_since_last": context.get("time_gap_days"),
                "ongoing_projects": ongoing_projects,
                "recent_experiences": recent_experiences,
                "should_acknowledge_gap": should_acknowledge_gap,
                "greeting_tone": greeting_tone
            }
            
            logger.info(
                f"[GREETING CONTEXT] {character_id}: {time_gap_context}, "
                f"gap={greeting_context['days_since_last']} days, "
                f"projects={len(ongoing_projects)}, experiences={len(recent_experiences)}"
            )
            
            return greeting_context
            
        except Exception as e:
            logger.error(f"Failed to build greeting context: {e}", exc_info=True)
            # Return safe default
            return {
                "time_gap_context": "first_time",
                "last_conversation_date": None,
                "days_since_last": None,
                "ongoing_projects": [],
                "recent_experiences": [],
                "should_acknowledge_gap": False,
                "greeting_tone": "warm"
            }
    
    def format_greeting_instructions(self, greeting_context: Dict[str, any]) -> str:
        """
        Format greeting context into instructions for system prompt.
        
        Args:
            greeting_context: Greeting context from build_greeting_context()
            
        Returns:
            Formatted instructions string for system prompt
        """
        time_gap = greeting_context["time_gap_context"]
        days_since = greeting_context.get("days_since_last")
        projects = greeting_context.get("ongoing_projects", [])
        experiences = greeting_context.get("recent_experiences", [])
        
        # Build instruction based on time gap context
        if time_gap == "first_time":
            instruction = "This is your first conversation with this user. Greet them warmly and naturally."
        
        elif time_gap == "continuing":
            instruction = "This is a continuation of your recent conversation (within 24 hours)."
            if projects:
                instruction += f" You were discussing: {', '.join(projects[:2])}."
        
        elif time_gap == "recent":
            instruction = f"You last spoke {days_since} days ago. Greet them warmly and pick up where you left off."
            if projects:
                instruction += f" You can reference ongoing topics: {', '.join(projects[:2])}."
        
        elif time_gap == "catching_up":
            instruction = f"It's been {days_since} days since you last spoke. "
            instruction += "Acknowledge the time gap naturally and ask how they've been."
            if projects:
                instruction += f" You might reference: {', '.join(projects[:2])}."
        
        elif time_gap == "welcoming_back":
            instruction = f"It's been {days_since} days since your last conversation. "
            instruction += "Welcome them back warmly and express interest in what they've been up to."
            if projects or experiences:
                context_items = projects[:1] + experiences[:1]
                instruction += f" You might recall: {', '.join(context_items)}."
        
        else:
            instruction = "Greet the user naturally."
        
        return instruction
    
    def _determine_greeting_tone(self, time_gap_context: str) -> str:
        """
        Determine appropriate greeting tone based on time gap.
        
        Args:
            time_gap_context: Time gap context from temporal service
            
        Returns:
            Greeting tone: "casual" | "warm" | "enthusiastic"
        """
        tone_map = {
            "first_time": "warm",
            "continuing": "casual",
            "recent": "warm",
            "catching_up": "warm",
            "welcoming_back": "enthusiastic"
        }
        return tone_map.get(time_gap_context, "warm")
    
    def _get_last_conversation(
        self,
        character_id: str,
        current_conversation_id: Optional[str] = None
    ) -> Optional[any]:
        """
        Get the last conversation for a character (excluding current).
        
        Args:
            character_id: Character ID
            current_conversation_id: Optional current conversation to exclude
            
        Returns:
            Last conversation or None
        """
        try:
            # Get recent conversations using temporal service
            conversations = self.temporal_service.get_recent_conversations(
                db=self.db,
                character_id=character_id,
                limit=10
            )
            
            # Filter out current conversation if specified
            if current_conversation_id:
                conversations = [c for c in conversations if c.id != current_conversation_id]
            
            # Return most recent
            return conversations[0] if conversations else None
            
        except Exception as e:
            logger.warning(f"Failed to get last conversation: {e}")
            return None
    
    def _get_ongoing_projects(self, character_id: str, limit: int = 3) -> List[str]:
        """
        Get ongoing projects from memory.
        
        Args:
            character_id: Character ID
            limit: Maximum number of projects to return
            
        Returns:
            List of project content strings
        """
        try:
            # Get PROJECT type memories
            project_memories = self.memory_repo.get_by_character_and_type(
                character_id=character_id,
                memory_type=MemoryType.PROJECT,
                limit=limit
            )
            
            return [m.content for m in project_memories]
        except Exception as e:
            logger.warning(f"Failed to get ongoing projects: {e}")
            return []
    
    def _get_recent_experiences(self, character_id: str, limit: int = 2) -> List[str]:
        """
        Get recent shared experiences from memory.
        
        Args:
            character_id: Character ID
            limit: Maximum number of experiences to return
            
        Returns:
            List of experience content strings
        """
        try:
            # Get EXPERIENCE type memories
            experience_memories = self.memory_repo.get_by_character_and_type(
                character_id=character_id,
                memory_type=MemoryType.EXPERIENCE,
                limit=limit
            )
            
            return [m.content for m in experience_memories]
        except Exception as e:
            logger.warning(f"Failed to get recent experiences: {e}")
            return []
