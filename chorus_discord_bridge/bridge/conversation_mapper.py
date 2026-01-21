"""
Conversation Mapper: Discord channel/DM -> Chorus conversation mapping.

Manages persistent mappings between Discord channels and Chorus Engine conversations,
enabling conversation continuity across bot restarts.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .database import get_database

logger = logging.getLogger(__name__)


class ConversationMapper:
    """Maps Discord channels to Chorus Engine conversations."""
    
    def __init__(self, db_path: str = "storage/state.db"):
        """Initialize conversation mapper.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db = get_database(db_path)
    
    def get_or_create_conversation(
        self,
        discord_channel_id: str,
        discord_guild_id: Optional[str],
        chorus_conversation_id: str,
        chorus_thread_id: int,
        is_dm: bool = False
    ) -> Dict[str, Any]:
        """Get existing conversation mapping or create new one.
        
        Args:
            discord_channel_id: Discord channel or DM user ID
            discord_guild_id: Discord server ID (None for DMs)
            chorus_conversation_id: Chorus conversation UUID
            chorus_thread_id: Chorus thread ID
            is_dm: True if DM conversation
            
        Returns:
            Conversation mapping dict with keys: id, discord_channel_id,
            chorus_conversation_id, chorus_thread_id, created_at, last_message_at
        """
        # Try to get existing mapping
        existing = self.get_conversation_mapping(discord_channel_id)
        
        if existing:
            logger.info(
                f"Found existing conversation mapping: "
                f"Discord {discord_channel_id} -> "
                f"Chorus {existing['chorus_conversation_id']}"
            )
            return existing
        
        # Create new mapping
        try:
            cursor = self.db.execute(
                """
                INSERT INTO conversation_mappings 
                (discord_channel_id, discord_guild_id, chorus_conversation_id, 
                 chorus_thread_id, is_dm, message_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    discord_channel_id,
                    discord_guild_id,
                    chorus_conversation_id,
                    chorus_thread_id,
                    1 if is_dm else 0
                )
            )
            self.db.commit()
            
            logger.info(
                f"Created new conversation mapping: "
                f"Discord {discord_channel_id} -> "
                f"Chorus {chorus_conversation_id}"
            )
            
            # Return newly created mapping
            return self.get_conversation_mapping(discord_channel_id)
            
        except Exception as e:
            logger.error(f"Failed to create conversation mapping: {e}", exc_info=True)
            raise
    
    def get_conversation_mapping(
        self, discord_channel_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get conversation mapping by Discord channel ID.
        
        Args:
            discord_channel_id: Discord channel or DM user ID
            
        Returns:
            Conversation mapping dict, or None if not found
        """
        try:
            cursor = self.db.execute(
                """
                SELECT id, discord_channel_id, discord_guild_id,
                       chorus_conversation_id, chorus_thread_id,
                       is_dm, created_at, last_message_at, message_count
                FROM conversation_mappings
                WHERE discord_channel_id = ?
                """,
                (discord_channel_id,)
            )
            
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get conversation mapping: {e}", exc_info=True)
            return None
    
    def update_last_message_time(self, discord_channel_id: str) -> bool:
        """Update last_message_at timestamp and increment message count.
        
        Args:
            discord_channel_id: Discord channel or DM user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db.execute(
                """
                UPDATE conversation_mappings
                SET last_message_at = CURRENT_TIMESTAMP,
                    message_count = message_count + 1
                WHERE discord_channel_id = ?
                """,
                (discord_channel_id,)
            )
            self.db.commit()
            
            if cursor.rowcount == 0:
                logger.warning(
                    f"No conversation mapping found for channel {discord_channel_id}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to update last message time: {e}", exc_info=True
            )
            return False
    
    def list_active_conversations(
        self, limit: int = 50, include_dms: bool = True
    ) -> List[Dict[str, Any]]:
        """List active conversations, ordered by most recent activity.
        
        Args:
            limit: Maximum number of conversations to return
            include_dms: Whether to include DM conversations
            
        Returns:
            List of conversation mapping dicts
        """
        try:
            dm_filter = "" if include_dms else "WHERE is_dm = 0"
            
            cursor = self.db.execute(
                f"""
                SELECT id, discord_channel_id, discord_guild_id,
                       chorus_conversation_id, chorus_thread_id,
                       is_dm, created_at, last_message_at, message_count
                FROM conversation_mappings
                {dm_filter}
                ORDER BY last_message_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to list active conversations: {e}", exc_info=True)
            return []
    
    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics.
        
        Returns:
            Dict with keys: total_conversations, dm_conversations, 
            channel_conversations, total_messages
        """
        try:
            cursor = self.db.execute(
                """
                SELECT 
                    COUNT(*) as total_conversations,
                    SUM(CASE WHEN is_dm = 1 THEN 1 ELSE 0 END) as dm_conversations,
                    SUM(CASE WHEN is_dm = 0 THEN 1 ELSE 0 END) as channel_conversations,
                    SUM(message_count) as total_messages
                FROM conversation_mappings
                """
            )
            
            row = cursor.fetchone()
            return dict(row) if row else {}
            
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {e}", exc_info=True)
            return {}
    
    def delete_conversation_mapping(self, discord_channel_id: str) -> bool:
        """Delete a conversation mapping.
        
        Args:
            discord_channel_id: Discord channel or DM user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db.execute(
                "DELETE FROM conversation_mappings WHERE discord_channel_id = ?",
                (discord_channel_id,)
            )
            self.db.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted conversation mapping for {discord_channel_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete conversation mapping: {e}", exc_info=True)
            return False
