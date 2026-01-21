"""
User Tracker: Discord user information and alias management.

Tracks Discord users across conversations, including username changes,
display names, known aliases, and activity metrics.
"""
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .database import get_database

logger = logging.getLogger(__name__)


class UserTracker:
    """Tracks Discord user information and aliases."""
    
    def __init__(self, db_path: str = "storage/state.db"):
        """Initialize user tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db = get_database(db_path)
    
    def track_user(
        self,
        discord_user_id: str,
        username: str,
        display_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track user activity, creating or updating user record.
        
        Args:
            discord_user_id: Discord user ID
            username: Current Discord username
            display_name: Current display name/nickname (optional)
            
        Returns:
            User info dict with keys: discord_user_id, username, display_name,
            known_aliases, first_seen, last_seen, message_count
        """
        existing = self.get_user_info(discord_user_id)
        
        if existing:
            # Update existing user
            return self._update_user(
                discord_user_id, username, display_name, existing
            )
        else:
            # Create new user
            return self._create_user(discord_user_id, username, display_name)
    
    def _create_user(
        self,
        discord_user_id: str,
        username: str,
        display_name: Optional[str]
    ) -> Dict[str, Any]:
        """Create new user record.
        
        Args:
            discord_user_id: Discord user ID
            username: Discord username
            display_name: Display name/nickname
            
        Returns:
            Created user info dict
        """
        try:
            # Initialize aliases with current username
            aliases = json.dumps([username])
            
            self.db.execute(
                """
                INSERT INTO discord_users
                (discord_user_id, username, display_name, known_aliases, message_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (discord_user_id, username, display_name, aliases)
            )
            self.db.commit()
            
            logger.info(f"Created user record: {username} ({discord_user_id})")
            
            return self.get_user_info(discord_user_id)
            
        except Exception as e:
            logger.error(f"Failed to create user record: {e}", exc_info=True)
            raise
    
    def _update_user(
        self,
        discord_user_id: str,
        username: str,
        display_name: Optional[str],
        existing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing user record.
        
        Args:
            discord_user_id: Discord user ID
            username: Current username
            display_name: Current display name
            existing: Existing user info dict
            
        Returns:
            Updated user info dict
        """
        try:
            # Check if username changed
            username_changed = existing['username'] != username
            
            # Update last_username_update if username changed
            update_username_timestamp = username_changed
            
            # Parse existing aliases (may already be a list from get_user_info)
            if isinstance(existing['known_aliases'], str):
                aliases = json.loads(existing['known_aliases']) if existing['known_aliases'] else []
            else:
                aliases = existing['known_aliases'] if existing['known_aliases'] else []
            
            # Add new username to aliases if changed and not already present
            if username_changed and username not in aliases:
                aliases.append(username)
                logger.info(
                    f"Username changed: {existing['username']} -> {username} "
                    f"({discord_user_id})"
                )
            
            # Limit aliases to last 10
            if len(aliases) > 10:
                aliases = aliases[-10:]
            
            aliases_json = json.dumps(aliases)
            
            # Update user record
            if update_username_timestamp:
                self.db.execute(
                    """
                    UPDATE discord_users
                    SET username = ?,
                        display_name = ?,
                        known_aliases = ?,
                        last_seen = CURRENT_TIMESTAMP,
                        message_count = message_count + 1,
                        last_username_update = CURRENT_TIMESTAMP
                    WHERE discord_user_id = ?
                    """,
                    (username, display_name, aliases_json, discord_user_id)
                )
            else:
                self.db.execute(
                    """
                    UPDATE discord_users
                    SET display_name = ?,
                        last_seen = CURRENT_TIMESTAMP,
                        message_count = message_count + 1
                    WHERE discord_user_id = ?
                    """,
                    (display_name, discord_user_id)
                )
            
            self.db.commit()
            
            return self.get_user_info(discord_user_id)
            
        except Exception as e:
            logger.error(f"Failed to update user record: {e}", exc_info=True)
            # Return existing data on failure
            return existing
    
    def get_user_info(self, discord_user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information by Discord user ID.
        
        Args:
            discord_user_id: Discord user ID
            
        Returns:
            User info dict, or None if not found
        """
        try:
            cursor = self.db.execute(
                """
                SELECT discord_user_id, username, display_name, known_aliases,
                       first_seen, last_seen, message_count, last_username_update
                FROM discord_users
                WHERE discord_user_id = ?
                """,
                (discord_user_id,)
            )
            
            row = cursor.fetchone()
            
            if row:
                user_dict = dict(row)
                # Parse aliases JSON
                if user_dict.get('known_aliases'):
                    user_dict['known_aliases'] = json.loads(user_dict['known_aliases'])
                else:
                    user_dict['known_aliases'] = []
                return user_dict
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}", exc_info=True)
            return None
    
    def update_aliases(
        self, discord_user_id: str, new_alias: str
    ) -> bool:
        """Add a new alias for a user.
        
        Args:
            discord_user_id: Discord user ID
            new_alias: New alias to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            user = self.get_user_info(discord_user_id)
            
            if not user:
                logger.warning(
                    f"Cannot add alias: user {discord_user_id} not found"
                )
                return False
            
            aliases = user['known_aliases']
            
            # Add new alias if not already present
            if new_alias not in aliases:
                aliases.append(new_alias)
                
                # Limit to last 10 aliases
                if len(aliases) > 10:
                    aliases = aliases[-10:]
                
                aliases_json = json.dumps(aliases)
                
                self.db.execute(
                    """
                    UPDATE discord_users
                    SET known_aliases = ?
                    WHERE discord_user_id = ?
                    """,
                    (aliases_json, discord_user_id)
                )
                self.db.commit()
                
                logger.info(f"Added alias '{new_alias}' for user {discord_user_id}")
                return True
            
            return True  # Already exists, still success
            
        except Exception as e:
            logger.error(f"Failed to update aliases: {e}", exc_info=True)
            return False
    
    def list_active_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List most recently active users.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            List of user info dicts, ordered by last_seen
        """
        try:
            cursor = self.db.execute(
                """
                SELECT discord_user_id, username, display_name, known_aliases,
                       first_seen, last_seen, message_count, last_username_update
                FROM discord_users
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (limit,)
            )
            
            users = []
            for row in cursor.fetchall():
                user_dict = dict(row)
                # Parse aliases JSON
                if user_dict.get('known_aliases'):
                    user_dict['known_aliases'] = json.loads(user_dict['known_aliases'])
                else:
                    user_dict['known_aliases'] = []
                users.append(user_dict)
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to list active users: {e}", exc_info=True)
            return []
    
    def get_user_stats(self) -> Dict[str, int]:
        """Get user statistics.
        
        Returns:
            Dict with keys: total_users, total_messages, 
            active_users_24h (placeholder for now)
        """
        try:
            cursor = self.db.execute(
                """
                SELECT 
                    COUNT(*) as total_users,
                    SUM(message_count) as total_messages
                FROM discord_users
                """
            )
            
            row = cursor.fetchone()
            stats = dict(row) if row else {}
            
            # Placeholder for active users in last 24h
            # (would need datetime comparison in real implementation)
            stats['active_users_24h'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}", exc_info=True)
            return {}
    
    def delete_user(self, discord_user_id: str) -> bool:
        """Delete a user record.
        
        Args:
            discord_user_id: Discord user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db.execute(
                "DELETE FROM discord_users WHERE discord_user_id = ?",
                (discord_user_id,)
            )
            self.db.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted user record for {discord_user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete user record: {e}", exc_info=True)
            return False
