"""
Message Processor for Discord to Chorus Format Conversion

Phase 3, Task 3.2: Message Format Conversion
Handles conversion of Discord-specific formatting to Chorus-friendly format.
"""

import re
import logging
from typing import Optional, Dict, Any
import discord

logger = logging.getLogger(__name__)


class MessageProcessor:
    """
    Processes Discord messages for Chorus Engine consumption.
    
    Handles:
    - User mentions: <@123456> → @username
    - Role mentions: <@&123456> → @rolename
    - Channel mentions: <#123456> → #channel-name
    - Custom emojis: <:emoji_name:id> → :emoji_name:
    - Animated emojis: <a:emoji_name:id> → :emoji_name:
    - URLs and embeds
    - Code blocks and inline code
    - Multi-line messages
    """
    
    def __init__(self, bot: discord.Client):
        """
        Initialize the message processor.
        
        Args:
            bot: Discord bot client for looking up entities
        """
        self.bot = bot
        
    def process_discord_message(self, message: discord.Message) -> str:
        """
        Convert a Discord message to Chorus-friendly format.
        
        Args:
            message: Discord message object
            
        Returns:
            Processed message content
        """
        content = message.content
        
        # Process mentions first (needs Discord API lookups)
        content = self._process_user_mentions(content, message)
        content = self._process_role_mentions(content, message)
        content = self._process_channel_mentions(content, message)
        
        # Process custom emojis (simple regex replacement)
        content = self._process_custom_emojis(content)
        
        # Process Discord-specific formatting
        content = self._process_spoilers(content)
        content = self._process_timestamps(content)
        
        # Clean up excessive whitespace while preserving intentional formatting
        content = self._clean_whitespace(content)
        
        return content
    
    def _process_user_mentions(self, content: str, message: discord.Message) -> str:
        """
        Convert user mentions from <@123456> to @username.
        
        Args:
            content: Message content
            message: Discord message for context
            
        Returns:
            Content with processed user mentions
        """
        # Pattern: <@123456> or <@!123456> (with nickname indicator)
        pattern = r'<@!?(\d+)>'
        
        def replace_mention(match):
            user_id = int(match.group(1))
            
            # Try to get user from message mentions first (most reliable)
            for mentioned_user in message.mentions:
                if mentioned_user.id == user_id:
                    return f"@{mentioned_user.display_name}"
            
            # Fallback: try to get from guild
            if message.guild:
                member = message.guild.get_member(user_id)
                if member:
                    return f"@{member.display_name}"
            
            # Last resort: try to get from cache
            user = self.bot.get_user(user_id)
            if user:
                return f"@{user.display_name}"
            
            # If we can't resolve it, keep the ID but make it readable
            return f"@user_{user_id}"
        
        return re.sub(pattern, replace_mention, content)
    
    def _process_role_mentions(self, content: str, message: discord.Message) -> str:
        """
        Convert role mentions from <@&123456> to @rolename.
        
        Args:
            content: Message content
            message: Discord message for context
            
        Returns:
            Content with processed role mentions
        """
        # Pattern: <@&123456>
        pattern = r'<@&(\d+)>'
        
        def replace_mention(match):
            role_id = int(match.group(1))
            
            # Try to get role from message role mentions
            for role in message.role_mentions:
                if role.id == role_id:
                    return f"@{role.name}"
            
            # Fallback: try to get from guild
            if message.guild:
                role = message.guild.get_role(role_id)
                if role:
                    return f"@{role.name}"
            
            # If we can't resolve it, keep the ID but make it readable
            return f"@role_{role_id}"
        
        return re.sub(pattern, replace_mention, content)
    
    def _process_channel_mentions(self, content: str, message: discord.Message) -> str:
        """
        Convert channel mentions from <#123456> to #channel-name.
        
        Args:
            content: Message content
            message: Discord message for context
            
        Returns:
            Content with processed channel mentions
        """
        # Pattern: <#123456>
        pattern = r'<#(\d+)>'
        
        def replace_mention(match):
            channel_id = int(match.group(1))
            
            # Try to get channel from guild
            if message.guild:
                channel = message.guild.get_channel(channel_id)
                if channel:
                    return f"#{channel.name}"
            
            # Fallback: try to get from bot cache
            channel = self.bot.get_channel(channel_id)
            if channel:
                return f"#{channel.name}"
            
            # If we can't resolve it, keep the ID but make it readable
            return f"#channel_{channel_id}"
        
        return re.sub(pattern, replace_mention, content)
    
    def _process_custom_emojis(self, content: str) -> str:
        """
        Convert custom emojis from <:emoji_name:id> to :emoji_name:.
        
        Args:
            content: Message content
            
        Returns:
            Content with processed custom emojis
        """
        # Pattern: <:emoji_name:123456> or <a:emoji_name:123456> (animated)
        pattern = r'<a?:([a-zA-Z0-9_]+):\d+>'
        return re.sub(pattern, r':\1:', content)
    
    def _process_spoilers(self, content: str) -> str:
        """
        Convert Discord spoilers ||text|| to (spoiler: text).
        
        Args:
            content: Message content
            
        Returns:
            Content with processed spoilers
        """
        # Pattern: ||text||
        pattern = r'\|\|([^|]+)\|\|'
        return re.sub(pattern, r'(spoiler: \1)', content)
    
    def _process_timestamps(self, content: str) -> str:
        """
        Convert Discord timestamps <t:1234567890:F> to human-readable format.
        
        Args:
            content: Message content
            
        Returns:
            Content with processed timestamps
        """
        # Pattern: <t:1234567890:F> (various format types: t, T, d, D, f, F, R)
        pattern = r'<t:(\d+)(?::[tTdDfFR])?>'
        
        def replace_timestamp(match):
            timestamp = int(match.group(1))
            # Convert to human-readable format
            from datetime import datetime
            try:
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError):
                return f"<timestamp:{timestamp}>"
        
        return re.sub(pattern, replace_timestamp, content)
    
    def _clean_whitespace(self, content: str) -> str:
        """
        Clean up excessive whitespace while preserving intentional formatting.
        
        Args:
            content: Message content
            
        Returns:
            Content with cleaned whitespace
        """
        # Don't touch code blocks
        if '```' in content:
            return content
        
        # Replace multiple spaces with single space (but preserve newlines)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Replace multiple spaces with single space
            cleaned_line = re.sub(r' {2,}', ' ', line)
            # Strip trailing whitespace
            cleaned_line = cleaned_line.rstrip()
            cleaned_lines.append(cleaned_line)
        
        # Join lines back together
        content = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Strip leading/trailing whitespace from entire message
        content = content.strip()
        
        return content
    
    def format_message_with_context(
        self,
        content: str,
        message: discord.Message,
        include_author: bool = False
    ) -> str:
        """
        Format message with optional context (author, reply info, etc.).
        
        Args:
            content: Processed message content
            message: Original Discord message
            include_author: Whether to prefix with author name
            
        Returns:
            Formatted message with context
        """
        parts = []
        
        # Add author prefix if requested
        if include_author:
            author_name = message.author.display_name
            parts.append(f"{author_name}: {content}")
        else:
            parts.append(content)
        
        # Add reply context if this is a reply
        if message.reference and message.reference.resolved:
            replied_msg = message.reference.resolved
            if isinstance(replied_msg, discord.Message):
                # Get first 100 chars of replied message
                reply_preview = replied_msg.content[:100]
                if len(replied_msg.content) > 100:
                    reply_preview += "..."
                parts.insert(0, f"[Replying to {replied_msg.author.display_name}: {reply_preview}]")
        
        return '\n'.join(parts)
    
    def process_attachments(self, message: discord.Message) -> Optional[str]:
        """
        Generate text description of message attachments.
        
        Args:
            message: Discord message
            
        Returns:
            Text description of attachments, or None if no attachments
        """
        if not message.attachments:
            return None
        
        attachment_texts = []
        for attachment in message.attachments:
            # Determine attachment type
            if attachment.content_type:
                if attachment.content_type.startswith('image/'):
                    attachment_texts.append(f"[Image: {attachment.filename}]")
                elif attachment.content_type.startswith('video/'):
                    attachment_texts.append(f"[Video: {attachment.filename}]")
                elif attachment.content_type.startswith('audio/'):
                    attachment_texts.append(f"[Audio: {attachment.filename}]")
                else:
                    attachment_texts.append(f"[File: {attachment.filename}]")
            else:
                # Guess based on extension
                ext = attachment.filename.lower().split('.')[-1]
                if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                    attachment_texts.append(f"[Image: {attachment.filename}]")
                elif ext in ['mp4', 'mov', 'avi', 'webm']:
                    attachment_texts.append(f"[Video: {attachment.filename}]")
                elif ext in ['mp3', 'wav', 'ogg', 'm4a']:
                    attachment_texts.append(f"[Audio: {attachment.filename}]")
                else:
                    attachment_texts.append(f"[File: {attachment.filename}]")
        
        return '\n'.join(attachment_texts)
    
    def process_embeds(self, message: discord.Message) -> Optional[str]:
        """
        Generate text description of message embeds.
        
        Args:
            message: Discord message
            
        Returns:
            Text description of embeds, or None if no embeds
        """
        if not message.embeds:
            return None
        
        embed_texts = []
        for embed in message.embeds:
            parts = []
            
            if embed.title:
                parts.append(f"[Embed: {embed.title}]")
            
            if embed.description:
                # Truncate long descriptions
                desc = embed.description[:200]
                if len(embed.description) > 200:
                    desc += "..."
                parts.append(desc)
            
            if embed.url:
                parts.append(f"URL: {embed.url}")
            
            if parts:
                embed_texts.append(' - '.join(parts))
            else:
                embed_texts.append("[Embed]")
        
        return '\n'.join(embed_texts)
    
    def process_complete_message(self, message: discord.Message) -> str:
        """
        Process complete Discord message including content, attachments, and embeds.
        
        Args:
            message: Discord message
            
        Returns:
            Complete processed message text
        """
        parts = []
        
        # Process main content
        if message.content:
            content = self.process_discord_message(message)
            parts.append(content)
        
        # Add attachments
        attachments = self.process_attachments(message)
        if attachments:
            parts.append(attachments)
        
        # Add embeds
        embeds = self.process_embeds(message)
        if embeds:
            parts.append(embeds)
        
        return '\n'.join(parts) if parts else "[Empty message]"
