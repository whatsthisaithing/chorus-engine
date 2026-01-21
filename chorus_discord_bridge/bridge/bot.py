"""Discord bot for Chorus Engine bridge."""

import discord
import logging
import asyncio
from typing import Optional
from discord.ext import commands

from .config import BridgeConfig
from .chorus_client import ChorusClient, ChorusAPIError
from .conversation_mapper import ConversationMapper
from .user_tracker import UserTracker
from .database import init_database
from .message_processor import MessageProcessor
from .response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class ChorusBot(commands.Bot):
    """Discord bot that bridges to Chorus Engine."""
    
    def __init__(self, config: BridgeConfig):
        """
        Initialize the Discord bot.
        
        Args:
            config: Bridge configuration
        """
        self.config = config
        
        # Setup Discord intents
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.messages = True
        intents.guilds = True
        intents.members = True  # Optional, for user tracking
        
        # Initialize bot with command prefix
        super().__init__(
            command_prefix=config.discord_command_prefix,
            intents=intents,
            help_command=None  # Disable default help command
        )
        
        # Initialize Chorus client
        self.chorus_client = ChorusClient(
            api_url=config.chorus_api_url,
            api_key=config.chorus_api_key,
            timeout=config.chorus_timeout,
            retry_attempts=config.chorus_retry_attempts,
            retry_delay=config.chorus_retry_delay
        )
        
        # Initialize state management (Phase 2)
        db_path = config.bridge_state_db_path or "storage/state.db"
        
        # Ensure database is initialized
        logger.info(f"Initializing state database: {db_path}")
        if not init_database(db_path):
            logger.error("Failed to initialize state database!")
        
        self.conversation_mapper = ConversationMapper(db_path)
        self.user_tracker = UserTracker(db_path)
        
        # Initialize message processor (Phase 3, Task 3.2)
        self.message_processor = MessageProcessor(self)
        
        # Initialize response formatter (Phase 3, Task 3.3)
        self.response_formatter = ResponseFormatter()
        
        logger.info("ChorusBot initialized with state persistence")
    
    async def setup_hook(self):
        """Called when the bot is setting up."""
        logger.info("Setting up bot...")
        
        # Verify Chorus Engine connection
        if not self.chorus_client.health_check():
            logger.warning("Cannot connect to Chorus Engine API!")
            logger.warning(f"Attempted URL: {self.config.chorus_api_url}")
        else:
            logger.info("✓ Connected to Chorus Engine")
        
        # Verify character exists
        try:
            character_info = self.chorus_client.get_character_info(
                self.config.chorus_character_id
            )
            character_name = character_info.get('name', self.config.chorus_character_id)
            logger.info(f"✓ Active character: {character_name}")
        except ChorusAPIError as e:
            logger.error(f"Failed to load character: {e}")
    
    async def on_ready(self):
        """Called when the bot is ready and connected to Discord."""
        logger.info(f"Bot connected as {self.user.name} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} server(s)")
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="@mentions"
            )
        )
        
        logger.info("Bot is ready!")
    
    async def on_message(self, message: discord.Message):
        """
        Handle incoming Discord messages.
        
        Args:
            message: Discord message object
        """
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Ignore messages from other bots
        if message.author.bot:
            return
        
        # Check if bot was mentioned or if it's a DM
        is_mentioned = self.user in message.mentions
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Only respond to mentions or DMs
        if not is_mentioned and not is_dm:
            # Process commands even if not mentioned
            await self.process_commands(message)
            return
        
        # Check if DM support is enabled
        if is_dm and not self.config.bridge_enable_dm_support:
            await message.channel.send("Sorry, I'm not configured to handle direct messages.")
            return
        
        logger.info(
            f"Message from {message.author.name} "
            f"in {'DM' if is_dm else f'#{message.channel.name}'}: "
            f"{message.content[:50]}..."
        )
        
        # Show typing indicator
        async with message.channel.typing():
            try:
                # Track user activity (Phase 2)
                display_name = message.author.display_name if hasattr(message.author, 'display_name') else None
                self.user_tracker.track_user(
                    discord_user_id=str(message.author.id),
                    username=message.author.name,
                    display_name=display_name
                )
                
                # Get or create conversation for this channel (Phase 2)
                conversation_id, thread_id = await self._get_or_create_conversation(message)
                
                # Phase 3: Fetch recent message history for context (sliding window)
                await self._sync_message_history(message, conversation_id, thread_id)
                
                # Clean message content (remove @mentions)
                clean_content = self._clean_message_content(message)
                
                if not clean_content.strip():
                    await message.channel.send("I need some text to respond to!")
                    return
                
                # Prepare metadata (for future deduplication)
                metadata = {
                    'discord_message_id': str(message.id),
                    'discord_channel_id': str(message.channel.id),
                    'discord_user_id': str(message.author.id),
                    'is_dm': is_dm,
                    'username': message.author.name,
                    'platform': 'discord'
                }
                
                # Send message to Chorus Engine
                logger.debug(f"Sending to Chorus: conversation={conversation_id}, thread={thread_id}")
                
                # Show typing indicator while processing (Phase 3, Task 3.3)
                async with message.channel.typing():
                    response = self.chorus_client.send_message(
                        conversation_id=conversation_id,
                        thread_id=thread_id,
                        message=clean_content,
                        user_name=message.author.name,
                        metadata=metadata,
                        primary_user=message.author.name,
                        conversation_source='discord'
                    )
                
                # Update conversation activity (Phase 2)
                discord_channel_id = str(message.channel.id) if not is_dm else str(message.author.id)
                self.conversation_mapper.update_last_message_time(discord_channel_id)
                
                # Extract assistant's response
                assistant_message = response.get('assistant_message', {})
                reply_content = assistant_message.get('content', '')
                
                if not reply_content:
                    logger.warning("Empty response from Chorus Engine")
                    await message.channel.send("*[No response generated]*")
                    return
                
                # Send response to Discord (handle 2000 char limit)
                await self._send_reply(message.channel, reply_content)
                
                logger.info(f"Sent reply ({len(reply_content)} chars)")
                
            except ChorusAPIError as e:
                logger.error(f"Chorus API error: {e}")
                await message.channel.send(
                    f"Sorry, I encountered an error communicating with my brain: {str(e)[:100]}"
                )
            
            except Exception as e:
                logger.exception(f"Unexpected error handling message: {e}")
                await message.channel.send(
                    "Sorry, something went wrong processing your message."
                )
        
        # Process commands after handling message
        await self.process_commands(message)
    
    async def _get_or_create_conversation(
        self,
        message: discord.Message
    ) -> tuple[str, int]:
        """
        Get or create a Chorus conversation for this Discord channel.
        
        Phase 2: Uses ConversationMapper for persistence across restarts.
        
        Args:
            message: Discord message object
            
        Returns:
            Tuple of (conversation_id, thread_id)
        """
        # Determine Discord IDs
        if isinstance(message.channel, discord.DMChannel):
            discord_channel_id = str(message.author.id)  # Use user ID for DMs
            discord_guild_id = None
            title = f"Discord DM with {message.author.name}"
            is_dm = True
        else:
            discord_channel_id = str(message.channel.id)
            discord_guild_id = str(message.guild.id) if message.guild else None
            channel_name = getattr(message.channel, 'name', 'unknown')
            guild_name = message.guild.name if message.guild else 'unknown'
            title = f"Discord: {guild_name} #{channel_name}"
            is_dm = False
        
        # Check if conversation mapping exists (Phase 2)
        mapping = self.conversation_mapper.get_conversation_mapping(discord_channel_id)
        
        if mapping:
            logger.debug(
                f"Using existing conversation: "
                f"Discord {discord_channel_id} -> Chorus {mapping['chorus_conversation_id']}"
            )
            return mapping['chorus_conversation_id'], mapping['chorus_thread_id']
        
        # Create new conversation in Chorus Engine
        # DMs are marked as private to prevent memory extraction/leakage
        logger.info(f"Creating new conversation: {title} (private={is_dm})")
        try:
            conv_data = self.chorus_client.create_conversation(
                character_id=self.config.chorus_character_id,
                title=title,
                is_private=is_dm  # DMs are private, channels are not
            )
            
            conversation_id = conv_data['conversation_id']
            thread_id = conv_data['thread_id']
            
            # Store mapping in database (Phase 2)
            self.conversation_mapper.get_or_create_conversation(
                discord_channel_id=discord_channel_id,
                discord_guild_id=discord_guild_id,
                chorus_conversation_id=conversation_id,
                chorus_thread_id=thread_id,
                is_dm=is_dm
            )
            
            logger.info(
                f"Created and mapped conversation: {conversation_id} "
                f"(thread: {thread_id})"
            )
            
            return conversation_id, thread_id
            
        except ChorusAPIError as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    def _clean_message_content(self, message: discord.Message) -> str:
        """
        Clean and convert Discord message content to Chorus format.
        
        Phase 3, Task 3.2: Use MessageProcessor for format conversion.
        
        Args:
            message: Discord message object
            
        Returns:
            Cleaned message content
        """
        # Remove @mentions of the bot first
        content = message.content
        content = content.replace(f'<@{self.user.id}>', '').strip()
        content = content.replace(f'<@!{self.user.id}>', '').strip()
        
        # Create a temporary message object with the cleaned content
        # for processing by MessageProcessor
        if content:
            # Process through MessageProcessor
            # Note: We need to temporarily modify message.content
            original_content = message.content
            message.content = content
            processed = self.message_processor.process_complete_message(message)
            message.content = original_content  # Restore original
            return processed
        
        # If no text content, still process attachments/embeds
        return self.message_processor.process_complete_message(message)
    
    async def _send_reply(
        self,
        channel: discord.abc.Messageable,
        content: str,
        max_length: int = 2000
    ):
        """
        Send a reply to Discord, splitting if necessary.
        
        Phase 3, Task 3.3: Use ResponseFormatter for intelligent splitting.
        
        Args:
            channel: Discord channel to send to
            content: Message content
            max_length: Maximum message length (Discord limit is 2000)
        """
        # Use response formatter to split message intelligently
        chunks = self.response_formatter.format_response(content)
        
        # Send each chunk
        for chunk in chunks:
            await channel.send(chunk)
    
    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle errors in event handlers."""
        logger.exception(f"Error in {event_method}")
    
    async def close(self):
        """Clean up resources when closing the bot."""
        logger.info("Closing bot...")
        await super().close()
    
    async def _sync_message_history(
        self,
        current_message: discord.Message,
        conversation_id: str,
        thread_id: int
    ):
        """
        Sync recent Discord message history to Chorus conversation.
        
        Phase 3, Task 3.1: Sliding window message fetching with deduplication.
        
        - Fetches last N messages from Discord channel (before current message)
        - Queries Chorus for existing Discord message IDs (stored in metadata)
        - Sends only NEW messages to Chorus (deduplication)
        - Maintains chronological order (oldest first)
        
        Args:
            current_message: The current Discord message being processed
            conversation_id: Chorus conversation ID
            thread_id: Chorus thread ID
        """
        try:
            # Configuration
            history_limit = self.config.bridge_history_limit or 10
            
            logger.debug(f"Syncing last {history_limit} messages for context...")
            
            # Fetch recent Discord messages (before current message)
            discord_messages = []
            async for msg in current_message.channel.history(
                limit=history_limit,
                before=current_message
            ):
                # Skip bot messages (they're already in Chorus as assistant messages)
                if msg.author == self.user:
                    continue
                
                # Skip other bot messages
                if msg.author.bot:
                    continue
                
                discord_messages.append(msg)
            
            # Reverse to get chronological order (oldest first)
            discord_messages.reverse()
            
            if not discord_messages:
                logger.debug("No history messages to sync")
                return
            
            logger.debug(f"Fetched {len(discord_messages)} Discord messages")
            
            # Get existing Discord message IDs from Chorus conversation
            existing_ids = await self._get_existing_discord_message_ids(
                conversation_id,
                thread_id
            )
            
            logger.debug(f"Chorus already has {len(existing_ids)} Discord messages")
            
            # Filter out messages already in Chorus (deduplication)
            new_messages = [
                msg for msg in discord_messages
                if str(msg.id) not in existing_ids
            ]
            
            if not new_messages:
                logger.debug("All history messages already in Chorus")
                return
            
            logger.info(f"Syncing {len(new_messages)} new messages to Chorus")
            
            # Send new messages to Chorus (chronologically)
            for msg in new_messages:
                await self._send_history_message_to_chorus(
                    msg,
                    conversation_id,
                    thread_id
                )
            
            logger.debug(f"✓ Synced {len(new_messages)} messages")
            
        except discord.Forbidden:
            logger.warning("Missing permissions to read message history")
        except Exception as e:
            logger.error(f"Error syncing message history: {e}", exc_info=True)
    
    async def _get_existing_discord_message_ids(
        self,
        conversation_id: str,
        thread_id: int
    ) -> set[str]:
        """
        Get set of Discord message IDs already in Chorus conversation.
        
        Queries Chorus API for message metadata to find discord_message_id values.
        
        Args:
            conversation_id: Chorus conversation ID
            thread_id: Chorus thread ID
            
        Returns:
            Set of Discord message ID strings
        """
        try:
            # Get conversation messages from Chorus
            messages = self.chorus_client.get_thread_messages(
                conversation_id=conversation_id,
                thread_id=thread_id
            )
            
            # Extract Discord message IDs from metadata
            existing_ids = set()
            for msg in messages:
                metadata = msg.get('metadata', {}) or {}
                discord_msg_id = metadata.get('discord_message_id')
                if discord_msg_id:
                    existing_ids.add(str(discord_msg_id))
            
            return existing_ids
            
        except ChorusAPIError as e:
            logger.warning(f"Failed to get existing message IDs: {e}")
            return set()  # Assume no existing messages on error
    
    async def _send_history_message_to_chorus(
        self,
        message: discord.Message,
        conversation_id: str,
        thread_id: int
    ):
        """
        Send a single Discord history message to Chorus (catch-up).
        
        This is used for syncing history, not for generating responses.
        
        Args:
            message: Discord message object
            conversation_id: Chorus conversation ID
            thread_id: Chorus thread ID
        """
        try:
            # Clean message content
            clean_content = self._clean_message_content(message)
            
            if not clean_content.strip():
                return  # Skip empty messages
            
            # Prepare metadata
            is_dm = isinstance(message.channel, discord.DMChannel)
            metadata = {
                'discord_message_id': str(message.id),
                'discord_channel_id': str(message.channel.id),
                'discord_user_id': str(message.author.id),
                'is_dm': is_dm,
                'username': message.author.name,
                'platform': 'discord',
                'is_history_sync': True  # Mark as history (not live message)
            }
            
            # Send to Chorus (without expecting response - this is catch-up)
            self.chorus_client.add_user_message(
                conversation_id=conversation_id,
                thread_id=thread_id,
                message=clean_content,
                user_name=message.author.name,
                metadata=metadata
            )
            
            logger.debug(f"Synced message {message.id[:8]}... from {message.author.name}")
            
        except ChorusAPIError as e:
            logger.warning(f"Failed to sync message {message.id}: {e}")
        except Exception as e:
            logger.error(f"Error syncing message {message.id}: {e}")
        self.chorus_client.close()
        await super().close()
