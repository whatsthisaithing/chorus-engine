"""Discord bot for Chorus Engine bridge."""

import discord
import logging
import asyncio
from typing import Optional
from discord.ext import commands

from .config import BridgeConfig
from .chorus_client import ChorusClient, ChorusAPIError

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
        
        # Cache for conversation mappings (will be replaced with DB in Phase 2)
        self._conversation_cache: dict[str, tuple[str, int]] = {}
        
        logger.info("ChorusBot initialized")
    
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
                # Get or create conversation for this channel
                conversation_id, thread_id = await self._get_or_create_conversation(message)
                
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
                    'is_dm': is_dm
                }
                
                # Send message to Chorus Engine
                logger.debug(f"Sending to Chorus: conversation={conversation_id}, thread={thread_id}")
                response = self.chorus_client.send_message(
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    message=clean_content,
                    user_name=message.author.name,
                    metadata=metadata
                )
                
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
        
        Args:
            message: Discord message object
            
        Returns:
            Tuple of (conversation_id, thread_id)
        """
        # Create a unique key for this channel/DM
        if isinstance(message.channel, discord.DMChannel):
            channel_key = f"dm_{message.author.id}"
            title = f"Discord DM with {message.author.name}"
        else:
            channel_key = f"channel_{message.channel.id}"
            channel_name = getattr(message.channel, 'name', 'unknown')
            guild_name = message.guild.name if message.guild else 'unknown'
            title = f"Discord: {guild_name} #{channel_name}"
        
        # Check cache (will be replaced with DB in Phase 2)
        if channel_key in self._conversation_cache:
            logger.debug(f"Using cached conversation for {channel_key}")
            return self._conversation_cache[channel_key]
        
        # Create new conversation
        logger.info(f"Creating new conversation for {channel_key}: {title}")
        try:
            conv_data = self.chorus_client.create_conversation(
                character_id=self.config.chorus_character_id,
                title=title,
                is_private=False
            )
            
            conversation_id = conv_data['conversation_id']
            thread_id = conv_data['thread_id']
            
            # Cache the mapping
            self._conversation_cache[channel_key] = (conversation_id, thread_id)
            
            logger.info(
                f"Created conversation: {conversation_id} "
                f"(thread: {thread_id})"
            )
            
            return conversation_id, thread_id
            
        except ChorusAPIError as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    def _clean_message_content(self, message: discord.Message) -> str:
        """
        Clean Discord message content for sending to Chorus.
        
        Args:
            message: Discord message object
            
        Returns:
            Cleaned message content
        """
        # Start with the content
        content = message.content
        
        # Remove @mentions of the bot
        content = content.replace(f'<@{self.user.id}>', '').strip()
        content = content.replace(f'<@!{self.user.id}>', '').strip()
        
        # TODO: Convert other @mentions to readable names
        # TODO: Handle Discord emojis
        # TODO: Handle attachments
        
        return content
    
    async def _send_reply(
        self,
        channel: discord.abc.Messageable,
        content: str,
        max_length: int = 2000
    ):
        """
        Send a reply to Discord, splitting if necessary.
        
        Args:
            channel: Discord channel to send to
            content: Message content
            max_length: Maximum message length (Discord limit is 2000)
        """
        if len(content) <= max_length:
            await channel.send(content)
            return
        
        # Split into multiple messages
        logger.info(f"Splitting long message ({len(content)} chars)")
        
        # Try to split on newlines first
        lines = content.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_length:
                # Send current chunk
                if current_chunk:
                    await channel.send(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        # Send remaining chunk
        if current_chunk:
            await channel.send(current_chunk)
    
    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle errors in event handlers."""
        logger.exception(f"Error in {event_method}")
    
    async def close(self):
        """Clean up resources when closing the bot."""
        logger.info("Closing bot...")
        self.chorus_client.close()
        await super().close()
