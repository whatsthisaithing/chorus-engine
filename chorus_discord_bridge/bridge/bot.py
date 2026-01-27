"""Discord bot for Chorus Engine bridge."""

import discord
import logging
import asyncio
import io
import re
from typing import Optional, List
from discord.ext import commands

from .config import BridgeConfig
from .chorus_client import ChorusClient, ChorusAPIError, ConversationNotFoundError
from .conversation_mapper import ConversationMapper
from .user_tracker import UserTracker
from .database import init_database, get_database
from .message_processor import MessageProcessor
from .response_formatter import ResponseFormatter
from .image_handler import ImageHandler  # Phase 3: Vision support

logger = logging.getLogger(__name__)


class ChorusBot(commands.Bot):
    """Discord bot that bridges to Chorus Engine."""
    
    def __init__(self, config: BridgeConfig, character_id: str, chorus_client: Optional[ChorusClient] = None):
        """
        Initialize the Discord bot.
        
        Args:
            config: Bridge configuration
            character_id: Character ID to use for this bot instance
            chorus_client: Optional shared Chorus client instance
        """
        self.config = config
        self.character_id = character_id
        
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
        
        # Use shared Chorus client or create new one
        if chorus_client:
            self.chorus_client = chorus_client
        else:
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
        
        # Get database instance for shared use
        self.database = get_database(db_path)
        
        self.conversation_mapper = ConversationMapper(db_path)
        self.user_tracker = UserTracker(db_path)
        
        # Initialize message processor (Phase 3, Task 3.2)
        self.message_processor = MessageProcessor(self)
        
        # Initialize response formatter (Phase 3, Task 3.3)
        self.response_formatter = ResponseFormatter()
        
        # Initialize image handler (Phase 3: Vision support)
        vision_config = getattr(config, 'vision', {})
        self.image_handler = ImageHandler(
            chorus_client=self.chorus_client,
            database=self.database,
            max_file_size_mb=vision_config.get('max_file_size_mb', 10),
            max_images_per_message=vision_config.get('max_images_per_message', 5)
        )
        
        # Rate limiting: Per-user cooldowns (Phase 4, Task 4.2)
        self.user_cooldowns = {}  # user_id -> last_message_timestamp
        
        # Loop detection: Track consecutive bot responses (Phase 4, Task 4.2)
        self.consecutive_responses = {}  # channel_id -> count
        
        logger.info(f"ChorusBot initialized for character '{self.character_id}' with state persistence")
    
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
            character_info = self.chorus_client.get_character_info(self.character_id)
            character_name = character_info.get('name', self.character_id)
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
        
        # Phase 4, Task 4.2: Reset consecutive response counter (human message received)
        channel_id = str(message.channel.id)
        self._reset_consecutive_responses(channel_id)
        
        # Phase 4, Task 4.2: Check per-user cooldown
        if self._is_on_cooldown(str(message.author.id), cooldown_seconds=1.0):
            logger.debug(f"User {message.author.name} on cooldown, skipping")
            return
        
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
                
                # Phase 4, Task 4.2: Check for runaway loop (bot-to-bot conversations)
                if self._check_runaway_loop(channel_id, max_consecutive=5):
                    # Silently pause - don't spam channel with meta-messages
                    logger.info(f"Pausing due to runaway loop detection in {channel_id}")
                    await message.add_reaction("🛑")  # Signal that loop was detected
                    return
                
                # Phase 3: Fetch recent message history for context (sliding window)
                await self._sync_message_history(message, conversation_id, thread_id)
                
                # Clean message content (remove @mentions)
                clean_content = self._clean_message_content(message)
                
                if not clean_content.strip():
                    await message.channel.send("I need some text to respond to!")
                    return
                
                # Phase 3: Process images in current message (if any)
                image_attachment_ids = []
                if message.attachments:
                    try:
                        logger.debug(f"Processing {len(message.attachments)} attachment(s) from current message")
                        image_attachment_ids = await self.image_handler.process_message_images(
                            message=message,
                            character_id=self.character_id
                        )
                        if image_attachment_ids:
                            logger.info(
                                f"Processed {len(image_attachment_ids)} image(s) from current message"
                            )
                        elif any(att.content_type and att.content_type.startswith('image/') for att in message.attachments):
                            # Had images but none were processed - notify user
                            await message.channel.send(
                                f"*{message.author.mention}, I had trouble processing the image(s) in your message. "
                                f"The message text will still be processed, but I won't be able to see the images.*",
                                reference=message
                            )
                    except Exception as e:
                        logger.error(f"Failed to process images in current message: {e}", exc_info=True)
                        # Notify user of failure
                        try:
                            await message.channel.send(
                                f"*{message.author.mention}, I encountered an error processing your image(s). "
                                f"I'll continue with the text message only.*",
                                reference=message
                            )
                        except:
                            pass  # Don't let notification failure block the main flow
                
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
                        conversation_source='discord',
                        image_attachment_ids=image_attachment_ids if image_attachment_ids else None
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
                
                # Phase 4, Task 4.6: Check for image generation preview
                # If confirmation is disabled, trigger generation immediately
                image_prompt_preview = response.get('image_prompt_preview')
                if image_prompt_preview and not image_prompt_preview.get('needs_confirmation', True):
                    logger.info(f"Image preview detected, triggering generation (confirmation disabled)")
                    try:
                        # Trigger image generation using the preview prompt
                        image_result = self.chorus_client.confirm_and_generate_image(
                            thread_id=thread_id,
                            prompt=image_prompt_preview['prompt'],
                            negative_prompt=image_prompt_preview.get('negative_prompt', ''),
                            disable_future_confirmations=False  # Already disabled
                        )
                        
                        if image_result and image_result.get('success'):
                            logger.info(f"Image generated successfully: {image_result.get('image_id')}")
                            # Update assistant_message with image metadata for download
                            if 'metadata' not in assistant_message:
                                assistant_message['metadata'] = {}
                            assistant_message['metadata']['image_path'] = image_result.get('file_path', '').replace('data/', '/')
                            assistant_message['metadata']['image_id'] = image_result.get('image_id')
                        else:
                            logger.warning(f"Image generation failed: {image_result.get('error') if image_result else 'Unknown error'}")
                    except Exception as e:
                        logger.error(f"Failed to generate image: {e}")
                
                # Check for image in assistant message metadata (Phase 4, Task 4.6)
                image_file = await self._download_image_if_present(assistant_message)
                
                # Convert <username> references to Discord @mentions (Phase 4, Tasks 4.7 & 4.8)
                reply_content = await self._process_username_mentions(message, reply_content)
                
                # Send response to Discord (handle 2000 char limit)
                discord_message_ids = await self._send_reply(message.channel, reply_content, image_file=image_file)
                
                # Update assistant message metadata with Discord message IDs to prevent duplication
                # when syncing history in future messages
                if discord_message_ids and assistant_message.get('id'):
                    try:
                        # Store first Discord message ID in the assistant message metadata
                        # This allows the history sync to recognize this message and skip it
                        update_metadata = {
                            'discord_message_id': discord_message_ids[0],  # Primary message ID
                            'discord_message_ids': discord_message_ids,  # All IDs if split
                            'discord_channel_id': str(message.channel.id),
                            'platform': 'discord'
                        }
                        
                        self.chorus_client.update_message_metadata(
                            message_id=assistant_message['id'],
                            metadata=update_metadata
                        )
                        logger.debug(f"Updated assistant message {assistant_message['id'][:8]} with Discord IDs")
                    except Exception as e:
                        logger.warning(f"Failed to update assistant message metadata: {e}")
                
                # Phase 4, Task 4.2: Increment consecutive response counter
                self._increment_consecutive_responses(channel_id)
                
                logger.info(f"Sent reply ({len(reply_content)} chars{', with image' if image_file else ''})")
            
            except ConversationNotFoundError as e:
                # Conversation was deleted in Chorus - rebuild silently and process message
                logger.warning(f"Conversation not found (deleted?): {e}")
                logger.info("Rebuilding conversation and processing message...")
                
                discord_channel_id = str(message.channel.id) if not is_dm else str(message.author.id)
                self.conversation_mapper.delete_conversation_mapping(discord_channel_id)
                
                # Clear user cooldown to allow immediate retry
                self._clear_cooldown(str(message.author.id))
                
                # Recursively call on_message to create new conversation and process message
                # This gives seamless UX - user doesn't need to resend
                try:
                    await self.on_message(message)
                except Exception as rebuild_error:
                    logger.error(f"Failed to rebuild conversation: {rebuild_error}")
                    await message.channel.send(
                        "Sorry, I encountered an error rebuilding my memory. Please try again."
                    )
                
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
        logger.info(f"[MAPPING CHECK] Looking up Discord channel: {discord_channel_id}")
        mapping = self.conversation_mapper.get_conversation_mapping(
            discord_channel_id, 
            character_id=self.character_id
        )
        logger.info(f"[MAPPING RESULT] Found mapping: {mapping is not None}")
        
        if mapping:
            logger.info(
                f"Using existing conversation: "
                f"Discord {discord_channel_id} ({self.character_id}) -> "
                f"Chorus {mapping['chorus_conversation_id']}"
            )
            return mapping['chorus_conversation_id'], mapping['chorus_thread_id']
        
        # Create new conversation in Chorus Engine
        # DMs are marked as private to prevent memory extraction/leakage
        logger.info(f"Creating new conversation: {title} (private={is_dm})")
        try:
            conv_data = self.chorus_client.create_conversation(
                character_id=self.character_id,
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
                character_id=self.character_id,
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
        
        Preserves @mentions (converted to @username by MessageProcessor) so bots
        can see who is being addressed in group conversations.
        
        Args:
            message: Discord message object
            
        Returns:
            Cleaned message content
        """
        # Let MessageProcessor handle all mentions (including this bot's)
        # This preserves "who's talking to whom" context in group chats
        return self.message_processor.process_complete_message(message)
    
    async def _download_image_if_present(
        self,
        assistant_message: dict
    ) -> Optional[discord.File]:
        """
        Download image from Chorus API if present in response metadata.
        
        Phase 4, Task 4.6: Image generation support with graceful fallback.
        
        Args:
            assistant_message: Assistant message dict from Chorus response
            
        Returns:
            discord.File object if image downloaded successfully, None otherwise
        """
        try:
            metadata = assistant_message.get('metadata', {})
            if not metadata:
                return None
            
            image_path = metadata.get('image_path')
            if not image_path:
                return None
            
            # Image was generated, try to download it
            logger.info(f"Downloading image from Chorus: {image_path}")
            
            # Build full URL
            image_url = f"{self.config.chorus_api_url}{image_path}"
            
            # Download image using ChorusClient session
            try:
                response = self.chorus_client.session.get(image_url, timeout=10)
                response.raise_for_status()
                
                image_data = response.content
                
                # Get filename from path or use default
                filename = image_path.split('/')[-1] if '/' in image_path else 'image.png'
                
                # Create Discord File object
                discord_file = discord.File(
                    io.BytesIO(image_data),
                    filename=filename
                )
                
                logger.info(f"Successfully downloaded image ({len(image_data)} bytes)")
                return discord_file
                
            except Exception as download_error:
                logger.error(f"Failed to download image from {image_url}: {download_error}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking for image: {e}")
            return None
    
    async def _process_username_mentions(
        self,
        message: discord.Message,
        content: str
    ) -> str:
        """
        Process LLM response to convert @username or <username> references to Discord @mentions.
        
        Phase 4, Tasks 4.7 & 4.8: Username mention formatting and case preservation.
        
        The LLM may output @username or <username> when referring to users.
        This method converts those to Discord mention format (<@user_id>) for notifications.
        Shortened/informal names without @ or brackets are left as-is (no notification).
        
        Args:
            message: Original Discord message (for channel context)
            content: Response text from LLM
            
        Returns:
            Processed content with @username converted to Discord mentions
        """
        logger.debug(f"Processing username mentions in: {content[:100]}...")
        
        # Find all @username or <username> patterns
        # Matches: @Username, @User123, <Username>, etc.
        username_pattern = r'(?:@|<)([A-Za-z0-9_]+)>?'
        matches = re.findall(username_pattern, content)
        
        logger.debug(f"Found {len(matches)} username patterns: {matches}")
        
        if not matches:
            return content  # No username references found
        
        # Build username → user_id mapping from recent channel messages
        username_map = await self._build_username_map(message.channel)
        
        logger.debug(f"Username map has {len(username_map)} entries: {list(username_map.keys())}")
        
        # Convert each @username or <username> to Discord mention
        def replace_username(match):
            full_match = match.group(0)  # Full match including @ or <
            username = match.group(1)     # Just the username part
            
            # Case-insensitive lookup
            user_id = username_map.get(username.lower())
            
            if user_id:
                # Found exact match - convert to Discord mention
                logger.debug(f"Converting {full_match} to <@{user_id}>")
                return f"<@{user_id}>"
            else:
                # No match - keep original format
                logger.debug(f"No match for {full_match}, keeping original")
                return full_match
        
        processed_content = re.sub(username_pattern, replace_username, content)
        logger.debug(f"Processed content: {processed_content[:100]}...")
        return processed_content
    
    async def _build_username_map(
        self,
        channel: discord.abc.Messageable
    ) -> dict[str, str]:
        """
        Build a mapping of username (lowercase) → discord_user_id from recent messages.
        
        Args:
            channel: Discord channel to fetch recent messages from
            
        Returns:
            Dict mapping lowercase username to discord_user_id
        """
        username_map = {}
        
        try:
            # Fetch last 50 messages (more than history sync limit to catch more participants)
            async for msg in channel.history(limit=50):
                # Map author's username (lowercase) to their user_id
                username_lower = msg.author.name.lower()
                username_map[username_lower] = str(msg.author.id)
            
            logger.debug(f"Built username map with {len(username_map)} users")
            
        except Exception as e:
            logger.warning(f"Failed to build username map: {e}")
        
        return username_map
    
    async def _send_reply(
        self,
        channel: discord.abc.Messageable,
        content: str,
        max_length: int = 2000,
        image_file: Optional[discord.File] = None
    ) -> List[str]:
        """
        Send a reply to Discord, splitting if necessary.
        
        Phase 3, Task 3.3: Use ResponseFormatter for intelligent splitting.
        Phase 4, Task 4.6: Image attachment support.
        
        Args:
            channel: Discord channel to send to
            content: Message content
            max_length: Maximum message length (Discord limit is 2000)
            image_file: Optional image file to attach (only to first message)
            
        Returns:
            List of Discord message IDs for sent messages
        """
        # Use response formatter to split message intelligently
        chunks = self.response_formatter.format_response(content)
        
        # Send each chunk and collect message IDs
        message_ids = []
        for i, chunk in enumerate(chunks):
            # Attach image only to first message chunk
            if i == 0 and image_file:
                sent_msg = await channel.send(chunk, file=image_file)
            else:
                sent_msg = await channel.send(chunk)
            message_ids.append(str(sent_msg.id))
        
        return message_ids
    
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
            # Configuration (Phase 3: Reduced to 5 for VRAM efficiency with vision)
            history_limit = 5
            
            logger.debug(f"Syncing last {history_limit} messages for context...")
            
            # Fetch recent Discord messages (before current message)
            # Include ALL messages for full group chat context (including other bots)
            discord_messages = []
            async for msg in current_message.channel.history(
                limit=history_limit,
                before=current_message
            ):
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
        Handles both user messages and bot's own assistant messages.
        
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
            
            # Determine if this is a bot message (assistant) or user message
            is_bot_message = message.author == self.user
            
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
            
            # Phase 3: Process images in historical message (if any)
            image_attachment_ids = []
            if not is_bot_message and message.attachments:
                try:
                    image_attachment_ids = await self.image_handler.process_message_images(
                        message=message,
                        character_id=self.character_id
                    )
                    if image_attachment_ids:
                        logger.info(
                            f"Processed {len(image_attachment_ids)} image(s) from history message {str(message.id)[:8]}"
                        )
                except Exception as e:
                    logger.error(f"Failed to process images in history message: {e}", exc_info=True)
            
            # Send to Chorus based on message type
            if is_bot_message:
                # This is the bot's own response - add as assistant message
                self.chorus_client.add_assistant_message(
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    message=clean_content,
                    metadata=metadata
                )
                logger.debug(f"Synced ASSISTANT message {str(message.id)[:8]}...")
            else:
                # This is a user message
                self.chorus_client.add_user_message(
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    message=clean_content,
                    user_name=message.author.name,
                    metadata=metadata,
                    image_attachment_ids=image_attachment_ids if image_attachment_ids else None
                )
                logger.debug(
                    f"Synced USER message {str(message.id)[:8]}... from {message.author.name}"
                    f"{f' with {len(image_attachment_ids)} image(s)' if image_attachment_ids else ''}"
                )
            
        except ChorusAPIError as e:
            logger.warning(f"Failed to sync message {message.id}: {e}")
        except Exception as e:
            logger.error(f"Error syncing message {message.id}: {e}")
    
    def _is_on_cooldown(self, user_id: str, cooldown_seconds: float = 1.0) -> bool:
        """
        Check if user is on cooldown.
        
        Args:
            user_id: Discord user ID
            cooldown_seconds: Cooldown duration in seconds
            
        Returns:
            True if user is on cooldown, False otherwise
        """
        import time
        now = time.time()
        
        if user_id in self.user_cooldowns:
            last_message = self.user_cooldowns[user_id]
            if now - last_message < cooldown_seconds:
                return True
        
        # Update timestamp
        self.user_cooldowns[user_id] = now
        return False
    
    def _clear_cooldown(self, user_id: str):
        """
        Clear cooldown for a user.
        Used when retrying after conversation rebuild.
        
        Args:
            user_id: Discord user ID
        """
        if user_id in self.user_cooldowns:
            del self.user_cooldowns[user_id]
    
    def _check_runaway_loop(self, channel_id: str, max_consecutive: int = 5) -> bool:
        """
        Check if bot is in a runaway loop (too many consecutive responses).
        
        Detects when bot responds many times in a row without human intervention,
        which can happen in bot-to-bot conversations.
        
        Args:
            channel_id: Discord channel ID
            max_consecutive: Maximum consecutive bot responses allowed
            
        Returns:
            True if runaway loop detected, False otherwise
        """
        consecutive = self.consecutive_responses.get(channel_id, 0)
        
        if consecutive >= max_consecutive:
            logger.warning(
                f"Runaway loop detected in channel {channel_id}: "
                f"{consecutive} consecutive responses"
            )
            return True
        
        return False
    
    def _increment_consecutive_responses(self, channel_id: str):
        """Increment consecutive response counter for channel."""
        self.consecutive_responses[channel_id] = \
            self.consecutive_responses.get(channel_id, 0) + 1
    
    def _reset_consecutive_responses(self, channel_id: str):
        """Reset consecutive response counter for channel (called on human message)."""
        self.consecutive_responses[channel_id] = 0
    
    async def close(self):
        """Clean shutdown."""
        self.chorus_client.close()
        await super().close()
