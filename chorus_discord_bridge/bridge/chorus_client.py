"""HTTP client for interacting with Chorus Engine API."""

import aiohttp
import asyncio
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ChorusAPIError(Exception):
    """Raised when Chorus API request fails."""
    pass


class ConversationNotFoundError(ChorusAPIError):
    """Raised when conversation/thread is not found (404)."""
    pass


class ChorusClient:
    """Async client for communicating with Chorus Engine API."""
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize Chorus API client.
        
        Args:
            api_url: Base URL of Chorus Engine API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            retry_delay: Seconds to wait between retries
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Setup headers
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ChorusDiscordBridge/0.1.0'
        }
        
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Session will be created lazily
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            )
        return self._session
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make async HTTP request to Chorus API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            json: JSON body for request
            params: URL parameters
            
        Returns:
            Response JSON as dictionary
            
        Raises:
            ChorusAPIError: If request fails
        """
        url = f"{self.api_url}{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"{method} {url} (attempt {attempt + 1}/{self.retry_attempts})")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params
                ) as response:
                    # Check for HTTP errors
                    if response.status >= 400:
                        try:
                            error_data = await response.json()
                            error_detail = error_data.get('detail', await response.text())
                        except:
                            error_detail = await response.text()
                        
                        # Special handling for 404 on conversation/thread endpoints
                        if response.status == 404 and ('/threads/' in url or '/conversations/' in url):
                            raise ConversationNotFoundError(f"Conversation or thread not found: {error_detail}")
                        
                        # Retry on 5xx errors
                        if response.status >= 500 and attempt < self.retry_attempts - 1:
                            logger.warning(f"HTTP {response.status} error, retrying in {self.retry_delay}s...")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        
                        raise ChorusAPIError(f"HTTP {response.status}: {error_detail}")
                    
                    # Parse JSON response
                    try:
                        return await response.json()
                    except:
                        # If response is not JSON, return empty dict
                        return {}
                        
            except aiohttp.ClientError as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Request failed: {e}, retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                logger.error(f"Request failed after {self.retry_attempts} attempts: {url}")
                raise ChorusAPIError(f"Request failed: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error calling Chorus API: {str(e)}")
                logger.exception("Full traceback:")  # This will log the full stack trace
                raise ChorusAPIError(f"Unexpected error: {str(e)}")
    
    async def get_character_info(self, character_id: str) -> Dict[str, Any]:
        """
        Get character information from Chorus Engine.
        
        Args:
            character_id: Character identifier
            
        Returns:
            Character information dictionary
        """
        logger.info(f"Fetching character info: {character_id}")
        return await self._request('GET', f'/characters/{character_id}')
    
    async def create_conversation(
        self,
        character_id: str,
        title: str,
        is_private: bool = False,
        source: str = 'discord',
        image_confirmation_disabled: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new conversation with a character.
        
        Args:
            character_id: Character to converse with
            title: Conversation title
            is_private: Whether conversation is private
            source: Source platform (discord, test, etc.)
            image_confirmation_disabled: Bypass image generation confirmation dialog (default True for Discord)
            
        Returns:
            Conversation data with conversation_id and thread_id
        """
        logger.info(f"Creating conversation: {title} (character: {character_id}, source: {source})")
        
        payload = {
            'character_id': character_id,
            'title': title,
            'source': source,
            'image_confirmation_disabled': image_confirmation_disabled
        }
        
        # Create conversation (automatically creates a "Main Thread")
        conversation = await self._request('POST', '/conversations', json=payload)
        conversation_id = conversation['id']
        
        # Fetch the auto-created thread
        threads = await self._request('GET', f'/conversations/{conversation_id}/threads')
        
        if not threads:
            raise ChorusAPIError("No threads found in newly created conversation")
        
        # Get the first (main) thread
        thread_id = threads[0]['id']
        
        logger.info(f"Created conversation {conversation_id} with thread {thread_id}")
        
        # Return in expected format for compatibility
        return {
            'conversation_id': conversation_id,
            'thread_id': thread_id,
            'id': conversation_id,  # Also include raw response
            **conversation
        }
    
    async def send_message(
        self,
        conversation_id: str,
        thread_id: int,
        message: str,
        user_name: str = "Discord User",
        metadata: Optional[Dict[str, Any]] = None,
        primary_user: Optional[str] = None,
        conversation_source: str = 'discord',
        image_attachment_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to an existing conversation.
        
        Args:
            conversation_id: Conversation identifier
            thread_id: Thread identifier within conversation
            message: Message content
            user_name: Name of the user sending the message
            metadata: Optional metadata to attach to the message
            primary_user: Name of the user who invoked the bot (for multi-user contexts)
            conversation_source: Platform source (default: 'discord')
            image_attachment_ids: List of Chorus attachment IDs to link (Phase 3)
            
        Returns:
            Response including assistant's reply
        """
        logger.info(f"Sending message to conversation {conversation_id} (thread {thread_id})")
        logger.debug(f"Message preview: {message[:100]}...")
        
        payload = {
            'message': message,
            'user_name': user_name,
            'conversation_source': conversation_source
        }
        
        if metadata:
            payload['metadata'] = metadata
        
        if primary_user:
            payload['primary_user'] = primary_user
        
        if image_attachment_ids:
            payload['image_attachment_ids'] = image_attachment_ids
            logger.info(f"Including {len(image_attachment_ids)} image attachment(s)")
        
        endpoint = f'/threads/{thread_id}/messages'
        return await self._request('POST', endpoint, json=payload)
    
    async def confirm_and_generate_image(
        self,
        thread_id: int,
        prompt: str,
        negative_prompt: str = '',
        disable_future_confirmations: bool = False
    ) -> Dict[str, Any]:
        """
        Confirm and generate an image for a thread.
        
        Args:
            thread_id: Thread identifier
            prompt: Image generation prompt
            negative_prompt: Negative prompt for image generation
            disable_future_confirmations: Whether to disable future confirmation dialogs
            
        Returns:
            Image generation result with success status and image details
        """
        logger.info(f"Confirming and generating image for thread {thread_id}")
        
        payload = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'disable_future_confirmations': disable_future_confirmations
        }
        
        endpoint = f'/threads/{thread_id}/generate-image'
        return await self._request('POST', endpoint, json=payload)
    
    async def update_message_metadata(
        self,
        message_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update metadata for an existing message.
        
        Args:
            message_id: Message identifier
            metadata: Metadata to merge with existing metadata
            
        Returns:
            Updated message data
        """
        logger.debug(f"Updating metadata for message {message_id[:8]}...")
        
        payload = {'metadata': metadata}
        endpoint = f'/messages/{message_id}/metadata'
        return await self._request('PATCH', endpoint, json=payload)
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        thread_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get message history from a conversation.
        
        Args:
            conversation_id: Conversation identifier
            thread_id: Thread identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of message dictionaries
        """
        logger.info(f"Fetching history for conversation {conversation_id} (thread {thread_id})")
        
        # Note: This endpoint may not exist in current API
        # Will need to be implemented or use alternative method
        endpoint = f'/threads/{thread_id}/messages'
        params = {'limit': limit} if limit else None
        
        response = await self._request('GET', endpoint, params=params)
        return response.get('messages', [])
    
    async def get_thread_messages(
        self,
        conversation_id: str,
        thread_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all messages from a thread.
        
        Phase 3, Task 3.1: Used for deduplication - fetches existing messages
        to check which Discord messages are already in Chorus.
        
        Args:
            conversation_id: Conversation identifier
            thread_id: Thread identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of message dictionaries with metadata
        """
        logger.debug(f"Fetching messages for thread {thread_id}")
        
        endpoint = f'/threads/{thread_id}/messages'
        params = {}
        if limit:
            params['limit'] = limit
        
        response = await self._request('GET', endpoint, params=params)
        # API returns list directly, not wrapped in dict
        return response if isinstance(response, list) else response.get('messages', [])
    
    async def add_user_message(
        self,
        conversation_id: str,
        thread_id: int,
        message: str,
        user_name: str,
        metadata: Optional[Dict] = None,
        image_attachment_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add a user message to a thread without generating a response.
        
        Phase 3, Task 3.1: Used for syncing message history (catch-up).
        Unlike send_message(), this doesn't trigger LLM generation.
        
        Args:
            conversation_id: Conversation identifier
            thread_id: Thread identifier
            message: Message content
            user_name: Name of the user
            metadata: Optional metadata (should include discord_message_id)
            image_attachment_ids: List of Chorus attachment IDs to link (Phase 3)
            
        Returns:
            Response from API
        """
        logger.debug(f"Adding history message to thread {thread_id}")
        
        payload = {
            'content': message,
            'role': 'user',
            'metadata': metadata or {}
        }
        
        # Add username to metadata if not already present
        if 'username' not in payload['metadata']:
            payload['metadata']['username'] = user_name
        
        # Add image attachments if present (Phase 3)
        if image_attachment_ids:
            payload['image_attachment_ids'] = image_attachment_ids
            logger.debug(f"Including {len(image_attachment_ids)} image attachment(s) in history")
        
        endpoint = f'/threads/{thread_id}/messages/add'
        return await self._request('POST', endpoint, json=payload)
    
    async def add_assistant_message(
        self,
        conversation_id: str,
        thread_id: int,
        message: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add an assistant message to a thread without generating a response.
        
        Phase 3: Used for syncing bot's own message history when rebuilding conversations.
        This ensures assistant responses from Discord are preserved in Chorus.
        
        Args:
            conversation_id: Conversation identifier
            thread_id: Thread identifier
            message: Message content from the bot
            metadata: Optional metadata (should include discord_message_id)
            
        Returns:
            Response from API
        """
        logger.debug(f"Adding assistant history message to thread {thread_id}")
        
        payload = {
            'content': message,
            'role': 'assistant',
            'metadata': metadata or {}
        }
        
        endpoint = f'/threads/{thread_id}/messages/add'
        return await self._request('POST', endpoint, json=payload)
    
    async def upload_image(self, file_path, filename: str) -> Optional[str]:
        """
        Upload an image to Chorus Engine (Phase 3 - Vision System).
        
        Uses the two-step upload process:
        1. Upload file to /api/attachments/upload
        2. Returns attachment_id for linking to messages
        
        Args:
            file_path: Path to image file
            filename: Original filename
            
        Returns:
            Attachment ID if successful, None otherwise
        """
        try:
            logger.debug(f"Uploading image {filename} to Chorus API")
            
            # Detect MIME type from filename
            mime_type = 'image/jpeg'
            if filename.lower().endswith('.png'):
                mime_type = 'image/png'
            elif filename.lower().endswith('.webp'):
                mime_type = 'image/webp'
            elif filename.lower().endswith('.gif'):
                mime_type = 'image/gif'
            
            # Read file content into memory
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Create multipart form data
            form = aiohttp.FormData()
            form.add_field(
                'file',
                file_content,
                filename=filename,
                content_type=mime_type
            )
            
            # Upload using aiohttp
            # Create temporary session without Content-Type header
            upload_headers = {}
            if self.api_key:
                upload_headers['Authorization'] = f'Bearer {self.api_key}'
            
            async with aiohttp.ClientSession() as upload_session:
                async with upload_session.post(
                    f"{self.api_url}/api/attachments/upload",
                    data=form,
                    headers=upload_headers,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Upload failed: {response.status} {error_text}")
                        return None
                    
                    data = await response.json()
                    attachment_id = data.get('attachment_id')
                    
                    if not attachment_id:
                        logger.error(f"No attachment_id in response: {data}")
                        return None
                    
                    logger.info(f"Successfully uploaded image: {attachment_id}")
                    return attachment_id
            
        except Exception as e:
            logger.error(f"Failed to upload image {filename}: {e}", exc_info=True)
            return None
    
    async def health_check(self) -> bool:
        """
        Check if Chorus Engine API is accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            logger.debug("Performing health check")
            async with self.session.get(
                f"{self.api_url}/characters",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
