"""HTTP client for interacting with Chorus Engine API."""

import requests
import time
import logging
from typing import Dict, Any, Optional, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ChorusAPIError(Exception):
    """Raised when Chorus API request fails."""
    pass


class ChorusClient:
    """Client for communicating with Chorus Engine API."""
    
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
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ChorusDiscordBridge/0.1.0'
        })
        
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Chorus API.
        
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
        
        try:
            logger.debug(f"{method} {url}")
            response = self.session.request(
                method=method,
                url=url,
                json=json,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse JSON response
            try:
                return response.json()
            except ValueError:
                # If response is not JSON, return empty dict
                return {}
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout calling Chorus API: {url}")
            raise ChorusAPIError(f"Request timeout: {str(e)}")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Chorus API: {url}")
            raise ChorusAPIError(f"Connection error: {str(e)}")
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Chorus API: {e.response.status_code} - {url}")
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            raise ChorusAPIError(f"HTTP {e.response.status_code}: {error_detail}")
        
        except Exception as e:
            logger.error(f"Unexpected error calling Chorus API: {str(e)}")
            raise ChorusAPIError(f"Unexpected error: {str(e)}")
    
    def get_character_info(self, character_id: str) -> Dict[str, Any]:
        """
        Get character information from Chorus Engine.
        
        Args:
            character_id: Character identifier
            
        Returns:
            Character information dictionary
        """
        logger.info(f"Fetching character info: {character_id}")
        return self._request('GET', f'/characters/{character_id}')
    
    def create_conversation(
        self,
        character_id: str,
        title: str,
        is_private: bool = False,
        source: str = 'discord'
    ) -> Dict[str, Any]:
        """
        Create a new conversation with a character.
        
        Args:
            character_id: Character to converse with
            title: Conversation title
            is_private: Whether conversation is private
            source: Source platform (discord, test, etc.)
            
        Returns:
            Conversation data with conversation_id and thread_id
        """
        logger.info(f"Creating conversation: {title} (character: {character_id}, source: {source})")
        
        payload = {
            'character_id': character_id,
            'title': title,
            'source': source
        }
        
        # Create conversation (automatically creates a "Main Thread")
        conversation = self._request('POST', '/conversations', json=payload)
        conversation_id = conversation['id']
        
        # Fetch the auto-created thread
        threads = self._request('GET', f'/conversations/{conversation_id}/threads')
        
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
    
    def send_message(
        self,
        conversation_id: str,
        thread_id: int,
        message: str,
        user_name: str = "Discord User",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to an existing conversation.
        
        Args:
            conversation_id: Conversation identifier
            thread_id: Thread identifier within conversation
            message: Message content
            user_name: Name of the user sending the message
            metadata: Optional metadata to attach to the message
            
        Returns:
            Response including assistant's reply
        """
        logger.info(f"Sending message to conversation {conversation_id} (thread {thread_id})")
        logger.debug(f"Message preview: {message[:100]}...")
        
        payload = {
            'message': message,
            'user_name': user_name
        }
        
        if metadata:
            payload['metadata'] = metadata
        
        endpoint = f'/threads/{thread_id}/messages'
        return self._request('POST', endpoint, json=payload)
    
    def get_conversation_history(
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
        
        response = self._request('GET', endpoint, params=params)
        return response.get('messages', [])
    
    def health_check(self) -> bool:
        """
        Check if Chorus Engine API is accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            logger.debug("Performing health check")
            response = self.session.get(
                f"{self.api_url}/characters",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
