"""
Audio Storage Service
Handles saving, retrieving, and serving audio files.
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class AudioStorageService:
    """
    Service for managing audio file storage.
    
    Responsibilities:
    - Save audio files with unique filenames
    - Retrieve audio file paths
    - Clean up old audio files
    - Organize audio by date (optional)
    """
    
    def __init__(self):
        """Initialize the audio storage service."""
        # Use simple default path structure
        self.audio_folder = Path("data/audio")
        self.audio_folder.mkdir(parents=True, exist_ok=True)
        
        # Maximum age for audio files (days)
        self.max_age_days = 30
    
    def save_audio(self, audio_data: bytes, message_id: str, extension: str = "wav") -> str:
        """
        Save audio file to disk and return filename.
        
        Args:
            audio_data: Raw audio file bytes
            message_id: ID of the message this audio belongs to
            extension: File extension (wav, mp3, etc.)
            
        Returns:
            Filename (not full path) of saved audio file
        """
        # Generate unique filename based on message_id and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"msg_{message_id}_{timestamp}.{extension}"
        
        file_path = self.audio_folder / filename
        
        # Write audio data
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        
        return filename
    
    def save_audio_from_path(self, source_path: Path, message_id: str) -> str:
        """
        Copy audio file from ComfyUI output to audio storage.
        
        Args:
            source_path: Path to the audio file from ComfyUI
            message_id: ID of the message this audio belongs to
            
        Returns:
            Filename (not full path) of saved audio file
        """
        # Detect extension from source file
        extension = source_path.suffix.lstrip('.')
        
        # Read source file
        with open(source_path, 'rb') as f:
            audio_data = f.read()
        
        return self.save_audio(audio_data, message_id, extension)
    
    def get_audio_path(self, filename: str) -> Path:
        """
        Get absolute path to audio file.
        
        Args:
            filename: The audio filename (without path)
            
        Returns:
            Absolute Path to the audio file
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        file_path = self.audio_folder / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {filename}")
        
        return file_path
    
    def audio_exists(self, filename: str) -> bool:
        """
        Check if audio file exists.
        
        Args:
            filename: The audio filename (without path)
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = self.audio_folder / filename
        return file_path.exists()
    
    def delete_audio(self, filename: str) -> bool:
        """
        Delete audio file.
        
        Args:
            filename: The audio filename (without path)
            
        Returns:
            True if file was deleted, False if file didn't exist
        """
        file_path = self.audio_folder / filename
        
        if file_path.exists():
            file_path.unlink()
            return True
        
        return False
    
    def cleanup_old_audio(self, days: Optional[int] = None) -> int:
        """
        Delete audio files older than N days.
        
        Args:
            days: Number of days (uses config default if None)
            
        Returns:
            Number of files deleted
        """
        if days is None:
            days = self.max_age_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for audio_file in self.audio_folder.glob("*.wav"):
            # Check file modification time
            file_time = datetime.fromtimestamp(audio_file.stat().st_mtime)
            
            if file_time < cutoff_date:
                audio_file.unlink()
                deleted_count += 1
        
        # Also check for other audio formats
        for extension in ["mp3", "flac", "ogg"]:
            for audio_file in self.audio_folder.glob(f"*.{extension}"):
                file_time = datetime.fromtimestamp(audio_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    audio_file.unlink()
                    deleted_count += 1
        
        return deleted_count
    
    def get_audio_size(self, filename: str) -> int:
        """
        Get audio file size in bytes.
        
        Args:
            filename: The audio filename (without path)
            
        Returns:
            File size in bytes
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        file_path = self.get_audio_path(filename)
        return file_path.stat().st_size
    
    def get_total_storage_used(self) -> int:
        """
        Calculate total storage used by audio files.
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        
        for audio_file in self.audio_folder.glob("*"):
            if audio_file.is_file():
                total_size += audio_file.stat().st_size
        
        return total_size
