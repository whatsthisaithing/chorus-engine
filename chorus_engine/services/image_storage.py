"""
Image storage service for managing generated image files.

Phase 5: Handles saving images to disk, creating thumbnails, and managing file paths.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageStorageError(Exception):
    """Base exception for image storage errors."""
    pass


class ImageStorageService:
    """
    Service for storing and managing generated images.
    
    Handles:
    - Saving full-size images to disk
    - Creating thumbnails
    - Organizing files by conversation
    - Generating file paths
    - File cleanup on deletion
    """
    
    def __init__(
        self,
        base_path: Path = Path("data/images"),
        thumbnail_size: int = 512,
        image_format: str = "PNG"
    ):
        """
        Initialize image storage service.
        
        Args:
            base_path: Root directory for image storage
            thumbnail_size: Maximum dimension for thumbnails (pixels)
            image_format: Image format (PNG, JPEG, etc.)
        """
        self.base_path = Path(base_path)
        self.thumbnail_size = thumbnail_size
        self.image_format = image_format.upper()
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Image storage initialized: {self.base_path} "
            f"(thumbnail: {thumbnail_size}px, format: {image_format})"
        )
    
    async def save_image(
        self,
        image_data: bytes,
        conversation_id: str,
        image_id: int,
        create_thumbnail: bool = True
    ) -> Tuple[Path, Optional[Path]]:
        """
        Save an image to disk and optionally create a thumbnail.
        
        Args:
            image_data: Image bytes
            conversation_id: Conversation ID for organization
            image_id: Unique image ID
            create_thumbnail: Whether to create a thumbnail
        
        Returns:
            Tuple of (full_image_path, thumbnail_path)
        
        Raises:
            ImageStorageError: Failed to save image
        """
        try:
            # Create conversation directory
            conv_dir = self.base_path / conversation_id
            conv_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate file paths
            full_path = conv_dir / f"{image_id}_full.{self.image_format.lower()}"
            thumb_path = conv_dir / f"{image_id}_thumb.{self.image_format.lower()}" if create_thumbnail else None
            
            # Open image with Pillow
            image = Image.open(io.BytesIO(image_data))
            
            # Save full image
            await asyncio.to_thread(image.save, full_path, format=self.image_format)
            logger.debug(f"Saved full image: {full_path}")
            
            # Create thumbnail if requested
            if create_thumbnail:
                thumbnail = await self._create_thumbnail(image)
                await asyncio.to_thread(thumbnail.save, thumb_path, format=self.image_format)
                logger.debug(f"Saved thumbnail: {thumb_path}")
            
            logger.info(f"Image saved successfully: {image_id} (conversation: {conversation_id})")
            
            return (full_path, thumb_path)
            
        except Exception as e:
            logger.error(f"Failed to save image {image_id}: {e}")
            raise ImageStorageError(f"Failed to save image: {str(e)}")
    
    async def _create_thumbnail(self, image: Image.Image) -> Image.Image:
        """
        Create a thumbnail from an image.
        
        Args:
            image: PIL Image object
        
        Returns:
            Thumbnail PIL Image object
        """
        # Create a copy to avoid modifying original
        thumbnail = image.copy()
        
        # Calculate thumbnail size (maintain aspect ratio)
        thumbnail.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)
        
        return thumbnail
    
    async def delete_image(self, full_path: Path, thumbnail_path: Optional[Path] = None) -> None:
        """
        Delete an image and its thumbnail from disk.
        
        Args:
            full_path: Path to full-size image
            thumbnail_path: Optional path to thumbnail
        """
        try:
            # Delete full image
            if full_path.exists():
                await asyncio.to_thread(full_path.unlink)
                logger.debug(f"Deleted image: {full_path}")
            
            # Delete thumbnail
            if thumbnail_path and thumbnail_path.exists():
                await asyncio.to_thread(thumbnail_path.unlink)
                logger.debug(f"Deleted thumbnail: {thumbnail_path}")
            
            # Clean up empty conversation directories
            conv_dir = full_path.parent
            if conv_dir.exists() and not any(conv_dir.iterdir()):
                await asyncio.to_thread(conv_dir.rmdir)
                logger.debug(f"Removed empty directory: {conv_dir}")
            
            logger.info(f"Image deleted successfully: {full_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to delete image {full_path}: {e}")
            raise ImageStorageError(f"Failed to delete image: {str(e)}")
    
    async def delete_conversation_images(self, conversation_id: str) -> int:
        """
        Delete all images for a conversation.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Number of files deleted
        """
        conv_dir = self.base_path / conversation_id
        
        if not conv_dir.exists():
            logger.debug(f"No images found for conversation: {conversation_id}")
            return 0
        
        try:
            # Count and delete all files
            files = list(conv_dir.glob("*"))
            count = len(files)
            
            for file_path in files:
                if file_path.is_file():
                    await asyncio.to_thread(file_path.unlink)
            
            # Remove directory
            await asyncio.to_thread(conv_dir.rmdir)
            
            logger.info(f"Deleted {count} files for conversation: {conversation_id}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete conversation images: {e}")
            raise ImageStorageError(f"Failed to delete conversation images: {str(e)}")
    
    def get_image_path(self, conversation_id: str, image_id: int, thumbnail: bool = False) -> Path:
        """
        Get the path to an image file.
        
        Args:
            conversation_id: Conversation ID
            image_id: Image ID
            thumbnail: Whether to get thumbnail path
        
        Returns:
            Path to image file
        """
        suffix = "thumb" if thumbnail else "full"
        filename = f"{image_id}_{suffix}.{self.image_format.lower()}"
        return self.base_path / conversation_id / filename
    
    def image_exists(self, file_path: Path) -> bool:
        """
        Check if an image file exists.
        
        Args:
            file_path: Path to image file
        
        Returns:
            True if file exists
        """
        return file_path.exists()
    
    async def get_image_info(self, file_path: Path) -> dict:
        """
        Get metadata about an image file.
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with width, height, size_bytes
        
        Raises:
            ImageStorageError: Failed to read image
        """
        if not file_path.exists():
            raise ImageStorageError(f"Image not found: {file_path}")
        
        try:
            # Get file size
            size_bytes = await asyncio.to_thread(file_path.stat)
            size_bytes = size_bytes.st_size
            
            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size
            
            return {
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
                "format": self.image_format
            }
            
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            raise ImageStorageError(f"Failed to get image info: {str(e)}")
