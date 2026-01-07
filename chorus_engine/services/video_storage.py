"""
Video storage service for managing generated video files.

Handles saving videos to disk, extracting thumbnails, and managing file paths.
Workflow-agnostic - accepts any video format ComfyUI produces.
"""

import logging
import asyncio
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io
import subprocess

logger = logging.getLogger(__name__)


class VideoStorageError(Exception):
    """Base exception for video storage errors."""
    pass


class VideoStorageService:
    """
    Service for storing and managing generated videos.
    
    Handles:
    - Saving videos to disk (any format)
    - Extracting first frame as thumbnail
    - Organizing files by conversation
    - Generating file paths
    - File cleanup on deletion
    """
    
    def __init__(
        self,
        base_path: Path = Path("data/videos"),
        thumbnail_size: int = 512,
        thumbnail_format: str = "PNG"
    ):
        """
        Initialize video storage service.
        
        Args:
            base_path: Root directory for video storage
            thumbnail_size: Maximum dimension for thumbnails (pixels)
            thumbnail_format: Thumbnail image format (PNG, JPEG, etc.)
        """
        self.base_path = Path(base_path)
        self.thumbnail_size = thumbnail_size
        self.thumbnail_format = thumbnail_format.upper()
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Video storage initialized: {self.base_path} "
            f"(thumbnail: {thumbnail_size}px, format: {thumbnail_format})"
        )
    
    async def save_video(
        self,
        video_path: Path,
        conversation_id: str,
        video_id: int,
        create_thumbnail: bool = False  # Videos display inline, thumbnails not needed by default
    ) -> Tuple[Path, Optional[Path]]:
        """
        Save a video file and optionally create a thumbnail.
        
        Args:
            video_path: Path to source video file from ComfyUI
            conversation_id: Conversation ID for organization
            video_id: Unique video ID
            create_thumbnail: Whether to extract first frame as thumbnail
        
        Returns:
            Tuple of (video_path, thumbnail_path)
        
        Raises:
            VideoStorageError: Failed to save video
        """
        try:
            # Create conversation directory
            conv_dir = self.base_path / conversation_id
            conv_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file extension from source
            extension = video_path.suffix.lower()
            
            # Generate file paths with timestamp to avoid conflicts
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            dest_video_path = conv_dir / f"{video_id}_{timestamp}{extension}"
            thumb_path = None
            
            # Copy video file
            await asyncio.to_thread(shutil.copy2, video_path, dest_video_path)
            logger.debug(f"Saved video: {dest_video_path}")
            
            # Create thumbnail if requested
            if create_thumbnail:
                thumb_path = conv_dir / f"{video_id}_thumb.{self.thumbnail_format.lower()}"
                success = await self._extract_first_frame(dest_video_path, thumb_path)
                if not success:
                    logger.warning(f"Failed to extract thumbnail for video {video_id}")
                    thumb_path = None
            
            # Return paths as-is (they already include data/videos prefix)
            return dest_video_path, thumb_path
        
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise VideoStorageError(f"Failed to save video: {e}")
    
    async def _extract_first_frame(
        self,
        video_path: Path,
        output_path: Path
    ) -> bool:
        """
        Extract first frame from video as thumbnail.
        
        Uses ffmpeg if available, falls back to Pillow for webp.
        
        Args:
            video_path: Path to video file
            output_path: Path for thumbnail output
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try ffmpeg first (most reliable)
            if await self._has_ffmpeg():
                return await self._extract_with_ffmpeg(video_path, output_path)
            
            # Fallback: Try Pillow for webp/gif
            return await self._extract_with_pillow(video_path, output_path)
        
        except Exception as e:
            logger.error(f"Failed to extract frame: {e}")
            return False
    
    async def _has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    async def _extract_with_ffmpeg(
        self,
        video_path: Path,
        output_path: Path
    ) -> bool:
        """
        Extract frame using ffmpeg.
        
        Args:
            video_path: Source video
            output_path: Output image
        
        Returns:
            True if successful
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vframes', '1',  # Extract 1 frame
                '-vf', f'scale={self.thumbnail_size}:{self.thumbnail_size}:force_original_aspect_ratio=decrease',
                '-y',  # Overwrite
                str(output_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.debug(f"Extracted thumbnail with ffmpeg: {output_path}")
                return True
            else:
                logger.warning(f"ffmpeg extraction failed: {result.stderr.decode()}")
                return False
        
        except Exception as e:
            logger.error(f"ffmpeg extraction error: {e}")
            return False
    
    async def _extract_with_pillow(
        self,
        video_path: Path,
        output_path: Path
    ) -> bool:
        """
        Extract frame using Pillow (for webp/gif).
        
        Args:
            video_path: Source video
            output_path: Output image
        
        Returns:
            True if successful
        """
        try:
            # Open video file (works for webp, gif)
            image = await asyncio.to_thread(Image.open, video_path)
            
            # Seek to first frame
            if hasattr(image, 'seek'):
                image.seek(0)
            
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            
            # Resize
            image.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)
            
            # Save
            await asyncio.to_thread(image.save, output_path, format=self.thumbnail_format)
            
            logger.debug(f"Extracted thumbnail with Pillow: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Pillow extraction error: {e}")
            return False
    
    async def delete_video(
        self,
        video_path: Path,
        thumbnail_path: Optional[Path] = None
    ) -> bool:
        """
        Delete video and thumbnail files.
        
        Args:
            video_path: Path to video file
            thumbnail_path: Path to thumbnail (if exists)
        
        Returns:
            True if successful
        """
        try:
            success = True
            
            # Delete video
            if video_path.exists():
                await asyncio.to_thread(video_path.unlink)
                logger.debug(f"Deleted video file: {video_path}")
            
            # Delete thumbnail
            if thumbnail_path and thumbnail_path.exists():
                await asyncio.to_thread(thumbnail_path.unlink)
                logger.debug(f"Deleted thumbnail: {thumbnail_path}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to delete video files: {e}")
            return False
    
    def get_absolute_path(self, relative_path: Path) -> Path:
        """
        Convert relative path to absolute.
        
        Args:
            relative_path: Relative path from project root
        
        Returns:
            Absolute path
        """
        # relative_path is like: data/videos/conv_id/1_video.webm
        # We need absolute path for serving
        return Path.cwd() / relative_path
