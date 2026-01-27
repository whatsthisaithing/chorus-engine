"""
Image Handler for Discord Bridge Vision Integration (Phase 3)

Handles downloading images from Discord, uploading to Chorus Engine,
and managing shared attachment cache to prevent duplicate processing
across multiple bots.
"""
import logging
import aiohttp
import tempfile
from pathlib import Path
from typing import List, Optional
import discord
from PIL import Image

from .database import Database
from .chorus_client import ChorusClient

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles Discord image download and Chorus upload with caching."""
    
    # Supported image MIME types
    SUPPORTED_TYPES = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/gif"
    ]
    
    def __init__(
        self,
        chorus_client: ChorusClient,
        database: Database,
        max_file_size_mb: int = 50,  # Increased from 10 - we resize anyway
        max_images_per_message: int = 5
    ):
        """Initialize image handler.
        
        Args:
            chorus_client: Chorus client for API calls
            database: Database instance for caching
            max_file_size_mb: Maximum file size in MB before download (default: 50, safety net)
            max_images_per_message: Maximum images per message (default: 5)
        """
        self.chorus_client = chorus_client
        self.db = database
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.max_images = max_images_per_message
    
    async def process_message_images(
        self,
        message: discord.Message,
        character_id: str
    ) -> List[str]:
        """Process all images in a Discord message.
        
        Returns list of Chorus attachment IDs. Uses shared cache to prevent
        duplicate uploads across bots.
        
        Args:
            message: Discord message object
            character_id: Character ID of the bot processing this
            
        Returns:
            List of Chorus attachment IDs (empty if no valid images)
        """
        if not message.attachments:
            return []
        
        chorus_attachment_ids = []
        processed_count = 0
        
        for attachment in message.attachments:
            # Enforce limit
            if processed_count >= self.max_images:
                logger.warning(
                    f"Message {message.id} has more than {self.max_images} images, "
                    f"skipping remaining attachments"
                )
                break
            
            # Check if it's an image
            if not self._is_image(attachment):
                logger.debug(f"Skipping non-image attachment: {attachment.filename}")
                continue
            
            # Check file size
            if attachment.size > self.max_file_size:
                logger.warning(
                    f"Image {attachment.filename} too large ({attachment.size} bytes), "
                    f"max is {self.max_file_size}"
                )
                continue
            
            # Check cache first (shared across all bots)
            cached = self.db.get_cached_attachment(str(attachment.id))
            if cached:
                logger.info(
                    f"Using cached attachment {attachment.id} -> {cached['chorus_attachment_id']} "
                    f"(uploaded by {cached['uploaded_by_bot']})"
                )
                chorus_attachment_ids.append(cached['chorus_attachment_id'])
                processed_count += 1
                continue
            
            # Not cached - download, upload, and cache
            try:
                chorus_id = await self._download_and_upload(
                    message=message,
                    attachment=attachment,
                    character_id=character_id
                )
                
                if chorus_id:
                    chorus_attachment_ids.append(chorus_id)
                    processed_count += 1
                    logger.info(
                        f"Successfully processed image {attachment.filename}: "
                        f"Discord {attachment.id} -> Chorus {chorus_id}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to process attachment {attachment.filename}: {e}",
                    exc_info=True
                )
                # Continue with other images even if one fails
        
        logger.info(
            f"Processed {processed_count} image(s) from message {message.id}, "
            f"returning {len(chorus_attachment_ids)} Chorus attachment ID(s)"
        )
        
        return chorus_attachment_ids
    
    async def _download_and_upload(
        self,
        message: discord.Message,
        attachment: discord.Attachment,
        character_id: str
    ) -> Optional[str]:
        """Download from Discord, upload to Chorus, cache mapping.
        
        Args:
            message: Discord message
            attachment: Discord attachment
            character_id: Character ID uploading this
            
        Returns:
            Chorus attachment ID, or None if failed
        """
        temp_path = None
        resized_path = None
        
        try:
            # Create temporary file
            suffix = Path(attachment.filename).suffix or '.jpg'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_path = Path(tmp.name)
            
            # Download from Discord
            logger.debug(f"Downloading {attachment.filename} to {temp_path}")
            await attachment.save(temp_path)
            
            # Verify download
            if not temp_path.exists():
                logger.error(f"Download failed: {temp_path} does not exist")
                return None
            
            file_size = temp_path.stat().st_size
            if file_size == 0:
                logger.error(f"Downloaded file is empty: {temp_path}")
                return None
            
            logger.debug(f"Downloaded {file_size} bytes")
            
            # Resize image to max 2048px on longest side
            resized_path = self._resize_image(temp_path, max_dimension=2048)
            if not resized_path:
                logger.error(f"Failed to resize image: {attachment.filename}")
                return None
            
            # Upload to Chorus (use resized version)
            logger.debug(f"Uploading to Chorus API: {attachment.filename}")
            chorus_id = await self.chorus_client.upload_image(
                file_path=resized_path,
                filename=attachment.filename
            )
            
            if not chorus_id:
                logger.error("Chorus upload returned no attachment ID")
                return None
            
            logger.info(f"Uploaded to Chorus: {chorus_id}")
            
            # Cache the mapping (using original Discord file size)
            self.db.cache_attachment(
                discord_attachment_id=str(attachment.id),
                discord_message_id=str(message.id),
                chorus_attachment_id=chorus_id,
                filename=attachment.filename,
                file_size=attachment.size,
                content_type=attachment.content_type or 'image/jpeg',
                uploaded_by_bot=character_id
            )
            
            logger.debug(f"Cached mapping: Discord {attachment.id} -> Chorus {chorus_id}")
            
            return chorus_id
            
        except Exception as e:
            logger.error(f"Failed to download/upload {attachment.filename}: {e}", exc_info=True)
            return None
            
        finally:
            # Cleanup temporary files
            for path in [temp_path, resized_path]:
                if path and path.exists():
                    try:
                        path.unlink()
                        logger.debug(f"Cleaned up temp file: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {path}: {e}")
    
    def _resize_image(self, image_path: Path, max_dimension: int = 2048) -> Optional[Path]:
        """Resize image to fit within max_dimension on longest side.
        
        Maintains aspect ratio. If image is already smaller, returns original.
        Creates a new temp file with resized image.
        
        Args:
            image_path: Path to original image
            max_dimension: Maximum size for longest side (default 2048px)
            
        Returns:
            Path to resized image (new temp file), or None if failed
        """
        try:
            # Open image
            img = Image.open(image_path)
            original_width, original_height = img.size
            
            # Check if resize needed
            if original_width <= max_dimension and original_height <= max_dimension:
                logger.debug(f"Image already within limits ({original_width}x{original_height}), no resize needed")
                return image_path  # Return original
            
            # Calculate new dimensions maintaining aspect ratio
            if original_width > original_height:
                # Landscape or square
                new_width = max_dimension
                new_height = int((max_dimension / original_width) * original_height)
            else:
                # Portrait
                new_height = max_dimension
                new_width = int((max_dimension / original_height) * original_width)
            
            logger.info(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")
            
            # Resize with high quality
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to new temp file
            suffix = image_path.suffix or '.jpg'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                resized_path = Path(tmp.name)
            
            # Determine format and save
            save_format = 'JPEG'
            save_kwargs = {'quality': 90, 'optimize': True}
            
            if suffix.lower() in ['.png']:
                save_format = 'PNG'
                save_kwargs = {'optimize': True}
            elif suffix.lower() in ['.webp']:
                save_format = 'WEBP'
                save_kwargs = {'quality': 90}
            elif suffix.lower() in ['.gif']:
                save_format = 'GIF'
                save_kwargs = {}
            
            # Convert RGBA to RGB for JPEG
            if save_format == 'JPEG' and resized.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', resized.size, (255, 255, 255))
                if resized.mode == 'P':
                    resized = resized.convert('RGBA')
                background.paste(resized, mask=resized.split()[-1] if resized.mode in ('RGBA', 'LA') else None)
                resized = background
            
            resized.save(resized_path, format=save_format, **save_kwargs)
            resized.close()
            img.close()
            
            new_size = resized_path.stat().st_size
            logger.debug(f"Resized image saved: {new_size} bytes")
            
            return resized_path
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}", exc_info=True)
            return None

    
    def _is_image(self, attachment: discord.Attachment) -> bool:
        """Check if attachment is a supported image type.
        
        Args:
            attachment: Discord attachment
            
        Returns:
            True if supported image type
        """
        # Check content type if available
        if attachment.content_type:
            if attachment.content_type.lower() in self.SUPPORTED_TYPES:
                return True
        
        # Fallback: check file extension
        filename_lower = attachment.filename.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        
        return any(filename_lower.endswith(ext) for ext in image_extensions)
