"""
PNG Metadata Handler
===================

Handles reading and writing tEXt chunks in PNG images for character card metadata.
"""

import base64
import logging
from typing import Optional
from io import BytesIO
from PIL import Image, PngImagePlugin

logger = logging.getLogger(__name__)


class PNGMetadataHandler:
    """Handle PNG tEXt chunk operations for character card metadata."""
    
    @staticmethod
    def read_text_chunk(png_data: bytes, keyword: str) -> Optional[str]:
        """
        Extract tEXt chunk with specific keyword from PNG data.
        
        Args:
            png_data: PNG file data as bytes
            keyword: tEXt chunk keyword to search for (e.g., 'chara', 'chorus_card')
            
        Returns:
            Decoded text data if found, None otherwise
        """
        try:
            image = Image.open(BytesIO(png_data))
            
            # Check if PNG has text metadata
            if not hasattr(image, 'text') or not isinstance(image.text, dict):
                logger.debug(f"No text metadata found in PNG")
                return None
            
            # Look for the specific keyword (case-insensitive search)
            for key, value in image.text.items():
                if key.lower() == keyword.lower():
                    logger.debug(f"Found tEXt chunk with keyword '{key}'")
                    # The value is already base64-decoded by PIL
                    # But character cards store base64-encoded data IN the tEXt chunk
                    # So we need to decode it again
                    try:
                        decoded = base64.b64decode(value).decode('utf-8')
                        return decoded
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 data from chunk '{key}': {e}")
                        # Try returning as-is in case it's not base64 encoded
                        return value
            
            logger.debug(f"tEXt chunk with keyword '{keyword}' not found")
            return None
            
        except Exception as e:
            logger.error(f"Error reading PNG metadata: {e}")
            return None
    
    @staticmethod
    def write_text_chunk(png_data: bytes, keyword: str, data: str) -> bytes:
        """
        Embed tEXt chunk with data into PNG image.
        
        Args:
            png_data: Original PNG file data as bytes
            keyword: tEXt chunk keyword (e.g., 'chorus_card')
            data: Text data to embed (will be base64-encoded)
            
        Returns:
            Modified PNG data with embedded metadata
        """
        try:
            image = Image.open(BytesIO(png_data))
            
            # Create PngInfo object for metadata
            png_info = PngImagePlugin.PngInfo()
            
            # Preserve existing metadata except the keyword we're replacing
            if hasattr(image, 'text') and isinstance(image.text, dict):
                for key, value in image.text.items():
                    if key.lower() != keyword.lower():
                        png_info.add_text(key, value)
            
            # Add our new metadata (base64-encoded)
            encoded_data = base64.b64encode(data.encode('utf-8')).decode('ascii')
            png_info.add_text(keyword, encoded_data)
            
            # Save to BytesIO
            output = BytesIO()
            image.save(output, format='PNG', pnginfo=png_info)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error writing PNG metadata: {e}")
            raise
    
    @staticmethod
    def extract_image(png_path: str) -> bytes:
        """
        Load PNG image data from file.
        
        Args:
            png_path: Path to PNG file
            
        Returns:
            PNG file data as bytes
        """
        try:
            with open(png_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading PNG file '{png_path}': {e}")
            raise
    
    @staticmethod
    def save_image(png_data: bytes, output_path: str) -> None:
        """
        Save PNG data to file.
        
        Args:
            png_data: PNG file data as bytes
            output_path: Path to save PNG file
        """
        try:
            with open(output_path, 'wb') as f:
                f.write(png_data)
        except Exception as e:
            logger.error(f"Error saving PNG file to '{output_path}': {e}")
            raise
