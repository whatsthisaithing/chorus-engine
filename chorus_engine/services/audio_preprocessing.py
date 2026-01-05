"""
Audio Preprocessing Service
Converts markdown text to plain English suitable for text-to-speech.
"""

import re
from typing import Optional


class AudioPreprocessingService:
    """
    Service for preparing text for TTS generation.
    
    Converts markdown-formatted text to plain English by:
    - Removing code blocks
    - Simplifying inline code
    - Converting links to link text only
    - Removing bold/italic markers
    - Simplifying headers
    - Basic table cleanup
    
    Known limitations (documented for future enhancement):
    - Tables become unstructured text (not natural speech)
    - Code blocks are entirely removed
    - Lists lose structure
    - Math notation not handled
    """
    
    def __init__(self):
        """Initialize the audio preprocessing service."""
        self.max_recommended_length = 2000  # Characters
    
    def preprocess_for_tts(self, markdown_text: str) -> str:
        """
        Convert markdown response to TTS-ready plain English.
        
        Args:
            markdown_text: The markdown-formatted text from LLM
            
        Returns:
            Plain English text suitable for TTS
        """
        if not markdown_text:
            return ""
        
        text = markdown_text
        
        # 1. Remove code blocks entirely (triple backticks)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 2. Remove inline code formatting (keep content)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 3. Remove image syntax entirely
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        
        # 4. Convert links to just the link text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 5. Remove bold/italic markers (keep content)
        # Handle ** and __ for bold
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # Handle * and _ for italic
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # 6. Convert headers to plain text (remove # symbols)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # 7. Remove table syntax
        # This is the weak point - tables become unstructured
        # TODO FUTURE: Parse tables and convert to natural speech
        text = re.sub(r'\|', '', text)  # Remove pipe characters
        text = re.sub(r'^[-\s:]+$', '', text, flags=re.MULTILINE)  # Remove separator rows
        
        # 8. Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # 9. Convert bullet points to plain text
        # TODO FUTURE: Convert to "First... Second... Third..."
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 10. Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' {2,}', ' ', text)  # Remove multiple spaces
        text = text.strip()
        
        return text
    
    def should_skip_tts(self, text: str) -> bool:
        """
        Determine if text is unsuitable for TTS.
        
        Returns True if:
        - Text is too long
        - Text is primarily code blocks
        - Text is empty after preprocessing
        
        Args:
            text: The original markdown text
            
        Returns:
            True if TTS should be skipped, False otherwise
        """
        if not text or not text.strip():
            return True
        
        # Check if text is too long
        if len(text) > self.max_recommended_length:
            return True
        
        # Check if text is primarily code blocks
        # Count code block content vs total content
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        code_length = sum(len(block) for block in code_blocks)
        
        # If more than 60% is code, skip TTS
        if code_length > len(text) * 0.6:
            return True
        
        # Check if preprocessing would leave nothing
        preprocessed = self.preprocess_for_tts(text)
        if len(preprocessed) < 10:  # Too short to be meaningful
            return True
        
        return False
    
    def estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration in seconds.
        
        Uses rough approximation:
        - Average speaking rate: ~150 words per minute
        - Average word length: ~5 characters
        
        Args:
            text: The preprocessed plain text
            
        Returns:
            Estimated duration in seconds
        """
        if not text:
            return 0.0
        
        # Rough word count (split on whitespace)
        words = len(text.split())
        
        # Average speaking rate: 150 words per minute
        minutes = words / 150.0
        seconds = minutes * 60.0
        
        # Add a small buffer (TTS might be slightly slower)
        return seconds * 1.1
    
    def validate_text_for_tts(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate text is suitable for TTS and provide reason if not.
        
        Args:
            text: The markdown text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            If is_valid is True, error_message is None
            If is_valid is False, error_message explains why
        """
        if not text or not text.strip():
            return False, "Text is empty"
        
        if len(text) > self.max_recommended_length:
            return False, f"Text too long ({len(text)} chars, max {self.max_recommended_length})"
        
        # Check if primarily code
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        code_length = sum(len(block) for block in code_blocks)
        
        if code_length > len(text) * 0.6:
            return False, "Text is primarily code blocks (not suitable for TTS)"
        
        # Check if preprocessing leaves meaningful content
        preprocessed = self.preprocess_for_tts(text)
        if len(preprocessed) < 10:
            return False, "No meaningful text after preprocessing"
        
        return True, None
