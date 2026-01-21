"""
Response Formatter for Chorus to Discord Format Conversion

Phase 3, Task 3.3: Response Formatting
Handles formatting of Chorus responses for Discord, including message splitting
and Discord-specific formatting preservation.
"""

import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formats Chorus Engine responses for Discord consumption.
    
    Handles:
    - 2000 character limit (Discord constraint)
    - Intelligent message splitting on newlines
    - Code block preservation (don't split mid-block)
    - Markdown formatting preservation
    - Special character escaping
    """
    
    # Discord's message length limit
    MAX_MESSAGE_LENGTH = 2000
    
    # Minimum chunk size for splitting
    MIN_CHUNK_SIZE = 100
    
    def __init__(self):
        """Initialize the response formatter."""
        pass
    
    def format_response(self, content: str) -> List[str]:
        """
        Format Chorus response for Discord, splitting if necessary.
        
        Args:
            content: Chorus response content
            
        Returns:
            List of message chunks (each under Discord limit)
        """
        # Clean up the content first
        content = self._clean_response(content)
        
        # If it fits in one message, return as-is
        if len(content) <= self.MAX_MESSAGE_LENGTH:
            return [content]
        
        # Need to split - use intelligent splitting
        logger.info(f"Splitting long response ({len(content)} chars)")
        return self._split_message(content)
    
    def _clean_response(self, content: str) -> str:
        """
        Clean up Chorus response for Discord.
        
        Args:
            content: Raw Chorus response
            
        Returns:
            Cleaned response
        """
        # Remove excessive whitespace at start/end
        content = content.strip()
        
        # Fix common Chorus output issues
        # (e.g., if Chorus outputs markdown that Discord doesn't like)
        
        return content
    
    def _split_message(self, content: str) -> List[str]:
        """
        Split a message into chunks that fit Discord's limit.
        
        Uses intelligent splitting:
        1. Preserve code blocks (never split inside them)
        2. Split on paragraph breaks first
        3. Split on sentence breaks if needed
        4. Force split on words as last resort
        
        Args:
            content: Content to split
            
        Returns:
            List of message chunks
        """
        # First, identify code blocks and split around them
        chunks = self._split_preserving_code_blocks(content)
        
        # Then, check each chunk and split further if needed
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.MAX_MESSAGE_LENGTH:
                final_chunks.append(chunk)
            else:
                # Need to split this chunk further
                final_chunks.extend(self._split_text_chunk(chunk))
        
        return final_chunks
    
    def _split_preserving_code_blocks(self, content: str) -> List[str]:
        """
        Split content while preserving code blocks.
        
        Args:
            content: Content to split
            
        Returns:
            List of chunks with code blocks intact
        """
        # Pattern: ```language\ncode\n```
        code_block_pattern = r'```[\s\S]*?```'
        
        # Find all code blocks
        code_blocks = []
        for match in re.finditer(code_block_pattern, content):
            code_blocks.append((match.start(), match.end(), match.group()))
        
        # If no code blocks, return as single chunk
        if not code_blocks:
            return [content]
        
        # Split content around code blocks
        chunks = []
        last_end = 0
        
        for start, end, code in code_blocks:
            # Add text before code block
            if start > last_end:
                text_before = content[last_end:start].strip()
                if text_before:
                    chunks.append(text_before)
            
            # Add code block (check if it fits)
            if len(code) <= self.MAX_MESSAGE_LENGTH:
                chunks.append(code)
            else:
                # Code block too large - need to split it
                # This is tricky, but we'll do our best
                logger.warning(f"Code block too large ({len(code)} chars), forcing split")
                chunks.extend(self._split_large_code_block(code))
            
            last_end = end
        
        # Add remaining text after last code block
        if last_end < len(content):
            text_after = content[last_end:].strip()
            if text_after:
                chunks.append(text_after)
        
        return chunks
    
    def _split_large_code_block(self, code_block: str) -> List[str]:
        """
        Split a code block that's too large for Discord.
        
        Args:
            code_block: Code block to split (including ``` markers)
            
        Returns:
            List of chunks
        """
        # Extract language and code
        lines = code_block.split('\n')
        first_line = lines[0]  # ```language
        language = first_line[3:].strip() if len(first_line) > 3 else ''
        
        # Get code lines (without first and last lines which are ```)
        code_lines = lines[1:-1] if len(lines) > 2 else lines[1:]
        
        # Split code into chunks
        chunks = []
        current_chunk = []
        current_length = len(f"```{language}\n") + len("```")  # Account for markers
        
        for line in code_lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > self.MAX_MESSAGE_LENGTH - 100:  # Leave buffer
                # Finalize current chunk
                chunk_code = '\n'.join(current_chunk)
                chunks.append(f"```{language}\n{chunk_code}\n```")
                
                # Start new chunk
                current_chunk = [line]
                current_length = len(f"```{language}\n") + len(line) + 1 + len("```")
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_code = '\n'.join(current_chunk)
            chunks.append(f"```{language}\n{chunk_code}\n```")
        
        return chunks
    
    def _split_text_chunk(self, text: str) -> List[str]:
        """
        Split a text chunk (no code blocks) intelligently.
        
        Priority:
        1. Split on double newlines (paragraphs)
        2. Split on single newlines
        3. Split on sentence endings
        4. Split on spaces (words)
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        # If it fits, return as-is
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return [text]
        
        # Try splitting on paragraphs first
        chunks = self._split_on_pattern(text, '\n\n')
        if chunks and all(len(c) <= self.MAX_MESSAGE_LENGTH for c in chunks):
            return chunks
        
        # Need to split some chunks further - try newlines
        final_chunks = []
        chunks_to_process = chunks if chunks else [text]
        
        for chunk in chunks_to_process:
            if len(chunk) <= self.MAX_MESSAGE_LENGTH:
                final_chunks.append(chunk)
            else:
                # Split on single newlines
                sub_chunks = self._split_on_pattern(chunk, '\n')
                if sub_chunks:
                    for sub_chunk in sub_chunks:
                        if len(sub_chunk) <= self.MAX_MESSAGE_LENGTH:
                            final_chunks.append(sub_chunk)
                        else:
                            # Still too large - split on sentences
                            final_chunks.extend(self._split_on_sentences(sub_chunk))
                else:
                    # Pattern split failed, use sentences
                    final_chunks.extend(self._split_on_sentences(chunk))
        
        return final_chunks
    
    def _split_on_pattern(self, text: str, pattern: str) -> List[str]:
        """
        Split text on a pattern, creating chunks under the limit.
        
        Args:
            text: Text to split
            pattern: Pattern to split on
            
        Returns:
            List of chunks
        """
        parts = text.split(pattern)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Check if adding this part would exceed limit
            test_chunk = current_chunk + pattern + part if current_chunk else part
            
            if len(test_chunk) <= self.MAX_MESSAGE_LENGTH:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Check if part itself is too large
                if len(part) > self.MAX_MESSAGE_LENGTH:
                    # Part is too large, needs further splitting
                    return None  # Signal that this split method failed
                
                current_chunk = part
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def _split_on_sentences(self, text: str) -> List[str]:
        """
        Split text on sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        # Pattern for sentence endings: . ! ? followed by space or end
        sentence_pattern = r'([.!?]+[\s\n]+)'
        
        # Split but keep the delimiters
        parts = re.split(sentence_pattern, text)
        
        # Recombine sentences
        sentences = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
            else:
                sentences.append(parts[i])
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence
            
            if len(test_chunk) <= self.MAX_MESSAGE_LENGTH:
                current_chunk = test_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Check if sentence itself is too large
                if len(sentence) > self.MAX_MESSAGE_LENGTH:
                    # Force split on spaces
                    chunks.extend(self._force_split_on_words(sentence))
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:self.MAX_MESSAGE_LENGTH]]
    
    def _force_split_on_words(self, text: str) -> List[str]:
        """
        Force split text on word boundaries (last resort).
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        words = text.split(' ')
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + ' ' + word if current_chunk else word
            
            if len(test_chunk) <= self.MAX_MESSAGE_LENGTH:
                current_chunk = test_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Check if word itself is too large
                if len(word) > self.MAX_MESSAGE_LENGTH:
                    # Force split the word (character by character)
                    for i in range(0, len(word), self.MAX_MESSAGE_LENGTH):
                        chunks.append(word[i:i + self.MAX_MESSAGE_LENGTH])
                    current_chunk = ""
                else:
                    current_chunk = word
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:self.MAX_MESSAGE_LENGTH]]
    
    def format_typing_indicator(self, is_thinking: bool = True) -> str:
        """
        Generate typing indicator text.
        
        Args:
            is_thinking: Whether to show "thinking" vs "typing"
            
        Returns:
            Typing indicator text
        """
        return "🤔 Thinking..." if is_thinking else "✍️ Typing..."
    
    def escape_special_characters(self, text: str) -> str:
        """
        Escape special Discord markdown characters if needed.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        # Discord markdown characters that might need escaping
        # Only escape if they're being used unintentionally
        # (Most of the time we want to preserve markdown)
        
        # For now, don't escape anything - preserve formatting
        return text
