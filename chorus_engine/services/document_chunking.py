"""Document chunking service for intelligent text splitting.

Supports multiple chunking strategies:
- Fixed-size chunking with overlap
- Semantic chunking (sentence-aware)
- Paragraph-based chunking
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkMethod(Enum):
    """Chunking strategy."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class DocumentChunk:
    """Container for a document chunk."""
    index: int
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        """Return character count."""
        return len(self.content)


class ChunkingService:
    """Service for intelligent document chunking."""
    
    DEFAULT_CHUNK_SIZE = 1000  # characters
    DEFAULT_OVERLAP = 200  # characters
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP
    ):
        """
        Initialize chunking service.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        logger.info(f"ChunkingService initialized (size: {chunk_size}, overlap: {overlap})")
    
    def chunk_document(
        self,
        content: str,
        method: ChunkMethod = ChunkMethod.SEMANTIC,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk document using specified method.
        
        Args:
            content: Document text content
            method: Chunking method to use
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for chunking")
            return []
        
        metadata = metadata or {}
        
        logger.info(f"Chunking document: {len(content)} chars using {method.value}")
        
        if method == ChunkMethod.FIXED_SIZE:
            chunks = self._chunk_fixed_size(content, metadata)
        elif method == ChunkMethod.SEMANTIC:
            chunks = self._chunk_semantic(content, metadata)
        elif method == ChunkMethod.PARAGRAPH:
            chunks = self._chunk_by_paragraph(content, metadata)
        elif method == ChunkMethod.SENTENCE:
            chunks = self._chunk_by_sentence(content, metadata)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _chunk_fixed_size(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            content: Text to chunk
            base_metadata: Metadata to attach to all chunks
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        index = 0
        
        while start < len(content):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at word boundary
            if end < len(content):
                # Look for space within last 10% of chunk
                search_start = end - int(self.chunk_size * 0.1)
                space_pos = content.rfind(' ', search_start, end)
                
                if space_pos != -1:
                    end = space_pos + 1  # Include the space
            
            # Extract chunk
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    **base_metadata,
                    'chunk_method': ChunkMethod.FIXED_SIZE.value,
                    'start_char': start,
                    'end_char': end,
                }
                
                chunks.append(DocumentChunk(
                    index=index,
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata=chunk_metadata
                ))
                index += 1
            
            # Move start position (with overlap)
            start = end - self.overlap
            
            # Avoid tiny overlapping chunks at the end
            if start >= len(content) - self.overlap:
                break
        
        return chunks
    
    def _chunk_semantic(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk by semantic units (sentences), respecting size limits.
        
        This method tries to keep sentences together while staying within
        the target chunk size.
        
        Args:
            content: Text to chunk
            base_metadata: Metadata to attach to all chunks
            
        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        start_char = 0
        index = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                chunk_metadata = {
                    **base_metadata,
                    'chunk_method': ChunkMethod.SEMANTIC.value,
                    'sentence_count': len(current_chunk),
                    'start_char': start_char,
                    'end_char': end_char,
                }
                
                chunks.append(DocumentChunk(
                    index=index,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata
                ))
                index += 1
                
                # Start new chunk with overlap (last sentence from previous chunk)
                if self.overlap > 0 and current_chunk:
                    # Calculate how many sentences fit in overlap
                    overlap_sentences = []
                    overlap_size = 0
                    
                    for prev_sentence in reversed(current_chunk):
                        if overlap_size + len(prev_sentence) <= self.overlap:
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_size += len(prev_sentence)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                    start_char = end_char - overlap_size
                else:
                    current_chunk = []
                    current_size = 0
                    start_char = end_char
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk if any content remains
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            end_char = start_char + len(chunk_text)
            
            chunk_metadata = {
                **base_metadata,
                'chunk_method': ChunkMethod.SEMANTIC.value,
                'sentence_count': len(current_chunk),
                'start_char': start_char,
                'end_char': end_char,
            }
            
            chunks.append(DocumentChunk(
                index=index,
                content=chunk_text,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _chunk_by_paragraph(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk by paragraphs, splitting large paragraphs if needed.
        
        Args:
            content: Text to chunk
            base_metadata: Metadata to attach to all chunks
            
        Returns:
            List of chunks
        """
        # Split into paragraphs (double newline or more)
        paragraphs = re.split(r'\n\s*\n+', content)
        
        chunks = []
        index = 0
        char_position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is small enough, make it a chunk
            if len(para) <= self.chunk_size:
                chunk_metadata = {
                    **base_metadata,
                    'chunk_method': ChunkMethod.PARAGRAPH.value,
                    'is_complete_paragraph': True,
                    'start_char': char_position,
                    'end_char': char_position + len(para),
                }
                
                chunks.append(DocumentChunk(
                    index=index,
                    content=para,
                    start_char=char_position,
                    end_char=char_position + len(para),
                    metadata=chunk_metadata
                ))
                index += 1
            else:
                # Large paragraph - split using semantic chunking
                para_chunks = self._chunk_semantic(para, base_metadata)
                
                for para_chunk in para_chunks:
                    # Adjust metadata
                    para_chunk.metadata['is_complete_paragraph'] = False
                    para_chunk.metadata['chunk_method'] = ChunkMethod.PARAGRAPH.value
                    para_chunk.index = index
                    para_chunk.start_char += char_position
                    para_chunk.end_char += char_position
                    
                    chunks.append(para_chunk)
                    index += 1
            
            char_position += len(para) + 2  # +2 for paragraph break
        
        return chunks
    
    def _chunk_by_sentence(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk by individual sentences (for very precise retrieval).
        
        Args:
            content: Text to chunk
            base_metadata: Metadata to attach to all chunks
            
        Returns:
            List of chunks
        """
        sentences = self._split_into_sentences(content)
        
        chunks = []
        char_position = 0
        
        for index, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            chunk_metadata = {
                **base_metadata,
                'chunk_method': ChunkMethod.SENTENCE.value,
                'is_single_sentence': True,
                'start_char': char_position,
                'end_char': char_position + len(sentence),
            }
            
            chunks.append(DocumentChunk(
                index=index,
                content=sentence.strip(),
                start_char=char_position,
                end_char=char_position + len(sentence),
                metadata=chunk_metadata
            ))
            
            char_position += len(sentence) + 1  # +1 for space
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristics.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLP libraries)
        # Split on . ! ? followed by space and capital letter
        sentence_endings = re.compile(r'([.!?]+[\s]+)(?=[A-Z])')
        
        sentences = []
        last_end = 0
        
        for match in sentence_endings.finditer(text):
            sentence = text[last_end:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            last_end = match.end()
        
        # Add final sentence
        if last_end < len(text):
            final_sentence = text[last_end:].strip()
            if final_sentence:
                sentences.append(final_sentence)
        
        # If no sentences found (no proper punctuation), split by newlines
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        # If still no sentences, return whole text
        if not sentences:
            sentences = [text]
        
        return sentences
    
    def estimate_chunks(self, content: str, method: ChunkMethod = ChunkMethod.SEMANTIC) -> int:
        """
        Estimate number of chunks without actually creating them.
        
        Args:
            content: Text to estimate
            method: Chunking method
            
        Returns:
            Estimated number of chunks
        """
        if method == ChunkMethod.FIXED_SIZE:
            # Simple calculation for fixed-size
            effective_chunk_size = self.chunk_size - self.overlap
            return max(1, (len(content) + effective_chunk_size - 1) // effective_chunk_size)
        
        elif method == ChunkMethod.SEMANTIC:
            # Estimate based on average sentence length
            sentences = self._split_into_sentences(content)
            if not sentences:
                return 1
            
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            sentences_per_chunk = max(1, self.chunk_size // avg_sentence_length)
            return max(1, (len(sentences) + sentences_per_chunk - 1) // sentences_per_chunk)
        
        elif method == ChunkMethod.PARAGRAPH:
            paragraphs = re.split(r'\n\s*\n+', content)
            return len([p for p in paragraphs if p.strip()])
        
        elif method == ChunkMethod.SENTENCE:
            sentences = self._split_into_sentences(content)
            return len([s for s in sentences if s.strip()])
        
        return 1
