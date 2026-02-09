"""
Conversation Export Service

Exports conversations to various formats for archival and sharing.

Phase 8 - Day 10: Export functionality for conversation preservation
"""

import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from chorus_engine.models.conversation import Conversation, Message, MessageRole, ConversationSummary
from chorus_engine.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class ConversationExportService:
    """
    Export conversations to various formats.
    
    Supports:
    - Markdown (.md) - Simple, readable, preserves formatting
    - Plain text (.txt) - Maximum compatibility
    - Future: PDF, HTML, JSON
    """
    
    def __init__(self):
        """Initialize export service."""
        self.config_loader = ConfigLoader()
    
    def export_to_markdown(
        self,
        conversation: Conversation,
        summaries: List[ConversationSummary] = None,
        include_metadata: bool = True,
        include_summary: bool = True,
        include_memories: bool = False
    ) -> str:
        """
        Export conversation to markdown format.
        
        Args:
            conversation: Conversation to export
            summaries: List of conversation summaries (if analyzed)
            include_metadata: Include title, dates, character info
            include_summary: Include conversation summary (if analyzed)
            include_memories: Include extracted memories list
        
        Returns:
            Markdown-formatted string
        """
        lines = []
        
        # Header
        if include_metadata:
            lines.append(f"# {conversation.title}")
            lines.append("")
            
            # Character info
            try:
                character = self.config_loader.load_character(conversation.character_id)
                lines.append(f"**Character:** {character.name}")
            except:
                lines.append(f"**Character ID:** {conversation.character_id}")
            
            lines.append(f"**Started:** {self._format_datetime(conversation.created_at)}")
            lines.append(f"**Last Updated:** {self._format_datetime(conversation.updated_at)}")
            
            # Message count
            message_count = sum(
                1 for thread in conversation.threads for m in thread.messages if m.deleted_at is None
            )
            lines.append(f"**Messages:** {message_count}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Conversation Summary (if available and requested)
        if include_summary and summaries:
            latest_summary = summaries[0]  # Already sorted desc in API
            
            lines.append("## Conversation Summary")
            lines.append("")
            lines.append(f"> {latest_summary.summary}")
            lines.append("")
            
            if latest_summary.key_topics:
                topics = latest_summary.key_topics if isinstance(latest_summary.key_topics, str) else ", ".join(latest_summary.key_topics)
                lines.append(f"**Key Topics:** {topics}")
            if latest_summary.emotional_arc:
                lines.append(f"**Emotional Arc:** {latest_summary.emotional_arc}")
            if latest_summary.open_questions:
                questions = latest_summary.open_questions if isinstance(latest_summary.open_questions, str) else ", ".join(latest_summary.open_questions)
                lines.append(f"**Open Questions:** {questions}")
            if latest_summary.message_count:
                lines.append(f"**Messages Covered:** {latest_summary.message_count}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Extracted Memories (if requested)
        if include_memories and conversation.memories:
            lines.append("## Extracted Memories")
            lines.append("")
            
            # Group by type
            memories_by_type = {}
            for memory in conversation.memories:
                mem_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                if mem_type not in memories_by_type:
                    memories_by_type[mem_type] = []
                memories_by_type[mem_type].append(memory)
            
            for mem_type, mems in sorted(memories_by_type.items()):
                lines.append(f"### {mem_type.title()}")
                lines.append("")
                for mem in mems:
                    confidence = f" (confidence: {mem.confidence:.2f})" if mem.confidence else ""
                    lines.append(f"- {mem.content}{confidence}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Conversation Messages
        lines.append("## Conversation")
        lines.append("")
        
        # Get all messages sorted by time
        all_messages = []
        for thread in conversation.threads:
            all_messages.extend([m for m in thread.messages if m.deleted_at is None])
        all_messages.sort(key=lambda m: m.created_at)
        
        # Format messages
        for msg in all_messages:
            lines.append(self._format_message_markdown(msg))
        
        # Footer
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*Exported from Chorus Engine on {self._format_datetime(datetime.now())}*")
        
        return "\n".join(lines)
    
    def export_to_text(
        self,
        conversation: Conversation,
        include_metadata: bool = True
    ) -> str:
        """
        Export conversation to plain text format.
        
        Args:
            conversation: Conversation to export
            include_metadata: Include title, dates, character info
        
        Returns:
            Plain text string
        """
        lines = []
        
        # Header
        if include_metadata:
            lines.append("=" * 70)
            lines.append(conversation.title)
            lines.append("=" * 70)
            lines.append("")
            
            try:
                character = self.config_loader.load_character(conversation.character_id)
                lines.append(f"Character: {character.name}")
            except:
                lines.append(f"Character ID: {conversation.character_id}")
            
            lines.append(f"Started: {self._format_datetime(conversation.created_at)}")
            lines.append(f"Last Updated: {self._format_datetime(conversation.updated_at)}")
            
            message_count = sum(
                1 for thread in conversation.threads for m in thread.messages if m.deleted_at is None
            )
            lines.append(f"Messages: {message_count}")
            
            lines.append("")
            lines.append("-" * 70)
            lines.append("")
        
        # Get all messages sorted by time
        all_messages = []
        for thread in conversation.threads:
            all_messages.extend([m for m in thread.messages if m.deleted_at is None])
        all_messages.sort(key=lambda m: m.created_at)
        
        # Format messages
        for msg in all_messages:
            lines.append(self._format_message_text(msg))
            lines.append("")
        
        # Footer
        lines.append("-" * 70)
        lines.append(f"Exported from Chorus Engine on {self._format_datetime(datetime.now())}")
        
        return "\n".join(lines)
    
    def save_to_file(
        self,
        conversation: Conversation,
        summaries: List[ConversationSummary] = None,
        output_dir: Path = None,
        format: str = "markdown",
        include_metadata: bool = True,
        include_summary: bool = True,
        include_memories: bool = False
    ) -> Path:
        """
        Export conversation to file.
        
        Args:
            conversation: Conversation to export
            summaries: List of conversation summaries (if analyzed)
            output_dir: Directory to save file
            format: Export format ('markdown' or 'text')
            include_metadata: Include metadata
            include_summary: Include summary (markdown only)
            include_memories: Include memories (markdown only)
        
        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate content
        if format == "markdown":
            content = self.export_to_markdown(
                conversation=conversation,
                summaries=summaries,
                include_metadata=include_metadata,
                include_summary=include_summary,
                include_memories=include_memories
            )
            extension = ".md"
        elif format == "text":
            content = self.export_to_text(
                conversation=conversation,
                include_metadata=include_metadata
            )
            extension = ".txt"
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._sanitize_filename(conversation.title)
        filename = f"{safe_title}_{timestamp}{extension}"
        
        # Write file
        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        
        logger.info(f"Exported conversation to {filepath}")
        
        return filepath
    
    def _format_message_markdown(self, message: Message) -> str:
        """Format a message for markdown export."""
        # Determine speaker
        if message.role == MessageRole.USER:
            speaker = "**User**"
        elif message.role == MessageRole.ASSISTANT:
            speaker = "**Assistant**"
        elif message.role == MessageRole.SYSTEM:
            speaker = "**System**"
        else:
            speaker = f"**{message.role}**"
        
        # Format timestamp
        timestamp = self._format_time(message.created_at)
        
        # Format content with proper indentation
        content = message.content.strip()
        
        # Build message block
        return f"{speaker} *({timestamp})*\n\n{content}\n"
    
    def _format_message_text(self, message: Message) -> str:
        """Format a message for plain text export."""
        # Determine speaker
        if message.role == MessageRole.USER:
            speaker = "USER"
        elif message.role == MessageRole.ASSISTANT:
            speaker = "ASSISTANT"
        elif message.role == MessageRole.SYSTEM:
            speaker = "SYSTEM"
        else:
            speaker = str(message.role).upper()
        
        # Format timestamp
        timestamp = self._format_time(message.created_at)
        
        # Build message
        lines = [
            f"[{timestamp}] {speaker}:",
            message.content.strip()
        ]
        
        return "\n".join(lines)
    
    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for display."""
        return dt.strftime("%B %d, %Y at %I:%M %p")
    
    def _format_time(self, dt: datetime) -> str:
        """Format time for message timestamps."""
        return dt.strftime("%I:%M %p")
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize conversation title for use in filename."""
        # Remove/replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            title = title.replace(char, '_')
        
        # Limit length
        max_length = 50
        if len(title) > max_length:
            title = title[:max_length]
        
        # Remove trailing spaces/underscores
        title = title.strip().rstrip('_')
        
        return title or "conversation"
