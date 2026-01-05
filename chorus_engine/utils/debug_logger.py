"""
Debug logging utility for LLM interactions.
Logs every prompt sent to any LLM with full metadata for debugging.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DebugLogger:
    """Logs all LLM interactions to conversation-specific files for debugging."""
    
    def __init__(self, debug_dir: Path = None, enabled: bool = True):
        """Initialize debug logger.
        
        Args:
            debug_dir: Directory for debug logs. Defaults to data/debug_logs/conversations/
            enabled: Whether debug logging is enabled (from system config)
        """
        if debug_dir is None:
            debug_dir = Path("data/debug_logs/conversations")
        
        self.debug_dir = debug_dir
        self.enabled = enabled
        
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug logger initialized: {self.debug_dir}")
        else:
            logger.info("Debug logging disabled (debug mode off in system config)")
    
    def log_llm_interaction(
        self,
        conversation_id: str,
        interaction_type: str,
        model: str,
        prompt: str,
        response: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log an LLM interaction with full context.
        
        Args:
            conversation_id: ID of conversation (or 'system' for non-conversation calls)
            interaction_type: Type of interaction (chat, intent_detection, image_prompt, etc.)
            model: Model name used
            prompt: Full prompt sent to LLM
            response: LLM response (if available)
            settings: LLM settings (temperature, max_tokens, etc.)
            metadata: Additional metadata (character_id, thread_id, etc.)
            error: Error message if interaction failed
        """
        if not self.enabled:
            return
        
        try:
            # Create conversation-specific directory
            conversation_dir = self.debug_dir / conversation_id
            conversation_dir.mkdir(parents=True, exist_ok=True)
            
            # Log to conversation.jsonl file
            log_file = conversation_dir / "conversation.jsonl"
            
            # Build interaction record
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction_type,
                "model": model,
                "prompt": prompt,
                "response": response,
                "settings": settings or {},
                "metadata": metadata or {},
                "error": error
            }
            
            # Append to JSONL file (one JSON object per line)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
            
            # Also log summary to regular logs
            logger.debug(
                f"[DEBUG LOG] {interaction_type} | model={model} | "
                f"conversation={conversation_id} | "
                f"prompt_len={len(prompt)} chars"
            )
            
        except Exception as e:
            logger.error(f"Failed to write debug log: {e}", exc_info=True)
    
    def get_conversation_log(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Read all logged interactions for a conversation.
        
        Args:
            conversation_id: ID of conversation
            
        Returns:
            List of interaction records
        """
        log_file = self.debug_dir / conversation_id / "conversation.jsonl"
        if not log_file.exists():
            return []
        
        interactions = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))
        
        return interactions
    
    def clear_conversation_log(self, conversation_id: str):
        """Delete debug log directory for a conversation.
        
        Args:
            conversation_id: ID of conversation
        """
        import shutil
        conversation_dir = self.debug_dir / conversation_id
        if conversation_dir.exists():
            shutil.rmtree(conversation_dir)
            logger.info(f"Cleared debug logs for conversation {conversation_id}")
    
    def enable(self):
        """Enable debug logging."""
        self.enabled = True
        logger.info("Debug logging enabled")
    
    def disable(self):
        """Disable debug logging."""
        self.enabled = False
        logger.info("Debug logging disabled")


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def initialize_debug_logger(enabled: bool = True) -> DebugLogger:
    """Initialize global debug logger with enabled flag.
    
    Args:
        enabled: Whether debug logging is enabled (from system config)
    
    Returns:
        Initialized DebugLogger instance
    """
    global _debug_logger
    _debug_logger = DebugLogger(enabled=enabled)
    return _debug_logger


def get_debug_logger() -> DebugLogger:
    """Get global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()  # Default to enabled if not initialized
    return _debug_logger


def log_llm_call(
    conversation_id: str,
    interaction_type: str,
    model: str,
    prompt: str,
    response: Optional[str] = None,
    **kwargs
):
    """Convenience function to log LLM interaction."""
    get_debug_logger().log_llm_interaction(
        conversation_id=conversation_id,
        interaction_type=interaction_type,
        model=model,
        prompt=prompt,
        response=response,
        **kwargs
    )
