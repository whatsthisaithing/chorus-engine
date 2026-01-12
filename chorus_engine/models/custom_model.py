"""Database model for downloaded LLM models (curated and custom)."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON
from chorus_engine.db.database import Base


class DownloadedModel(Base):
    """Track all downloaded models (curated and custom HF) in unified table."""
    __tablename__ = "downloaded_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(200), nullable=False, unique=True)  # e.g., "llama-3.3-8b-Q4_K_M" or "hf.co/user/repo:Q4_K_M"
    display_name = Column(String(500), nullable=False)  # Human-readable name
    repo_id = Column(String(500), nullable=False)  # HuggingFace repo_id
    filename = Column(String(500), nullable=True)  # GGUF filename (null for Ollama models)
    quantization = Column(String(100), nullable=False)  # e.g., "Q4_K_M"
    parameters = Column(Float, nullable=True)  # Billion parameters (e.g., 8.0)
    context_window = Column(Integer, nullable=True)  # Context window size
    file_size_mb = Column(Float, nullable=True)  # File size in MB
    file_path = Column(String(1000), nullable=True)  # Local file path (null for Ollama)
    ollama_model_name = Column(String(500), nullable=True)  # Ollama model name if imported
    source = Column(String(50), nullable=False)  # 'curated' or 'custom_hf'
    tags = Column(JSON, nullable=True)  # List of tags
    downloaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'display_name': self.display_name,
            'repo_id': self.repo_id,
            'filename': self.filename,
            'quantization': self.quantization,
            'parameters': self.parameters,
            'context_window': self.context_window,
            'file_size_mb': self.file_size_mb,
            'file_path': self.file_path,
            'ollama_model_name': self.ollama_model_name,
            'source': self.source,
            'tags': self.tags or [],
            'downloaded_at': self.downloaded_at.isoformat() if self.downloaded_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'custom': self.source == 'custom_hf'  # For backward compatibility
        }
