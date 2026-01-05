"""Database models for ComfyUI workflows."""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer
from chorus_engine.db.database import Base


class Workflow(Base):
    """
    A workflow represents a ComfyUI workflow configuration for generation tasks.
    
    Phase 6.5: Supports multiple workflow types (image, audio, video).
    
    Each workflow:
    - Belongs to a specific character
    - Has a unique name within that character
    - Points to a JSON workflow file
    - Can be marked as the default workflow for its type
    - Contains type-specific configuration
    """
    __tablename__ = "workflows"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_name = Column(String(50), nullable=False)
    workflow_name = Column(String(100), nullable=False)
    workflow_file_path = Column(String(500), nullable=False)  # workflows/{character}/{type}/{name}.json
    workflow_type = Column(String(20), nullable=False, default='image')  # Phase 6.5: image, audio, video
    is_default = Column(Boolean, nullable=False, default=False)
    
    # Image generation configuration
    trigger_word = Column(String(100), nullable=True)  # e.g., "nova"
    default_style = Column(Text, nullable=True)  # e.g., "photorealistic portrait, dramatic lighting"
    negative_prompt = Column(Text, nullable=True)  # e.g., "cartoon, anime, illustration"
    self_description = Column(Text, nullable=True)  # e.g., "25 year old woman with auburn hair"
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Workflow(id={self.id}, character={self.character_name}, name={self.workflow_name}, default={self.is_default})>"
