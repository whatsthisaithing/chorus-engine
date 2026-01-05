"""Repository for workflow database operations."""

from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from chorus_engine.models.workflow import Workflow


class WorkflowRepository:
    """Repository for managing workflow database operations."""
    
    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db
    
    def create(self, character_name: str, workflow_name: str, workflow_file_path: str,
               workflow_type: str = 'image', is_default: bool = False, trigger_word: Optional[str] = None,
               default_style: Optional[str] = None, negative_prompt: Optional[str] = None,
               self_description: Optional[str] = None) -> Workflow:
        """
        Create a new workflow record.
        
        Args:
            character_name: Character this workflow belongs to
            workflow_name: Unique name for this workflow
            workflow_file_path: Path to the JSON workflow file
            workflow_type: Type of workflow (image, audio, video) - Phase 6.5
            is_default: Whether this is the default workflow for this type
            trigger_word: Trigger word for image generation
            default_style: Default style prompt
            negative_prompt: Negative prompt
            self_description: Character self-description
            
        Returns:
            Created workflow
        """
        # If setting as default, unset other defaults for this character AND type
        if is_default:
            self.db.query(Workflow).filter(
                Workflow.character_name == character_name,
                Workflow.workflow_type == workflow_type,
                Workflow.is_default == True
            ).update({"is_default": False})
        
        workflow = Workflow(
            character_name=character_name,
            workflow_name=workflow_name,
            workflow_file_path=workflow_file_path,
            workflow_type=workflow_type,
            is_default=is_default,
            trigger_word=trigger_word,
            default_style=default_style,
            negative_prompt=negative_prompt,
            self_description=self_description
        )
        
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)
        
        return workflow
    
    def get_by_id(self, workflow_id: int) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    def get_by_name(self, character_name: str, workflow_name: str) -> Optional[Workflow]:
        """Get workflow by character and name."""
        return self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.workflow_name == workflow_name
        ).first()
    
    def get_all_for_character(self, character_name: str) -> List[Workflow]:
        """Get all workflows for a character."""
        return self.db.query(Workflow).filter(
            Workflow.character_name == character_name
        ).order_by(Workflow.workflow_name).all()
    
    def get_default(self, character_name: str) -> Optional[Workflow]:
        """Get the default workflow for a character (legacy - for image workflows)."""
        return self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.is_default == True
        ).first()
    
    def get_default_for_character_and_type(self, character_name: str, workflow_type: str) -> Optional[Workflow]:
        """Get the default workflow for a character and type (Phase 6.5)."""
        return self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.workflow_type == workflow_type,
            Workflow.is_default == True
        ).first()
    
    def set_default(self, character_name: str, workflow_name: str) -> bool:
        """
        Set a workflow as the default for a character.
        
        Args:
            character_name: Character name
            workflow_name: Workflow to set as default
            
        Returns:
            True if successful, False if workflow not found
        """
        # Unset all defaults for this character
        self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.is_default == True
        ).update({"is_default": False})
        
        # Set new default
        result = self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.workflow_name == workflow_name
        ).update({"is_default": True})
        
        self.db.commit()
        return result > 0
    
    def update_config(self, workflow_id: int, trigger_word: Optional[str] = None,
                     default_style: Optional[str] = None, negative_prompt: Optional[str] = None,
                     self_description: Optional[str] = None) -> Optional[Workflow]:
        """
        Update workflow configuration.
        
        Note: This method ALWAYS updates all fields. Pass empty string or None to clear a field.
        
        Args:
            workflow_id: Workflow ID
            trigger_word: New trigger word
            default_style: New default style
            negative_prompt: New negative prompt
            self_description: New self description
            
        Returns:
            Updated workflow or None if not found
        """
        workflow = self.get_by_id(workflow_id)
        if not workflow:
            return None
        
        # Always update all fields (allows clearing values)
        workflow.trigger_word = trigger_word
        workflow.default_style = default_style
        workflow.negative_prompt = negative_prompt
        workflow.self_description = self_description
        
        self.db.commit()
        self.db.refresh(workflow)
        
        return workflow
    
    def rename(self, character_name: str, old_name: str, new_name: str) -> bool:
        """
        Rename a workflow.
        
        Args:
            character_name: Character name
            old_name: Current workflow name
            new_name: New workflow name
            
        Returns:
            True if successful, False if workflow not found
        """
        result = self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.workflow_name == old_name
        ).update({"workflow_name": new_name})
        
        self.db.commit()
        return result > 0
    
    def delete(self, character_name: str, workflow_name: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            character_name: Character name
            workflow_name: Workflow name
            
        Returns:
            True if deleted, False if not found
        """
        result = self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.workflow_name == workflow_name
        ).delete()
        
        self.db.commit()
        return result > 0
    
    def count_for_character(self, character_name: str) -> int:
        """Get count of workflows for a character."""
        return self.db.query(Workflow).filter(
            Workflow.character_name == character_name
        ).count()
    
    def count_for_character_and_type(self, character_name: str, workflow_type: str) -> int:
        """Get count of workflows for a character and type (Phase 6.5)."""
        return self.db.query(Workflow).filter(
            Workflow.character_name == character_name,
            Workflow.workflow_type == workflow_type
        ).count()
    
    def get_default_config(self, character_name: str) -> Optional[Dict[str, Optional[str]]]:
        """
        Get default workflow configuration for a character.
        
        Returns:
            Dict with trigger_word, default_style, negative_prompt, self_description
            or None if no default workflow found
        """
        workflow = self.get_default(character_name)
        if not workflow:
            return None
        
        return {
            "trigger_word": workflow.trigger_word,
            "default_style": workflow.default_style,
            "negative_prompt": workflow.negative_prompt,
            "self_description": workflow.self_description,
            "workflow_file_path": workflow.workflow_file_path
        }
