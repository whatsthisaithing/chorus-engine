"""
Workflow management for ComfyUI integration.

Phase 5: Handles loading character-specific workflows and injecting prompts.
Phase 6: Extended to support workflow types (image, audio, video).
"""

import json
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """
    Types of ComfyUI workflows supported.
    
    Phase 5: IMAGE (image generation)
    Phase 6: AUDIO (text-to-speech)
    Future: VIDEO (video generation)
    """
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""
    pass


class WorkflowNotFoundError(WorkflowError):
    """Workflow file not found."""
    pass


class WorkflowValidationError(WorkflowError):
    """Workflow file is invalid."""
    pass


class WorkflowManager:
    """
    Manages ComfyUI workflow loading and prompt injection.
    
    Handles:
    - Loading character-specific workflows
    - Validating workflow structure
    - Injecting prompts into workflow nodes
    - Injecting seeds and other parameters
    """
    
    def __init__(self, workflows_dir: Path = Path("workflows")):
        """
        Initialize workflow manager.
        
        Args:
            workflows_dir: Root directory containing character workflow folders
        """
        self.workflows_dir = Path(workflows_dir)
        
        if not self.workflows_dir.exists():
            logger.warning(f"Workflows directory does not exist: {self.workflows_dir}")
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Workflow manager initialized: {self.workflows_dir}")
    
    def load_character_workflow(
        self,
        character_id: str,
        workflow_filename: str = "workflow.json"
    ) -> Dict[str, Any]:
        """
        Load a character's ComfyUI workflow.
        
        Args:
            character_id: Character ID (subfolder name)
            workflow_filename: Workflow file name (default: workflow.json)
        
        Returns:
            Workflow data dictionary
        
        Raises:
            WorkflowNotFoundError: Workflow file doesn't exist
            WorkflowValidationError: Workflow file is invalid
        """
        workflow_path = self.workflows_dir / character_id / workflow_filename
        
        if not workflow_path.exists():
            logger.error(f"Workflow not found: {workflow_path}")
            raise WorkflowNotFoundError(
                f"Workflow file not found: {workflow_path}. "
                f"Please create a workflow for character '{character_id}' at this location."
            )
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            logger.debug(f"Loaded workflow: {workflow_path}")
            
            # Validate workflow structure
            self._validate_workflow(workflow_data, workflow_path)
            
            return workflow_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in workflow file: {workflow_path} - {e}")
            raise WorkflowValidationError(
                f"Workflow file contains invalid JSON: {e}. "
                f"Please re-export the workflow from ComfyUI."
            )
        
        except Exception as e:
            logger.error(f"Error loading workflow: {e}")
            raise WorkflowError(f"Failed to load workflow: {str(e)}")
    
    def list_character_workflows(self, character_id: str) -> List[str]:
        """
        List all workflow files for a character.
        
        Args:
            character_id: Character ID (subfolder name)
        
        Returns:
            List of workflow filenames (without .json extension)
        """
        character_dir = self.workflows_dir / character_id
        
        if not character_dir.exists():
            logger.debug(f"No workflow directory for character: {character_id}")
            return []
        
        workflows = []
        for file in character_dir.glob("*.json"):
            workflows.append(file.stem)  # filename without extension
        
        logger.debug(f"Found {len(workflows)} workflows for {character_id}: {workflows}")
        return sorted(workflows)
    
    def delete_workflow(self, character_id: str, workflow_name: str) -> None:
        """
        Delete a workflow file.
        
        Args:
            character_id: Character ID (subfolder name)
            workflow_name: Workflow name (without .json extension)
        
        Raises:
            WorkflowNotFoundError: Workflow doesn't exist
        """
        workflow_path = self.workflows_dir / character_id / f"{workflow_name}.json"
        
        if not workflow_path.exists():
            raise WorkflowNotFoundError(f"Workflow not found: {workflow_name}")
        
        workflow_path.unlink()
        logger.info(f"Deleted workflow: {workflow_path}")
    
    def rename_workflow(
        self,
        character_id: str,
        old_name: str,
        new_name: str
    ) -> None:
        """
        Rename a workflow file.
        
        Args:
            character_id: Character ID (subfolder name)
            old_name: Current workflow name (without .json extension)
            new_name: New workflow name (without .json extension)
        
        Raises:
            WorkflowNotFoundError: Old workflow doesn't exist
            WorkflowError: New workflow name already exists
        """
        character_dir = self.workflows_dir / character_id
        old_path = character_dir / f"{old_name}.json"
        new_path = character_dir / f"{new_name}.json"
        
        if not old_path.exists():
            raise WorkflowNotFoundError(f"Workflow not found: {old_name}")
        
        if new_path.exists():
            raise WorkflowError(f"Workflow already exists: {new_name}")
        
        old_path.rename(new_path)
        logger.info(f"Renamed workflow: {old_name} -> {new_name}")
    
    def save_workflow(
        self,
        character_id: str,
        workflow_name: str,
        workflow_data: Dict[str, Any]
    ) -> None:
        """
        Save a workflow file for a character.
        
        Args:
            character_id: Character ID (subfolder name)
            workflow_name: Workflow name (without .json extension)
            workflow_data: Workflow JSON data
        
        Raises:
            WorkflowValidationError: Invalid workflow data
        """
        # Validate before saving
        self._validate_workflow(workflow_data, Path(f"{character_id}/{workflow_name}.json"))
        
        character_dir = self.workflows_dir / character_id
        character_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = character_dir / f"{workflow_name}.json"
        
        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2)
        
        logger.info(f"Saved workflow: {workflow_path}")
    
    def get_character_workflows_dir(self, character_id: str) -> Path:
        """
        Get the workflows directory path for a character.
        
        Args:
            character_id: Character ID (subfolder name)
        
        Returns:
            Path to character's workflows directory
        """
        return self.workflows_dir / character_id
    
    def _validate_workflow(self, workflow_data: Dict[str, Any], path: Path) -> None:
        """
        Validate workflow structure.
        
        Args:
            workflow_data: Workflow dictionary
            path: Path to workflow file (for error messages)
        
        Raises:
            WorkflowValidationError: Workflow is invalid
        """
        # ComfyUI API format can be either:
        # 1. {"prompt": {nodes...}} - wrapped format
        # 2. {nodes...} - direct format (older exports or manual edits)
        
        # If it has a "prompt" key, use that as the nodes
        if "prompt" in workflow_data:
            nodes = workflow_data["prompt"]
        else:
            # Otherwise treat the whole thing as nodes
            nodes = workflow_data
        
        # Check if we have at least some node-like structures
        if not isinstance(nodes, dict) or len(nodes) == 0:
            raise WorkflowValidationError(
                f"Workflow {path.name} contains no valid nodes. "
                f"Please export workflow in API format from ComfyUI."
            )
        
        # Metadata is now optional (placeholder-based injection doesn't need it)
        if "metadata" in workflow_data:
            metadata = workflow_data["metadata"]
            
            # If metadata exists, validate its structure
            if "prompt_node_id" in metadata:
                prompt_node_id = str(metadata["prompt_node_id"])
                
                if prompt_node_id not in nodes:
                    raise WorkflowValidationError(
                        f"Workflow {path.name} metadata references prompt_node_id '{prompt_node_id}', "
                        f"but this node doesn't exist in the workflow."
                    )
                    
                logger.debug(f"Workflow has metadata-based prompt injection: {path.name}")
        
        # Check if workflow has placeholders (alternative to metadata)
        has_placeholders = self._check_for_placeholders(nodes)
        
        if not has_placeholders and "metadata" not in workflow_data:
            logger.warning(
                f"Workflow {path.name} has neither __CHORUS_PROMPT__ placeholders nor metadata. "
                f"Prompt injection may not work. Add __CHORUS_PROMPT__ to your positive prompt text."
            )
        
        logger.debug(f"Workflow validation passed: {path.name}")
    
    def _check_for_placeholders(self, prompt_nodes: Dict[str, Any]) -> bool:
        """
        Check if workflow contains Chorus Engine placeholders.
        
        Checks for:
        - Image workflow placeholders: __CHORUS_PROMPT__, __CHORUS_NEGATIVE__, __CHORUS_SEED__
        - Audio workflow placeholders: __CHORUS_TEXT__, __CHORUS_VOICE_SAMPLE__, __CHORUS_VOICE_TRANSCRIPT__
        """
        for node_data in prompt_nodes.values():
            if not isinstance(node_data, dict):
                continue
            
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            
            for value in inputs.values():
                if isinstance(value, str):
                    # Check for image or audio placeholders
                    if any(placeholder in value for placeholder in [
                        "__CHORUS_PROMPT__", 
                        "__CHORUS_NEGATIVE__", 
                        "__CHORUS_SEED__",
                        "__CHORUS_TEXT__",
                        "__CHORUS_VOICE_SAMPLE__",
                        "__CHORUS_VOICE_TRANSCRIPT__"
                    ]):
                        return True
        
        return False
    
    def inject_prompt(
        self,
        workflow_data: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Inject prompts and parameters into a workflow.
        
        Supports two methods:
        1. Placeholder-based: Searches for {{CHORUS_PROMPT}}, {{CHORUS_NEGATIVE}}, {{CHORUS_SEED}}
        2. Metadata-based: Uses node IDs from metadata section (legacy)
        
        Args:
            workflow_data: Original workflow dictionary
            positive_prompt: Positive prompt text
            negative_prompt: Optional negative prompt text
            seed: Optional seed for reproducibility
        
        Returns:
            Modified workflow data (copy, doesn't modify original)
        
        Raises:
            WorkflowValidationError: Workflow structure doesn't support injection
        """
        # Work with a copy to avoid modifying the original
        workflow = json.loads(json.dumps(workflow_data))
        
        # Extract nodes (handle both formats)
        if "prompt" in workflow:
            prompt_nodes = workflow["prompt"]
        else:
            # Direct format - the whole workflow is the nodes
            prompt_nodes = workflow
        
        # Try placeholder-based injection first (simpler, more flexible)
        placeholders_found = self._inject_via_placeholders(
            prompt_nodes,
            positive_prompt,
            negative_prompt,
            seed
        )
        
        if placeholders_found > 0:
            logger.debug(f"Injected prompts via placeholders ({placeholders_found} replacements)")
            return workflow
        
        # Fall back to metadata-based injection (legacy method)
        metadata = workflow_data.get("metadata", {})
        if metadata and "prompt_node_id" in metadata:
            logger.debug("Using metadata-based prompt injection (legacy)")
            self._inject_via_metadata(
                prompt_nodes,
                metadata,
                positive_prompt,
                negative_prompt,
                seed
            )
            return workflow
        
        # Neither method available
        raise WorkflowValidationError(
            "Workflow doesn't support prompt injection. "
            "Please add __CHORUS_PROMPT__ placeholder to your positive prompt node, "
            "or add metadata with node IDs."
        )
    
    def _inject_via_placeholders(
        self,
        prompt_nodes: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: Optional[str],
        seed: Optional[int]
    ) -> int:
        """
        Inject prompts by finding and replacing placeholder text.
        
        Searches all text fields in the workflow for:
        - __CHORUS_PROMPT__ → replaced with positive_prompt
        - __CHORUS_NEGATIVE__ → replaced with negative_prompt
        - __CHORUS_SEED__ → replaced with seed
        
        Returns:
            Number of replacements made (0 if no placeholders found)
        """
        replacements = 0
        
        # Recursively search for placeholders in all nodes
        for node_id, node_data in prompt_nodes.items():
            if not isinstance(node_data, dict):
                continue
            
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            
            # Check all text inputs
            for key, value in inputs.items():
                if not isinstance(value, str):
                    continue
                
                original_value = value
                
                # Replace placeholders
                if "__CHORUS_PROMPT__" in value:
                    value = value.replace("__CHORUS_PROMPT__", positive_prompt)
                    replacements += 1
                    logger.debug(f"Replaced __CHORUS_PROMPT__ in node {node_id}/{key}")
                
                if negative_prompt and "__CHORUS_NEGATIVE__" in value:
                    value = value.replace("__CHORUS_NEGATIVE__", negative_prompt)
                    replacements += 1
                    logger.debug(f"Replaced __CHORUS_NEGATIVE__ in node {node_id}/{key}")
                
                if seed is not None and "__CHORUS_SEED__" in value:
                    # If entire value is the placeholder, replace with integer
                    if value.strip() == "__CHORUS_SEED__":
                        inputs[key] = seed
                        replacements += 1
                        logger.debug(f"Replaced __CHORUS_SEED__ in node {node_id}/{key} (as integer)")
                        continue  # Don't process as string below
                    else:
                        # Placeholder mixed with other text, replace as string
                        value = value.replace("__CHORUS_SEED__", str(seed))
                        replacements += 1
                        logger.debug(f"Replaced __CHORUS_SEED__ in node {node_id}/{key} (in string)")
                
                # Update if changed
                if value != original_value:
                    inputs[key] = value
            
            # Also check for seed as numeric field (not string)
            if seed is not None and "seed" in inputs:
                # Check if it's a placeholder string
                if isinstance(inputs["seed"], str) and "__CHORUS_SEED__" in inputs["seed"]:
                    inputs["seed"] = seed
                    replacements += 1
                    logger.debug(f"Replaced __CHORUS_SEED__ in node {node_id}/seed (numeric)")
        
        return replacements
    
    def _inject_via_metadata(
        self,
        prompt_nodes: Dict[str, Any],
        metadata: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: Optional[str],
        seed: Optional[int]
    ) -> None:
        """
        Inject prompts using metadata node IDs (legacy method).
        
        Args:
            prompt_nodes: Workflow prompt nodes
            metadata: Workflow metadata with node IDs
            positive_prompt: Positive prompt text
            negative_prompt: Optional negative prompt
            seed: Optional seed
        
        Raises:
            WorkflowValidationError: Required nodes not found
        """
        # Inject positive prompt
        prompt_node_id = str(metadata["prompt_node_id"])
        
        if prompt_node_id not in prompt_nodes:
            raise WorkflowValidationError(
                f"Cannot inject prompt: node {prompt_node_id} not found"
            )
        
        # Find the text field in the node inputs
        node = prompt_nodes[prompt_node_id]
        
        if "inputs" not in node:
            raise WorkflowValidationError(
                f"Node {prompt_node_id} has no 'inputs' field"
            )
        
        # Inject positive prompt (usually in 'text' field for CLIPTextEncode)
        if "text" in node["inputs"]:
            node["inputs"]["text"] = positive_prompt
            logger.debug(f"Injected positive prompt into node {prompt_node_id}")
        else:
            logger.warning(
                f"Node {prompt_node_id} has no 'text' input field. "
                f"Available fields: {list(node['inputs'].keys())}"
            )
        
        # Inject negative prompt if provided
        if negative_prompt and "negative_prompt_node_id" in metadata:
            neg_node_id = str(metadata["negative_prompt_node_id"])
            
            if neg_node_id in prompt_nodes:
                neg_node = prompt_nodes[neg_node_id]
                
                if "inputs" in neg_node and "text" in neg_node["inputs"]:
                    neg_node["inputs"]["text"] = negative_prompt
                    logger.debug(f"Injected negative prompt into node {neg_node_id}")
        
        # Inject seed if provided
        if seed is not None and "seed_node_id" in metadata:
            seed_node_id = str(metadata["seed_node_id"])
            
            if seed_node_id in prompt_nodes:
                seed_node = prompt_nodes[seed_node_id]
                
                if "inputs" in seed_node and "seed" in seed_node["inputs"]:
                    seed_node["inputs"]["seed"] = seed
                    logger.debug(f"Injected seed {seed} into node {seed_node_id}")
    
    def get_workflow_path(
        self,
        character_id: str,
        workflow_filename: str = "workflow.json"
    ) -> Path:
        """
        Get the full path to a character's workflow file.
        
        Args:
            character_id: Character ID
            workflow_filename: Workflow filename
        
        Returns:
            Path to workflow file
        """
        return self.workflows_dir / character_id / workflow_filename
    
    def workflow_exists(
        self,
        character_id: str,
        workflow_filename: str = "workflow.json"
    ) -> bool:
        """
        Check if a workflow file exists.
        
        Args:
            character_id: Character ID
            workflow_filename: Workflow filename
        
        Returns:
            True if file exists
        """
        return self.get_workflow_path(character_id, workflow_filename).exists()
    
    def create_example_workflow(self, character_id: str) -> Path:
        """
        Create an example workflow file for a character.
        
        Args:
            character_id: Character ID
        
        Returns:
            Path to created file
        """
        workflow_dir = self.workflows_dir / character_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = workflow_dir / "workflow.json"
        
        # Create a minimal example workflow
        example_workflow = {
            "metadata": {
                "name": f"{character_id} Image Generation",
                "description": "Example workflow - replace with your ComfyUI workflow",
                "prompt_node_id": "6",
                "negative_prompt_node_id": "7",
                "seed_node_id": "3"
            },
            "prompt": {
                "3": {
                    "inputs": {
                        "seed": 42,
                        "control_after_generate": "randomize"
                    },
                    "class_type": "KSampler"
                },
                "6": {
                    "inputs": {
                        "text": "PLACEHOLDER_PROMPT",
                        "clip": ["11", 0]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "7": {
                    "inputs": {
                        "text": "PLACEHOLDER_NEGATIVE",
                        "clip": ["11", 0]
                    },
                    "class_type": "CLIPTextEncode"
                }
            }
        }
        
        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(example_workflow, f, indent=2)
        
        logger.info(f"Created example workflow at: {workflow_path}")
        return workflow_path    
    def load_workflow_by_type(
        self,
        character_id: str,
        workflow_type: WorkflowType,
        workflow_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a workflow by type (Phase 6+).
        
        Workflows are now organized in subfolders by type:
        workflows/<character_id>/image/workflow.json
        workflows/<character_id>/audio/workflow.json
        
        Falls back to workflows/defaults/<type>/<workflow> if character-specific not found.
        
        Args:
            character_id: Character ID (subfolder name, or "defaults" for default workflows)
            workflow_type: Type of workflow (IMAGE, AUDIO, VIDEO)
            workflow_name: Specific workflow name (default: workflow.json)
        
        Returns:
            Workflow data dictionary
        
        Raises:
            WorkflowNotFoundError: Workflow file doesn't exist
            WorkflowValidationError: Workflow file is invalid
        """
        if workflow_name is None:
            workflow_name = "workflow.json"
        elif not workflow_name.endswith('.json'):
            workflow_name = f"{workflow_name}.json"
        
        # Try character-specific workflow first (in type subfolder)
        workflow_path = self.workflows_dir / character_id / workflow_type.value / workflow_name
        
        # If not found and not already looking in defaults, try defaults folder
        if not workflow_path.exists() and character_id != "defaults":
            logger.debug(f"Character-specific workflow not found at {workflow_path}, trying defaults")
            # Defaults folder doesn't use type subfolders - workflows are directly in defaults/
            workflow_path = self.workflows_dir / "defaults" / workflow_name
        
        if not workflow_path.exists():
            logger.error(f"Workflow not found: {workflow_path}")
            raise WorkflowNotFoundError(
                f"Workflow file not found: {workflow_path}. "
                f"Please create a {workflow_type.value} workflow named '{workflow_name}' in workflows/defaults/{workflow_type.value}/ "
                f"or in workflows/{character_id}/{workflow_type.value}/."
            )
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            logger.debug(f"Loaded {workflow_type.value} workflow: {workflow_path}")
            
            # Validate workflow structure
            self._validate_workflow(workflow_data, workflow_path)
            
            return workflow_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in workflow file: {workflow_path} - {e}")
            raise WorkflowValidationError(
                f"Workflow file contains invalid JSON: {e}. "
                f"Please re-export the workflow from ComfyUI."
            )
        
        except Exception as e:
            logger.error(f"Error loading workflow: {e}")
            raise WorkflowError(f"Failed to load workflow: {str(e)}")
    
    def list_workflows_by_type(
        self,
        character_id: str,
        workflow_type: WorkflowType
    ) -> List[str]:
        """
        List all workflows of a specific type for a character (Phase 6+).
        
        Args:
            character_id: Character ID (subfolder name)
            workflow_type: Type of workflow (IMAGE, AUDIO, VIDEO)
        
        Returns:
            List of workflow filenames (without .json extension)
        """
        type_dir = self.workflows_dir / character_id / workflow_type.value
        
        if not type_dir.exists():
            logger.debug(f"No {workflow_type.value} workflow directory for character: {character_id}")
            return []
        
        workflows = []
        for file in type_dir.glob("*.json"):
            workflows.append(file.stem)  # filename without extension
        
        logger.debug(f"Found {len(workflows)} {workflow_type.value} workflows for {character_id}: {workflows}")
        return sorted(workflows)
    
    def save_workflow_by_type(
        self,
        character_id: str,
        workflow_type: WorkflowType,
        workflow_name: str,
        workflow_data: Dict[str, Any]
    ) -> None:
        """
        Save a workflow file of a specific type for a character (Phase 6+).
        
        Args:
            character_id: Character ID (subfolder name)
            workflow_type: Type of workflow (IMAGE, AUDIO, VIDEO)
            workflow_name: Workflow name (without .json extension)
            workflow_data: Workflow JSON data
        
        Raises:
            WorkflowValidationError: Invalid workflow data
        """
        # Validate before saving
        self._validate_workflow(workflow_data, Path(f"{character_id}/{workflow_type.value}/{workflow_name}.json"))
        
        type_dir = self.workflows_dir / character_id / workflow_type.value
        type_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = type_dir / f"{workflow_name}.json"
        
        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2)
        
        logger.info(f"Saved {workflow_type.value} workflow: {workflow_path}")
    
    def inject_audio_placeholders(
        self,
        workflow_data: Dict[str, Any],
        text: str,
        voice_sample_path: Optional[str] = None,
        transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inject TTS-specific placeholders into an audio workflow (Phase 6).
        
        Placeholders:
        - __CHORUS_TEXT__: The text to convert to speech
        - __CHORUS_VOICE_SAMPLE__: Path to voice sample audio file
        - __CHORUS_VOICE_TRANSCRIPT__: Transcript of the voice sample
        
        If voice_sample_path or transcript are None, they're replaced with empty strings.
        
        Args:
            workflow_data: The workflow JSON to modify
            text: The text to speak
            voice_sample_path: Path to voice sample audio file (optional)
            transcript: Transcript of the voice sample (optional)
        
        Returns:
            Modified workflow with placeholders replaced
        """
        # Deep copy to avoid modifying original
        import copy
        workflow = copy.deepcopy(workflow_data)
        
        # Helper function to properly escape strings for JSON
        def json_escape(s: str) -> str:
            """Escape a string for safe insertion into JSON."""
            # Use json.dumps to properly escape, then remove the surrounding quotes
            return json.dumps(s)[1:-1]
        
        # Convert to JSON string for replacement
        workflow_str = json.dumps(workflow)
        
        # Replace text placeholder with properly escaped text
        workflow_str = workflow_str.replace('__CHORUS_TEXT__', json_escape(text))
        
        # Replace voice sample placeholders (or empty string if not provided)
        voice_sample = voice_sample_path or ''
        transcript_text = transcript or ''
        
        workflow_str = workflow_str.replace('__CHORUS_VOICE_SAMPLE__', json_escape(voice_sample))
        workflow_str = workflow_str.replace('__CHORUS_VOICE_TRANSCRIPT__', json_escape(transcript_text))
        
        # Convert back to dictionary
        return json.loads(workflow_str)