"""Model management service for downloading and tracking GGUF models."""

import asyncio
import json
import logging
import subprocess
import httpx
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    logger.warning("huggingface_hub not installed. Model downloads will not be available.")


@dataclass
class ModelInfo:
    """Information about a downloaded model."""
    model_id: str
    display_name: str
    repo_id: str
    filename: str
    quantization: str
    parameters_billions: float
    context_window: int
    file_size_mb: int
    file_path: str
    downloaded_at: str
    last_used: Optional[str] = None
    custom: bool = False
    source: str = "curated"
    tags: List[str] = None
    ollama_model_name: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelManager:
    """
    Manages local model storage, downloads, and metadata.
    
    Features:
    - Download GGUF models from HuggingFace
    - Cache models in data/models/
    - Track download progress
    - Validate model files
    - List available models
    - Delete unused models
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory for model storage (default: data/models/)
        """
        self.models_dir = models_dir or Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache directory for HuggingFace downloads
        self.cache_dir = self.models_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"ModelManager initialized: {self.models_dir}")
    
    async def download_model(
        self,
        repo_id: str,
        filename: str,
        model_id: str,
        display_name: str,
        quantization: str,
        parameters_billions: float,
        context_window: int,
        tags: List[str] = None,
        custom: bool = False,
        expected_file_size_mb: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """
        Download a GGUF model from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
            filename: Model filename in the repo
            model_id: Unique identifier for this model
            display_name: Human-readable name
            quantization: Quantization type (Q4_K_M, etc.)
            parameters_billions: Model size in billions
            context_window: Context window size
            tags: Optional tags for categorization
            custom: Whether this is a custom (non-curated) model
            expected_file_size_mb: Expected file size in MB (for progress tracking)
            progress_callback: Optional callback(downloaded_bytes, total_bytes)
            
        Returns:
            Path to downloaded model file
            
        Raises:
            Exception: If download fails
        """
        if not HAS_HF_HUB:
            raise Exception("huggingface_hub not installed. Cannot download models.")
        
        logger.info(f"Downloading model: {display_name}")
        logger.info(f"  Repo: {repo_id}")
        logger.info(f"  File: {filename}")
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download using huggingface_hub
            # Note: hf_hub_download is synchronous, run in executor
            # We'll monitor file size if callback provided
            loop = asyncio.get_event_loop()
            
            # Start download in background
            download_task = loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.cache_dir),
                    resume_download=True,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False
                )
            )
            
            # Monitor progress if callback provided
            if progress_callback:
                # Expected final path
                expected_path = model_dir / filename
                
                # HuggingFace downloads to cache first with .incomplete suffix
                # Look for the actual download file in the cache
                cache_download_dir = model_dir / ".cache" / "huggingface" / "download"
                
                # Convert expected file size to bytes
                expected_total_bytes = int(expected_file_size_mb * 1024 * 1024) if expected_file_size_mb else 0
                
                logger.info(f"Starting progress monitoring for: {expected_path}")
                logger.info(f"Expected total size: {expected_file_size_mb} MB")
                logger.info(f"Monitoring cache directory: {cache_download_dir}")
                
                # Poll for file size updates
                import time
                while not download_task.done():
                    await asyncio.sleep(1)  # Check every second
                    
                    current_size = 0
                    
                    # First check if file is at final location (fast path after move)
                    if expected_path.exists():
                        current_size = expected_path.stat().st_size
                    # Otherwise look in cache for .incomplete file
                    elif cache_download_dir.exists():
                        for temp_file in cache_download_dir.glob("*.incomplete"):
                            current_size = temp_file.stat().st_size
                            break
                    
                    if current_size > 0:
                        # Use expected total or fallback to current size * 2
                        total_estimate = expected_total_bytes if expected_total_bytes > 0 else current_size * 2
                        progress_callback(current_size, total_estimate)
            
            # Wait for download to complete
            local_path = await download_task
            local_path = Path(local_path)
            
            # Get file size
            file_size_mb = local_path.stat().st_size // (1024 * 1024)
            
            # Save metadata
            metadata = ModelInfo(
                model_id=model_id,
                display_name=display_name,
                repo_id=repo_id,
                filename=filename,
                quantization=quantization,
                parameters_billions=parameters_billions,
                context_window=context_window,
                file_size_mb=file_size_mb,
                file_path=str(local_path),
                downloaded_at=datetime.now().isoformat(),
                last_used=None,
                custom=custom,
                source="custom" if custom else "curated",
                tags=tags or []
            )
            
            self._save_metadata(model_dir, metadata)
            
            logger.info(f"Model downloaded successfully: {local_path}")
            logger.info(f"  Size: {file_size_mb} MB")
            
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}", exc_info=True)
            # Clean up partial download
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
            raise
    
    def generate_modelfile(
        self,
        model_path: Path,
        model_name: str,
        template: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate Modelfile content for Ollama import.
        
        Args:
            model_path: Path to GGUF file
            model_name: Name for the model in Ollama
            template: Optional chat template
            parameters: Optional model parameters (temperature, etc.)
            
        Returns:
            Modelfile content as string
        """
        # Use absolute path for FROM directive
        abs_path = model_path.resolve()
        
        modelfile = f"FROM {abs_path}\n\n"
        
        # Add template if provided
        if template:
            modelfile += f'TEMPLATE """{template}"""\n\n'
        
        # Add parameters if provided
        if parameters:
            for key, value in parameters.items():
                modelfile += f"PARAMETER {key} {value}\n"
        
        return modelfile
    
    async def import_to_ollama(
        self,
        model_path: Path,
        model_name: str,
        ollama_base_url: str = "http://localhost:11434"
    ) -> bool:
        """
        Import downloaded GGUF model into Ollama.
        
        Args:
            model_path: Path to downloaded GGUF file
            model_name: Name to give model in Ollama (e.g., "qwen2.5-14b-q5")
            ollama_base_url: Ollama API base URL
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            # Generate Modelfile
            modelfile_content = self.generate_modelfile(
                model_path=model_path,
                model_name=model_name,
                parameters={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            # Save Modelfile temporarily
            modelfile_path = model_path.parent / "Modelfile"
            modelfile_path.write_text(modelfile_content)
            
            logger.info(f"Importing model to Ollama: {model_name}")
            logger.debug(f"Modelfile path: {modelfile_path}")
            
            # Run ollama create command
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for import
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully imported model to Ollama: {model_name}")
                logger.debug(f"Ollama output: {result.stdout}")
                
                # Clean up Modelfile
                modelfile_path.unlink(missing_ok=True)
                
                return True
            else:
                logger.error(f"Failed to import model to Ollama: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama import timed out after 5 minutes")
            return False
        except FileNotFoundError:
            logger.error("Ollama CLI not found. Is Ollama installed and in PATH?")
            return False
        except Exception as e:
            logger.error(f"Failed to import model to Ollama: {e}", exc_info=True)
            return False
    
    async def check_ollama_availability(
        self,
        ollama_base_url: str = "http://localhost:11434"
    ) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Args:
            ollama_base_url: Ollama API base URL
            
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ollama_base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all downloaded models with metadata.
        
        Returns:
            List of ModelInfo objects
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == ".cache":
                continue
            
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        data = json.load(f)
                    models.append(ModelInfo(**data))
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_dir.name}: {e}")
        
        # Sort by last_used (most recent first), then by downloaded_at
        models.sort(
            key=lambda m: (m.last_used or m.downloaded_at, m.downloaded_at),
            reverse=True
        )
        
        return models
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Get local path for a downloaded model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model file, or None if not found
        """
        model_dir = self.models_dir / model_id
        metadata_path = model_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path) as f:
                data = json.load(f)
            
            file_path = Path(data["file_path"])
            if file_path.exists():
                return file_path
            else:
                logger.warning(f"Model file missing: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model path for {model_id}: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get metadata for a downloaded model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelInfo or None if not found
        """
        model_dir = self.models_dir / model_id
        metadata_path = model_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path) as f:
                data = json.load(f)
            return ModelInfo(**data)
        except Exception as e:
            logger.error(f"Failed to load model info for {model_id}: {e}")
            return None
    
    def update_last_used(self, model_id: str) -> bool:
        """
        Update the last_used timestamp for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if updated, False otherwise
        """
        model_dir = self.models_dir / model_id
        metadata_path = model_dir / "metadata.json"
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path) as f:
                data = json.load(f)
            
            data["last_used"] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last_used for {model_id}: {e}")
            return False
    
    def update_ollama_model_name(self, model_id: str, ollama_model_name: str) -> bool:
        """
        Update the ollama_model_name in model metadata.
        
        Args:
            model_id: Model identifier
            ollama_model_name: Name of model in Ollama
            
        Returns:
            True if updated, False otherwise
        """
        model_dir = self.models_dir / model_id
        metadata_path = model_dir / "metadata.json"
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path) as f:
                data = json.load(f)
            
            data["ollama_model_name"] = ollama_model_name
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Updated ollama_model_name for {model_id}: {ollama_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ollama_model_name for {model_id}: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from disk.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False if not found or error
        """
        model_dir = self.models_dir / model_id
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_id}")
            return False
        
        try:
            # Delete all files in the directory
            import shutil
            shutil.rmtree(model_dir)
            
            logger.info(f"Deleted model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def get_model_size(self, repo_id: str, filename: str) -> int:
        """
        Get model file size without downloading.
        
        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename
            
        Returns:
            File size in bytes, or 0 if unable to determine
        """
        if not HAS_HF_HUB:
            return 0
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Run in executor (API call is synchronous)
            loop = asyncio.get_event_loop()
            repo_info = await loop.run_in_executor(
                None,
                lambda: api.repo_info(repo_id=repo_id)
            )
            
            # Find the file in siblings
            for sibling in repo_info.siblings:
                if sibling.rfilename == filename:
                    return sibling.size
            
            logger.warning(f"File not found in repo: {filename}")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get model size: {e}")
            return 0
    
    async def list_repo_gguf_files(self, repo_id: str) -> List[str]:
        """
        List all GGUF files in a HuggingFace repository.
        
        Args:
            repo_id: HuggingFace repository ID
            
        Returns:
            List of GGUF filenames
        """
        if not HAS_HF_HUB:
            return []
        
        try:
            # Run in executor
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(
                None,
                lambda: list_repo_files(repo_id=repo_id)
            )
            
            # Filter for GGUF files
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            return gguf_files
            
        except Exception as e:
            logger.error(f"Failed to list repo files: {e}")
            return []
    
    def get_total_storage_used(self) -> int:
        """
        Get total storage used by all models in MB.
        
        Returns:
            Total size in MB
        """
        total_mb = 0
        
        for model_info in self.list_models():
            total_mb += model_info.file_size_mb
        
        return total_mb
    
    def _save_metadata(self, model_dir: Path, metadata: ModelInfo) -> None:
        """Save model metadata to JSON file."""
        metadata_path = model_dir / "metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.debug(f"Saved metadata: {metadata_path}")
    
    # ========== Ollama HuggingFace Integration ==========
    
    async def pull_ollama_model(
        self,
        model_name: str,
        ollama_url: str = "http://localhost:11434",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bool:
        """
        Pull a model using Ollama's pull endpoint with streaming progress.
        Supports both Ollama native models and HuggingFace models via hf.co/ format.
        
        Args:
            model_name: Full Ollama model name (e.g., "qwen2.5:14b" or "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
            ollama_url: Ollama API URL
            progress_callback: Optional async callback(status_dict) for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Pulling Ollama model: {model_name}")
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{ollama_url}/api/pull",
                    json={"name": model_name, "stream": True}
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        
                        try:
                            status = json.loads(line)
                            logger.debug(f"Pull status: {status}")
                            
                            if progress_callback:
                                if asyncio.iscoroutinefunction(progress_callback):
                                    await progress_callback(status)
                                else:
                                    progress_callback(status)
                            
                            # Check for completion
                            if status.get("status") == "success":
                                logger.info(f"Successfully pulled model: {model_name}")
                                return True
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in pull response: {line}")
                            continue
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def parse_hf_url(self, url: str) -> Optional[str]:
        """
        Parse a HuggingFace URL to extract repo_id.
        
        Args:
            url: HuggingFace URL or repo_id
            
        Returns:
            repo_id (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF") or None
        """
        url = url.strip()
        
        # Handle different URL formats
        if url.startswith("https://huggingface.co/"):
            repo_id = url.replace("https://huggingface.co/", "")
        elif url.startswith("hf.co/"):
            repo_id = url.replace("hf.co/", "")
        else:
            # Assume it's already a repo_id
            repo_id = url
        
        # Remove trailing slashes and query params
        repo_id = repo_id.split("?")[0].rstrip("/")
        
        # Validate format (should be username/reponame)
        if "/" not in repo_id:
            return None
            
        return repo_id
    
    async def get_hf_quantizations(self, repo_id: str) -> List[Dict[str, str]]:
        """
        Get available GGUF quantizations from a HuggingFace repo.
        
        Args:
            repo_id: HuggingFace repo (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
            
        Returns:
            List of dicts: [{"quant": "Q4_K_M", "filename": "..."}, ...]
        """
        try:
            hf_api_url = "https://huggingface.co/api"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{hf_api_url}/models/{repo_id}")
                response.raise_for_status()
                data = response.json()
                
                # Extract GGUF files from siblings array
                siblings = data.get("siblings", [])
                quantizations = []
                
                for sibling in siblings:
                    filename = sibling.get("rfilename", "")
                    if not filename.endswith(".gguf"):
                        continue
                    
                    # Extract quantization from filename
                    # Common patterns: "model-Q4_K_M.gguf", "model.Q4_K_M.gguf"
                    parts = filename.upper().replace(".GGUF", "").split("-")
                    quant = None
                    
                    for part in parts:
                        if any(q in part for q in ["Q4", "Q5", "Q6", "Q8", "IQ", "F16", "F32"]):
                            quant = part
                            break
                    
                    if quant:
                        quantizations.append({
                            "quant": quant,
                            "filename": filename
                        })
                
                return quantizations
                
        except Exception as e:
            logger.error(f"Failed to get HuggingFace quantizations for {repo_id}: {e}")
            return []
    
    def create_hf_model_name(self, repo_id: str, quantization: str) -> str:
        """
        Create an Ollama model name from HuggingFace repo and quantization.
        
        Args:
            repo_id: HuggingFace repo (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
            quantization: Quantization type (e.g., "Q4_K_M")
            
        Returns:
            Full Ollama model name (e.g., "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
        """
        return f"hf.co/{repo_id}:{quantization}"
