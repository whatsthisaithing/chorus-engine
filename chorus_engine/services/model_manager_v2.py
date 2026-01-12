"""Model management service using Ollama's HuggingFace integration."""

import asyncio
import json
import logging
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Information about a model (Ollama-native or HuggingFace)."""
    model_name: str  # Full Ollama name: "qwen2.5:14b" or "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
    display_name: str
    source: str  # "ollama" or "huggingface"
    repo_id: Optional[str] = None  # HuggingFace repo (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
    quantization: Optional[str] = None
    parameters_billions: Optional[float] = None
    context_window: Optional[int] = None
    estimated_size_gb: Optional[float] = None
    tags: List[str] = None
    custom: bool = False
    added_at: Optional[str] = None
    last_used: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelManager:
    """
    Manages Ollama models using native Ollama + HuggingFace integration.
    
    Features:
    - List Ollama-native models
    - Add HuggingFace GGUF models via hf.co/ format
    - Pull models with progress tracking
    - Query HuggingFace API for available quantizations
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize model manager."""
        self.ollama_url = ollama_url
        self.hf_api_url = "https://huggingface.co/api"
        logger.info(f"ModelManager initialized (Ollama: {ollama_url})")
    
    async def list_ollama_models(self) -> List[Dict[str, Any]]:
        """List all models currently in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/show",
                    json={"name": model_name}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    async def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Pull a model using Ollama's pull endpoint with streaming progress.
        
        Args:
            model_name: Full Ollama model name (e.g., "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
            progress_callback: Optional callback(status_dict) for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Pulling model: {model_name}")
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/pull",
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
                                await progress_callback(status)
                            
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
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.ollama_url}/api/delete",
                    json={"name": model_name}
                )
                response.raise_for_status()
                logger.info(f"Deleted model: {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    async def get_hf_repo_files(self, repo_id: str) -> List[str]:
        """
        Get list of files in a HuggingFace repository.
        
        Args:
            repo_id: HuggingFace repo (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
            
        Returns:
            List of filenames
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.hf_api_url}/models/{repo_id}")
                response.raise_for_status()
                data = response.json()
                
                # Extract filenames from siblings array
                siblings = data.get("siblings", [])
                return [s["rfilename"] for s in siblings]
                
        except Exception as e:
            logger.error(f"Failed to get HuggingFace repo files for {repo_id}: {e}")
            return []
    
    def parse_hf_url(self, url: str) -> Optional[str]:
        """
        Parse a HuggingFace URL to extract repo_id.
        
        Args:
            url: HuggingFace URL (e.g., "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
            
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
    
    async def get_available_quantizations(self, repo_id: str) -> List[Dict[str, Any]]:
        """
        Get available GGUF quantizations from a HuggingFace repo.
        
        Args:
            repo_id: HuggingFace repo (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
            
        Returns:
            List of dicts with quantization info: [{"quant": "Q4_K_M", "filename": "...", "size_mb": ...}, ...]
        """
        files = await self.get_hf_repo_files(repo_id)
        
        quantizations = []
        for filename in files:
            if not filename.endswith(".gguf"):
                continue
            
            # Try to extract quantization from filename
            # Common patterns: "model-Q4_K_M.gguf", "model.Q4_K_M.gguf", "Q4_K_M.gguf"
            parts = filename.upper().replace(".GGUF", "").split("-")
            quant = None
            
            for part in parts:
                if any(q in part for q in ["Q4", "Q5", "Q6", "Q8", "IQ", "F16", "F32"]):
                    quant = part.replace("_", "_")  # Normalize underscores
                    break
            
            if not quant:
                # Try extracting from last part before .gguf
                last_part = filename.rsplit(".", 1)[0].split("-")[-1]
                if any(q in last_part.upper() for q in ["Q4", "Q5", "Q6", "Q8", "IQ", "F16", "F32"]):
                    quant = last_part.upper()
            
            if quant:
                quantizations.append({
                    "quant": quant,
                    "filename": filename,
                    "size_mb": None  # Could fetch from file metadata if needed
                })
        
        return quantizations
    
    def create_ollama_model_name(self, repo_id: str, quantization: str) -> str:
        """
        Create an Ollama model name from HuggingFace repo and quantization.
        
        Args:
            repo_id: HuggingFace repo (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
            quantization: Quantization type (e.g., "Q4_K_M")
            
        Returns:
            Full Ollama model name (e.g., "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
        """
        return f"hf.co/{repo_id}:{quantization}"
    
    async def check_ollama_availability(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
