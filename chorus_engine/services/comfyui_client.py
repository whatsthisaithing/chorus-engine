"""
ComfyUI API client for image generation.

Phase 5: Handles communication with ComfyUI server for workflow submission,
status polling, and result retrieval.
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Base exception for ComfyUI-related errors."""
    pass


class ComfyUIConnectionError(ComfyUIError):
    """ComfyUI server is not reachable."""
    pass


class ComfyUIJobError(ComfyUIError):
    """Job submission or execution failed."""
    pass


class ComfyUIClient:
    """
    Client for interacting with ComfyUI API.
    
    Handles:
    - Workflow submission
    - Job status polling
    - Result retrieval
    - Health checks
    - Job cancellation
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8188",
        timeout: float = 300.0,
        poll_interval: float = 2.0
    ):
        """
        Initialize ComfyUI client.
        
        Args:
            base_url: ComfyUI server URL
            timeout: Maximum time to wait for generation (seconds)
            poll_interval: Time between status checks (seconds)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client = httpx.AsyncClient(timeout=30.0)  # HTTP request timeout
        
        logger.info(f"ComfyUI client initialized: {self.base_url}")
    
    async def health_check(self) -> bool:
        """
        Check if ComfyUI server is reachable.
        
        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/system_stats")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"ComfyUI health check failed: {e}")
            return False
    
    async def submit_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """
        Submit a workflow to ComfyUI for generation.
        
        Args:
            workflow_data: Complete workflow JSON with prompts injected
        
        Returns:
            Job ID (prompt_id) for tracking
        
        Raises:
            ComfyUIConnectionError: Server not reachable
            ComfyUIJobError: Job submission failed
        """
        # Generate a client_id for this request
        client_id = str(uuid.uuid4())
        
        # Prepare the prompt request
        payload = {
            "prompt": workflow_data,
            "client_id": client_id
        }
        
        try:
            logger.debug(f"Submitting workflow to ComfyUI (client_id: {client_id})")
            response = await self.client.post(
                f"{self.base_url}/prompt",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if not prompt_id:
                raise ComfyUIJobError("No prompt_id returned from ComfyUI")
            
            logger.info(f"Workflow submitted successfully: {prompt_id}")
            return prompt_id
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to ComfyUI: {e}")
            raise ComfyUIConnectionError(f"ComfyUI server not reachable at {self.base_url}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"ComfyUI returned error: {e.response.status_code} - {e.response.text}")
            raise ComfyUIJobError(f"Workflow submission failed: {e.response.text}")
        
        except Exception as e:
            logger.error(f"Unexpected error submitting workflow: {e}")
            raise ComfyUIJobError(f"Workflow submission failed: {str(e)}")
    
    async def check_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a running job.
        
        Args:
            job_id: Job ID (prompt_id) from submit_workflow
        
        Returns:
            Status dictionary with keys:
                - status: "queued" | "running" | "completed" | "failed"
                - progress: Float 0.0-1.0 (if available)
                - current_node: String node name (if available)
                - error: Error message (if failed)
        
        Raises:
            ComfyUIConnectionError: Server not reachable
        """
        try:
            response = await self.client.get(f"{self.base_url}/history/{job_id}")
            response.raise_for_status()
            
            history = response.json()
            
            # If job_id not in history, it might still be queued
            if job_id not in history:
                return {
                    "status": "queued",
                    "progress": 0.0,
                    "current_node": None,
                    "error": None
                }
            
            job_data = history[job_id]
            
            # Check if job completed
            if "outputs" in job_data:
                return {
                    "status": "completed",
                    "progress": 1.0,
                    "current_node": None,
                    "error": None,
                    "outputs": job_data["outputs"]
                }
            
            # Check for errors
            if "status" in job_data:
                status_info = job_data["status"]
                if status_info.get("status_str") == "error":
                    error_msg = status_info.get("messages", ["Unknown error"])[0]
                    return {
                        "status": "failed",
                        "progress": 0.0,
                        "current_node": None,
                        "error": error_msg
                    }
            
            # Job is running
            return {
                "status": "running",
                "progress": 0.5,  # Approximate, ComfyUI doesn't provide fine-grained progress
                "current_node": None,
                "error": None
            }
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to ComfyUI: {e}")
            raise ComfyUIConnectionError(f"ComfyUI server not reachable at {self.base_url}")
        
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return {
                "status": "unknown",
                "progress": 0.0,
                "current_node": None,
                "error": str(e)
            }
    
    async def get_result(self, job_id: str, output_node: str = "SaveImage") -> Optional[bytes]:
        """
        Retrieve the generated image from a completed job.
        
        Args:
            job_id: Job ID (prompt_id)
            output_node: Name of the output node (default: SaveImage)
        
        Returns:
            Image bytes, or None if not found
        
        Raises:
            ComfyUIConnectionError: Server not reachable
            ComfyUIJobError: Job not completed or image not found
        """
        try:
            # Get job history to find output files
            response = await self.client.get(f"{self.base_url}/history/{job_id}")
            response.raise_for_status()
            
            history = response.json()
            
            if job_id not in history:
                raise ComfyUIJobError(f"Job {job_id} not found in history")
            
            job_data = history[job_id]
            
            if "outputs" not in job_data:
                raise ComfyUIJobError(f"Job {job_id} has no outputs (may not be completed)")
            
            # Find the output image
            outputs = job_data["outputs"]
            
            # Look for SaveImage node output
            image_info = None
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    image_info = node_output["images"][0]  # Get first image
                    break
            
            if not image_info:
                raise ComfyUIJobError(f"No image found in job {job_id} outputs")
            
            # Download the image
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")
            image_type = image_info.get("type", "output")
            
            # Build download URL
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": image_type
            }
            
            logger.debug(f"Downloading image: {filename} from subfolder: {subfolder}")
            
            response = await self.client.get(
                f"{self.base_url}/view",
                params=params
            )
            response.raise_for_status()
            
            logger.info(f"Successfully retrieved image for job {job_id}")
            return response.content
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to ComfyUI: {e}")
            raise ComfyUIConnectionError(f"ComfyUI server not reachable at {self.base_url}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Error retrieving image: {e.response.status_code}")
            raise ComfyUIJobError(f"Failed to retrieve image: {e.response.text}")
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving image: {e}")
            raise ComfyUIJobError(f"Failed to retrieve image: {str(e)}")
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Attempt to cancel a running job.
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            True if cancellation successful, False otherwise
        
        Note:
            ComfyUI doesn't have a direct cancel API, so this is best-effort
        """
        logger.warning(f"Cancel requested for job {job_id}, but ComfyUI has no cancel API")
        return False
    
    async def wait_for_completion(
        self,
        job_id: str,
        callback=None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete, polling status periodically.
        
        Args:
            job_id: Job ID to wait for
            callback: Optional async callback(status_dict) called on each poll
        
        Returns:
            Final status dictionary
        
        Raises:
            ComfyUIJobError: Job failed or timed out
            ComfyUIConnectionError: Server not reachable
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.timeout:
                raise ComfyUIJobError(f"Job {job_id} timed out after {self.timeout}s")
            
            # Check status
            status = await self.check_status(job_id)
            
            # Call callback if provided
            if callback:
                await callback(status)
            
            # Check if job is done
            if status["status"] == "completed":
                logger.info(f"Job {job_id} completed in {elapsed:.1f}s")
                return status
            
            elif status["status"] == "failed":
                error_msg = status.get("error", "Unknown error")
                raise ComfyUIJobError(f"Job {job_id} failed: {error_msg}")
            
            # Wait before polling again
            await asyncio.sleep(self.poll_interval)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.debug("ComfyUI client closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
