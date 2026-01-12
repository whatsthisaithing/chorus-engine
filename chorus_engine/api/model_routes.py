"""API routes for integrated LLM model management (Phase 10)."""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from chorus_engine.services.model_manager import ModelManager, ModelInfo
from chorus_engine.services.model_library import ModelLibrary, CuratedModel, QuantizationInfo
from chorus_engine.services.vram_estimator import VRAMEstimator, VRAMEstimate, GPUInfo
from chorus_engine.db.database import get_db
from chorus_engine.models.custom_model import DownloadedModel

logger = logging.getLogger(__name__)

# Initialize services
model_manager = ModelManager()
model_library = ModelLibrary()

# Download job tracking
download_jobs: Dict[str, Dict[str, Any]] = {}

# Create router
router = APIRouter(prefix="/api", tags=["models"])


# Request/Response models

class CuratedModelResponse(BaseModel):
    """Response model for curated model info."""
    id: str
    name: str
    description: str
    repo_id: str
    filename_template: str
    parameters: float
    context_window: int
    category: str
    tags: List[str]
    recommended_quant: Dict[str, str]
    tested: bool
    default: bool
    performance: Dict[str, str]
    quantizations: List[Dict[str, Any]]
    warning: Optional[str] = None


class DownloadedModelResponse(BaseModel):
    """Response model for downloaded model info."""
    model_id: str
    display_name: str
    repo_id: str
    filename: Optional[str] = None  # None for custom HF models
    quantization: str
    parameters: Optional[float] = None  # None for custom HF models
    context_window: Optional[int] = None  # None for custom HF models
    file_size_mb: Optional[int] = None  # None for custom HF models
    file_path: Optional[str] = None  # None for custom HF models
    downloaded_at: str
    last_used: Optional[str] = None
    custom: bool
    source: str
    tags: List[str]
    ollama_model_name: Optional[str] = None


class ModelDownloadRequest(BaseModel):
    """Request to download a model."""
    model_id: str = Field(..., description="Curated model ID")
    quantization: str = Field(..., description="Quantization to download (e.g., 'Q4_K_M')")
    display_name: Optional[str] = Field(None, description="Custom display name (optional)")


class ModelDownloadStatus(BaseModel):
    """Download job status."""
    job_id: str
    status: str  # 'pending', 'downloading', 'completed', 'failed'
    progress: Optional[float] = None
    current_size_mb: Optional[float] = None
    total_size_mb: Optional[float] = None
    error: Optional[str] = None
    model_id: Optional[str] = None
    file_path: Optional[str] = None


class GPUInfoResponse(BaseModel):
    """GPU information response."""
    gpus: List[Dict[str, Any]]
    total_vram_mb: int
    cuda_available: bool


class VRAMEstimateRequest(BaseModel):
    """Request for VRAM estimation."""
    model_id: str = Field(..., description="Curated model ID")
    quantization: str = Field(..., description="Quantization level")
    context_window: Optional[int] = Field(None, description="Override context window")
    n_gpu_layers: Optional[int] = Field(-1, description="Number of GPU layers (-1 = all)")


class VRAMEstimateResponse(BaseModel):
    """VRAM estimation response."""
    total_vram_mb: int
    model_weights_mb: int
    kv_cache_mb: int
    activation_mb: int
    overhead_mb: int
    will_fit: bool
    available_vram_mb: Optional[int]
    recommended_quantization: Optional[str] = None
    alternative_quantizations: List[str] = []


class SwitchModelRequest(BaseModel):
    """Request to switch active model."""
    model_path: str = Field(..., description="Absolute path to GGUF file")


# Routes

@router.get("/models/curated", response_model=List[CuratedModelResponse])
async def list_curated_models(
    category: Optional[str] = None,
    max_params: Optional[float] = None,
    tested_only: bool = False,
):
    """
    List pre-tested models from curated library.
    
    Args:
        category: Filter by category (balanced, creative, technical, advanced)
        max_params: Maximum model size in billions
        tested_only: Only include tested models
    """
    try:
        models = model_library.search_models(
            category=category,
            max_params=max_params,
            tested_only=tested_only,
            include_user=False
        )
        
        # Convert to response models
        response = []
        for model in models:
            response.append(CuratedModelResponse(
                id=model.id,
                name=model.name,
                description=model.description,
                repo_id=model.repo_id,
                filename_template=model.filename_template,
                parameters=model.parameters,
                context_window=model.context_window,
                category=model.category,
                tags=model.tags,
                recommended_quant=model.recommended_quant,
                tested=model.tested,
                default=model.default,
                performance=model.performance,
                quantizations=[
                    {
                        "quant": q.quant,
                        "filename": q.filename,
                        "file_size_mb": q.file_size_mb,
                        "min_vram_mb": q.min_vram_mb
                    }
                    for q in model.quantizations
                ],
                warning=model.warning
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list curated models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/downloaded", response_model=List[DownloadedModelResponse])
async def list_downloaded_models(db: Session = Depends(get_db)):
    """List all downloaded models from database."""
    try:
        # Query from database
        models = db.query(DownloadedModel).order_by(DownloadedModel.downloaded_at.desc()).all()
        
        # Convert to response models
        response = []
        for model in models:
            response.append(DownloadedModelResponse(
                model_id=model.model_id,
                display_name=model.display_name,
                repo_id=model.repo_id,
                filename=model.filename,
                quantization=model.quantization,
                parameters=model.parameters,
                context_window=model.context_window,
                file_size_mb=model.file_size_mb,
                file_path=model.file_path or "",
                downloaded_at=model.downloaded_at.isoformat() if model.downloaded_at else None,
                last_used=model.last_used.isoformat() if model.last_used else None,
                custom=model.source == 'custom_hf',
                source=model.source,
                tags=model.tags or [],
                ollama_model_name=model.ollama_model_name
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list downloaded models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/import-ollama")
async def import_model_to_ollama(model_id: str):
    """Import an already-downloaded model to Ollama."""
    try:
        # Get model metadata
        models = model_manager.list_models()
        model = next((m for m in models if m.model_id == model_id), None)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Generate Ollama model name (convert underscores to dashes for consistency)
        ollama_model_name = f"{model.model_id.replace('.', '-').replace('_', '-')}"
        
        # Check if Ollama is available
        ollama_available = await model_manager.check_ollama_availability()
        if not ollama_available:
            raise HTTPException(status_code=503, detail="Ollama is not running. Start Ollama and try again.")
        
        # Import to Ollama
        import_success = await model_manager.import_to_ollama(
            model_path=Path(model.file_path),
            model_name=ollama_model_name
        )
        
        if not import_success:
            raise HTTPException(status_code=500, detail="Failed to import model to Ollama")
        
        # Update metadata
        model_manager.update_ollama_model_name(model_id, ollama_model_name)
        
        return {"success": True, "ollama_model_name": ollama_model_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import model to Ollama: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/download", response_model=ModelDownloadStatus)
async def start_model_download(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks
):
    """
    Start downloading a model from HuggingFace.
    
    Returns a job_id to track progress.
    """
    try:
        # Get curated model info
        curated_model = model_library.get_model_info(request.model_id)
        if not curated_model:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found in curated library")
        
        # Find quantization info
        quant_info = None
        for q in curated_model.quantizations:
            if q.quant == request.quantization:
                quant_info = q
                break
        
        if not quant_info:
            raise HTTPException(
                status_code=400,
                detail=f"Quantization '{request.quantization}' not available for model '{request.model_id}'"
            )
        
        # Check if already downloaded
        existing_model_id = f"{request.model_id}-{request.quantization.lower()}"
        existing_path = model_manager.get_model_path(existing_model_id)
        if existing_path:
            return ModelDownloadStatus(
                job_id="already-downloaded",
                status="completed",
                model_id=existing_model_id,
                file_path=str(existing_path)
            )
        
        # Create download job
        job_id = str(uuid4())
        download_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "model_id": request.model_id,
            "quantization": request.quantization,
            "error": None
        }
        
        # Start background download
        background_tasks.add_task(
            _download_model_background,
            job_id,
            curated_model,
            quant_info,
            request.display_name
        )
        
        return ModelDownloadStatus(
            job_id=job_id,
            status="pending",
            progress=0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model download: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/download/{job_id}", response_model=ModelDownloadStatus)
async def get_download_status(job_id: str):
    """Get status of a download job."""
    if job_id == "already-downloaded":
        raise HTTPException(status_code=400, detail="Model already downloaded")
    
    job = download_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Download job '{job_id}' not found")
    
    return ModelDownloadStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        current_size_mb=job.get("current_size_mb"),
        total_size_mb=job.get("total_size_mb"),
        error=job.get("error"),
        model_id=job.get("completed_model_id"),
        file_path=job.get("file_path")
    )


@router.delete("/models")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """
    Remove a curated model from tracking (doesn't delete from Ollama).
    Curated models now use Ollama's hf.co/ format, so deletion just removes
    from Chorus Engine's database, not from Ollama's model store.
    """
    try:
        model = db.query(DownloadedModel).filter(
            DownloadedModel.model_id == model_id,
            DownloadedModel.source == 'curated'
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        db.delete(model)
        db.commit()
        
        logger.info(f"Removed curated model: {model_id}")
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/gpu", response_model=GPUInfoResponse)
async def get_gpu_info():
    """Get GPU information and available VRAM."""
    try:
        gpus = VRAMEstimator.detect_gpu_vram()
        
        cuda_available = len(gpus) > 0
        total_vram = sum(gpu.total_vram_mb for gpu in gpus)
        
        gpu_list = [
            {
                "name": gpu.name,
                "vram_mb": gpu.total_vram_mb,
                "vram_free_mb": gpu.available_vram_mb,
                "vram_used_mb": gpu.total_vram_mb - gpu.available_vram_mb,
                "gpu_id": gpu.id
            }
            for gpu in gpus
        ]
        
        return GPUInfoResponse(
            gpus=gpu_list,
            total_vram_mb=total_vram,
            cuda_available=cuda_available
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}", exc_info=True)
        # Return empty response if CUDA not available
        return GPUInfoResponse(
            gpus=[],
            total_vram_mb=0,
            cuda_available=False
        )


@router.get("/system/gpu/diagnostics")
async def get_gpu_diagnostics():
    """Get detailed GPU detection diagnostics for troubleshooting."""
    from chorus_engine.services.vram_estimator import HAS_PYNVML, NVML_ERROR
    import sys
    
    diagnostics = {
        "pynvml_imported": HAS_PYNVML,
        "import_error": NVML_ERROR,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
    }
    
    # Check for DLL files
    if sys.platform == 'win32':
        import os
        search_paths = [
            os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32'),
            os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'NVIDIA Corporation', 'NVSMI'),
            'C:\\Windows\\System32',
            'C:\\Program Files\\NVIDIA Corporation\\NVSMI',
        ]
        
        dll_files_found = []
        for path in search_paths:
            if os.path.exists(path):
                for dll_name in ['nvml.dll', 'nvml64.dll']:
                    dll_path = os.path.join(path, dll_name)
                    if os.path.exists(dll_path):
                        dll_files_found.append(dll_path)
        
        diagnostics["dll_search_paths"] = search_paths
        diagnostics["dll_files_found"] = dll_files_found
        diagnostics["system_path"] = os.environ.get('PATH', '').split(';')[:10]  # First 10 PATH entries
    
    return diagnostics


@router.post("/models/estimate", response_model=VRAMEstimateResponse)
async def estimate_vram(request: VRAMEstimateRequest):
    """Estimate VRAM requirements for a model."""
    try:
        # Get curated model info
        curated_model = model_library.get_model_info(request.model_id)
        if not curated_model:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # Find quantization info
        quant_info = None
        for q in curated_model.quantizations:
            if q.quant == request.quantization:
                quant_info = q
                break
        
        if not quant_info:
            raise HTTPException(
                status_code=400,
                detail=f"Quantization '{request.quantization}' not available"
            )
        
        # Use request context window or model default
        context_window = request.context_window or curated_model.context_window
        
        # Get VRAM estimation
        estimate = VRAMEstimator.estimate_with_recommendation(
            model_params_billions=curated_model.parameters,
            quantization=request.quantization,
            context_window=context_window,
            n_gpu_layers=request.n_gpu_layers or -1
        )
        
        # Get available VRAM
        gpus = VRAMEstimator.detect_gpu_vram()
        available_vram = sum(gpu.available_vram_mb for gpu in gpus) if gpus else None
        
        # Check if it will fit
        will_fit = available_vram >= estimate.estimated_vram_mb if available_vram else False
        
        # Get alternative quantizations if it doesn't fit
        alternatives = []
        if not will_fit and available_vram:
            recommended_quant = VRAMEstimator.recommend_quantization(
                model_params_billions=curated_model.parameters,
                context_window=context_window,
                available_vram_mb=available_vram,
                n_gpu_layers=request.n_gpu_layers or -1
            )
            if recommended_quant:
                alternatives = [recommended_quant]
        
        # Build response
        return VRAMEstimateResponse(
            total_vram_mb=estimate.estimated_vram_mb,
            model_weights_mb=estimate.breakdown.get('model_weights', 0),
            kv_cache_mb=estimate.breakdown.get('kv_cache', 0),
            activation_mb=estimate.breakdown.get('activation', 0),
            overhead_mb=estimate.breakdown.get('overhead', 0),
            will_fit=will_fit,
            available_vram_mb=available_vram,
            recommended_quantization=recommended_quant if (not will_fit and recommended_quant) else None,
            alternative_quantizations=alternatives
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to estimate VRAM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/status")
async def get_llm_status():
    """
    Get current LLM status: provider, model path, whether model is loaded.
    
    This is used to detect missing models and show warnings.
    """
    try:
        # Import config loader
        from chorus_engine.config.loader import ConfigLoader
        
        config_loader = ConfigLoader()
        config = config_loader.load_system_config()
        provider = config.llm.provider
        
        # Try to get app_state, but handle if not available yet
        llm_client = None
        try:
            from chorus_engine.api.app import app_state
            llm_client = app_state.get("llm_client")
        except (ImportError, KeyError, AttributeError):
            # App state not ready yet, that's ok
            pass
        
        # For integrated provider, check if model is actually loaded or available
        model_loaded = False
        model_path = None
        error_message = None
        
        if provider == "integrated":
            model_path = config.llm.model
            
            # Check if model file exists (ready for lazy loading)
            if model_path:
                model_file_exists = Path(model_path).exists()
                
                if not model_file_exists:
                    error_message = f"Model file not found: {model_path}"
                    model_loaded = False
                else:
                    # Model file exists - either loaded or ready to load
                    if llm_client and hasattr(llm_client, "is_loaded"):
                        # Check if already loaded, or if file exists (ready for lazy load)
                        model_loaded = llm_client.is_loaded() or (hasattr(llm_client, "model_exists") and llm_client.model_exists)
                    elif llm_client:
                        # If client exists but no is_loaded method, assume loaded
                        model_loaded = True
                    else:
                        # Client not initialized yet, but file exists = ready
                        model_loaded = True
            else:
                error_message = "No model path configured"
                model_loaded = False
        else:
            # For other providers, assume working if client exists
            model_loaded = llm_client is not None
        
        return {
            "provider": provider,
            "model_path": model_path,
            "model_loaded": model_loaded,
            "error": error_message
        }
        
    except Exception as e:
        logger.error(f"Failed to get LLM status: {e}", exc_info=True)
        return {
            "provider": "unknown",
            "model_path": None,
            "model_loaded": False,
            "error": str(e)
        }


@router.post("/llm/switch-model")
async def switch_model(request: SwitchModelRequest):
    """
    Switch the active LLM model (integrated provider only).
    
    This allows changing models without restarting the server.
    Only works with integrated provider.
    
    Note: This endpoint accesses app_state which is imported from app module.
    """
    try:
        # Import app_state from app module to avoid circular imports
        from chorus_engine.api.app import app_state
        
        llm_client = app_state.get("llm_client")
        if not llm_client:
            raise HTTPException(status_code=503, detail="LLM client not initialized")
        
        # Check if integrated provider
        if not hasattr(llm_client, "switch_model"):
            raise HTTPException(
                status_code=400,
                detail="Model switching only supported with integrated provider"
            )
        
        # Validate model path
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        if not model_path.suffix == ".gguf":
            raise HTTPException(status_code=400, detail="Model must be a GGUF file")
        
        # Switch model
        success = await llm_client.switch_model(str(model_path))
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to switch model")
        
        # Update app state
        app_state["current_model"] = str(model_path)
        
        return {
            "success": True,
            "message": f"Switched to model: {model_path.name}",
            "model_path": str(model_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks

async def _download_model_background(
    job_id: str,
    curated_model: CuratedModel,
    quant_info: QuantizationInfo,
    custom_display_name: Optional[str]
):
    """Background task to pull a curated model via Ollama (same as HF import)."""
    try:
        # Update status
        download_jobs[job_id]["status"] = "downloading"
        download_jobs[job_id]["total_size_mb"] = quant_info.file_size_mb
        download_jobs[job_id]["current_size_mb"] = 0
        
        # Generate Ollama model name using hf.co/ format
        # Format: hf.co/<repo_id>:<quantization>
        ollama_model_name = f"hf.co/{curated_model.repo_id}:{quant_info.quant}"
        model_id = ollama_model_name  # Use Ollama name as model_id
        display_name = custom_display_name or f"{curated_model.name} ({quant_info.quant})"
        
        download_jobs[job_id]["model_name"] = ollama_model_name
        
        logger.info(f"Starting Ollama pull for curated model: {ollama_model_name}")
        
        # Define progress callback
        def update_progress(status: Dict[str, Any]):
            download_jobs[job_id]["status"] = "downloading"
            
            # Update progress from Ollama status
            if "completed" in status and "total" in status:
                completed = status["completed"]
                total = status["total"]
                if total > 0:
                    progress_pct = (completed / total) * 100
                    current_mb = completed / (1024 * 1024)
                    download_jobs[job_id]["progress"] = progress_pct
                    download_jobs[job_id]["current_size_mb"] = current_mb
                    logger.debug(f"Ollama pull progress: {current_mb:.0f}MB ({progress_pct:.1f}%)")
        
        # Pull via Ollama with progress callback
        success = await model_manager.pull_ollama_model(
            model_name=ollama_model_name,
            progress_callback=update_progress
        )
        
        if not success:
            raise Exception("Ollama pull failed")
        
        # Mark as completed
        download_jobs[job_id]["status"] = "completed"
        download_jobs[job_id]["progress"] = 100
        download_jobs[job_id]["ollama_model_name"] = ollama_model_name
        
        logger.info(f"Curated model pull completed: {ollama_model_name}")
        
        # Save to database
        try:
            from chorus_engine.db.database import SessionLocal
            db = SessionLocal()
            try:
                # Check if already exists
                existing = db.query(DownloadedModel).filter(
                    DownloadedModel.model_id == model_id
                ).first()
                
                if not existing:
                    # Add to database
                    downloaded_model = DownloadedModel(
                        model_id=model_id,
                        display_name=display_name,
                        repo_id=curated_model.repo_id,
                        filename=quant_info.filename,  # Store for reference
                        quantization=quant_info.quant,
                        parameters=curated_model.parameters,
                        context_window=curated_model.context_window,
                        file_size_mb=quant_info.file_size_mb,
                        file_path=None,  # No local file path, managed by Ollama
                        ollama_model_name=ollama_model_name,
                        source='curated',
                        tags=curated_model.tags
                    )
                    db.add(downloaded_model)
                    db.commit()
                    logger.info(f"Saved curated model to database: {model_id}")
                else:
                    logger.info(f"Curated model already exists in database: {model_id}")
            finally:
                db.close()
        except Exception as db_error:
            logger.error(f"Failed to save curated model to database: {db_error}", exc_info=True)
            # Don't fail the download if DB save fails
        
    except Exception as e:
        logger.error(f"Curated model pull failed: {e}", exc_info=True)
        download_jobs[job_id]["status"] = "failed"
        download_jobs[job_id]["error"] = str(e)


# ========== Ollama HuggingFace Integration ==========

class HFModelPullRequest(BaseModel):
    """Request to pull a HuggingFace model via Ollama."""
    hf_url: str = Field(..., description="HuggingFace repository URL or repo_id")
    quantization: str = Field(..., description="Quantization to pull (e.g., 'Q4_K_M')")


class HFQuantizationsResponse(BaseModel):
    """Response with available quantizations."""
    repo_id: str
    quantizations: List[Dict[str, str]]


@router.get("/models/hf-quantizations")
async def get_hf_quantizations(hf_url: str):
    """
    Get available quantizations for a HuggingFace GGUF repository.
    
    Args:
        hf_url: HuggingFace URL or repo_id
    """
    try:
        # Parse URL to get repo_id
        repo_id = model_manager.parse_hf_url(hf_url)
        if not repo_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid HuggingFace URL. Expected format: https://huggingface.co/username/reponame"
            )
        
        # Get available quantizations
        quantizations = await model_manager.get_hf_quantizations(repo_id)
        
        if not quantizations:
            raise HTTPException(
                status_code=404,
                detail=f"No GGUF files found in repository: {repo_id}"
            )
        
        return HFQuantizationsResponse(
            repo_id=repo_id,
            quantizations=quantizations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get HF quantizations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/pull-hf")
async def pull_hf_model(request: HFModelPullRequest):
    """
    Pull a HuggingFace GGUF model via Ollama's hf.co/ integration.
    Returns immediately with a job ID for progress tracking.
    """
    try:
        # Parse HF URL
        repo_id = model_manager.parse_hf_url(request.hf_url)
        if not repo_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid HuggingFace URL"
            )
        
        # Create Ollama model name
        model_name = model_manager.create_hf_model_name(repo_id, request.quantization)
        
        # Create pull job
        job_id = str(uuid4())
        download_jobs[job_id] = {
            "job_id": job_id,
            "status": "pulling",
            "progress": 0.0,
            "model_name": model_name,
            "repo_id": repo_id,
            "quantization": request.quantization,
            "error": None
        }
        
        # Start pull in background
        asyncio.create_task(_pull_hf_model_background(job_id, model_name))
        
        return {"job_id": job_id, "model_name": model_name, "status": "pulling"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start HF model pull: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _pull_hf_model_background(job_id: str, model_name: str):
    """Background task to pull HF model via Ollama."""
    from chorus_engine.db.database import SessionLocal
    
    try:
        logger.info(f"Starting HF model pull: {model_name}")
        
        async def progress_callback(status: Dict[str, Any]):
            """Update job status with Ollama pull progress."""
            # Ollama returns: {"status": "pulling", "digest": "...", "total": ..., "completed": ...}
            download_jobs[job_id]["last_status"] = status.get("status", "pulling")
            
            if "total" in status and "completed" in status:
                total = status["total"]
                completed = status["completed"]
                if total > 0:
                    progress = (completed / total) * 100
                    download_jobs[job_id]["progress"] = progress
                    download_jobs[job_id]["current_size_mb"] = completed / (1024 * 1024)
                    download_jobs[job_id]["total_size_mb"] = total / (1024 * 1024)
        
        # Pull the model
        success = await model_manager.pull_ollama_model(
            model_name=model_name,
            progress_callback=progress_callback
        )
        
        if success:
            download_jobs[job_id]["status"] = "completed"
            download_jobs[job_id]["progress"] = 100.0
            logger.info(f"HF model pull completed: {model_name}")
            
            # Automatically save to database
            try:
                db = SessionLocal()
                try:
                    # Check if already exists
                    existing = db.query(DownloadedModel).filter(
                        DownloadedModel.model_id == model_name
                    ).first()
                    
                    if not existing:
                        # Add to database
                        downloaded_model = DownloadedModel(
                            model_id=model_name,
                            display_name=model_name,
                            repo_id=download_jobs[job_id].get("repo_id", "unknown"),
                            quantization=download_jobs[job_id].get("quantization", "unknown"),
                            ollama_model_name=model_name,
                            source='custom_hf'
                        )
                        db.add(downloaded_model)
                        db.commit()
                        logger.info(f"Auto-saved custom model to database: {model_name}")
                finally:
                    db.close()
            except Exception as db_error:
                logger.error(f"Failed to save custom model to database: {db_error}", exc_info=True)
                # Don't fail the whole pull if DB save fails
            
        else:
            download_jobs[job_id]["status"] = "failed"
            download_jobs[job_id]["error"] = "Pull failed - check Ollama logs"
            logger.error(f"HF model pull failed: {model_name}")
        
    except Exception as e:
        logger.error(f"HF model pull failed: {e}", exc_info=True)
        download_jobs[job_id]["status"] = "failed"
        download_jobs[job_id]["error"] = str(e)


# ==================== Custom Model Management ====================

@router.get("/models/custom", response_model=List[DownloadedModelResponse])
async def get_custom_models(db: Session = Depends(get_db)):
    """Get all custom HF models."""
    try:
        models = db.query(DownloadedModel).filter(
            DownloadedModel.source == 'custom_hf'
        ).order_by(DownloadedModel.downloaded_at.desc()).all()
        
        response = []
        for model in models:
            response.append(DownloadedModelResponse(
                model_id=model.model_id,
                display_name=model.display_name,
                repo_id=model.repo_id,
                filename=model.filename,
                quantization=model.quantization,
                parameters=model.parameters,
                context_window=model.context_window,
                file_size_mb=model.file_size_mb,
                file_path=model.file_path or "",
                downloaded_at=model.downloaded_at.isoformat() if model.downloaded_at else None,
                last_used=model.last_used.isoformat() if model.last_used else None,
                custom=True,
                source=model.source,
                tags=model.tags or [],
                ollama_model_name=model.ollama_model_name
            ))
        return response
    except Exception as e:
        logger.error(f"Failed to get custom models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/custom")
async def delete_custom_model(model_id: str, db: Session = Depends(get_db)):
    """Remove a custom model from tracking (doesn't delete from Ollama)."""
    try:
        logger.info(f"Attempting to delete custom model with ID: {model_id}")
        
        model = db.query(DownloadedModel).filter(
            DownloadedModel.model_id == model_id,
            DownloadedModel.source == 'custom_hf'
        ).first()
        
        if not model:
            logger.warning(f"Model not found with ID: {model_id}")
            raise HTTPException(status_code=404, detail="Model not found")
        
        db.delete(model)
        db.commit()
        
        logger.info(f"Removed custom model: {model_id}")
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete custom model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
