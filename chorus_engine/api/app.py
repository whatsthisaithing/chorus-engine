"""FastAPI application and routes."""

import logging
import asyncio
import subprocess
import json
import yaml
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from chorus_engine.config import ConfigLoader, SystemConfig, CharacterConfig
from chorus_engine.config import IMMUTABLE_CHARACTERS
from chorus_engine.llm import create_llm_client, LLMError
from chorus_engine.db import get_db, init_db
from chorus_engine.models import Conversation, Thread, Message, Memory, MessageRole, MemoryType, ConversationSummary
from chorus_engine.repositories import (
    ConversationRepository,
    ThreadRepository,
    MessageRepository,
    MemoryRepository,
)
from chorus_engine.services.core_memory_loader import CoreMemoryLoader
from chorus_engine.services.prompt_assembly import PromptAssemblyService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.system_prompt_generator import SystemPromptGenerator
from chorus_engine.services.memory_extraction import MemoryExtractionService
# Phase 7: Background extraction worker removed - replaced by intent detection system
# from chorus_engine.services.background_extraction import BackgroundExtractionManager
from chorus_engine.db.vector_store import VectorStore
from pathlib import Path

# Phase 5 imports
from chorus_engine.services.comfyui_client import ComfyUIClient
from chorus_engine.services.workflow_manager import WorkflowManager
from chorus_engine.services.image_prompt_service import ImagePromptService
from chorus_engine.services.image_storage import ImageStorageService
from chorus_engine.services.image_generation_orchestrator import ImageGenerationOrchestrator
from chorus_engine.repositories.image_repository import ImageRepository

# Phase 6 imports
from chorus_engine.services.audio_generation_orchestrator import AudioGenerationOrchestrator
from chorus_engine.services.audio_preprocessing import AudioPreprocessingService
from chorus_engine.services.audio_storage import AudioStorageService
from chorus_engine.repositories.voice_sample_repository import VoiceSampleRepository
from chorus_engine.services.tts.provider_factory import TTSProviderFactory
from chorus_engine.services.tts.tts_service import TTSService

# Phase 8 imports
from chorus_engine.services.conversation_export_service import ConversationExportService
from chorus_engine.services.conversation_analysis_service import ConversationAnalysisService
from chorus_engine.services.memory_profile_service import MemoryProfileService
from chorus_engine.repositories.audio_repository import AudioRepository

# Debugging imports
from chorus_engine.utils.debug_logger import log_llm_call

# Phase 7 imports
from chorus_engine.services.intent_detection_service import IntentDetectionService, IntentResult

logger = logging.getLogger(__name__)
# Set level to DEBUG for our app logger, let it propagate to parent handlers
logger.setLevel(logging.DEBUG)
logger.debug(f"[STARTUP] Logger '{__name__}' configured: level={logger.level}, handlers={len(logger.handlers)}, propagate={logger.propagate}")


def _strip_image_tags(text: str) -> str:
    """
    Strip markdown and HTML image tags from text.
    
    Used to clean up LLM responses when generating images, since the LLM
    sometimes adds image tags despite instructions not to.
    
    Removes:
    - Markdown images: ![alt](url)
    - HTML images: <img src="url" ... />
    """
    import re
    
    # Remove markdown image syntax: ![anything](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    
    # Remove HTML img tags: <img ... />  or <img ...>
    text = re.sub(r'<img[^>]*/?>', '', text, flags=re.IGNORECASE)
    
    return text


# Global state
app_state = {
    "system_config": None,
    "characters": {},
    "llm_client": None,
    "vector_store": None,
    "embedding_service": None,
    "extraction_service": None,  # Phase 4.1
    # "extraction_manager": None,  # Phase 4.1 - REMOVED in Phase 7
    "image_orchestrator": None,  # Phase 5
    "comfyui_client": None,  # Phase 5
    "current_model": None,  # Track currently loaded model for VRAM management
    "comfyui_lock": asyncio.Lock(),  # Phase 6: Prevent concurrent ComfyUI operations
    "audio_orchestrator": None,  # Phase 6
    "intent_detection_service": None,  # Phase 7
    "analysis_service": None,  # Phase 8
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    logger.info("Starting Chorus Engine...")
    
    # Logging is already configured by main.py - no need to reconfigure here
    
    try:
        # Initialize database
        init_db()
        logger.info("✓ Database initialized")
        
        # Load configuration
        loader = ConfigLoader()
        system_config = loader.load_system_config()
        characters = loader.load_all_characters()
        
        # Initialize debug logger with system config debug flag
        from chorus_engine.utils.debug_logger import initialize_debug_logger
        initialize_debug_logger(enabled=system_config.debug)
        logger.info(f"Debug logging: {'enabled' if system_config.debug else 'disabled'}")
        
        if not characters:
            logger.warning("No characters loaded - you'll need to add characters to use the system")
        
        # Note: Workflow configs are now managed in database and fetched dynamically when needed
        # (Not stored in character config objects anymore - Phase 5 change)
        
        # Initialize LLM client (provider-agnostic factory)
        llm_client = create_llm_client(system_config.llm)
        
        # Check LLM availability
        llm_available = await llm_client.health_check()
        if llm_available:
            logger.info(f"✓ Connected to LLM: {system_config.llm.model}")
        else:
            logger.warning(f"⚠ LLM not available at {system_config.llm.base_url}")
        
        # Initialize Phase 3 services
        vector_store = VectorStore(Path("data/vector_store"))
        embedding_service = EmbeddingService()
        
        # Phase 7.5: Fast keyword-based intent detection (no VRAM overhead)
        from chorus_engine.services.keyword_intent_detection import KeywordIntentDetector
        keyword_detector = KeywordIntentDetector()
        logger.info("✓ Keyword-based intent detection initialized (no LLM, instant)")
        
        # Initialize persistent DB session for background services
        # Note: This session stays open for the lifetime of the application
        db_session = next(get_db())
        
        # Load core memories for all characters
        core_memory_loader = CoreMemoryLoader(db_session)
        for character_id in characters.keys():
            try:
                loaded_count = core_memory_loader.load_character_core_memories(character_id)
                logger.info(f"✓ Loaded {loaded_count} core memories for {character_id}")
            except Exception as e:
                logger.warning(f"Could not load core memories for {character_id}: {e}")
        
        # Phase 4.1: Initialize memory extraction services
        memory_repo = MemoryRepository(db_session)
        conversation_repo = ConversationRepository(db_session)
        
        extraction_service = MemoryExtractionService(
            llm_client=llm_client,
            memory_repository=memory_repo,
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
        # Create LLM usage lock early for coordination between background tasks and image generation
        llm_usage_lock = asyncio.Lock()
        
        # Phase 7.5: Background memory extraction worker (async, using character's model)
        from chorus_engine.services.background_memory_extractor import BackgroundMemoryExtractor
        background_extractor = BackgroundMemoryExtractor(
            llm_client=llm_client,
            extraction_service=extraction_service,
            temperature=0.2,
            llm_usage_lock=llm_usage_lock
        )
        
        # Start background extraction worker
        await background_extractor.start()
        logger.info("✓ Background memory extraction worker started (async, uses character model)")
        
        # Phase 8: Initialize conversation analysis service
        analysis_service = ConversationAnalysisService(
            db=db_session,
            llm_client=llm_client,
            vector_store=vector_store,
            embedding_service=embedding_service,
            temperature=0.1
        )
        logger.info("✓ Conversation analysis service initialized")
        
        # Phase 5: Initialize image generation services
        comfyui_client = None
        image_orchestrator = None
        workflow_manager = None
        
        if system_config.comfyui.enabled:
            try:
                comfyui_client = ComfyUIClient(
                    base_url=system_config.comfyui.url,
                    timeout=system_config.comfyui.timeout_seconds,
                    poll_interval=system_config.comfyui.polling_interval_seconds
                )
                
                # Check ComfyUI availability
                comfyui_available = await comfyui_client.health_check()
                if comfyui_available:
                    logger.info("✓ Connected to ComfyUI")
                    
                    # Initialize workflow manager (shared by image and audio)
                    workflow_manager = WorkflowManager(
                        workflows_dir=system_config.paths.workflows
                    )
                    
                    image_prompt_service = ImagePromptService(
                        llm_client=llm_client
                    )
                    
                    image_storage_service = ImageStorageService(
                        base_path=Path("data/images")
                    )
                    
                    image_orchestrator = ImageGenerationOrchestrator(
                        system_config=system_config,
                        comfyui_client=comfyui_client,
                        workflow_manager=workflow_manager,
                        prompt_service=image_prompt_service,
                        storage_service=image_storage_service
                    )
                    
                    logger.info("✓ Image generation services initialized")
                else:
                    logger.warning("⚠ ComfyUI not available - image generation disabled")
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize image generation: {e}")
        else:
            logger.info("Image generation disabled in config")
        
        # Phase 6: Initialize audio generation services
        audio_orchestrator = None
        audio_storage = None
        
        if system_config.comfyui.enabled and comfyui_client and workflow_manager:
            try:
                # Initialize audio services
                audio_preprocessing = AudioPreprocessingService()
                audio_storage = AudioStorageService()
                
                # Initialize repositories with the persistent db_session
                voice_sample_repo = VoiceSampleRepository(db_session)
                audio_repo = AudioRepository(db_session)
                
                # Initialize audio orchestrator with correct parameter names
                audio_orchestrator = AudioGenerationOrchestrator(
                    comfyui_client=comfyui_client,
                    workflow_manager=workflow_manager,
                    preprocessing_service=audio_preprocessing,
                    storage_service=audio_storage
                )
                
                logger.info("✓ Audio generation services initialized (legacy orchestrator)")
                
                # Initialize TTS provider system
                # This supports both ComfyUI workflows and embedded Chatterbox TTS
                comfyui_lock = asyncio.Lock()  # Lock for ComfyUI to prevent concurrent jobs
                # Note: llm_usage_lock created earlier and passed to BackgroundMemoryExtractor
                
                TTSProviderFactory.initialize_providers(
                    comfyui_client=comfyui_client,
                    comfyui_lock=comfyui_lock,
                    workflow_manager=workflow_manager,
                    audio_storage=audio_storage,
                    system_config=system_config
                )
                
                available_providers = TTSProviderFactory.get_available_providers()
                if available_providers:
                    logger.info(f"✓ TTS providers initialized: {list(available_providers.keys())}")
                else:
                    logger.warning("⚠ No TTS providers available")
                    
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize audio generation: {e}")
        else:
            logger.info("Audio generation disabled (requires ComfyUI)")
        
        
        # Store in app state
        app_state["system_config"] = system_config
        app_state["characters"] = characters
        app_state["llm_client"] = llm_client
        app_state["vector_store"] = vector_store
        app_state["embedding_service"] = embedding_service
        app_state["extraction_service"] = extraction_service
        app_state["keyword_detector"] = keyword_detector  # Phase 7.5: Keyword-based intent detection
        app_state["background_extractor"] = background_extractor  # Phase 7.5: Async memory extraction
        app_state["llm_usage_lock"] = llm_usage_lock  # Coordination lock for LLM access
        app_state["image_orchestrator"] = image_orchestrator
        app_state["comfyui_client"] = comfyui_client
        app_state["audio_orchestrator"] = audio_orchestrator
        app_state["audio_storage"] = audio_storage
        app_state["analysis_service"] = analysis_service  # Phase 8
        app_state["db_session"] = db_session  # Store for cleanup on shutdown
        
        # Initialize title generation service
        from chorus_engine.services.title_generation import TitleGenerationService
        title_service = TitleGenerationService(llm_client=llm_client)
        app_state["title_service"] = title_service
        logger.info("✓ Title generation service initialized")
        
        logger.info(f"✓ Chorus Engine ready with {len(characters)} character(s)")
        
    except Exception as e:
        logger.error(f"Failed to start Chorus Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chorus Engine...")
    
    # Phase 7.5: Stop background extraction worker
    if app_state.get("background_extractor"):
        await app_state["background_extractor"].stop()
        logger.info("✓ Background extraction stopped")
    
    # Close persistent database session
    if app_state.get("db_session"):
        app_state["db_session"].close()
        logger.info("✓ Database session closed")
    
    if app_state["llm_client"]:
        await app_state["llm_client"].close()


async def get_gpu_vram_usage() -> Optional[int]:
    """
    Get actual GPU VRAM usage in MB using nvidia-smi.
    Returns None if nvidia-smi not available or fails.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


async def unload_all_models_except(keep_model: Optional[str] = None):
    """
    Aggressively unload all models currently loaded in Ollama, optionally keeping one.
    Uses actual generation requests with keep_alive=0 to force unload, then verifies.
    
    Args:
        keep_model: Model name to keep loaded (all others will be unloaded)
    """
    llm_client = app_state.get("llm_client")
    if not llm_client:
        return
    
    try:
        # Query Ollama for currently loaded models
        response = await llm_client.client.get(f"{llm_client.base_url}/api/ps")
        ps_data = response.json()
        loaded_models = [m["name"] for m in ps_data.get("models", [])]
        
        logger.info(f"[VRAM CLEANUP] Ollama reports {len(loaded_models)} loaded models: {loaded_models}")
        
        # Unload all except the one we want to keep
        for model_name in loaded_models:
            if keep_model and model_name == keep_model:
                logger.info(f"[VRAM CLEANUP] Keeping {model_name} loaded")
                continue
            
            logger.info(f"[VRAM CLEANUP] Force unloading {model_name} with empty generation request")
            try:
                # Send actual generation request with keep_alive=0 to force processing
                await llm_client.client.post(
                    f"{llm_client.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",  # Empty prompt
                        "keep_alive": 0  # Force unload after this request
                    }
                )
                logger.info(f"[VRAM CLEANUP SUCCESS] Force unloaded {model_name}")
            except Exception as e:
                logger.warning(f"[VRAM CLEANUP FAILED] Could not unload {model_name}: {e}")
        
        # Verify unload with polling (wait up to 5 seconds)
        logger.info("[VRAM CLEANUP] Verifying unload by polling ps endpoint...")
        for attempt in range(10):
            await asyncio.sleep(0.5)
            verify_response = await llm_client.client.get(f"{llm_client.base_url}/api/ps")
            verify_data = verify_response.json()
            remaining_models = [m["name"] for m in verify_data.get("models", [])]
            
            # Check if only our keep_model remains (or nothing if no keep_model)
            expected_count = 1 if keep_model else 0
            if len(remaining_models) <= expected_count:
                if keep_model and keep_model not in remaining_models and len(remaining_models) > 0:
                    # Wrong model loaded
                    logger.warning(f"[VRAM CLEANUP] Unexpected models still loaded: {remaining_models}")
                    continue
                logger.info(f"[VRAM CLEANUP VERIFIED] Ollama VRAM cleared after {(attempt + 1) * 0.5}s - {len(remaining_models)} models remain: {remaining_models}")
                break
            else:
                logger.debug(f"[VRAM CLEANUP] Polling attempt {attempt + 1}/10: {len(remaining_models)} models still loaded: {remaining_models}")
        else:
            # Loop completed without break - models still loaded
            logger.warning(f"[VRAM CLEANUP] Verification timeout - {len(remaining_models)} models still reported loaded: {remaining_models}")
                
    except Exception as e:
        logger.warning(f"[VRAM CLEANUP] Failed to query/unload models: {e}")


async def preload_model(model: str, character_id: str):
    """
    Explicitly preload a specific model by making a small generation request.
    This forces Ollama to load the model into VRAM right now.
    
    Args:
        model: The model name to preload
        character_id: The character ID requesting the model
    """
    llm_client = app_state.get("llm_client")
    if not llm_client:
        return
    
    try:
        logger.info(f"[PRELOAD] Explicitly loading {model} for {character_id}...")
        # Make a tiny generation request to force model load
        await llm_client.client.post(
            f"{llm_client.base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Hi",
                "stream": False
            },
            timeout=30.0
        )
        logger.info(f"[PRELOAD] Successfully loaded {model}")
        app_state["current_model"] = model
    except Exception as e:
        logger.warning(f"[PRELOAD] Failed to preload {model}: {e}")


async def ensure_model_loaded(model: str, character_id: str):
    """
    Ensure the specified model is loaded for the given character.
    
    VRAM Management Strategy (Phase 7):
    - Keep BOTH intent detection model AND character model loaded simultaneously
    - Only unload when switching characters (unload old character's model)
    - Keep intent model resident across character switches
    - Only unload ALL models for ComfyUI image generation (handled separately)
    
    Args:
        model: The model name to load
        character_id: The character ID requesting the model
    """
    llm_client = app_state.get("llm_client")
    if not llm_client:
        return
    
    current_model = app_state.get("current_model")
    last_character = app_state.get("last_character")
    intent_model = app_state.get("intent_model", "gemma2:9b")
    
    logger.info(f"[MODEL TRACKING] ensure_model_loaded called: model={model}, character={character_id}, last_character={last_character}")
    
    # Check what's actually loaded using provider abstraction
    try:
        loaded_models = await llm_client.get_loaded_models()
        logger.info(f"[MODEL TRACKING] LLM provider reports: {len(loaded_models)} loaded: {loaded_models}")
        
        # If the requested model is already loaded, we're good
        if model in loaded_models:
            logger.info(f"[MODEL TRACKING] Model {model} already loaded for {character_id}")
            app_state["current_model"] = model
            app_state["last_character"] = character_id
            return
        
        # Only unload if switching characters (not for intent detection -> chat)
        character_switched = last_character and last_character != character_id
        if character_switched:
            # Character switch: unload old character's model, keep intent model
            logger.info(f"[MODEL TRACKING] Character switched from {last_character} to {character_id}")
            models_to_unload = [m for m in loaded_models if m != intent_model]
            if models_to_unload:
                logger.info(f"[MODEL TRACKING] Unloading old character models: {models_to_unload} (keeping intent model)")
                for model_name in models_to_unload:
                    try:
                        # Use provider's unload method
                        await llm_client.unload_model(model_name)
                        logger.info(f"[MODEL TRACKING] Unloaded {model_name}")
                    except Exception as e:
                        logger.warning(f"[MODEL TRACKING] Failed to unload {model_name}: {e}")
        else:
            # Same character: keep both intent model and character model loaded
            logger.info(f"[MODEL TRACKING] Same character, keeping all loaded models: {loaded_models}")
    except Exception as e:
        logger.warning(f"[MODEL TRACKING] Could not check loaded models: {e}")
    
    # Update trackers
    app_state["last_character"] = character_id
    
    # Model not loaded - it will load on first request
    logger.info(f"[MODEL TRACKING] Model {model} will load on first request for {character_id}")
    app_state["current_model"] = model


async def log_llm_status(context: str):
    """
    Log the current LLM provider's model status and GPU VRAM usage for debugging.
    
    Args:
        context: Description of when this is being called (e.g., "AFTER MESSAGE", "AFTER IMAGE GEN")
    """
    llm_client = app_state.get("llm_client")
    if not llm_client:
        return
    
    try:
        # Get loaded models using provider abstraction
        loaded_models = await llm_client.get_loaded_models()
        
        # Get GPU VRAM
        vram_mb = await get_gpu_vram_usage()
        vram_str = f"{vram_mb}MB" if vram_mb else "N/A"
        
        logger.info(f"[{context}] LLM: {len(loaded_models)} models {loaded_models} | GPU VRAM: {vram_str}")
    except Exception as e:
        logger.warning(f"[{context}] Could not get LLM status: {e}")


# Create FastAPI app
app = FastAPI(
    title="Chorus Engine",
    description="Local AI Character Orchestration System",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm_available: bool
    characters_loaded: int


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    character_id: str


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    character_name: str


# Conversation models
class ConversationCreate(BaseModel):
    """Create conversation request."""
    character_id: str
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    """Conversation response."""
    id: str
    character_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationUpdate(BaseModel):
    """Update conversation request."""
    title: str


# Thread models
class ThreadCreate(BaseModel):
    """Create thread request."""
    title: str = "New Thread"


class ThreadResponse(BaseModel):
    """Thread response."""
    id: str
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ThreadUpdate(BaseModel):
    """Update thread request."""
    title: str


# Message models
class MessageCreate(BaseModel):
    """Create message request."""
    content: str


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime
    is_private: str = "false"  # "true" or "false" as string
    metadata: Optional[dict] = None
    # Phase 6: TTS audio fields
    has_audio: str = "false"  # "true" or "false" as string
    audio_url: Optional[str] = None
    audio_emotion: Optional[str] = None
    
    class Config:
        from_attributes = True
        populate_by_name = True
        
    @classmethod
    def from_orm(cls, obj, db_session=None):
        """
        Convert ORM object to response model.
        
        Args:
            obj: Message ORM object
            db_session: Optional database session to check for audio
        """
        # Check if message has audio (Phase 6)
        has_audio = "false"
        audio_url = None
        
        if db_session:
            from chorus_engine.repositories import AudioRepository
            audio_repo = AudioRepository(db_session)
            audio_record = audio_repo.get_by_message_id(obj.id)
            if audio_record:
                has_audio = "true"
                audio_url = f"/audio/{audio_record.audio_filename}"
        
        return cls(
            id=obj.id,
            thread_id=obj.thread_id,
            role=obj.role.value if hasattr(obj.role, 'value') else obj.role,
            content=obj.content,
            created_at=obj.created_at,
            is_private=obj.is_private,
            metadata=obj.meta_data,
            has_audio=has_audio,
            audio_url=audio_url,
            audio_emotion=None  # Reserved for future use
        )


class ChatInThreadRequest(BaseModel):
    """Send message in a thread."""
    message: str


class ChatInThreadResponse(BaseModel):
    """Chat in thread response."""
    user_message: MessageResponse
    assistant_message: MessageResponse
    image_request_detected: bool = False
    image_prompt_preview: Optional[dict] = None
    conversation_title_updated: Optional[str] = None  # New title if auto-generated


# Memory models
class MemoryCreate(BaseModel):
    """Create memory request."""
    content: str
    memory_type: str = "explicit"
    thread_id: Optional[str] = None
    tags: Optional[List[str]] = None
    priority: Optional[int] = None


class MemoryResponse(BaseModel):
    """Memory response."""
    id: str
    conversation_id: Optional[str]
    thread_id: Optional[str]
    memory_type: str
    content: str
    created_at: datetime
    character_id: Optional[str] = None
    vector_id: Optional[str] = None
    embedding_model: Optional[str] = None
    priority: Optional[int] = None
    tags: Optional[List[str]] = None
    # Phase 4.1: Implicit memory fields
    confidence: Optional[float] = None
    category: Optional[str] = None
    status: Optional[str] = None
    source_messages: Optional[List[str]] = None
    
    class Config:
        from_attributes = True


class ImageGenerationConfirmRequest(BaseModel):
    """Request to confirm and generate an image."""
    message_id: Optional[int] = None
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    disable_future_confirmations: bool = False
    workflow_id: Optional[str] = None  # Selected workflow ID from dropdown


class SceneCaptureRequest(BaseModel):
    """Request to capture scene image."""
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    workflow_id: Optional[str] = None  # Selected workflow ID from dropdown


class ImageGenerationResponse(BaseModel):
    """Response from image generation."""
    success: bool
    image_id: Optional[int] = None
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    prompt: Optional[str] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None


# Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    llm_available = False
    if app_state["llm_client"]:
        llm_available = await app_state["llm_client"].health_check()
    
    return HealthResponse(
        status="ok",
        llm_available=llm_available,
        characters_loaded=len(app_state["characters"]),
    )


@app.get("/characters")
async def list_characters():
    """List all available characters."""
    characters = app_state["characters"]
    return {
        "characters": [
            {
                "id": char.id,
                "name": char.name,
                "role": char.role,
                "personality_traits": char.personality_traits,
                "immersion_level": char.immersion_level,
                "profile_image": char.profile_image,
                "profile_image_url": char.profile_image_url,
                "system_prompt": char.system_prompt,
                "visual_identity": {
                    "default_workflow": char.visual_identity.default_workflow if char.visual_identity else None,
                    "prompt_context": char.visual_identity.prompt_context if char.visual_identity else None,
                } if char.visual_identity else None,
                "preferred_llm": {
                    "model": char.preferred_llm.model if char.preferred_llm else None,
                    "temperature": char.preferred_llm.temperature if char.preferred_llm else None,
                } if char.preferred_llm else None,
                "memory": {
                    "scope": char.memory.scope if char.memory else None,
                } if char.memory else None,
                "capabilities": {
                    "image_generation": char.image_generation.enabled if char.image_generation else False,
                    "audio_generation": char.voice is not None,
                },
            }
            for char in characters.values()
        ]
    }


@app.get("/characters/{character_id}/stats")
async def get_character_stats(character_id: str, db: Session = Depends(get_db)):
    """Get statistics for a character."""
    
    # Count conversations
    conv_repo = ConversationRepository(db)
    conversations = conv_repo.list_by_character(character_id)
    
    # Count messages across all conversations
    message_count = 0
    for conv in conversations:
        threads = conv.threads
        for thread in threads:
            message_count += len(thread.messages)
    
    # Count memories (handle potential database enum errors)
    memory_count = 0
    try:
        memory_repo = MemoryRepository(db)
        memories = memory_repo.list_by_character(character_id)
        memory_count = len(memories)
    except (LookupError, Exception) as e:
        # If there's a database error (e.g., invalid enum values), just return 0
        logger.warning(f"Failed to count memories for {character_id}: {e}")
        memory_count = 0
    
    # Count core memories from character config
    characters = app_state["characters"]
    character = characters.get(character_id)
    core_memories_count = len(character.core_memories) if character and character.core_memories else 0
    
    return {
        "character_id": character_id,
        "conversation_count": len(conversations),
        "message_count": message_count,
        "memory_count": memory_count,
        "core_memory_count": core_memories_count,
    }


@app.post("/characters/{character_id}/profile-image")
async def set_character_profile_image(character_id: str, request: dict):
    """Set a character's profile image by copying from conversation images."""
    image_filename = request.get("image_filename")
    
    if not image_filename:
        raise HTTPException(status_code=400, detail="image_filename is required")
    
    # Get source and destination paths
    images_dir = Path(__file__).parent.parent.parent / "data" / "images"
    character_images_dir = Path(__file__).parent.parent.parent / "data" / "character_images"
    
    # Find the source image in any conversation folder
    source_path = None
    for conv_dir in images_dir.iterdir():
        if conv_dir.is_dir():
            potential_source = conv_dir / image_filename
            if potential_source.exists():
                source_path = potential_source
                break
    
    if not source_path or not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_filename}")
    
    # Copy to character_images with character_id prefix
    dest_filename = f"{character_id}_{image_filename}"
    dest_path = character_images_dir / dest_filename
    
    import shutil
    shutil.copy2(source_path, dest_path)
    
    # Update character config in memory
    characters = app_state["characters"]
    character = characters.get(character_id)
    
    if not character:
        raise HTTPException(status_code=404, detail=f"Character not found: {character_id}")
    
    character.profile_image = dest_filename
    
    # Update the YAML file
    character_yaml_path = Path(__file__).parent.parent.parent / "characters" / f"{character_id}.yaml"
    if not character_yaml_path.exists():
        # Try with .yaml extension variations
        for yaml_file in (Path(__file__).parent.parent.parent / "characters").glob(f"{character_id}*.yaml"):
            character_yaml_path = yaml_file
            break
    
    if character_yaml_path.exists():
        import yaml
        with open(character_yaml_path, 'r', encoding='utf-8') as f:
            char_data = yaml.safe_load(f)
        
        char_data['profile_image'] = dest_filename
        
        with open(character_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(char_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Updated profile image for {character_id}: {dest_filename}")
    
    return {"success": True, "profile_image": dest_filename, "profile_image_url": f"/character_images/{dest_filename}"}


# Debug endpoint
@app.get("/debug/conversation/{conversation_id}")
async def get_conversation_debug_log(conversation_id: str):
    """Get debug log for a conversation showing all LLM interactions."""
    from chorus_engine.utils.debug_logger import get_debug_logger
    
    debug_logger = get_debug_logger()
    interactions = debug_logger.get_conversation_log(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "interaction_count": len(interactions),
        "interactions": interactions
    }


@app.delete("/debug/conversation/{conversation_id}")
async def clear_conversation_debug_log(conversation_id: str):
    """Clear debug log for a conversation."""
    from chorus_engine.utils.debug_logger import get_debug_logger
    
    debug_logger = get_debug_logger()
    debug_logger.clear_conversation_log(conversation_id)
    
    return {"status": "cleared", "conversation_id": conversation_id}


@app.get("/characters/{character_id}")
async def get_character(character_id: str):
    """Get details for a specific character."""
    characters = app_state["characters"]
    
    if character_id not in characters:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    char = characters[character_id]
    
    # Build response with optional preferred_llm
    response = {
        "id": char.id,
        "name": char.name,
        "role": char.role,
        "personality_traits": char.personality_traits,
        "system_prompt": char.system_prompt,
        "immersion_level": char.immersion_level,
    }
    
    # Include preferred_llm if any values are set
    if char.preferred_llm.model or char.preferred_llm.temperature is not None:
        response["preferred_llm"] = {
            "model": char.preferred_llm.model,
            "temperature": char.preferred_llm.temperature
        }
    
    return response


@app.get("/characters/{character_id}/immersion-notice")
async def get_character_immersion_notice(character_id: str):
    """
    Get the immersion notice text for a character.
    
    Returns None if no notice is needed (minimal/balanced characters).
    Used by UI to display one-time notice for full/unbounded characters.
    """
    characters = app_state["characters"]
    
    if character_id not in characters:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    character = characters[character_id]
    generator = SystemPromptGenerator()
    
    should_show = generator.should_show_immersion_notice(character)
    notice_text = generator.get_immersion_notice_text(character) if should_show else None
    
    return {
        "character_id": character_id,
        "immersion_level": character.immersion_level,
        "should_show_notice": should_show,
        "notice_text": notice_text,
    }


@app.post("/characters")
async def create_character(character_data: dict):
    """
    Create a new user character.
    
    Validates that ID doesn't conflict with immutable characters.
    """
    loader = ConfigLoader()
    
    # Validate character data
    try:
        character = CharacterConfig(**character_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid character data: {e}")
    
    # Check if ID conflicts with immutable characters
    if character.id in IMMUTABLE_CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot use reserved character ID '{character.id}'"
        )
    
    # Check if character already exists
    try:
        existing = loader.load_character(character.id)
        raise HTTPException(
            status_code=409,
            detail=f"Character '{character.id}' already exists"
        )
    except:
        pass  # Character doesn't exist, good to create
    
    # Save character
    try:
        file_path = loader.save_character(character)
        
        # Reload characters in app state
        app_state["characters"] = loader.load_all_characters()
        
        return {
            "id": character.id,
            "name": character.name,
            "message": "Character created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save character: {e}")


@app.patch("/characters/{character_id}")
async def update_character(character_id: str, updates: dict):
    """
    Update an existing user character.
    
    Cannot update immutable default characters.
    """
    if character_id in IMMUTABLE_CHARACTERS:
        raise HTTPException(
            status_code=403,
            detail=f"Cannot modify immutable character '{character_id}'"
        )
    
    loader = ConfigLoader()
    
    try:
        # Load existing character
        character = loader.load_character(character_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Character not found: {e}")
    
    # Apply updates
    try:
        # Convert to dict, update, and recreate
        char_dict = character.model_dump()
        char_dict.update(updates)
        
        # Don't allow ID changes
        if "id" in updates and updates["id"] != character_id:
            raise HTTPException(status_code=400, detail="Cannot change character ID")
        
        # Recreate with updates
        updated_character = CharacterConfig(**char_dict)
        
        # Save
        loader.save_character(updated_character)
        
        # Reload characters in app state
        app_state["characters"] = loader.load_all_characters()
        
        return {
            "id": character_id,
            "message": "Character updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update character: {e}")


@app.delete("/characters/{character_id}")
async def delete_character(character_id: str):
    """
    Delete a user character.
    
    Cannot delete immutable default characters.
    """
    if character_id in IMMUTABLE_CHARACTERS:
        raise HTTPException(
            status_code=403,
            detail=f"Cannot delete immutable character '{character_id}'. Clone it instead."
        )
    
    loader = ConfigLoader()
    
    try:
        loader.delete_character(character_id)
        
        # Reload characters in app state
        app_state["characters"] = loader.load_all_characters()
        
        return {"message": f"Character '{character_id}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete character: {e}")


@app.post("/characters/{character_id}/clone")
async def clone_character(character_id: str, new_id: str):
    """
    Clone a character (allows cloning immutable defaults).
    
    Creates a new character with the same configuration but a new ID.
    """
    loader = ConfigLoader()
    
    try:
        # Load source character
        source = loader.load_character(character_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Source character not found: {e}")
    
    # Validate new ID
    if new_id in IMMUTABLE_CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot use reserved character ID '{new_id}'"
        )
    
    # Check if new ID already exists
    try:
        existing = loader.load_character(new_id)
        raise HTTPException(
            status_code=409,
            detail=f"Character '{new_id}' already exists"
        )
    except:
        pass  # New ID doesn't exist, good to clone
    
    try:
        # Create cloned character
        cloned = source.model_copy(deep=True)
        cloned.id = new_id
        cloned.name = f"{source.name} (Clone)"
        
        # Save
        loader.save_character(cloned)
        
        # Reload characters in app state
        app_state["characters"] = loader.load_all_characters()
        
        return {
            "id": new_id,
            "name": cloned.name,
            "message": f"Cloned '{character_id}' to '{new_id}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clone character: {e}")


# ===== Export/Import Endpoints =====

@app.get("/characters/{character_id}/export")
async def export_character(character_id: str):
    """
    Export a character configuration as YAML.
    
    Returns the YAML file for download.
    """
    characters = app_state["characters"]
    
    if character_id not in characters:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    character = characters[character_id]
    
    try:
        # Convert to dict, excluding computed fields
        data = character.model_dump(
            exclude_none=True,
            exclude={"created_at", "updated_at"},
            mode='json'
        )
        
        # Convert to YAML
        yaml_content = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120
        )
        
        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": f"attachment; filename={character_id}.yaml"
            }
        )
    except Exception as e:
        logger.error(f"Failed to export character {character_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export character: {e}")


@app.post("/characters/import")
async def import_character(file: UploadFile = File(...)):
    """
    Import a character configuration from YAML file.
    
    If a character with the same ID exists, it will be overwritten (unless immutable).
    """
    if not file.filename.endswith('.yaml') and not file.filename.endswith('.yml'):
        raise HTTPException(status_code=400, detail="File must be a YAML file (.yaml or .yml)")
    
    try:
        # Read file content
        content = await file.read()
        yaml_str = content.decode('utf-8')
        
        # Parse YAML
        data = yaml.safe_load(yaml_str)
        
        if not data:
            raise HTTPException(status_code=400, detail="Empty YAML file")
        
        # Validate character data
        try:
            character = CharacterConfig(**data)
        except ValidationError as e:
            errors = [f"{' → '.join(str(l) for l in err['loc'])}: {err['msg']}" 
                     for err in e.errors()]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid character configuration: {', '.join(errors)}"
            )
        
        # Check if character is immutable
        if character.id in IMMUTABLE_CHARACTERS:
            raise HTTPException(
                status_code=403,
                detail=f"Cannot overwrite immutable character '{character.id}'. Change the ID in the YAML file."
            )
        
        # Save character
        loader = ConfigLoader()
        loader.save_character(character)
        
        # Reload characters in app state
        app_state["characters"] = loader.load_all_characters()
        
        return {
            "id": character.id,
            "name": character.name,
            "message": f"Character '{character.id}' imported successfully"
        }
        
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import character: {e}")


@app.get("/config/system/export")
async def export_system_config():
    """
    Export the system configuration as YAML.
    
    Returns the YAML file for download.
    """
    system_config = app_state["system_config"]
    
    try:
        # Convert to dict
        data = system_config.model_dump(
            exclude_none=True,
            mode='json'
        )
        
        # Convert to YAML
        yaml_content = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120
        )
        
        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": "attachment; filename=system.yaml"
            }
        )
    except Exception as e:
        logger.error(f"Failed to export system config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export system config: {e}")


@app.post("/config/system/import")
async def import_system_config(file: UploadFile = File(...)):
    """
    Import system configuration from YAML file.
    
    WARNING: This will overwrite the current system configuration and restart the system.
    """
    if not file.filename.endswith('.yaml') and not file.filename.endswith('.yml'):
        raise HTTPException(status_code=400, detail="File must be a YAML file (.yaml or .yml)")
    
    try:
        # Read file content
        content = await file.read()
        yaml_str = content.decode('utf-8')
        
        # Parse YAML
        data = yaml.safe_load(yaml_str)
        
        if not data:
            raise HTTPException(status_code=400, detail="Empty YAML file")
        
        # Validate system config data
        try:
            system_config = SystemConfig(**data)
        except ValidationError as e:
            errors = [f"{' → '.join(str(l) for l in err['loc'])}: {err['msg']}" 
                     for err in e.errors()]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid system configuration: {', '.join(errors)}"
            )
        
        # Save system config
        config_file = Path("config/system.yaml")
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=120
            )
        
        # Update app state
        app_state["system_config"] = system_config
        
        # Note: In a production system, you might want to trigger a reload of dependent services
        # For now, we'll just update the app state and recommend a restart
        
        return {
            "message": "System configuration imported successfully. Please restart the server for all changes to take effect.",
            "restart_recommended": True
        }
        
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import system config: {e}")


# ===== Log Viewing Endpoints =====

@app.get("/logs/server")
async def get_server_logs(lines: int = Query(default=500, le=10000)):
    """
    Get recent server log entries.
    
    Returns the most recent log lines from the current server log file.
    If no log file exists (debug mode off), returns console output notice.
    
    Args:
        lines: Number of recent lines to return (max 10000)
    """
    try:
        log_dir = Path("data/debug_logs/server")
        
        if not log_dir.exists():
            return {
                "logs": "Debug mode is disabled. Server logs are only written to console.\nEnable debug mode in system.yaml to save logs to file.",
                "file": None,
                "lines_returned": 0
            }
        
        # Find the most recent log file
        log_files = sorted(log_dir.glob("server_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not log_files:
            return {
                "logs": "No server log files found. Server logs are being written to console only.",
                "file": None,
                "lines_returned": 0
            }
        
        log_file = log_files[0]
        
        # Read the last N lines efficiently
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": ''.join(recent_lines),
            "file": log_file.name,
            "lines_returned": len(recent_lines),
            "total_lines": len(all_lines)
        }
        
    except Exception as e:
        logger.error(f"Failed to read server logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read server logs: {e}")


@app.get("/logs/conversations")
async def list_conversation_logs(db: Session = Depends(get_db)):
    """
    List all available conversation debug logs with metadata.
    
    Returns a list of conversation IDs with character and title info.
    """
    try:
        debug_dir = Path("data/debug_logs/conversations")
        
        if not debug_dir.exists():
            return {"conversations": []}
        
        # Get all conversation directories (skip intent_detection)
        conv_dirs = [d for d in debug_dir.iterdir() if d.is_dir() and d.name != "intent_detection"]
        
        conv_repo = ConversationRepository(db)
        characters = app_state["characters"]
        
        conversations = []
        for conv_dir in conv_dirs:
            log_file = conv_dir / "conversation.jsonl"
            if log_file.exists():
                stat = log_file.stat()
                
                # Try to get conversation details from database
                conv_id = conv_dir.name
                character_name = "Unknown"
                title = "Unknown"
                
                try:
                    conversation = conv_repo.get(conv_id)
                    if conversation:
                        title = conversation.title or "Untitled"
                        character = characters.get(conversation.character_id)
                        if character:
                            character_name = character.name
                except:
                    pass  # Conversation might be deleted
                
                conversations.append({
                    "conversation_id": conv_id,
                    "character_name": character_name,
                    "title": title,
                    "log_file": str(log_file.relative_to(Path("."))),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by modification time, newest first
        conversations.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Failed to list conversation logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list conversation logs: {e}")


@app.get("/logs/conversations/{conversation_id}")
async def get_conversation_log(conversation_id: str, prettify: bool = Query(default=True)):
    """
    Get debug log for a specific conversation.
    
    Returns the conversation debug log (all LLM interactions).
    
    Args:
        conversation_id: ID of the conversation
        prettify: Whether to prettify the JSON (default: true)
    """
    try:
        log_file = Path(f"data/debug_logs/conversations/{conversation_id}/conversation.jsonl")
        
        if not log_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"No debug log found for conversation {conversation_id}"
            )
        
        # Read all interactions
        interactions = []
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))
        
        if prettify:
            # Return prettified JSON
            return {
                "conversation_id": conversation_id,
                "interactions": interactions,
                "count": len(interactions)
            }
        else:
            # Return raw JSONL for download
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/x-ndjson",
                headers={
                    "Content-Disposition": f"attachment; filename={conversation_id}_debug.jsonl"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read conversation log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read conversation log: {e}")


@app.get("/logs/extractions/{conversation_id}")
async def get_extraction_log(conversation_id: str, prettify: bool = Query(default=True)):
    """
    Get extraction log for a specific conversation.
    
    Returns the extraction log (memory extractions from conversations).
    
    Args:
        conversation_id: ID of the conversation
        prettify: Whether to prettify the JSON (default: true)
    """
    try:
        log_file = Path(f"data/debug_logs/conversations/{conversation_id}/extractions.jsonl")
        
        if not log_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"No extraction log found for conversation {conversation_id}"
            )
        
        # Read all extractions
        extractions = []
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    extractions.append(json.loads(line))
        
        if prettify:
            # Return prettified JSON
            return {
                "conversation_id": conversation_id,
                "extractions": extractions,
                "count": len(extractions)
            }
        else:
            # Return raw JSONL for download
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/x-ndjson",
                headers={
                    "Content-Disposition": f"attachment; filename={conversation_id}_extractions.jsonl"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read extraction log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read extraction log: {e}")


@app.get("/logs/image-prompts/{conversation_id}")
async def get_image_prompts_log(conversation_id: str, prettify: bool = Query(default=True)):
    """
    Get image prompt generation log for a specific conversation.
    
    Returns the image prompt log (scene capture and in-conversation image requests).
    
    Args:
        conversation_id: ID of the conversation
        prettify: Whether to prettify the JSON (default: true)
    """
    try:
        log_file = Path(f"data/debug_logs/conversations/{conversation_id}/image_prompts.jsonl")
        
        if not log_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"No image prompt log found for conversation {conversation_id}"
            )
        
        # Read all image requests
        requests = []
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    requests.append(json.loads(line))
        
        if prettify:
            # Return prettified JSON
            return {
                "conversation_id": conversation_id,
                "image_requests": requests,
                "count": len(requests)
            }
        else:
            # Return raw JSONL for download
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/x-ndjson",
                headers={
                    "Content-Disposition": f"attachment; filename={conversation_id}_image_prompts.jsonl"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read image prompt log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read image prompt log: {e}")


@app.get("/logs/intent-detection")
async def get_intent_detection_log(lines: int = Query(default=100, le=1000)):
    """
    Get recent intent detection log entries.
    
    Returns the most recent intent detection interactions.
    
    Args:
        lines: Number of recent entries to return (max 1000)
    """
    try:
        log_file = Path("data/debug_logs/conversations/intent_detection/conversation.jsonl")
        
        if not log_file.exists():
            return {
                "interactions": [],
                "count": 0,
                "message": "No intent detection log found. Debug mode may be disabled."
            }
        
        # Read all interactions
        all_interactions = []
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    all_interactions.append(json.loads(line))
        
        # Return the most recent N interactions
        recent = all_interactions[-lines:] if len(all_interactions) > lines else all_interactions
        
        return {
            "interactions": recent,
            "count": len(recent),
            "total": len(all_interactions)
        }
        
    except Exception as e:
        logger.error(f"Failed to read intent detection log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read intent detection log: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Simple chat endpoint - send a message to a character.
    
    This is a basic implementation without conversation history or memory.
    Use /conversations for persistent conversations.
    """
    characters = app_state["characters"]
    llm_client = app_state["llm_client"]
    
    # Validate character exists
    if request.character_id not in characters:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{request.character_id}' not found"
        )
    
    character = characters[request.character_id]
    
    # Check LLM availability
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    llm_available = await llm_client.health_check()
    if not llm_available:
        raise HTTPException(
            status_code=503,
            detail=f"LLM not available at {app_state['system_config'].llm.base_url}"
        )
    
    # Generate response
    try:
        # Use character's preferred model
        model = character.preferred_llm.model or app_state['system_config'].llm.model
        
        response = await llm_client.generate(
            prompt=request.message,
            system_prompt=character.system_prompt,
            model=model  # Pass character's model
        )
        
        return ChatResponse(
            response=response.content,
            character_name=character.name,
        )
        
    except LLMError as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


# === Conversation Management Endpoints ===

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation with a character."""
    # Validate character exists
    if request.character_id not in app_state["characters"]:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{request.character_id}' not found"
        )
    
    repo = ConversationRepository(db)
    conversation = repo.create(
        character_id=request.character_id,
        title=request.title
    )
    
    # Create default thread
    thread_repo = ThreadRepository(db)
    thread_repo.create(conversation_id=conversation.id, title="Main Thread")
    
    return conversation


@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    character_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all conversations."""
    repo = ConversationRepository(db)
    
    if character_id:
        conversations = repo.list_by_character(character_id, skip=skip, limit=limit)
    else:
        conversations = repo.list_all(skip=skip, limit=limit)
    
    return conversations


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get conversation details."""
    repo = ConversationRepository(db)
    conversation = repo.get_by_id(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@app.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = Query("markdown", regex="^(markdown|text)$"),
    include_metadata: bool = Query(True),
    include_summary: bool = Query(True),
    include_memories: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Export conversation to markdown or text format.
    
    Args:
        conversation_id: Conversation ID to export
        format: Export format ('markdown' or 'text')
        include_metadata: Include title, dates, character info
        include_summary: Include conversation summary if analyzed (markdown only)
        include_memories: Include extracted memories list (markdown only)
    
    Returns:
        FileResponse with exported conversation
    """
    # Get conversation
    repo = ConversationRepository(db)
    conversation = repo.get_by_id(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Query summaries if needed
    summaries = []
    if include_summary:
        summaries = db.query(ConversationSummary).filter(
            ConversationSummary.conversation_id == conversation_id
        ).order_by(ConversationSummary.created_at.desc()).all()
    
    # Export to temporary file
    export_service = ConversationExportService()
    export_dir = Path("data/exports")
    
    try:
        filepath = export_service.save_to_file(
            conversation=conversation,
            summaries=summaries,
            output_dir=export_dir,
            format=format,
            include_metadata=include_metadata,
            include_summary=include_summary,
            include_memories=include_memories
        )
        
        # Determine media type
        media_type = "text/markdown" if format == "markdown" else "text/plain"
        
        # Return file for download
        return FileResponse(
            path=str(filepath),
            media_type=media_type,
            filename=filepath.name,
            headers={
                "Content-Disposition": f'attachment; filename="{filepath.name}"'
            }
        )
    except Exception as e:
        logger.exception(f"Failed to export conversation {conversation_id}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/conversations/{conversation_id}/analyze")
async def analyze_conversation_now(
    conversation_id: str,
    force: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Manually trigger conversation analysis ("Analyze Now" button).
    
    Runs synchronously and returns detailed results immediately.
    
    Soft minimums:
    - 5 messages or 100 tokens
    - Can be bypassed with force=true
    """
    config_loader = app_state["system_config"]._loader if hasattr(app_state["system_config"], "_loader") else ConfigLoader()
    analysis_service = app_state["analysis_service"]
    
    # Get conversation
    repo = ConversationRepository(db)
    conversation = repo.get_by_id(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Load character
    character = config_loader.load_character(conversation.character_id)
    
    # Get all threads and messages for this conversation
    thread_repo = ThreadRepository(db)
    message_repo = MessageRepository(db)
    threads = thread_repo.list_by_conversation(conversation_id)
    
    all_messages = []
    for thread in threads:
        messages = message_repo.list_by_thread(thread.id)
        all_messages.extend(messages)
    
    # Check soft minimums (unless forced)
    if not force:
        message_count = len(all_messages)
        token_count = sum(len(m.content.split()) * 1.3 for m in all_messages)
        
        if message_count < 5 or token_count < 100:
            return {
                "status": "warning",
                "message": f"Conversation might be too short ({message_count} messages, ~{int(token_count)} tokens). Analysis may not be meaningful.",
                "can_force": True,
                "message_count": message_count,
                "estimated_tokens": int(token_count)
            }
    
    # Run analysis synchronously
    try:
        analysis = await analysis_service.analyze_conversation(
            conversation_id=conversation_id,
            character=character,
            manual=True  # Mark as manual analysis
        )
        
        if not analysis:
            return {
                "status": "error",
                "message": "Analysis returned no results"
            }
        
        # Save analysis to database
        await analysis_service.save_analysis(
            conversation_id=conversation_id,
            character_id=conversation.character_id,
            analysis=analysis,
            manual=True
        )
        
        # Build memory counts
        memory_counts = {}
        for memory in analysis.memories:
            mem_type = memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type)
            memory_counts[mem_type] = memory_counts.get(mem_type, 0) + 1
        
        return {
            "status": "success",
            "analysis_type": "manual",
            "memories_extracted": len(analysis.memories),
            "memory_counts": memory_counts,
            "memories": [
                {
                    "type": m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                    "content": m.content,
                    "confidence": m.confidence,
                    "emotional_weight": m.emotional_weight,
                    "reasoning": m.reasoning
                }
                for m in analysis.memories
            ],
            "summary": analysis.summary,
            "themes": analysis.themes,
            "tone": analysis.tone,
            "emotional_arc": analysis.emotional_arc,
            "participants": analysis.participants,
            "key_topics": analysis.key_topics
        }
    except Exception as e:
        logger.exception(f"Manual analysis failed for conversation {conversation_id}")
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }


@app.get("/conversations/{conversation_id}/analyses")
async def get_analysis_history(
    conversation_id: str,
    include_memories: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Get analysis history for a conversation.
    
    Shows all ConversationSummary records with memory counts.
    """
    # Get conversation
    repo = ConversationRepository(db)
    conversation = repo.get_by_id(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Query all summaries for this conversation
    summaries = db.query(ConversationSummary).filter(
        ConversationSummary.conversation_id == conversation_id
    ).order_by(ConversationSummary.created_at.desc()).all()
    
    # Build analysis history
    analyses = []
    for summary in summaries:
        # Get memories associated with this analysis (within the message range)
        memory_repo = MemoryRepository(db)
        memories = memory_repo.list_by_conversation(conversation_id)
        
        # Count memories by type
        memory_counts = {}
        analysis_memories = []
        for memory in memories:
            mem_type = memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type)
            memory_counts[mem_type] = memory_counts.get(mem_type, 0) + 1
            
            if include_memories:
                analysis_memories.append({
                    "id": memory.id,
                    "type": mem_type,
                    "category": memory.category,
                    "content": memory.content,
                    "confidence": memory.confidence,
                    "emotional_weight": memory.emotional_weight,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None
                })
        
        # Parse emotional_arc safely
        emotional_arc = []
        if summary.emotional_arc:
            try:
                emotional_arc = json.loads(summary.emotional_arc)
            except (json.JSONDecodeError, ValueError):
                emotional_arc = []
        
        analysis = {
            "id": summary.id,
            "analyzed_at": summary.created_at.isoformat() if summary.created_at else None,
            "manual": summary.manual == "true",
            "summary": summary.summary,
            "themes": summary.key_topics if summary.key_topics else [],
            "tone": summary.tone,
            "emotional_arc": emotional_arc,
            "message_range": {
                "start": summary.message_range_start,
                "end": summary.message_range_end,
                "count": summary.message_count
            },
            "memory_counts": memory_counts,
            "total_memories": len(memories)
        }
        
        if include_memories:
            analysis["memories"] = analysis_memories
        
        analyses.append(analysis)
    
    return {
        "conversation_id": conversation_id,
        "character_id": conversation.character_id,
        "total_analyses": len(analyses),
        "analyses": analyses
    }


@app.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    db: Session = Depends(get_db)
):
    """Update conversation title."""
    repo = ConversationRepository(db)
    conversation = repo.update(conversation_id, title=request.title)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    delete_memories: bool = False,
    db: Session = Depends(get_db)
):
    """
    Delete a conversation and all its threads/messages.
    
    Args:
        conversation_id: ID of conversation to delete
        delete_memories: If True, also delete associated memories. If False (default), orphan them.
    
    Returns:
        Deletion status with memory handling info
    """
    conv_repo = ConversationRepository(db)
    mem_repo = MemoryRepository(db)
    
    # Check conversation exists
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Handle memories based on user choice
    memory_count = mem_repo.count_by_conversation(conversation_id)
    if delete_memories:
        # Get memories before deleting to access vector_ids
        memories = db.query(Memory).filter(Memory.conversation_id == conversation_id).all()
        vector_ids = [m.vector_id for m in memories if m.vector_id]
        
        # Delete from SQL
        deleted_count = mem_repo.delete_by_conversation(conversation_id)
        
        # Delete from vector store
        if vector_ids and conversation.character_id:
            try:
                vector_store = app_state.get("vector_store")
                if vector_store:
                    vector_store.delete_memories(
                        character_id=conversation.character_id,
                        memory_ids=vector_ids
                    )
                    logger.info(f"Deleted {len(vector_ids)} memories from vector store for conversation {conversation_id}")
            except Exception as e:
                logger.error(f"Failed to delete memories from vector store: {e}")
                # Don't fail the request - memories are already deleted from DB
        
        memory_action = "deleted"
    else:
        orphaned_count = mem_repo.orphan_conversation_memories(conversation_id)
        memory_action = "orphaned"
        deleted_count = orphaned_count
    
    # Delete conversation
    conv_repo.delete(conversation_id)
    
    return {
        "status": "deleted",
        "id": conversation_id,
        "memories": {
            "count": memory_count,
            "action": memory_action
        }
    }


# === Thread Management Endpoints ===

@app.post("/conversations/{conversation_id}/threads", response_model=ThreadResponse)
async def create_thread(
    conversation_id: str,
    request: ThreadCreate,
    db: Session = Depends(get_db)
):
    """Create a new thread in a conversation."""
    # Verify conversation exists
    conv_repo = ConversationRepository(db)
    if not conv_repo.get_by_id(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    repo = ThreadRepository(db)
    thread = repo.create(conversation_id=conversation_id, title=request.title)
    return thread


@app.get("/conversations/{conversation_id}/threads", response_model=List[ThreadResponse])
async def list_threads(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """List all threads in a conversation."""
    repo = ThreadRepository(db)
    threads = repo.list_by_conversation(conversation_id)
    return threads


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: str,
    db: Session = Depends(get_db)
):
    """Get thread details."""
    repo = ThreadRepository(db)
    thread = repo.get_by_id(thread_id)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return thread


@app.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: str,
    request: ThreadUpdate,
    db: Session = Depends(get_db)
):
    """Update thread title."""
    repo = ThreadRepository(db)
    thread = repo.update(thread_id, title=request.title)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return thread


@app.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    db: Session = Depends(get_db)
):
    """Delete a thread and all its messages."""
    repo = ThreadRepository(db)
    deleted = repo.delete(thread_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {"status": "deleted", "id": thread_id}


# === Message Management Endpoints ===

@app.get("/threads/{thread_id}/messages", response_model=List[MessageResponse])
async def list_messages(
    thread_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
    db: Session = Depends(get_db)
):
    """Get all messages in a thread."""
    repo = MessageRepository(db)
    messages = repo.list_by_thread(thread_id, skip=skip, limit=limit)
    return [MessageResponse.from_orm(msg, db_session=db) for msg in messages]


@app.post("/threads/{thread_id}/messages", response_model=ChatInThreadResponse)
async def send_message(
    thread_id: str,
    request: ChatInThreadRequest,
    db: Session = Depends(get_db)
):
    """Send a message in a thread and get AI response."""
    # Verify thread exists and get conversation
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_by_id(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Get character for this conversation
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(thread.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    character_id = conversation.character_id
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    character = app_state["characters"][character_id]
    
    # Check LLM availability
    llm_client = app_state["llm_client"]
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    llm_available = await llm_client.health_check()
    if not llm_available:
        raise HTTPException(
            status_code=503,
            detail=f"LLM not available at {app_state['system_config'].llm.base_url}"
        )
    
    # Save user message (capture privacy status at send time)
    msg_repo = MessageRepository(db)
    user_message = msg_repo.create(
        thread_id=thread_id,
        role=MessageRole.USER,
        content=request.message,
        is_private=(conversation.is_private == "true")
    )
    
    # Phase 7: Detect intents using unified intent detection service
    intent_service = app_state.get("intent_detection_service")
    detected_intents: Optional[IntentResult] = None
    
    if intent_service:
        try:
            # Detect all intents in one pass (no context - analyze only current message)
            # Use dedicated intent detection model (gemma2:9b) for better performance
            intent_model = app_state.get("intent_model", "gemma2:9b")
            detected_intents = await intent_service.detect_intents(
                message=request.message,
                character=character,
                model=intent_model,
                context=None
            )
            
            logger.info(
                f"[INTENT DETECTION] Detected intents for thread {thread_id}",
                extra={
                    "generate_image": detected_intents.generate_image,
                    "generate_video": detected_intents.generate_video,
                    "record_memory": detected_intents.record_memory,
                    "contains_recordable_facts": detected_intents.contains_recordable_facts,
                    "query_ambient": detected_intents.query_ambient,
                    "fact_count": len(detected_intents.extracted_facts),
                    "processing_time_ms": detected_intents.processing_time_ms
                }
            )
            
            # Phase 7: Handle implicit memory extraction (replaces Phase 4 background worker)
            if detected_intents.contains_recordable_facts and detected_intents.extracted_facts:
                # Only save if conversation is not in privacy mode
                if conversation.is_private != "true":
                    extraction_service = app_state.get("extraction_service")
                    
                    if extraction_service:
                        for fact in detected_intents.extracted_facts:
                            try:
                                # Create ExtractedMemory format
                                from chorus_engine.services.memory_extraction import ExtractedMemory
                                extracted = ExtractedMemory(
                                    content=fact.content,
                                    category=fact.category,
                                    confidence=fact.confidence,
                                    reasoning=fact.reasoning,
                                    source_message_ids=[]  # Intent detection doesn't track source messages
                                )
                                
                                # Use extraction service which handles vector store integration
                                memory = await extraction_service.save_extracted_memory(
                                    extracted=extracted,
                                    character_id=character.id,
                                    conversation_id=conversation.id
                                )
                                
                                if memory:
                                    logger.info(
                                        f"[IMPLICIT MEMORY] Saved fact: {memory.content[:50]}... (status={memory.status})",
                                        extra={
                                            "category": memory.category,
                                            "confidence": memory.confidence,
                                            "status": memory.status,
                                            "indexed": memory.status == "auto_approved"
                                        }
                                    )
                            except Exception as e:
                                logger.error(f"Failed to save implicit memory: {e}", exc_info=True)
                    else:
                        logger.warning("[IMPLICIT MEMORY] Extraction service not available")
                else:
                    logger.info(f"[IMPLICIT MEMORY] Skipped {len(detected_intents.extracted_facts)} facts due to privacy mode")
            
            # Phase 7: Handle explicit memory requests (future - requires UI confirmation)
            if detected_intents.record_memory and detected_intents.extracted_facts:
                logger.info(
                    f"[EXPLICIT MEMORY] User requested memory recording: {len(detected_intents.extracted_facts)} facts",
                    extra={"facts": [f.content for f in detected_intents.extracted_facts]}
                )
                # TODO: Return facts to UI for confirmation dialog
                # TODO: Save with high priority (95) after user confirmation
            
        except Exception as e:
            logger.error(f"Intent detection failed: {e}", exc_info=True)
            # Continue with message processing even if intent detection fails
    
    # Phase 5: Check if user is requesting an image
    image_request_detected = False
    image_prompt_preview = None
    orchestrator = app_state.get("image_orchestrator")
    
    if orchestrator and character.image_generation.enabled:
        # Load fresh workflow config from database
        # Note: Workflow config is now fetched by orchestrator from database as needed
        # (Phase 5 cleanup: workflow fields no longer stored on character.image_generation)
        
        # Check if confirmations are disabled for this conversation
        needs_confirmation = conversation.image_confirmation_disabled != "true"
        
        # Use character's preferred model for image prompt generation (if available)
        model_for_prompt = character.preferred_llm.model if character.preferred_llm.model else None
        logger.info(f"[IMAGE DETECTION] Using model for prompt generation: {model_for_prompt}")
        
        # Get recent conversation context for better image prompt generation (last 10 messages)
        recent_messages = msg_repo.get_thread_history(thread_id, limit=10)
        conversation_context = [{
            "role": msg["role"],
            "content": msg["content"]
        } for msg in recent_messages] if recent_messages else None
        
        try:
            image_info = await orchestrator.detect_and_prepare(
                message=request.message,
                character=character,
                conversation_context=conversation_context,
                model=model_for_prompt,
                db_session=db
            )
            
            if image_info:
                image_request_detected = True
                image_prompt_preview = {
                    "prompt": image_info["prompt"],
                    "negative_prompt": image_info["negative_prompt"],
                    "needs_trigger": image_info.get("needs_trigger", False),
                    "needs_confirmation": needs_confirmation
                }
                logger.info(f"Image request detected in thread {thread_id}: {image_info['prompt'][:50]}...")
                
                # Log to image_prompts.jsonl
                try:
                    conv_dir = Path("data/debug_logs/conversations") / conversation.id
                    conv_dir.mkdir(parents=True, exist_ok=True)
                    log_file = conv_dir / "image_prompts.jsonl"
                    
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "in_conversation",
                        "thread_id": thread_id,
                        "model": model_for_prompt or "default",
                        "user_request": request.message,
                        "context_messages": len(recent_messages) if recent_messages else 0,
                        "generated_prompt": image_info["prompt"],
                        "negative_prompt": image_info["negative_prompt"],
                        "needs_trigger": image_info.get("needs_trigger", False),
                        "reasoning": image_info.get("reasoning", ""),
                        "needs_confirmation": needs_confirmation
                    }
                    
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        
                except Exception as log_error:
                    logger.warning(f"Failed to log image request: {log_error}")
        except KeyError as e:
            logger.error(f"Failed to access image info field: {e}. Image info: {image_info if 'image_info' in locals() else 'None'}")
        except Exception as e:
            logger.error(f"Failed to detect image request: {e}", exc_info=True)
    
    # Get conversation history (includes the user message we just saved)
    history = msg_repo.get_thread_history(thread_id)
    
    # Generate AI response
    try:
        # Assemble prompt with memories (Phase 4.1)
        from chorus_engine.services.prompt_assembly import PromptAssemblyService
        
        prompt_assembler = PromptAssemblyService(
            db=db,
            character_id=character_id,
            model_name=app_state["system_config"].llm.model,
            context_window=32768
        )
        
        # If an image is being generated, pass the prompt to the character so they can reference it
        image_context = image_prompt_preview["prompt"] if (image_request_detected and image_prompt_preview) else None
        
        prompt_components = prompt_assembler.assemble_prompt(
            thread_id=thread_id,
            include_memories=True,  # Enable memory retrieval
            image_prompt_context=image_context
        )
        
        # Format for LLM API (includes memories in system prompt)
        messages = prompt_assembler.format_for_api(prompt_components)
        
        # Debug logging to see what we're actually sending
        print(f"\n{'='*80}")
        print(f"SENDING TO LLM: {len(messages)} total messages")
        print(f"  - 1 system prompt (with memories)")
        print(f"  - {len(prompt_components.messages)} history messages")
        print(f"Token breakdown: system={prompt_components.token_breakdown['system']}, memories={prompt_components.token_breakdown['memories']}, history={prompt_components.token_breakdown['history']}")
        print(f"{'='*80}")
        for i, msg in enumerate(messages):
            print(f"{i+1}. {msg['role']}: {msg['content'][:100]}...")
        print(f"{'='*80}\n")
        
        # Use character-specific LLM settings if available, otherwise use system defaults
        temperature = character.preferred_llm.temperature
        max_tokens = character.preferred_llm.max_tokens
        model = character.preferred_llm.model or app_state['system_config'].llm.model
        
        # Ensure correct model is loaded for this character (handles character switches)
        await ensure_model_loaded(model, character_id)
        
        print(f"LLM Settings for {character.name}:")
        print(f"  Model: {model} {'(character override)' if character.preferred_llm.model else '(system default)'}")
        print(f"  Temperature: {temperature if temperature is not None else app_state['system_config'].llm.temperature} {'(character override)' if temperature is not None else '(system default)'}")
        print(f"  Max tokens: {max_tokens if max_tokens is not None else app_state['system_config'].llm.max_response_tokens} {'(character override)' if max_tokens is not None else '(system default)'}\n")
        
        # Use generate_with_history to maintain conversation context
        response = await llm_client.generate_with_history(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model
        )
        
        # Save assistant message
        assistant_message = msg_repo.create(
            thread_id=thread_id,
            role=MessageRole.ASSISTANT,
            content=response.content,
            metadata={
                "model": app_state["system_config"].llm.model,
                "character": character.name
            }
        )
        
        # Phase 7: Memory extraction now handled by intent detection (above)
        # Phase 4 background extraction worker removed
        
        # Phase 6: Generate audio if TTS is enabled for this conversation
        audio_filename = None
        
        # Check if any TTS providers are available
        available_providers = TTSProviderFactory.get_available_providers()
        
        # Determine if TTS should be generated
        should_generate_tts = False
        if available_providers:
            # Check conversation-level TTS setting
            if conversation.tts_enabled is not None:
                should_generate_tts = bool(conversation.tts_enabled)
            else:
                # Fall back to character voice config
                if character.voice and character.voice.enabled:
                    should_generate_tts = character.voice.always_on
        
        if should_generate_tts:
            # Initialize repositories for audio generation
            audio_repo = AudioRepository(db)
            
            try:
                logger.info(f"[TTS] Auto-generating audio for message {assistant_message.id}")
                
                # Determine provider and check if we need VRAM management
                provider_name = "comfyui"  # default
                if character.voice and hasattr(character.voice, 'tts_provider'):
                    provider_name = character.voice.tts_provider.provider
                
                # Phase 7: Unload ALL models before audio generation (for ComfyUI or embedded models)
                character_model = model  # Already determined above
                system_config = app_state.get("system_config")
                use_intent_model = system_config.intent_detection.enabled if system_config else False
                intent_model = app_state.get("intent_model", "gemma2:9b") if use_intent_model else None
                
                logger.info(f"[TTS - VRAM] Unloading ALL models to maximize VRAM for {provider_name}...")
                
                if llm_client:
                    try:
                        await llm_client.unload_all_models()
                        logger.info(f"[TTS - VRAM] All models unloaded successfully")
                    except Exception as e:
                        logger.warning(f"[TTS - VRAM] Failed to unload models: {e}")
                
                try:
                    # Use unified TTS service instead of direct orchestrator
                    tts_service = TTSService(db)
                    
                    result = await tts_service.generate_audio(
                        text=assistant_message.content,
                        character=character,
                        message_id=assistant_message.id
                    )
                    
                    if result.success:
                        audio_filename = result.audio_filename
                        
                        # Save to database
                        logger.info(f"[TTS] Saving audio record to database for message {assistant_message.id}")
                        try:
                            audio_record = audio_repo.create(
                                message_id=assistant_message.id,
                                audio_filename=result.audio_filename,
                                workflow_name=result.metadata.get('workflow_name') if result.metadata else None,
                                generation_duration=result.generation_duration
                            )
                            logger.info(f"[TTS] Audio record saved successfully: {audio_record.id}")
                        except Exception as e:
                            logger.error(f"[TTS] Failed to save audio record: {e}", exc_info=True)
                        
                        logger.info(f"[TTS] Audio generated using {result.provider_name}: {audio_filename}")
                    else:
                        logger.error(f"[TTS] Audio generation failed: {result.error_message}")
                
                finally:
                    # Phase 7: Reload models after audio generation
                    if llm_client:
                        try:
                            logger.info(f"[TTS - VRAM] Reloading models after generation...")
                            if use_intent_model and intent_model:
                                # Reload both intent and character models
                                await llm_client.reload_models_after_generation(
                                    character_model=character_model,
                                    intent_model=intent_model
                                )
                            else:
                                # Only reload character model (keyword-based detection in use)
                                await llm_client.reload_model()
                            logger.info(f"[TTS - VRAM] Models reloaded successfully")
                        except Exception as e:
                            logger.error(f"[TTS - VRAM] Failed to reload models: {e}")
                        
                        logger.info("[TTS] Releasing ComfyUI lock")
            
            except Exception as tts_error:
                logger.error(f"[TTS] Failed to generate audio: {tts_error}", exc_info=True)
                # Don't fail the entire request if TTS fails
        
        response_data = ChatInThreadResponse(
            user_message=MessageResponse.from_orm(user_message, db_session=db),
            assistant_message=MessageResponse.from_orm(assistant_message, db_session=db),
            image_request_detected=image_request_detected,
            image_prompt_preview=image_prompt_preview
        )
        
        # Note: audio_url is now automatically populated by from_orm if audio exists
        # (including newly generated audio that was just saved to the database)
        
        # Auto-generate conversation title after 2 turns (4 messages)
        # Only if title is still auto-generated (not user-set)
        title_service = app_state.get("title_service")
        if title_service and conversation.title_auto_generated:
            # Count total messages in conversation (across all threads)
            thread_repo = ThreadRepository(db)
            threads = thread_repo.list_by_conversation(conversation.id)
            total_messages = 0
            for t in threads:
                total_messages += msg_repo.count_thread_messages(t.id)
            
            # Trigger after 4 messages (2 turns)
            if total_messages == 4:
                logger.info(f"[TITLE GEN] Triggering auto-title generation for conversation {conversation.id}")
                
                # Generate title synchronously (quick operation, ~1-2 seconds)
                try:
                    # Respect ComfyUI lock (don't interfere with image/audio generation)
                    comfyui_lock = app_state.get("comfyui_lock")
                    
                    # Get all messages for context
                    all_messages = []
                    for t in threads:
                        t_messages = msg_repo.list_by_thread(t.id)
                        all_messages.extend(t_messages)
                    
                    # Sort by creation time
                    all_messages.sort(key=lambda m: m.created_at)
                    
                    # Generate title using character's loaded model (fast: ~1-2 sec)
                    result = await title_service.generate_title(
                        messages=all_messages,
                        character_name=character.name,
                        model=model,
                        comfyui_lock=comfyui_lock
                    )
                    
                    if result.success and result.title:
                        # Update conversation title
                        conv_repo.update(conversation.id, title=result.title)
                        response_data.conversation_title_updated = result.title
                        logger.info(f"[TITLE GEN] Updated conversation title: {result.title}")
                    else:
                        logger.warning(f"[TITLE GEN] Failed: {result.error}")
                
                except Exception as e:
                    logger.error(f"[TITLE GEN] Title generation failed: {e}", exc_info=True)
                    # Don't fail the request, just log the error
        
        return response_data
        
    except LLMError as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


@app.post("/threads/{thread_id}/messages/stream")
async def send_message_stream(
    thread_id: str,
    request: ChatInThreadRequest,
    db: Session = Depends(get_db)
):
    """Send a message and stream the response."""
    from fastapi.responses import StreamingResponse
    import json
    
    # Verify thread exists and get conversation
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_by_id(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(thread.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    character_id = conversation.character_id
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    character = app_state["characters"][character_id]
    
    # Check LLM availability
    llm_client = app_state["llm_client"]
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    llm_available = await llm_client.health_check()
    if not llm_available:
        raise HTTPException(
            status_code=503,
            detail=f"LLM not available at {app_state['system_config'].llm.base_url}"
        )
    
    # Save user message (capture privacy status at send time)
    msg_repo = MessageRepository(db)
    user_message = msg_repo.create(
        thread_id=thread_id,
        role=MessageRole.USER,
        content=request.message,
        is_private=(conversation.is_private == "true")
    )
    
    # Phase 7.5: Fast keyword-based intent detection (no LLM, instant)
    keyword_detector = app_state.get("keyword_detector")
    detected_intents = None
    
    if keyword_detector:
        detected_intents = keyword_detector.detect(request.message)
        logger.info(
            f"[KEYWORD INTENT - STREAM] Message: '{request.message[:50]}...'"
        )
        logger.info(
            f"[KEYWORD INTENT - STREAM] Detected intents: image={detected_intents.generate_image}, "
            f"video={detected_intents.generate_video}, memory={detected_intents.record_memory}, "
            f"ambient={detected_intents.query_ambient}"
        )
    
    # Phase 5: Check if user is requesting an image (based on keyword detection)
    image_request_detected = False
    image_prompt_preview = None
    orchestrator = app_state.get("image_orchestrator")
    
    # Gate behind Phase 7 intent detection - only run orchestrator if image intent detected
    if orchestrator and character.image_generation.enabled and detected_intents and detected_intents.generate_image:
        # Note: Workflow config is fetched from database and used directly by orchestrator
        # (Not stored on character config object anymore - Phase 5 change)
        
        # Check if confirmations are disabled for this conversation
        needs_confirmation = conversation.image_confirmation_disabled != "true"
        
        # Use character's preferred model for image prompt generation (if available)
        model_for_prompt = character.preferred_llm.model if character.preferred_llm.model else None
        logger.info(f"[IMAGE DETECTION] Using model for prompt generation: {model_for_prompt}")
        
        # Get recent conversation context for better image prompt generation (last 10 messages)
        recent_messages = msg_repo.get_thread_history(thread_id, limit=10)
        conversation_context = [{
            "role": msg["role"],
            "content": msg["content"]
        } for msg in recent_messages] if recent_messages else None
        
        try:
            image_info = await orchestrator.detect_and_prepare(
                message=request.message,
                character=character,
                conversation_context=conversation_context,
                model=model_for_prompt,
                db_session=db
            )
            
            if image_info:
                image_request_detected = True
                image_prompt_preview = {
                    "prompt": image_info["prompt"],
                    "negative_prompt": image_info["negative_prompt"],
                    "needs_trigger": image_info.get("needs_trigger", False),
                    "needs_confirmation": needs_confirmation
                }
                logger.info(f"Image request detected in thread {thread_id}: {image_info['prompt'][:50]}...")
                
                # Log to image_prompts.jsonl
                try:
                    conv_dir = Path("data/debug_logs/conversations") / conversation.id
                    conv_dir.mkdir(parents=True, exist_ok=True)
                    log_file = conv_dir / "image_prompts.jsonl"
                    
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "in_conversation",
                        "thread_id": thread_id,
                        "model": model_for_prompt or "default",
                        "user_request": request.message,
                        "context_messages": len(recent_messages) if recent_messages else 0,
                        "generated_prompt": image_info["prompt"],
                        "negative_prompt": image_info["negative_prompt"],
                        "needs_trigger": image_info.get("needs_trigger", False),
                        "reasoning": image_info.get("reasoning", ""),
                        "needs_confirmation": needs_confirmation
                    }
                    
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        
                except Exception as log_error:
                    logger.warning(f"Failed to log image request: {log_error}")
        except KeyError as e:
            logger.error(f"Failed to access image info field: {e}. Image info: {image_info if 'image_info' in locals() else 'None'}")
        except Exception as e:
            logger.error(f"Failed to detect image request: {e}", exc_info=True)
    
    # Use PromptAssemblyService for memory-aware prompt building
    try:
        assembler = PromptAssemblyService(
            db=db,
            character_id=character_id,
            model_name=app_state["system_config"].llm.model,
            context_window=32768,  # TODO: Make configurable
        )
        
        # If an image is being generated, pass the prompt to the character so they can reference it
        image_context = image_prompt_preview["prompt"] if (image_request_detected and image_prompt_preview) else None
        
        # Assemble prompt with memory retrieval
        components = assembler.assemble_prompt(
            thread_id=thread_id,
            include_memories=True,
            image_prompt_context=image_context
        )
        
        # Format for LLM API
        messages = assembler.format_for_api(components)
        
        logger.info(f"Assembled prompt: {components.token_breakdown['total_used']} tokens (system: {components.token_breakdown['system']}, memories: {components.token_breakdown['memories']}, history: {components.token_breakdown['history']})")
        
    except Exception as e:
        logger.warning(f"Could not use memory-aware assembly, falling back to simple history: {e}")
        # Fallback to simple history if memory system fails
        history = msg_repo.get_thread_history(thread_id)
        messages = [{"role": "system", "content": character.system_prompt}] + history
    
    # Use character-specific LLM settings
    temperature = character.preferred_llm.temperature
    max_tokens = character.preferred_llm.max_tokens
    model = character.preferred_llm.model or app_state['system_config'].llm.model
    
    # Log LLM settings BEFORE loading
    logger.info(f"[CHAT] LLM Settings for {character.name}: model={model} {'(override)' if character.preferred_llm.model else '(default)'}, temp={temperature if temperature is not None else app_state['system_config'].llm.temperature} {'(override)' if temperature is not None else '(default)'}, max_tokens={max_tokens if max_tokens is not None else app_state['system_config'].llm.max_response_tokens} {'(override)' if max_tokens is not None else '(default)'}")
    
    # Ensure correct model is loaded for this character (handles character switches)
    await ensure_model_loaded(model, character_id)
    
    # Capture all values needed in generator before session closes (to avoid detached instance errors)
    conversation_id_for_stream = conversation.id
    last_extracted_count = conversation.last_extracted_message_count
    user_message_content = user_message.content
    user_message_id = user_message.id
    character_name = character.name
    character_id_for_stream = character.id
    conversation_is_private = conversation.is_private
    model_for_extraction = model  # Use character's model for background extraction
    character_config_for_stream = character  # Phase 8: Character config for memory profile
    title_auto_generated = conversation.title_auto_generated  # For title generation
    
    async def generate_stream():
        """Stream generator that yields SSE-formatted chunks."""
        try:
            full_content = ""
            
            # Send user message first
            yield f"data: {json.dumps({'type': 'user_message', 'content': user_message_content, 'id': user_message_id})}\n\n"
            
            # Send image detection info if found
            if image_request_detected and image_prompt_preview:
                yield f"data: {json.dumps({'type': 'image_request', 'image_info': image_prompt_preview})}\n\n"
            
            # Stream assistant response
            async for chunk in llm_client.stream_with_history(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model
            ):
                # Strip any image tags (markdown or HTML) if this is an image generation request
                # LLM sometimes adds them despite instructions not to
                if image_request_detected:
                    chunk = _strip_image_tags(chunk)
                
                full_content += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Log the full interaction to debug file
            log_llm_call(
                conversation_id=conversation_id_for_stream,
                interaction_type="chat_stream",
                model=model,
                prompt=json.dumps(messages, indent=2),  # Full messages array
                response=full_content,
                settings={
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                metadata={
                    "character_id": character.id,
                    "character_name": character_name,
                    "thread_id": thread_id
                }
            )
            
            # Save complete assistant message (use new DB session)
            from chorus_engine.db.database import get_db
            stream_db = next(get_db())
            try:
                stream_msg_repo = MessageRepository(stream_db)
                assistant_message = stream_msg_repo.create(
                    thread_id=thread_id,
                    role=MessageRole.ASSISTANT,
                    content=full_content,
                    metadata={
                        "model": app_state["system_config"].llm.model,
                        "character": character_name
                    }
                )
                stream_db.commit()
                
                # Capture assistant message ID before closing session
                assistant_message_id = assistant_message.id
                
                # Phase 7.5: Queue background memory extraction (async, uses character's model)
                # Only extract if conversation is not private
                if conversation_is_private != "true":
                    background_extractor = app_state.get("background_extractor")
                    if background_extractor:
                        # Get only the new user message for extraction
                        new_user_message = stream_msg_repo.get_by_id(user_message_id)
                        if new_user_message:
                            await background_extractor.queue_extraction(
                                conversation_id=conversation_id_for_stream,
                                character_id=character_id_for_stream,
                                messages=[new_user_message],  # Extract from this message
                                model=model_for_extraction,  # Use character's model (already loaded)
                                character_name=character_name,
                                character=character_config_for_stream  # Phase 8: Pass character config for memory profile
                            )
                            logger.debug(
                                f"[BACKGROUND MEMORY] Queued extraction for message {user_message_id} "
                                f"using model {model_for_extraction}"
                            )
                else:
                    logger.debug("[BACKGROUND MEMORY] Skipping extraction for private conversation")
                    
            finally:
                stream_db.close()
            
            # Auto-generate conversation title after 2 turns (4 messages)
            # Only if title is still auto-generated (not user-set)
            updated_title = None
            title_service = app_state.get("title_service")
            if title_service and title_auto_generated:
                try:
                    # Open new DB session for title generation
                    title_db = next(get_db())
                    try:
                        title_msg_repo = MessageRepository(title_db)
                        
                        # Count total messages
                        total_messages = title_msg_repo.count_thread_messages(thread_id)
                        
                        # Trigger after 4 messages (2 turns)
                        if total_messages == 4:
                            logger.info(f"[TITLE GEN - STREAM] Generating title for conversation {conversation_id_for_stream}")
                            
                            # Get all messages for context
                            all_messages = title_msg_repo.list_by_thread(thread_id)
                            all_messages.sort(key=lambda m: m.created_at)
                            
                            # Generate title (respects ComfyUI lock)
                            comfyui_lock = app_state.get("comfyui_lock")
                            result = await title_service.generate_title(
                                messages=all_messages,
                                character_name=character_name,
                                model=model_for_extraction,
                                comfyui_lock=comfyui_lock
                            )
                            
                            if result.success and result.title:
                                # Update conversation title
                                title_conv_repo = ConversationRepository(title_db)
                                title_conv_repo.update(conversation_id_for_stream, title=result.title)
                                updated_title = result.title
                                logger.info(f"[TITLE GEN - STREAM] Updated title: {result.title}")
                    finally:
                        title_db.close()
                except Exception as e:
                    logger.error(f"[TITLE GEN - STREAM] Failed: {e}", exc_info=True)
            
            # Send completion event with optional title update
            done_data = {'type': 'done', 'message_id': assistant_message_id}
            if updated_title:
                done_data['conversation_title_updated'] = updated_title
            yield f"data: {json.dumps(done_data)}\n\n"
            
            # Log LLM status after message generation
            await log_llm_status("AFTER MESSAGE")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# === Scene Capture Endpoints (Phase 9) ===

@app.post("/threads/{thread_id}/capture-scene-prompt")
async def generate_scene_capture_prompt(
    thread_id: str,
    db: Session = Depends(get_db)
):
    """
    Generate a scene capture prompt from third-person observer perspective.
    
    This provides a preview prompt that the user can edit before confirming
    image generation.
    
    Args:
        thread_id: Thread to capture scene from
        
    Returns:
        Prompt data for confirmation dialog
    """
    # Verify thread exists
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_by_id(thread_id)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Get conversation and character
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(thread.conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    character_id = conversation.character_id
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    character = app_state["characters"][character_id]
    
    # Verify character supports scene capture (unbounded only)
    if character.immersion_level != "unbounded":
        raise HTTPException(
            status_code=400,
            detail="Scene capture only available for unbounded immersion level"
        )
    
    if not character.image_generation.enabled:
        raise HTTPException(
            status_code=400,
            detail=f"Image generation not enabled for character {character.name}"
        )
    
    # Get recent messages (last 10 for context - matches in-conversation image generation)
    msg_repo = MessageRepository(db)
    # Use get_thread_history to get messages in chronological order (oldest first)
    # Then take the LAST 10 to get most recent messages
    all_messages = msg_repo.get_thread_history(thread_id)
    messages_dicts = all_messages[-10:] if len(all_messages) > 10 else all_messages
    
    # Convert dicts to Message objects for the service
    from chorus_engine.models.conversation import Message, MessageRole
    messages = []
    for msg_dict in messages_dicts:
        msg = Message(
            id=msg_dict.get("id"),
            thread_id=thread_id,
            role=MessageRole(msg_dict["role"]),
            content=msg_dict["content"],
            created_at=msg_dict.get("created_at")
        )
        messages.append(msg)
    
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="No messages in thread to capture scene from"
        )
    
    # Generate scene capture prompt
    from chorus_engine.services.scene_capture_prompt_service import SceneCapturePromptService
    
    scene_prompt_service = SceneCapturePromptService(
        llm_client=app_state["llm_client"]
    )
    
    model = character.preferred_llm.model if character.preferred_llm.model else None
    
    try:
        prompt_data = await scene_prompt_service.generate_prompt(
            messages=messages,
            character=character,
            model=model
        )
        
        # Log the preview generation to image_prompts.jsonl
        try:
            conv_dir = Path("data/debug_logs/conversations") / conversation.id
            conv_dir.mkdir(parents=True, exist_ok=True)
            log_file = conv_dir / "image_prompts.jsonl"
            
            # Also capture the system prompt and context string for debugging
            from chorus_engine.repositories import WorkflowRepository
            workflow_repo = WorkflowRepository(db)
            workflow_config = workflow_repo.get_default_config(character.id)
            
            # Rebuild the context string to see what was passed
            context_lines = []
            for i, msg in enumerate(messages, 1):
                role = msg.role.value.upper()
                content = msg.content[:500]  # Same limit as service
                context_lines.append(f"[Message {i}/{len(messages)}] {role}: {content}")
            context_string = "\n\n".join(context_lines)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "scene_capture_preview",
                "thread_id": thread_id,
                "model": model or "default",
                "context_messages": len(messages),
                "context_string": context_string,
                "generated_prompt": prompt_data["prompt"],
                "negative_prompt": prompt_data["negative_prompt"],
                "needs_trigger": prompt_data.get("needs_trigger", False),
                "reasoning": prompt_data.get("reasoning", "")
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as log_error:
            logger.warning(f"Failed to log scene capture preview: {log_error}")
        
        return {
            "prompt": prompt_data["prompt"],
            "negative_prompt": prompt_data["negative_prompt"],
            "reasoning": prompt_data.get("reasoning", ""),
            "needs_trigger": prompt_data.get("needs_trigger", False),
            "type": "scene_capture"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate scene capture prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate scene capture prompt: {str(e)}"
        )


@app.post("/threads/{thread_id}/capture-scene", response_model=ImageGenerationResponse)
async def capture_scene(
    thread_id: str,
    request: SceneCaptureRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a scene capture image from observer perspective.
    
    Same flow as normal image generation - waits for ComfyUI to complete
    and returns the generated image immediately.
    
    Creates a SCENE_CAPTURE message with generating status, triggers image
    generation, and returns the message. Frontend polls message status to
    get the final image.
    
    Args:
        thread_id: Thread to capture scene from
        request: Scene capture request with optional prompt, negative_prompt, seed, and workflow_id
        
    Returns:
        Message with generating status and prompt data
    """
    # Import at top of function scope
    from chorus_engine.models.conversation import Message as MessageModel, MessageRole
    
    # Verify thread exists
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_by_id(thread_id)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Get conversation and character
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(thread.conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    character_id = conversation.character_id
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    character = app_state["characters"][character_id]
    
    # Verify character supports scene capture
    if character.immersion_level != "unbounded":
        raise HTTPException(
            status_code=400,
            detail="Scene capture only available for unbounded immersion level"
        )
    
    if not character.image_generation.enabled:
        raise HTTPException(
            status_code=400,
            detail=f"Image generation not enabled for character {character.name}"
        )
    
    # If no prompt provided, generate one
    if not request.prompt:
        msg_repo = MessageRepository(db)
        # Use get_thread_history to get messages in chronological order (oldest first)
        # Then take the LAST 10 to get most recent messages
        all_messages_dicts = msg_repo.get_thread_history(thread_id)
        recent_messages_dicts = all_messages_dicts[-10:] if len(all_messages_dicts) > 10 else all_messages_dicts
        
        # Convert dicts to Message objects for the service
        messages = []
        for msg_dict in recent_messages_dicts:
            msg = MessageModel(
                id=msg_dict.get("id"),
                thread_id=thread_id,
                role=MessageRole(msg_dict["role"]),
                content=msg_dict["content"],
                created_at=msg_dict.get("created_at")
            )
            messages.append(msg)
        
        if not messages:
            raise HTTPException(
                status_code=400,
                detail="No messages in thread to capture scene from"
            )
        
        # Fetch workflow config for generation
        from chorus_engine.repositories import WorkflowRepository
        workflow_repo = WorkflowRepository(db)
        workflow_config = workflow_repo.get_default_config(character.id)
        
        from chorus_engine.services.scene_capture_prompt_service import SceneCapturePromptService
        
        scene_prompt_service = SceneCapturePromptService(
            llm_client=app_state["llm_client"]
        )
        
        try:
            prompt_data = await scene_prompt_service.generate_prompt(
                messages=messages,
                character=character,
                model=character.preferred_llm.model if character.preferred_llm.model else None,
                workflow_config=workflow_config
            )
            
            prompt = prompt_data["prompt"]
            if not request.negative_prompt:
                negative_prompt = prompt_data["negative_prompt"]
            else:
                negative_prompt = request.negative_prompt
            needs_trigger = prompt_data.get("needs_trigger", False)
            
            # Log image request to image_prompts.jsonl
            try:
                conv_dir = Path("data/debug_logs/conversations") / conversation.id
                conv_dir.mkdir(parents=True, exist_ok=True)
                log_file = conv_dir / "image_prompts.jsonl"
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "scene_capture",
                    "thread_id": thread_id,
                    "model": character.preferred_llm.model if character.preferred_llm.model else "default",
                    "context_messages": len(messages),
                    "generated_prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "needs_trigger": needs_trigger,
                    "reasoning": prompt_data.get("reasoning", ""),
                    "workflow_id": request.workflow_id or "default"
                }
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    
            except Exception as log_error:
                logger.warning(f"Failed to log image request: {log_error}")
            
        except Exception as e:
            logger.error(f"Failed to generate scene capture prompt: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate scene capture prompt: {str(e)}"
            )
    else:
        # User provided edited prompt, check if trigger needed
        prompt = request.prompt
        negative_prompt = request.negative_prompt
        needs_trigger = character.name.lower() in prompt.lower()
        
        # Fetch workflow config for trigger word and negative prompt
        from chorus_engine.repositories import WorkflowRepository
        workflow_repo = WorkflowRepository(db)
        workflow_config = workflow_repo.get_default_config(character.id)
        
        # Log user-provided prompt to image_prompts.jsonl
        try:
            conv_dir = Path("data/debug_logs/conversations") / conversation.id
            conv_dir.mkdir(parents=True, exist_ok=True)
            log_file = conv_dir / "image_prompts.jsonl"
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "scene_capture_user_edited",
                "thread_id": thread_id,
                "user_provided_prompt": prompt,
                "negative_prompt": negative_prompt,
                "needs_trigger": needs_trigger,
                "workflow_id": request.workflow_id or "default"
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as log_error:
            logger.warning(f"Failed to log image request: {log_error}")
    
    # Add trigger word if needed
    final_prompt = prompt
    if needs_trigger and workflow_config and workflow_config.get("trigger_word"):
        final_prompt = f"{workflow_config['trigger_word']}, {prompt}"
    
    # Use default negative if not provided
    if not negative_prompt and workflow_config:
        negative_prompt = workflow_config.get("negative_prompt")
    
    # Get orchestrator
    image_orchestrator = app_state.get("image_orchestrator")
    if not image_orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Image generation service not available"
        )
    
    # Acquire ComfyUI lock (same as normal image generation)
    comfyui_lock = app_state.get("comfyui_lock")
    if not comfyui_lock:
        raise HTTPException(status_code=500, detail="ComfyUI coordination lock not initialized")
    
    async with comfyui_lock:
        logger.info("[SCENE CAPTURE] Acquired ComfyUI lock - no other operations can use ComfyUI")
        
        # Unload ALL models before image generation to free VRAM (same as normal flow)
        # Acquire LLM usage lock to ensure no background tasks are using the model
        llm_usage_lock = app_state.get("llm_usage_lock")
        llm_client = app_state.get("llm_client")
        system_config = app_state.get("system_config")
        use_intent_model = system_config.intent_detection.enabled if system_config else False
        intent_model = app_state.get("intent_model", "gemma2:9b") if use_intent_model else None
        character_model = character.preferred_llm.model if character.preferred_llm.model else system_config.llm.model
        
        if not llm_usage_lock:
            raise HTTPException(status_code=500, detail="LLM usage lock not initialized")
        
        logger.info(f"[SCENE CAPTURE - VRAM] Waiting for LLM usage lock...")
        async with llm_usage_lock:
            logger.info(f"[SCENE CAPTURE - VRAM] Acquired LLM usage lock - unloading ALL models to maximize VRAM for ComfyUI...")
            
            if llm_client:
                try:
                    await llm_client.unload_all_models()
                    logger.info(f"[SCENE CAPTURE - VRAM] All LLM models unloaded successfully")
                except Exception as e:
                    logger.warning(f"[SCENE CAPTURE - VRAM] Failed to unload LLM models: {e}")
            
            # Unload TTS models if loaded
            from chorus_engine.services.tts.provider_factory import TTSProviderFactory
            unloaded_tts_providers = []  # Track which providers we unload
            try:
                # Use _providers directly to access all providers, not just available ones
                all_providers = TTSProviderFactory._providers
                for provider_name, provider in all_providers.items():
                    if provider.is_model_loaded():
                        logger.info(f"[SCENE CAPTURE - VRAM] Unloading TTS provider: {provider_name}")
                        provider.unload_model()
                        unloaded_tts_providers.append(provider_name)
            except Exception as e:
                logger.warning(f"[SCENE CAPTURE - VRAM] Failed to unload TTS models: {e}")
            
            try:
                # Create SCENE_CAPTURE message with generating status
                msg_repo = MessageRepository(db)
                scene_message = msg_repo.create(
                    thread_id=thread_id,
                    role=MessageRole.SCENE_CAPTURE,
                    content="",  # No text content for scene capture
                    metadata={
                        "image_prompt": prompt,
                        "final_prompt": final_prompt,
                        "negative_prompt": negative_prompt,
                        "seed": request.seed,
                        "workflow_id": request.workflow_id or "default",
                        "status": "generating"
                    }
                )
                
                # Generate image synchronously (waits for completion like normal flow)
                # Convert workflow_id to workflow_name if provided
                workflow_name = None
                if request.workflow_id:
                    from chorus_engine.repositories import WorkflowRepository
                    workflow_repo = WorkflowRepository(db)
                    workflow_entry = workflow_repo.get_by_id(request.workflow_id)
                    if workflow_entry:
                        workflow_name = workflow_entry.workflow_name
                
                result = await image_orchestrator.generate_image(
                    db=db,
                    conversation_id=conversation.id,
                    thread_id=thread_id,
                    character=character,
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    seed=request.seed,
                    message_id=scene_message.id,
                    workflow_name=workflow_name
                )
                
                # Convert filesystem path to HTTP URL
                full_path = Path(result["file_path"])
                relative_path = full_path.relative_to(Path("data/images"))
                http_path = f"/images/{relative_path.as_posix()}"
                
                http_thumb_path = None
                if result.get("thumbnail_path"):
                    thumb_path = Path(result["thumbnail_path"])
                    relative_thumb = thumb_path.relative_to(Path("data/images"))
                    http_thumb_path = f"/images/{relative_thumb.as_posix()}"
                
                # Update message metadata with completion
                from sqlalchemy.orm.attributes import flag_modified
                message = msg_repo.get_by_id(scene_message.id)
                if message and message.meta_data:
                    message.meta_data["status"] = "completed"
                    message.meta_data["image_id"] = result["image_id"]
                    message.meta_data["image_path"] = http_path
                    message.meta_data["thumbnail_path"] = http_thumb_path
                    message.meta_data["generation_time"] = result["generation_time"]
                    # Flag the JSON column as modified so SQLAlchemy knows to update it
                    flag_modified(message, "meta_data")
                    db.commit()
                
                logger.info(f"Scene capture image generated for message {scene_message.id}")
                
                # Return same format as normal image generation
                return ImageGenerationResponse(
                    success=True,
                    image_id=result["image_id"],
                    file_path=http_path,
                    thumbnail_path=http_thumb_path,
                    prompt=prompt,
                    generation_time=result["generation_time"]
                )
                
            except Exception as e:
                logger.error(f"Scene capture generation error: {e}", exc_info=True)
                
                # Update message with error if it was created
                try:
                    if scene_message:
                        from sqlalchemy.orm.attributes import flag_modified
                        message = msg_repo.get_by_id(scene_message.id)
                        if message and message.meta_data:
                            message.meta_data["status"] = "failed"
                            message.meta_data["error"] = str(e)
                            # Flag the JSON column as modified
                            flag_modified(message, "meta_data")
                            db.commit()
                except:
                    pass
                
                return ImageGenerationResponse(
                    success=False,
                    error=str(e)
                )
            
            finally:
                # Reload models after image generation (same as normal flow)
                if llm_client:
                    try:
                        logger.info(f"[SCENE CAPTURE - VRAM] Reloading LLM models after generation...")
                        if use_intent_model and intent_model:
                            await llm_client.reload_models_after_generation(
                                character_model=character_model,
                                intent_model=intent_model
                            )
                        else:
                            await llm_client.reload_model()
                        logger.info(f"[SCENE CAPTURE - VRAM] LLM models reloaded successfully")
                    except Exception as e:
                        logger.error(f"[SCENE CAPTURE - VRAM] Failed to reload LLM models: {e}")
                    
                    # Reload TTS models that were unloaded
                    from chorus_engine.services.tts.provider_factory import TTSProviderFactory
                    try:
                        all_providers = TTSProviderFactory._providers
                        for provider_name in unloaded_tts_providers:
                            if provider_name in all_providers:
                                provider = all_providers[provider_name]
                                logger.info(f"[SCENE CAPTURE - VRAM] Reloading TTS provider: {provider_name}")
                                provider.reload_model()
                                if provider.is_model_loaded():
                                    logger.info(f"[SCENE CAPTURE - VRAM] Successfully reloaded TTS provider: {provider_name}")
                                else:
                                    logger.warning(f"[SCENE CAPTURE - VRAM] Failed to reload TTS provider: {provider_name} - model still not loaded")
                    except Exception as e:
                        logger.warning(f"[SCENE CAPTURE - VRAM] Failed to reload TTS models: {e}")
                    
                    logger.info("[SCENE CAPTURE] Image generation complete - releasing locks")


# === Memory Management Endpoints ===

@app.post("/conversations/{conversation_id}/memories", response_model=MemoryResponse)
async def create_memory(
    conversation_id: str,
    request: MemoryCreate,
    db: Session = Depends(get_db)
):
    """Create an explicit memory for a conversation."""
    # Verify conversation exists
    conv_repo = ConversationRepository(db)
    if not conv_repo.get_by_id(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    repo = MemoryRepository(db)
    memory = repo.create(
        content=request.content,
        memory_type=MemoryType.EXPLICIT,
        conversation_id=conversation_id,
        thread_id=request.thread_id,
        tags=request.tags,
        priority=request.priority
    )
    return memory


@app.get("/conversations/{conversation_id}/memories", response_model=List[MemoryResponse])
async def list_memories(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """List all memories for a conversation."""
    repo = MemoryRepository(db)
    memories = repo.list_by_conversation(conversation_id)
    return memories


@app.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    db: Session = Depends(get_db)
):
    """Delete a memory from both database and vector store."""
    repo = MemoryRepository(db)
    
    # Get memory before deleting to access vector_id and character_id
    memory = repo.get_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Delete from database
    deleted = repo.delete(memory_id)
    
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete memory from database")
    
    # Delete from vector store if it has a vector_id
    if memory.vector_id:
        try:
            vector_store = app_state.get("vector_store")
            if vector_store:
                vector_store.delete_memories(
                    character_id=memory.character_id,
                    memory_ids=[memory.vector_id]
                )
                logger.info(f"Deleted memory {memory_id} from vector store (vector_id: {memory.vector_id})")
        except Exception as e:
            logger.error(f"Failed to delete memory from vector store: {e}")
            # Don't fail the request - memory is already deleted from DB
    
    return {"status": "deleted", "id": memory_id}


@app.get("/characters/{character_id}/memory-stats")
async def get_character_memory_stats(
    character_id: str,
    db: Session = Depends(get_db)
):
    """Get memory statistics for a character, including count and last update time."""
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    repo = MemoryRepository(db)
    memories = repo.list_by_character(character_id, memory_type=MemoryType.IMPLICIT)
    
    # Get count and most recent timestamp
    count = len(memories)
    last_updated = None
    if memories:
        # Find most recent created_at timestamp
        last_updated = max(mem.created_at for mem in memories)
    
    return {
        "character_id": character_id,
        "implicit_memory_count": count,
        "last_updated": last_updated.isoformat() if last_updated else None
    }


@app.get("/characters/{character_id}/core-memories", response_model=List[MemoryResponse])
async def get_character_core_memories(
    character_id: str,
    db: Session = Depends(get_db)
):
    """Get all core memories for a character."""
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    repo = MemoryRepository(db)
    core_memories = repo.list_by_character(character_id, memory_type=MemoryType.CORE)
    return core_memories


@app.get("/characters/{character_id}/memories", response_model=List[MemoryResponse])
async def get_character_memories(
    character_id: str,
    memory_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all memories for a character, optionally filtered by type."""
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    repo = MemoryRepository(db)
    
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
            memories = repo.list_by_character(character_id, memory_type=mem_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid memory type: {memory_type}")
    else:
        memories = repo.list_by_character(character_id)
    
    return memories


class CoreMemoryCreate(BaseModel):
    """Create core memory request."""
    content: str
    tags: Optional[List[str]] = None
    priority: Optional[int] = 2


@app.post("/characters/{character_id}/core-memories", response_model=MemoryResponse)
async def create_core_memory(
    character_id: str,
    request: CoreMemoryCreate,
    db: Session = Depends(get_db)
):
    """Create a core memory for a character. Only allowed for user-created characters."""
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    # Check if character is immutable
    if character_id in IMMUTABLE_CHARACTERS:
        raise HTTPException(
            status_code=403,
            detail=f"Cannot create core memories for immutable character '{character_id}'"
        )
    
    # Create core memory
    repo = MemoryRepository(db)
    embedding_service = app_state["embedding_service"]
    vector_store = app_state["vector_store"]
    
    # Generate embedding
    embedding = embedding_service.embed_text(request.content)
    
    # Store in vector database
    vector_id = vector_store.add_memory(
        character_id=character_id,
        content=request.content,
        memory_type=MemoryType.CORE,
        metadata={
            "tags": request.tags or [],
            "priority": request.priority or 2,
        }
    )
    
    # Store in SQL database
    memory = repo.create(
        content=request.content,
        memory_type=MemoryType.CORE,
        character_id=character_id,
        vector_id=vector_id,
        embedding_model=embedding_service.model_name,
        tags=request.tags or [],
        priority=request.priority or 2,
    )
    
    return memory


class SemanticSearchRequest(BaseModel):
    """Semantic search request."""
    query: str
    character_id: str
    memory_types: Optional[List[str]] = None
    limit: int = 10


class SemanticSearchResult(BaseModel):
    """Semantic search result with similarity score."""
    memory: MemoryResponse
    similarity: float
    rank_score: float


@app.post("/memories/search", response_model=List[SemanticSearchResult])
async def semantic_search_memories(
    request: SemanticSearchRequest,
    db: Session = Depends(get_db)
):
    """Semantic search across memories."""
    if request.character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")
    
    # Import here to avoid circular dependency
    from chorus_engine.services.memory_retrieval import MemoryRetrievalService
    
    # Initialize retrieval service
    retrieval_service = MemoryRetrievalService(
        db=db,
        vector_store=app_state["vector_store"],
        embedder=app_state["embedding_service"],
    )
    
    # Parse memory types
    memory_types = None
    if request.memory_types:
        memory_types = [MemoryType(t) for t in request.memory_types]
    
    # Retrieve memories
    retrieved = retrieval_service.retrieve_memories(
        query=request.query,
        character_id=request.character_id,
        token_budget=10000,  # High budget for search
        max_memories=request.limit,
        include_types=memory_types,
    )
    
    # Format results
    results = []
    for mem in retrieved:
        results.append(SemanticSearchResult(
            memory=MemoryResponse(
                id=str(mem.memory.id),
                conversation_id=mem.memory.conversation_id,
                thread_id=mem.memory.thread_id,
                memory_type=mem.memory.memory_type.value,
                content=mem.memory.content,
                created_at=mem.memory.created_at,
                character_id=mem.memory.character_id,
                vector_id=mem.memory.vector_id,
                embedding_model=mem.memory.embedding_model,
                priority=mem.memory.priority,
                tags=mem.memory.tags,
            ),
            similarity=mem.similarity,
            rank_score=mem.rank_score,
        ))
    
    return results


@app.post("/characters/{character_id}/reload-core-memories")
async def reload_character_core_memories(
    character_id: str,
    db: Session = Depends(get_db)
):
    """Reload core memories from character YAML configuration."""
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    # Create a fresh core memory loader
    core_loader = CoreMemoryLoader(db)
    
    try:
        # Delete existing core memories
        core_loader.delete_core_memories(character_id)
        
        # Reload from YAML
        loaded_count = core_loader.load_character_core_memories(character_id)
        
        logger.info(f"Reloaded {loaded_count} core memories for {character_id}")
        
        return {
            "status": "reloaded",
            "character_id": character_id,
            "count": loaded_count
        }
    except Exception as e:
        logger.error(f"Failed to reload core memories for {character_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload core memories: {str(e)}")


# === Phase 4.1: Implicit Memory Extraction Endpoints ===

@app.get("/characters/{character_id}/pending-memories", response_model=List[MemoryResponse])
async def get_pending_memories(
    character_id: str,
    db: Session = Depends(get_db)
):
    """Get pending implicit memories awaiting review for a character."""
    if character_id not in app_state["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    
    memory_repo = MemoryRepository(db)
    pending_memories = memory_repo.get_pending(character_id=character_id)
    
    return [MemoryResponse(
        id=str(m.id),
        conversation_id=m.conversation_id,
        thread_id=m.thread_id,
        memory_type=m.memory_type.value,
        content=m.content,
        created_at=m.created_at,
        character_id=m.character_id,
        vector_id=m.vector_id,
        embedding_model=m.embedding_model,
        priority=m.priority,
        tags=m.tags,
        confidence=m.confidence,
        category=m.category,
        status=m.status,
        source_messages=m.source_messages
    ) for m in pending_memories]


@app.post("/memories/{memory_id}/approve")
async def approve_memory(
    memory_id: str,
    db: Session = Depends(get_db)
):
    """Approve a pending memory and add it to vector store."""
    extraction_service = app_state.get("extraction_service")
    if not extraction_service:
        raise HTTPException(status_code=503, detail="Extraction service not available")
    
    success = await extraction_service.approve_pending_memory(memory_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or not pending")
    
    return {"status": "approved", "memory_id": memory_id}


@app.post("/memories/{memory_id}/reject")
async def reject_memory(
    memory_id: str,
    db: Session = Depends(get_db)
):
    """Reject and delete a pending memory."""
    memory_repo = MemoryRepository(db)
    success = memory_repo.delete(memory_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"status": "rejected", "memory_id": memory_id}


class BatchApproveRequest(BaseModel):
    """Batch approve memories request."""
    memory_ids: List[str]


@app.post("/memories/batch-approve")
async def batch_approve_memories(
    request: BatchApproveRequest,
    db: Session = Depends(get_db)
):
    """Approve multiple pending memories at once."""
    extraction_service = app_state.get("extraction_service")
    if not extraction_service:
        raise HTTPException(status_code=503, detail="Extraction service not available")
    
    approved_count = 0
    failed = []
    
    for memory_id in request.memory_ids:
        success = await extraction_service.approve_pending_memory(memory_id)
        if success:
            approved_count += 1
        else:
            failed.append(memory_id)
    
    return {
        "status": "completed",
        "approved_count": approved_count,
        "failed": failed
    }


class PrivacyUpdateRequest(BaseModel):
    """Update conversation privacy request."""
    is_private: bool


@app.put("/conversations/{conversation_id}/privacy")
async def update_conversation_privacy(
    conversation_id: str,
    request: PrivacyUpdateRequest,
    db: Session = Depends(get_db)
):
    """Toggle conversation privacy flag (prevents memory extraction when private)."""
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.set_private(conversation_id, request.is_private)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "status": "updated",
        "conversation_id": conversation_id,
        "is_private": request.is_private
    }


@app.get("/conversations/{conversation_id}/privacy")
async def get_conversation_privacy(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get conversation privacy status."""
    conv_repo = ConversationRepository(db)
    is_private = conv_repo.is_private(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "is_private": is_private
    }


class ManualExtractionRequest(BaseModel):
    """Manual extraction trigger request."""
    message_count: int = 10  # Number of recent messages to analyze


@app.post("/conversations/{conversation_id}/extract-memories")
async def manually_trigger_extraction(
    conversation_id: str,
    request: ManualExtractionRequest,
    db: Session = Depends(get_db)
):
    """Manually trigger memory extraction for a conversation."""
    extraction_manager = app_state.get("extraction_manager")
    if not extraction_manager:
        raise HTTPException(status_code=503, detail="Extraction manager not available")
    
    # Get conversation and verify it exists
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get recent messages from all threads in conversation
    thread_repo = ThreadRepository(db)
    msg_repo = MessageRepository(db)
    threads = thread_repo.list_by_conversation(conversation_id)
    
    messages = []
    for thread in threads:
        thread_messages = msg_repo.get_thread_history_objects(thread.id)
        messages.extend(thread_messages[-request.message_count:])
    
    if not messages:
        return {"status": "no_messages", "extracted_count": 0}
    
    # Get character's preferred model
    character_id = conversation.character_id
    character_config = app_state["characters"].get(character_id)
    model = None
    character_name = None
    if character_config and character_config.preferred_llm:
        model = character_config.preferred_llm.model or app_state['system_config'].llm.model
        character_name = character_config.name
    else:
        model = app_state['system_config'].llm.model
    
    # Queue extraction (Phase 8: Pass character config)
    await extraction_manager.queue_extraction(
        conversation_id=conversation_id,
        character_id=conversation.character_id,
        messages=messages,
        model=model,  # Use character's preferred model
        character_name=character_name,  # Pass character name for prompt context
        character=character_config  # Phase 8: Pass character config for memory profile
    )
    
    return {
        "status": "queued",
        "conversation_id": conversation_id,
        "message_count": len(messages)
    }


# === Phase 6: Voice Interaction Endpoints ===
# === Phase 5: Image Generation Endpoints ===

@app.post("/threads/{thread_id}/generate-image", response_model=ImageGenerationResponse)
async def generate_image(
    thread_id: str,
    request: ImageGenerationConfirmRequest,
    db: Session = Depends(get_db)
):
    """Generate an image for a thread."""
    orchestrator = app_state.get("image_orchestrator")
    
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Image generation not available (ComfyUI not configured or not running)"
        )
    
    try:
        # Get thread
        thread_repo = ThreadRepository(db)
        thread = thread_repo.get_by_id(thread_id)
        
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Get conversation
        conv_repo = ConversationRepository(db)
        conversation = conv_repo.get_by_id(thread.conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get character config
        character_id = conversation.character_id
        character = app_state["characters"].get(character_id)
        
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        
        if not character.image_generation.enabled:
            raise HTTPException(
                status_code=400,
                detail=f"Image generation not enabled for character {character_id}"
            )
        
        # Get workflow - use selected workflow_id or fall back to default
        from chorus_engine.repositories import WorkflowRepository
        from chorus_engine.models.conversation import MessageRole
        workflow_repo = WorkflowRepository(db)
        
        if request.workflow_id:
            # User selected a specific workflow
            workflow_entry = workflow_repo.get_by_id(request.workflow_id)
            if not workflow_entry:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workflow {request.workflow_id} not found"
                )
            workflow_name = workflow_entry.workflow_name
        else:
            # Use default workflow
            workflow_config = workflow_repo.get_default_config(character_id)
            if not workflow_config:
                # Instead of raising an error, send a helpful system message
                msg_repo = MessageRepository(db)
                system_message = msg_repo.create(
                    thread_id=thread_id,
                    role=MessageRole.SYSTEM,
                    content=f"Image generation is not available for {character.name}. Please upload and configure a workflow in the Workflow Manager before requesting images.",
                    metadata={"error_type": "no_workflow_configured"}
                )
                
                # Return error response - frontend will handle displaying the message
                return ImageGenerationResponse(
                    success=False,
                    error=f"No workflow configured for {character.name}. Please upload and configure a workflow in the Workflow Manager."
                )
            workflow_name = None  # orchestrator will load default
        
        # Note: Workflow config is used directly by orchestrator from database
        # (Phase 5 cleanup: workflow fields no longer stored on character.image_generation)
        
        # Update confirmation preference if requested
        if request.disable_future_confirmations:
            conversation.image_confirmation_disabled = "true"
            db.commit()
            logger.info(f"Disabled image confirmations for conversation {conversation.id}")
        
        # Acquire ComfyUI lock to prevent concurrent operations (Phase 6)
        # This ensures audio generation and image generation never run simultaneously
        comfyui_lock = app_state.get("comfyui_lock")
        
        if not comfyui_lock:
            raise HTTPException(status_code=500, detail="ComfyUI coordination lock not initialized")
        
        async with comfyui_lock:
            logger.info("[IMAGE GEN] Acquired ComfyUI lock - no other operations can use ComfyUI")
            
            # Phase 7: Unload ALL models before image generation to free VRAM
            # Acquire LLM usage lock to ensure no background tasks are using the model
            llm_usage_lock = app_state.get("llm_usage_lock")
            llm_client = app_state.get("llm_client")
            system_config = app_state.get("system_config")
            use_intent_model = system_config.intent_detection.enabled if system_config else False
            intent_model = app_state.get("intent_model", "gemma2:9b") if use_intent_model else None
            character_model = character.preferred_llm.model if character.preferred_llm.model else system_config.llm.model
            
            if not llm_usage_lock:
                raise HTTPException(status_code=500, detail="LLM usage lock not initialized")
            
            logger.info(f"[IMAGE GEN - VRAM] Waiting for LLM usage lock...")
            async with llm_usage_lock:
                logger.info(f"[IMAGE GEN - VRAM] Acquired LLM usage lock - unloading ALL models to maximize VRAM for ComfyUI...")
                
                if llm_client:
                    try:
                        # Unload all loaded models
                        await llm_client.unload_all_models()
                        logger.info(f"[IMAGE GEN - VRAM] All LLM models unloaded successfully")
                    except Exception as e:
                        logger.warning(f"[IMAGE GEN - VRAM] Failed to unload LLM models: {e}")
                
                # Unload TTS models if loaded
                from chorus_engine.services.tts.provider_factory import TTSProviderFactory
                unloaded_tts_providers = []  # Track which providers we unload
                try:
                    # Use _providers directly to access all providers, not just available ones
                    all_providers = TTSProviderFactory._providers
                    for provider_name, provider in all_providers.items():
                        if provider.is_model_loaded():
                            logger.info(f"[IMAGE GEN - VRAM] Unloading TTS provider: {provider_name}")
                            provider.unload_model()
                            unloaded_tts_providers.append(provider_name)
                except Exception as e:
                    logger.warning(f"[IMAGE GEN - VRAM] Failed to unload TTS models: {e}")
                
                try:
                    # Generate image
                    result = await orchestrator.generate_image(
                        db=db,
                        conversation_id=conversation.id,
                        thread_id=thread_id,
                        character=character,
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        seed=request.seed,
                        message_id=request.message_id,
                        workflow_name=workflow_name
                    )
                    
                    # Convert filesystem paths to HTTP URLs
                    # Path format: data/images/{conversation_id}/{image_id}_full.png
                    # URL format: /images/{conversation_id}/{image_id}_full.png
                    full_path = Path(result["file_path"])
                    relative_path = full_path.relative_to(Path("data/images"))
                    http_path = f"/images/{relative_path.as_posix()}"
                    
                    http_thumb_path = None
                    if result.get("thumbnail_path"):
                        thumb_path = Path(result["thumbnail_path"])
                        relative_thumb = thumb_path.relative_to(Path("data/images"))
                        http_thumb_path = f"/images/{relative_thumb.as_posix()}"
                    
                    # Attach image metadata to the most recent assistant message for persistence
                    # This allows images to show in conversation history after page refresh
                    msg_repo = MessageRepository(db)
                    messages = msg_repo.get_thread_history_objects(thread_id)
                    
                    # Find the most recent assistant message
                    assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]
                    if assistant_messages:
                        last_assistant_msg = assistant_messages[-1]
                        
                        # Handle metadata - it's stored as JSON, so we need to update it properly
                        # Get existing metadata or create new dict
                        if last_assistant_msg.meta_data:
                            metadata_dict = last_assistant_msg.meta_data.copy() if isinstance(last_assistant_msg.meta_data, dict) else {}
                        else:
                            metadata_dict = {}
                        
                        # Add image information
                        metadata_dict["image_id"] = result["image_id"]
                        metadata_dict["image_path"] = http_path
                        metadata_dict["thumbnail_path"] = http_thumb_path
                        metadata_dict["prompt"] = request.prompt
                        metadata_dict["generation_time"] = result["generation_time"]
                        
                        # Assign back to meta_data column
                        last_assistant_msg.meta_data = metadata_dict
                        db.commit()
                        logger.info(f"Attached image {result['image_id']} to assistant message {last_assistant_msg.id}")
                    else:
                        logger.warning(f"No assistant message found to attach image to in thread {thread_id}")
                    
                    return ImageGenerationResponse(
                        success=True,
                        image_id=result["image_id"],
                        file_path=http_path,
                        thumbnail_path=http_thumb_path,
                        prompt=request.prompt,
                        generation_time=result["generation_time"]
                    )
                    
                except Exception as e:
                    logger.error(f"Image generation failed: {e}", exc_info=True)
                    return ImageGenerationResponse(
                        success=False,
                        error=str(e)
                    )
                
                finally:
                    # Phase 7: Reload models after image generation
                    if llm_client:
                        try:
                            logger.info(f"[IMAGE GEN - VRAM] Reloading LLM models after generation...")
                            if use_intent_model and intent_model:
                                # Reload both intent and character models
                                await llm_client.reload_models_after_generation(
                                    character_model=character_model,
                                    intent_model=intent_model
                                )
                            else:
                                # Only reload character model (keyword-based detection in use)
                                await llm_client.reload_model()
                            logger.info(f"[IMAGE GEN - VRAM] LLM models reloaded successfully")
                        except Exception as e:
                            logger.error(f"[IMAGE GEN - VRAM] Failed to reload LLM models: {e}")
                    
                    # Reload TTS models that were unloaded
                    from chorus_engine.services.tts.provider_factory import TTSProviderFactory
                    try:
                        all_providers = TTSProviderFactory._providers
                        for provider_name in unloaded_tts_providers:
                            if provider_name in all_providers:
                                provider = all_providers[provider_name]
                                logger.info(f"[IMAGE GEN - VRAM] Reloading TTS provider: {provider_name}")
                                provider.reload_model()
                                if provider.is_model_loaded():
                                    logger.info(f"[IMAGE GEN - VRAM] Successfully reloaded TTS provider: {provider_name}")
                                else:
                                    logger.warning(f"[IMAGE GEN - VRAM] Failed to reload TTS provider: {provider_name} - model still not loaded")
                    except Exception as e:
                        logger.warning(f"[IMAGE GEN - VRAM] Failed to reload TTS models: {e}")
                    
                    logger.info("[IMAGE GEN] Image generation complete - releasing locks")
    
    except HTTPException:
        # Re-raise HTTP exceptions (404, 400, etc.)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_image endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.get("/conversations/{conversation_id}/images")
async def get_conversation_images(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get all images for a conversation, including scene captures."""
    image_repo = ImageRepository(db)
    msg_repo = MessageRepository(db)
    images = image_repo.get_by_conversation(conversation_id)
    
    result_images = []
    for img in images:
        # Convert filesystem paths to HTTP URLs
        full_path = Path(img.file_path)
        relative_path = full_path.relative_to(Path("data/images"))
        http_path = f"/images/{relative_path.as_posix()}"
        
        http_thumb_path = None
        if img.thumbnail_path:
            thumb_path = Path(img.thumbnail_path)
            relative_thumb = thumb_path.relative_to(Path("data/images"))
            http_thumb_path = f"/images/{relative_thumb.as_posix()}"
        
        # Check if this is a scene capture
        is_scene = False
        if img.message_id:
            msg = msg_repo.get_by_id(img.message_id)
            if msg and msg.role == MessageRole.SCENE_CAPTURE:
                is_scene = True
        
        result_images.append({
            "id": img.id,
            "message_id": img.message_id,
            "prompt": img.prompt,
            "negative_prompt": img.negative_prompt,
            "file_path": http_path,
            "thumbnail_path": http_thumb_path,
            "width": img.width,
            "height": img.height,
            "seed": img.seed,
            "generation_time": img.generation_time,
            "created_at": img.created_at.isoformat() if img.created_at else None,
            "is_scene_capture": is_scene
        })
    
    return {"images": result_images}


@app.delete("/images/{image_id}")
async def delete_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Delete a generated image and clean up message metadata."""
    image_repo = ImageRepository(db)
    image = image_repo.get_by_id(image_id)
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # If associated with a message, clean up metadata
        if image.message_id:
            msg_repo = MessageRepository(db)
            message = msg_repo.get_by_id(image.message_id)
            if message and message.meta_data:
                # Remove image references from metadata
                metadata = message.meta_data.copy() if isinstance(message.meta_data, dict) else {}
                metadata.pop('image_id', None)
                metadata.pop('image_path', None)
                metadata.pop('thumbnail_path', None)
                metadata.pop('image_prompt', None)
                metadata.pop('final_prompt', None)
                
                message.meta_data = metadata
                db.commit()
        
        # Delete from storage
        storage_service = ImageStorageService()
        await storage_service.delete_image(
            full_path=Path(image.file_path),
            thumbnail_path=Path(image.thumbnail_path) if image.thumbnail_path else None
        )
        
        # Delete from database
        image_repo.delete(image_id)
        
        return {"success": True, "message": "Image deleted"}
        
    except Exception as e:
        logger.error(f"Failed to delete image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Voice Sample Management Endpoints (Phase 6) ===

class VoiceSampleResponse(BaseModel):
    """Response model for voice sample data."""
    id: int
    character_id: str
    filename: str
    transcript: str
    is_default: bool
    uploaded_at: str


class TTSUpdateRequest(BaseModel):
    """Request model for updating TTS settings."""
    enabled: bool


@app.post("/characters/{character_id}/voice-samples", response_model=VoiceSampleResponse)
async def upload_voice_sample(
    character_id: str,
    file: UploadFile = File(...),
    transcript: str = Form(...),
    is_default: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Upload a voice sample for a character.
    
    The voice sample will be used for TTS voice cloning if the character
    has TTS generation enabled.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an audio file."
            )
        
        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Validate transcript not empty
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")
        
        # Create storage directory
        voice_samples_dir = Path("data/voice_samples") / character_id
        voice_samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{character_id}_sample_{timestamp}{file_extension}"
        file_path = voice_samples_dir / safe_filename
        
        # Save file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved voice sample: {file_path} ({len(content)} bytes)")
        
        # Create database record
        voice_sample_repo = VoiceSampleRepository(db)
        voice_sample = voice_sample_repo.create(
            character_id=character_id,
            filename=safe_filename,
            transcript=transcript.strip(),
            is_default=is_default
        )
        
        return VoiceSampleResponse(
            id=voice_sample.id,
            character_id=voice_sample.character_id,
            filename=voice_sample.filename,
            transcript=voice_sample.transcript,
            is_default=bool(voice_sample.is_default),
            uploaded_at=voice_sample.uploaded_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload voice sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters/{character_id}/voice-samples", response_model=List[VoiceSampleResponse])
async def list_voice_samples(
    character_id: str,
    db: Session = Depends(get_db)
):
    """List all voice samples for a character."""
    try:
        voice_sample_repo = VoiceSampleRepository(db)
        samples = voice_sample_repo.get_all_for_character(character_id)
        
        return [
            VoiceSampleResponse(
                id=sample.id,
                character_id=sample.character_id,
                filename=sample.filename,
                transcript=sample.transcript,
                is_default=bool(sample.is_default),
                uploaded_at=sample.uploaded_at.isoformat()
            )
            for sample in samples
        ]
        
    except Exception as e:
        logger.error(f"Failed to list voice samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/characters/{character_id}/voice-samples/{sample_id}")
async def update_voice_sample(
    character_id: str,
    sample_id: int,
    is_default: Optional[bool] = None,
    transcript: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Update a voice sample.
    
    Can update the transcript or set as default.
    """
    try:
        voice_sample_repo = VoiceSampleRepository(db)
        
        # Verify sample exists and belongs to character
        sample = voice_sample_repo.get_by_id(sample_id)
        if not sample:
            raise HTTPException(status_code=404, detail="Voice sample not found")
        
        if sample.character_id != character_id:
            raise HTTPException(
                status_code=403,
                detail="Voice sample does not belong to this character"
            )
        
        # Update transcript if provided
        if transcript is not None:
            if not transcript.strip():
                raise HTTPException(status_code=400, detail="Transcript cannot be empty")
            sample = voice_sample_repo.update_transcript(sample_id, transcript.strip())
        
        # Update default status if provided
        if is_default is not None and is_default:
            sample = voice_sample_repo.set_default(sample_id)
        
        return {
            "success": True,
            "message": "Voice sample updated",
            "sample": {
                "id": sample.id,
                "is_default": bool(sample.is_default),
                "transcript": sample.transcript
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update voice sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/characters/{character_id}/voice-samples/{sample_id}")
async def delete_voice_sample(
    character_id: str,
    sample_id: int,
    db: Session = Depends(get_db)
):
    """Delete a voice sample."""
    try:
        voice_sample_repo = VoiceSampleRepository(db)
        
        # Verify sample exists and belongs to character
        sample = voice_sample_repo.get_by_id(sample_id)
        if not sample:
            raise HTTPException(status_code=404, detail="Voice sample not found")
        
        if sample.character_id != character_id:
            raise HTTPException(
                status_code=403,
                detail="Voice sample does not belong to this character"
            )
        
        # Delete file
        voice_sample_file = Path("data/voice_samples") / character_id / sample.filename
        if voice_sample_file.exists():
            voice_sample_file.unlink()
            logger.info(f"Deleted voice sample file: {voice_sample_file}")
        
        # Delete database record
        voice_sample_repo.delete(sample_id)
        
        return {"success": True, "message": "Voice sample deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Audio Generation Endpoints (Phase 6.3) ===

@app.post("/conversations/{conversation_id}/messages/{message_id}/audio")
async def generate_message_audio(
    conversation_id: str,
    message_id: str,
    workflow_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Generate audio for a specific message using TTS."""
    from chorus_engine.repositories import ConversationRepository, MessageRepository, WorkflowRepository
    
    try:
        # Get repositories
        conv_repo = ConversationRepository(db)
        msg_repo = MessageRepository(db)
        audio_orchestrator = app_state.get("audio_orchestrator")
        comfyui_lock = app_state.get("comfyui_lock")
        
        if not audio_orchestrator:
            raise HTTPException(status_code=500, detail="Audio orchestrator not initialized")
        
        if not comfyui_lock:
            raise HTTPException(status_code=500, detail="ComfyUI coordination lock not initialized")
        
        # Get conversation and message
        conversation = conv_repo.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        message = msg_repo.get_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Verify message belongs to this conversation (through thread relationship)
        if message.thread.conversation_id != conversation_id:
            raise HTTPException(status_code=404, detail="Message not found in this conversation")
        
        # Only generate audio for assistant messages
        if message.role != "assistant":
            raise HTTPException(
                status_code=400,
                detail="Audio generation only supported for assistant messages"
            )
        
        # Check if audio already exists
        audio_repo = AudioRepository(db)
        existing_audio = audio_repo.get_by_message_id(message_id)
        if existing_audio:
            raise HTTPException(
                status_code=409,
                detail=f"Audio already exists for this message: {existing_audio.audio_filename}"
            )
        
        # Get voice sample repository
        voice_sample_repo = VoiceSampleRepository(db)
        
        # Determine workflow name (Phase 6.5: check database for default audio workflow)
        if workflow_name is None:
            # First, check database for character's default audio workflow
            workflow_repo = WorkflowRepository(db)
            db_default = workflow_repo.get_default_for_character_and_type(
                conversation.character_id,
                'audio'
            )
            
            if db_default:
                workflow_name = db_default.workflow_name
                logger.info(f"[AUDIO GEN] Using database default audio workflow: {workflow_name}")
            else:
                # Fall back to character YAML config
                character = app_state["characters"].get(conversation.character_id)
                if character and hasattr(character, 'tts_generation') and hasattr(character.tts_generation, 'default_workflow'):
                    workflow_name = character.tts_generation.default_workflow
                    logger.info(f"[AUDIO GEN] Using character config workflow: {workflow_name}")
                else:
                    workflow_name = "default_tts_workflow"
                    logger.info(f"[AUDIO GEN] Using system default workflow: {workflow_name}")
        logger.info(f"[AUDIO GEN] Starting audio generation for message {message_id}")
        
        # Phase 7: Unload ALL models before audio generation to free VRAM
        llm_client = app_state.get("llm_client")
        system_config = app_state.get("system_config")
        use_intent_model = system_config.intent_detection.enabled if system_config else False
        intent_model = app_state.get("intent_model", "gemma2:9b") if use_intent_model else None
        character = app_state["characters"].get(conversation.character_id)
        character_model = character.preferred_llm.model if character and character.preferred_llm.model else system_config.llm.model
        
        logger.info(f"[AUDIO GEN - VRAM] Unloading ALL models to maximize VRAM for TTS...")
        
        if llm_client:
            try:
                # Unload all loaded models
                await llm_client.unload_all_models()
                logger.info(f"[AUDIO GEN - VRAM] All models unloaded successfully")
            except Exception as e:
                logger.warning(f"[AUDIO GEN - VRAM] Failed to unload models: {e}")
        
        try:
            # Use unified TTS service
            tts_service = TTSService(db)
            
            result = await tts_service.generate_audio(
                text=message.content,
                character=character,
                message_id=message_id
            )
            
            if not result.success:
                raise HTTPException(status_code=500, detail=result.error_message)
            
            # Save to database
            logger.info(f"[AUDIO GEN] Saving audio record to database for message {message_id}")
            try:
                audio_record = audio_repo.create(
                    message_id=message_id,
                    audio_filename=result.audio_filename,
                    workflow_name=result.metadata.get('workflow_name') if result.metadata else None,
                    generation_duration=result.generation_duration
                )
                logger.info(f"[AUDIO GEN] Audio record saved successfully: {audio_record.id}")
            except Exception as e:
                logger.error(f"[AUDIO GEN] Failed to save audio record: {e}", exc_info=True)
                # Don't fail the request - audio file was generated successfully
                audio_record = None
            
            logger.info(f"[AUDIO GEN] Audio generated successfully using {result.provider_name}: {result.audio_filename}")
            
            return {
                "success": True,
                "audio_filename": result.audio_filename,
                "provider": result.provider_name,
                "generation_duration": result.generation_duration,
                "created_at": audio_record.created_at.isoformat() if audio_record else None
            }
        
        finally:
            # Phase 7: Reload models after audio generation
            if llm_client:
                try:
                    logger.info(f"[AUDIO GEN - VRAM] Reloading models after generation...")
                    if use_intent_model and intent_model:
                        # Reload both intent and character models
                        await llm_client.reload_models_after_generation(
                            character_model=character_model,
                            intent_model=intent_model
                        )
                    else:
                        # Only reload character model (keyword-based detection in use)
                        await llm_client.reload_model()
                        logger.info(f"[AUDIO GEN - VRAM] Models reloaded successfully")
                except Exception as e:
                    logger.error(f"[AUDIO GEN - VRAM] Failed to reload models: {e}")
            
            logger.info("[AUDIO GEN] Audio generation complete")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/messages/{message_id}/audio")
async def get_message_audio(
    conversation_id: str,
    message_id: str,
    db: Session = Depends(get_db)
):
    """Get audio metadata for a message."""
    from chorus_engine.repositories import MessageRepository
    
    try:
        # Verify message exists and belongs to conversation
        msg_repo = MessageRepository(db)
        message = msg_repo.get_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Verify message belongs to this conversation (through thread relationship)
        if message.thread.conversation_id != conversation_id:
            raise HTTPException(status_code=404, detail="Message not found in this conversation")
        
        # Get audio record
        audio_repo = AudioRepository(db)
        audio_record = audio_repo.get_by_message_id(message_id)
        
        if not audio_record:
            raise HTTPException(status_code=404, detail="No audio found for this message")
        
        return {
            "audio_filename": audio_record.audio_filename,
            "workflow_name": audio_record.workflow_name,
            "generation_duration": audio_record.generation_duration,
            "text_preprocessed": audio_record.text_preprocessed,
            "voice_sample_id": audio_record.voice_sample_id,
            "created_at": audio_record.created_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get audio metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}/messages/{message_id}/audio")
async def delete_message_audio(
    conversation_id: str,
    message_id: str,
    db: Session = Depends(get_db)
):
    """Delete generated audio for a message."""
    from chorus_engine.repositories import MessageRepository
    
    try:
        # Verify message exists and belongs to conversation
        msg_repo = MessageRepository(db)
        message = msg_repo.get_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Verify message belongs to this conversation (through thread relationship)
        if message.thread.conversation_id != conversation_id:
            raise HTTPException(status_code=404, detail="Message not found in this conversation")
        
        # Get and delete audio
        audio_repo = AudioRepository(db)
        audio_storage = app_state.get("audio_storage")
        
        audio_record = audio_repo.get_by_message_id(message_id)
        if not audio_record:
            raise HTTPException(status_code=404, detail="No audio found for this message")
        
        # Delete file
        if audio_storage:
            try:
                await audio_storage.delete_audio(audio_record.audio_filename)
                logger.info(f"Deleted audio file: {audio_record.audio_filename}")
            except FileNotFoundError:
                logger.warning(f"Audio file not found: {audio_record.audio_filename}")
        
        # Delete database record
        audio_repo.delete_by_message_id(message_id)
        
        return {"success": True, "message": "Audio deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serve an audio file."""
    try:
        audio_storage = app_state.get("audio_storage")
        if not audio_storage:
            raise HTTPException(status_code=500, detail="Audio storage not initialized")
        
        # Get file path (validates existence) - not async
        audio_path = audio_storage.get_audio_path(filename)
        
        # Determine content type from extension
        extension = audio_path.suffix.lower()
        content_type_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4"
        }
        content_type = content_type_map.get(extension, "application/octet-stream")
        
        return FileResponse(
            path=str(audio_path),
            media_type=content_type,
            filename=filename
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        logger.error(f"Failed to serve audio file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === TTS Settings Endpoints (Phase 6.3) ===

@app.patch("/conversations/{conversation_id}/tts")
async def update_conversation_tts(
    conversation_id: str,
    request: TTSUpdateRequest,
    db: Session = Depends(get_db)
):
    """Toggle TTS for a conversation."""
    from chorus_engine.repositories import ConversationRepository
    
    try:
        conv_repo = ConversationRepository(db)
        conversation = conv_repo.get_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update TTS setting
        conversation.tts_enabled = 1 if request.enabled else 0
        db.commit()
        
        logger.info(f"TTS {'enabled' if request.enabled else 'disabled'} for conversation {conversation_id}")
        
        return {
            "success": True,
            "tts_enabled": request.enabled,
            "conversation_id": conversation_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update TTS setting: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/tts")
async def get_conversation_tts(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get TTS status for a conversation."""
    from chorus_engine.repositories import ConversationRepository
    from chorus_engine.config.loader import ConfigLoader
    
    try:
        conv_repo = ConversationRepository(db)
        conversation = conv_repo.get_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get character TTS default from config
        config_loader = app_state.get("config_loader")
        character_default = False
        
        if config_loader:
            try:
                character_config = config_loader.load_character_config(conversation.character_id)
                # Safely check for tts_generation section
                if hasattr(character_config, 'tts_generation'):
                    character_default = (
                        character_config.tts_generation.enabled and
                        character_config.tts_generation.always_on
                    )
            except Exception as e:
                logger.warning(f"Could not load character config for TTS: {e}")
        
        # Determine effective TTS status
        if conversation.tts_enabled is not None:
            tts_enabled = bool(conversation.tts_enabled)
        else:
            tts_enabled = character_default
        
        return {
            "tts_enabled": tts_enabled,
            "conversation_override": conversation.tts_enabled,
            "character_default": character_default,
            "conversation_id": conversation_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get TTS status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Workflow Management Endpoints ===

@app.get("/characters/{character_id}/workflows")
async def list_character_workflows(character_id: str, db: Session = Depends(get_db)):
    """List all workflow files for a character."""
    from chorus_engine.repositories import WorkflowRepository
    
    workflow_repo = WorkflowRepository(db)
    workflows = workflow_repo.get_all_for_character(character_id)
    
    # Convert to dict format expected by frontend (Phase 6.5: include workflow_type)
    workflow_list = [
        {
            "id": w.id,
            "name": w.workflow_name,
            "workflow_type": w.workflow_type if hasattr(w, 'workflow_type') else 'image',
            "is_default": w.is_default,
            "trigger_word": w.trigger_word,
            "default_style": w.default_style,
            "negative_prompt": w.negative_prompt,
            "self_description": w.self_description,
            "created_at": w.created_at.isoformat() if w.created_at else None
        }
        for w in workflows
    ]
    
    return {"workflows": workflow_list}


@app.post("/characters/{character_id}/workflows/{workflow_name}")
async def upload_workflow(
    character_id: str,
    workflow_name: str,
    workflow_data: dict,
    workflow_type: str = Query(default="image", regex="^(image|audio|video)$"),
    db: Session = Depends(get_db)
):
    """
    Upload a new workflow file for a character (Phase 6.5: supports workflow types).
    
    Args:
        character_id: Character ID
        workflow_name: Workflow name (without .json extension)
        workflow_data: ComfyUI workflow JSON
        workflow_type: Type of workflow (image, audio, video)
    """
    from chorus_engine.repositories import WorkflowRepository
    from chorus_engine.services.workflow_manager import WorkflowType
    
    # Verify character exists
    character = app_state["characters"].get(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    try:
        # Map string to WorkflowType enum
        workflow_type_enum = WorkflowType(workflow_type)
        
        # Save workflow file using type-based folder structure
        workflow_manager = WorkflowManager()
        workflow_manager.save_workflow_by_type(
            character_id=character_id,
            workflow_type=workflow_type_enum,
            workflow_name=workflow_name,
            workflow_data=workflow_data
        )
        
        # Create database record
        workflow_repo = WorkflowRepository(db)
        workflow_file_path = f"workflows/{character_id}/{workflow_type}/{workflow_name}.json"
        
        # Auto-set as default if this is the first workflow of this type
        is_first = workflow_repo.count_for_character_and_type(character_id, workflow_type) == 0
        
        workflow = workflow_repo.create(
            character_name=character_id,
            workflow_name=workflow_name,
            workflow_file_path=workflow_file_path,
            workflow_type=workflow_type,
            is_default=is_first
        )
        
        message = f"{workflow_type.capitalize()} workflow '{workflow_name}' uploaded successfully"
        if is_first:
            message += " and set as default"
        
        return {
            "success": True,
            "message": message,
            "workflow": {
                "id": workflow.id,
                "name": workflow.workflow_name,
                "is_default": workflow.is_default
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to upload workflow: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/characters/{character_id}/workflows/{workflow_name}")
async def delete_workflow(character_id: str, workflow_name: str, db: Session = Depends(get_db)):
    """Delete a workflow file (Phase 6.5: supports type-based folders)."""
    from chorus_engine.repositories import WorkflowRepository
    
    try:
        # Get workflow from database to determine its type
        workflow_repo = WorkflowRepository(db)
        workflow = workflow_repo.get_by_name(character_id, workflow_name)
        
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_name}")
        
        # Delete file from type-specific folder
        workflow_type = workflow.workflow_type if hasattr(workflow, 'workflow_type') else 'image'
        workflow_path = Path(f"workflows/{character_id}/{workflow_type}/{workflow_name}.json")
        
        if workflow_path.exists():
            workflow_path.unlink()
            logger.info(f"Deleted workflow file: {workflow_path}")
        else:
            logger.warning(f"Workflow file not found at {workflow_path}, continuing with DB deletion")
        
        # Delete from database
        if not workflow_repo.delete(character_id, workflow_name):
            raise HTTPException(status_code=404, detail="Workflow not found in database")
        
        return {
            "success": True,
            "message": f"Workflow '{workflow_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/characters/{character_id}/workflows/{old_name}/rename")
async def rename_workflow(
    character_id: str,
    old_name: str,
    new_name: str = Query(...),
    db: Session = Depends(get_db)
):
    """Rename a workflow file."""
    from chorus_engine.repositories import WorkflowRepository
    
    try:
        # Update database
        workflow_repo = WorkflowRepository(db)
        if not workflow_repo.rename(character_id, old_name, new_name):
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Rename file
        workflow_manager = WorkflowManager()
        workflow_manager.rename_workflow(character_id, old_name, new_name)
        
        return {
            "success": True,
            "message": f"Workflow renamed from '{old_name}' to '{new_name}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rename workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/characters/{character_id}/default-workflow")
async def set_default_workflow(
    character_id: str,
    workflow_name: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    Set the default workflow for a character.
    
    Updates the database to mark the specified workflow as default.
    """
    from chorus_engine.repositories import WorkflowRepository
    
    character = app_state["characters"].get(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    try:
        workflow_repo = WorkflowRepository(db)
        
        if not workflow_repo.set_default(character_id, workflow_name):
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "success": True,
            "message": f"Default workflow set to '{workflow_name}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class WorkflowConfigUpdate(BaseModel):
    """Request body for updating workflow configuration."""
    trigger_word: Optional[str] = None
    default_style: Optional[str] = None
    negative_prompt: Optional[str] = None
    self_description: Optional[str] = None


@app.put("/workflows/{workflow_id}/config")
async def update_workflow_config(
    workflow_id: int,
    config: WorkflowConfigUpdate,
    db: Session = Depends(get_db)
):
    """
    Update workflow configuration (trigger word, style, negative prompt, self-description).
    
    Args:
        workflow_id: Workflow ID
        config: Configuration update data
    """
    from chorus_engine.repositories import WorkflowRepository
    
    try:
        workflow_repo = WorkflowRepository(db)
        
        workflow = workflow_repo.update_config(
            workflow_id=workflow_id,
            trigger_word=config.trigger_word,
            default_style=config.default_style,
            negative_prompt=config.negative_prompt,
            self_description=config.self_description
        )
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "success": True,
            "message": "Workflow configuration updated",
            "workflow": {
                "id": workflow.id,
                "name": workflow.workflow_name,
                "trigger_word": workflow.trigger_word,
                "default_style": workflow.default_style,
                "negative_prompt": workflow.negative_prompt,
                "self_description": workflow.self_description
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workflow config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_databases(
    db: Session = Depends(get_db)
):
    """
    DANGEROUS: Completely reset all databases (SQLite and ChromaDB).
    This deletes ALL conversations, messages, memories, images, and workflows.
    Character definitions are preserved.
    """
    try:
        # Import repositories
        conv_repo = ConversationRepository(db)
        thread_repo = ThreadRepository(db)
        msg_repo = MessageRepository(db)
        mem_repo = MemoryRepository(db)
        from chorus_engine.repositories import ImageRepository, WorkflowRepository
        image_repo = ImageRepository(db)
        workflow_repo = WorkflowRepository(db)
        
        # Get all conversations
        conversations = conv_repo.list_all()
        
        # Delete all conversations (cascades to threads, messages, memories)
        for conv in conversations:
            conv_repo.delete(conv.id)
        
        # Delete all generated images
        from chorus_engine.models.conversation import GeneratedImage
        db.query(GeneratedImage).delete()
        
        # Delete all workflows
        from chorus_engine.models.workflow import Workflow
        db.query(Workflow).delete()
        
        # Commit deletions
        db.commit()
        
        # Reset vector store - recreate all character collections
        vector_store = app_state["vector_store"]
        characters = app_state["characters"]
        
        for character_id in characters.keys():
            # Delete existing collection
            try:
                vector_store.client.delete_collection(f"character_{character_id}")
            except Exception:
                pass  # Collection might not exist
            
            # Recreate empty collection
            vector_store.client.create_collection(
                name=f"character_{character_id}",
                metadata={"hnsw:space": "cosine"}
            )
        
        return {
            "success": True,
            "message": "All databases reset successfully. Character definitions preserved. Workflows and images deleted."
        }
        
    except Exception as e:
        logger.error(f"Error resetting databases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset databases: {str(e)}")


# === Static Files ===
# Mount generated images directory
from pathlib import Path
images_dir = Path(__file__).parent.parent.parent / "data" / "images"
if images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
    logger.info(f"Mounted images directory: {images_dir}")

# Mount character profile images
character_images_dir = Path(__file__).parent.parent.parent / "data" / "character_images"
if character_images_dir.exists():
    app.mount("/character_images", StaticFiles(directory=str(character_images_dir)), name="character_images")
    logger.info(f"Mounted character_images directory: {character_images_dir}")

# Mount web UI static files LAST so API routes take precedence
web_dir = Path(__file__).parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
