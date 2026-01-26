"""
Vision service for image analysis using vision-language models.

Phase 1: Core Vision Foundation
- Image preprocessing (resize, optimize)
- Vision model inference via Ollama
- Structured observation extraction
- VRAM-aware model management
"""

import logging
import asyncio
import json
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image
import aiohttp

logger = logging.getLogger(__name__)


class VisionServiceError(Exception):
    """Base exception for vision service errors."""
    pass


class VisionModelError(VisionServiceError):
    """Error during vision model inference."""
    pass


class ImageProcessingError(VisionServiceError):
    """Error during image preprocessing."""
    pass


@dataclass
class VisionAnalysisResult:
    """Result of vision analysis."""
    observation: str
    main_subject: str = ""
    objects: List[str] = field(default_factory=list)
    text_content: str = ""
    mood: str = ""
    confidence: float = 0.8
    tags: List[str] = field(default_factory=list)
    structured_data: Optional[Dict] = None
    processing_time_ms: int = 0
    model: str = ""
    backend: str = ""


class VisionService:
    """
    Manages vision model integration and image analysis.
    
    Coordinates with ModelManager for VRAM management and uses
    vision-language models to analyze images before character interpretation.
    
    Architecture:
    - Vision model observes (perception layer)
    - Character LLM interprets (personality layer)
    """
    
    def __init__(
        self,
        vision_config: Dict[str, Any],
        llm_config: Dict[str, Any]
    ):
        """
        Initialize vision service.
        
        Args:
            vision_config: Vision configuration from system.yaml
            llm_config: LLM configuration from system.yaml (for backend detection)
        """
        self.config = vision_config
        self.llm_config = llm_config
        self.enabled = vision_config.get("enabled", True)
        
        # Detect backend from LLM provider
        self.backend = llm_config.get("provider", "ollama")
        self.base_url = llm_config.get("base_url", "http://localhost:11434")
        
        # Validate backend support
        supported_backends = ["ollama"]  # LM Studio support coming soon
        if self.backend not in supported_backends:
            logger.warning(
                f"Vision backend '{self.backend}' not yet supported. "
                f"Supported: {supported_backends}. Vision features will be disabled."
            )
            self.enabled = False
        
        # Model configuration
        model_config = vision_config.get("model", {})
        self.model_name = model_config.get("name", "qwen3-vl:4b")
        
        # Processing configuration
        processing_config = vision_config.get("processing", {})
        self.max_retries = processing_config.get("max_retries", 2)
        self.timeout_seconds = processing_config.get("timeout_seconds", 30)
        self.resize_target = processing_config.get("resize_target", 1024)
        self.supported_formats = processing_config.get("supported_formats", 
            ["jpg", "jpeg", "png", "webp", "gif"])
        self.max_file_size_mb = processing_config.get("max_file_size_mb", 10)
        
        # Output configuration
        output_config = vision_config.get("output", {})
        self.output_format = output_config.get("format", "structured")
        self.include_confidence = output_config.get("include_confidence", True)
        
        logger.info(f"Vision service initialized (backend={self.backend}, model={self.model_name})")
    
    async def analyze_image(
        self,
        image_path: Path,
        context: Optional[str] = None,
        character_id: Optional[str] = None
    ) -> VisionAnalysisResult:
        """
        Analyze image with vision model.
        
        Args:
            image_path: Path to image file
            context: Optional user-provided context
            character_id: Optional character for personality-aware prompts
            
        Returns:
            VisionAnalysisResult with observation, confidence, tags
            
        Raises:
            VisionServiceError: If analysis fails
        """
        if not self.enabled:
            raise VisionServiceError("Vision service is disabled")
        
        logger.info(f"Analyzing image: {image_path}")
        
        try:
            # 1. Validate and preprocess image
            processed_path = await self._preprocess_image(image_path)
            
            # 2. Build vision prompt
            prompt = self._build_vision_prompt(context, character_id)
            
            # 3. Run inference with retries
            start_time = time.time()
            observation = await self._run_vision_inference_with_retry(processed_path, prompt)
            processing_time = int((time.time() - start_time) * 1000)
            
            # 4. Parse structured output
            result = self._parse_vision_output(observation)
            result.processing_time_ms = processing_time
            result.model = self.model_name
            result.backend = self.backend
            
            logger.info(f"Image analysis complete in {processing_time}ms (confidence={result.confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise VisionServiceError(f"Failed to analyze image: {e}") from e
    
    async def _preprocess_image(self, image_path: Path) -> Path:
        """
        Resize and optimize image for vision model.
        
        Args:
            image_path: Path to original image
            
        Returns:
            Path to processed image
            
        Raises:
            ImageProcessingError: If preprocessing fails
        """
        try:
            # Check file exists
            if not image_path.exists():
                raise ImageProcessingError(f"Image file not found: {image_path}")
            
            # Check file size
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                raise ImageProcessingError(
                    f"Image too large: {file_size_mb:.1f}MB (max {self.max_file_size_mb}MB)"
                )
            
            # Check format
            img = Image.open(image_path)
            format_lower = img.format.lower() if img.format else ""
            if format_lower not in self.supported_formats:
                raise ImageProcessingError(
                    f"Unsupported image format: {img.format} (supported: {self.supported_formats})"
                )
            
            # Resize if needed
            max_dim = self.resize_target
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {image_path.stem}: {img.size}")
            
            # Save processed version
            processed_path = image_path.parent / f"processed_{image_path.name}"
            
            # Convert RGBA to RGB if saving as JPEG
            if img.mode == 'RGBA' and format_lower in ['jpg', 'jpeg']:
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            
            img.save(processed_path, optimize=True, quality=85)
            logger.debug(f"Preprocessed image saved: {processed_path}")
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ImageProcessingError(f"Failed to preprocess image: {e}") from e
    
    def _build_vision_prompt(
        self,
        context: Optional[str],
        character_id: Optional[str]
    ) -> str:
        """
        Build structured vision analysis prompt.
        
        Args:
            context: Optional user-provided context
            character_id: Optional character ID
            
        Returns:
            Formatted prompt string
        """
        prompt = """Analyze this image and provide a detailed structured observation.

Output Format (JSON):
{
  "main_subject": "Primary focus of the image",
  "objects": ["list", "of", "objects"],
  "people": {
    "count": 0,
    "descriptions": []
  },
  "text_content": "Any readable text (OCR)",
  "spatial_layout": "Description of spatial arrangement",
  "mood": "Emotional tone or atmosphere",
  "colors": ["dominant", "colors"],
  "notable_details": ["interesting", "details"],
  "confidence": 0.0-1.0
}

Be specific, factual, and objective. Focus on what is visibly present."""

        if context:
            prompt += f"\n\nUser context: {context}"
        
        return prompt
    
    async def _run_vision_inference_with_retry(
        self,
        image_path: Path,
        prompt: str
    ) -> str:
        """
        Run vision inference with retry logic.
        
        Args:
            image_path: Path to processed image
            prompt: Vision prompt
            
        Returns:
            Raw vision model response
            
        Raises:
            VisionModelError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.backend == "ollama":
                    return await self._ollama_vision_inference(image_path, prompt)
                elif self.backend == "lmstudio":
                    raise NotImplementedError("LM Studio backend not yet implemented")
                else:
                    raise VisionModelError(f"Unsupported backend: {self.backend}")
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Vision inference failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Vision inference failed after {self.max_retries + 1} attempts")
        
        raise VisionModelError(f"Vision inference failed: {last_error}") from last_error
    
    async def _ollama_vision_inference(
        self,
        image_path: Path,
        prompt: str
    ) -> str:
        """
        Call Ollama API with vision model.
        
        Args:
            image_path: Path to processed image
            prompt: Vision prompt
            
        Returns:
            Raw vision model response
            
        Raises:
            VisionModelError: If API call fails
        """
        try:
            # Convert image to base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Build request payload
            payload = {
                "model": self.model_name,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }],
                "stream": False
            }
            
            # Make API request
            url = f"{self.base_url}/api/chat"
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise VisionModelError(
                            f"Ollama API error (status {response.status}): {error_text}"
                        )
                    
                    result = await response.json()
                    
                    # Extract message content
                    if "message" in result and "content" in result["message"]:
                        return result["message"]["content"]
                    else:
                        raise VisionModelError(f"Unexpected Ollama response format: {result}")
            
        except aiohttp.ClientError as e:
            raise VisionModelError(f"Ollama API request failed: {e}") from e
        except Exception as e:
            raise VisionModelError(f"Ollama vision inference failed: {e}") from e
    
    def _parse_vision_output(self, raw_output: str) -> VisionAnalysisResult:
        """
        Parse JSON output from vision model into structured result.
        
        Args:
            raw_output: Raw text from vision model
            
        Returns:
            Structured VisionAnalysisResult
        """
        try:
            # Try to parse as JSON
            data = json.loads(raw_output)
            
            # Extract fields
            tags = self._extract_tags(data)
            
            # Create human-readable observation from structured data
            observation_parts = []
            
            # Main subject
            main_subject = data.get("main_subject", "")
            if main_subject:
                observation_parts.append(main_subject)
            
            # Objects
            objects = data.get("objects", [])
            if objects:
                observation_parts.append(f"Objects present: {', '.join(objects[:10])}")
            
            # People
            people_data = data.get("people", {})
            if isinstance(people_data, dict):
                count = people_data.get("count", 0)
                descriptions = people_data.get("descriptions", [])
                if count > 0:
                    if descriptions:
                        observation_parts.append(f"{count} person(s): {'; '.join(descriptions)}")
                    else:
                        observation_parts.append(f"{count} person(s) visible")
            
            # Text content (OCR)
            text_content = data.get("text_content", "")
            if text_content:
                observation_parts.append(f"Text visible: {text_content}")
            
            # Spatial layout
            spatial = data.get("spatial_layout", "")
            if spatial:
                observation_parts.append(f"Layout: {spatial}")
            
            # Mood/atmosphere
            mood = data.get("mood", "")
            if mood:
                observation_parts.append(f"Mood: {mood}")
            
            # Colors
            colors = data.get("colors", [])
            if colors:
                observation_parts.append(f"Dominant colors: {', '.join(colors[:5])}")
            
            # Notable details
            details = data.get("notable_details", [])
            if details:
                observation_parts.append(f"Notable details: {'; '.join(details[:5])}")
            
            # Combine into readable observation
            observation = ". ".join(observation_parts) + "."
            
            return VisionAnalysisResult(
                observation=observation,
                main_subject=main_subject,
                objects=objects,
                text_content=text_content,
                mood=mood,
                confidence=float(data.get("confidence", 0.8)),
                tags=tags,
                structured_data=data
            )
            
        except json.JSONDecodeError:
            # Fallback: treat as plain text observation
            logger.warning("Vision output is not JSON, treating as plain text")
            return VisionAnalysisResult(
                observation=raw_output,
                confidence=0.7,
                tags=[]
            )
    
    def _extract_tags(self, data: Dict) -> List[str]:
        """
        Extract simple tags from structured data for quick reference.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            List of tag strings
        """
        tags = []
        
        # Add objects
        tags.extend(data.get("objects", [])[:5])  # Limit to 5 objects
        
        # Add people tag if present
        people_data = data.get("people", {})
        if isinstance(people_data, dict) and people_data.get("count", 0) > 0:
            tags.append("people")
        
        # Add text tag if text content present
        if data.get("text_content"):
            tags.append("text")
        
        # Add mood if present
        if data.get("mood"):
            tags.append(data["mood"])
        
        return tags[:10]  # Limit to 10 tags total
    
    def should_analyze_image(
        self,
        message_content: str,
        source: str = "web"
    ) -> bool:
        """
        Determine if image should be analyzed based on intent detection.
        
        Phase 1-2: Simple heuristic approach
        
        Args:
            message_content: The message text
            source: Source platform (web, discord, slack)
            
        Returns:
            True if image should be analyzed
        """
        # Web UI always analyzes (explicit upload)
        if source == "web":
            return True
        
        # Check configuration overrides
        intent_config = self.config.get("intent", {})
        if intent_config.get("bridge_always_analyze", False):
            return True
        if intent_config.get("bridge_never_analyze", False):
            return False
        
        # Bridge with no message text = skip (likely social sharing)
        if not message_content.strip():
            return False
        
        # Check for vision trigger phrases
        trigger_phrases = intent_config.get("trigger_phrases", [
            "what do you see", "look at", "what's in",
            "describe", "check this", "what does this",
            "can you see", "tell me about", "what is this",
            "help with this", "analyze", "show me"
        ])
        
        message_lower = message_content.lower()
        return any(phrase in message_lower for phrase in trigger_phrases)
