"""
Chatterbox TTS Provider

Embedded TTS using Chatterbox Turbo model with chunking support.
"""

import logging
import time
import gc
import re
from pathlib import Path
from typing import Optional, List

try:
    import torch
    import soundfile as sf
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_provider import BaseTTSProvider, TTSRequest, TTSResult
from ..audio_preprocessing import AudioPreprocessingService

logger = logging.getLogger(__name__)


class ChatterboxTTSProvider(BaseTTSProvider):
    """TTS provider using embedded Chatterbox model with intelligent chunking."""
    
    # Chunking thresholds (characters)
    DEFAULT_CHUNK_THRESHOLD = 200  # ~13-15 seconds of speech
    MAX_CHUNK_SIZE = 300  # Hard limit per chunk
    
    def __init__(self, audio_storage, system_config=None):
        """
        Initialize Chatterbox TTS provider.
        
        Args:
            audio_storage: Audio storage service for saving files
            system_config: System configuration (optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available - required for Chatterbox TTS")
        
        self.audio_storage = audio_storage
        self.system_config = system_config
        self.preprocessor = AudioPreprocessingService()
        self._model = None
        self._device = None
        
        # Model will be lazy-loaded on first TTS request
        logger.info("[Chatterbox] Provider initialized (model will load on first use)")
    
    def _initialize_model(self):
        """Load Chatterbox model."""
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("[Chatterbox] Using CUDA (GPU acceleration)")
            else:
                self._device = "cpu"
                logger.warning("[Chatterbox] Using CPU (slower performance)")
            
            # Load model
            logger.info(f"[Chatterbox] Loading Chatterbox Turbo model on {self._device}...")
            self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)
            logger.info("[Chatterbox] Model loaded successfully")
            
            # CRITICAL FIX: Monkey-patch prepare_conditionals to fix librosa float64 issue
            # librosa.resample() returns float64, which causes "expected Float but found Double"
            # errors when CUDA operations are used (CUDA is strict, CPU was forgiving)
            # This issue appeared after reinstalling PyTorch with CUDA support
            import librosa
            import numpy as np
            
            original_prepare_conditionals = self._model.prepare_conditionals
            
            def patched_prepare_conditionals(audio_prompt_path, exaggeration=1.0, norm_loudness=True):
                """Patched to ensure float32 for CUDA compatibility."""
                import librosa
                
                # Monkey-patch librosa.resample to return float32
                original_resample = librosa.resample
                
                def float32_resample(y, **kwargs):
                    result = original_resample(y, **kwargs)
                    return result.astype(np.float32) if isinstance(result, np.ndarray) else result
                
                librosa.resample = float32_resample
                
                try:
                    # Call original with patched librosa
                    return original_prepare_conditionals(audio_prompt_path, exaggeration, norm_loudness)
                finally:
                    # Restore original librosa.resample
                    librosa.resample = original_resample
            
            self._model.prepare_conditionals = patched_prepare_conditionals
            logger.info("[Chatterbox] Applied CUDA float32 compatibility patch for voice cloning")
            
        except ImportError as e:
            logger.error(f"[Chatterbox] Package not installed: {e}")
            logger.error("[Chatterbox] Install with: pip install chatterbox-tts --no-deps")
            raise
        except Exception as e:
            logger.error(f"[Chatterbox] Failed to load model: {e}")
            raise
    
    @property
    def provider_name(self) -> str:
        return "chatterbox"
    
    async def generate_audio(self, request: TTSRequest) -> TTSResult:
        """
        Generate audio using Chatterbox Turbo.
        
        Supports voice cloning if voice_sample_path provided.
        Automatically chunks long text to avoid distortion (>30s audio).
        Lazy-loads model if it was unloaded.
        """
        # Lazy load model if it was unloaded
        if not self._model:
            logger.info("[Chatterbox] Model not loaded, lazy-loading...")
            try:
                self._initialize_model()
            except Exception as e:
                return TTSResult(
                    success=False,
                    error_message=f"Failed to load Chatterbox model: {e}",
                    provider_name=self.provider_name
                )
        
        start_time = time.time()
        
        try:
            # Get chunking threshold from provider config
            chunk_threshold = self.DEFAULT_CHUNK_THRESHOLD
            if request.provider_config:
                chunk_threshold = request.provider_config.get('chunk_threshold', chunk_threshold)
            
            # Check if text needs chunking
            text_length = len(request.text)
            
            if text_length > chunk_threshold:
                logger.info(f"[Chatterbox] Text length ({text_length} chars) exceeds threshold ({chunk_threshold}), chunking...")
                return await self._generate_chunked(request, chunk_threshold, start_time)
            else:
                # Single generation for short text
                return await self._generate_single(request, start_time)
        
        except Exception as e:
            logger.error(f"[Chatterbox] TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return TTSResult(
                success=False,
                error_message=str(e),
                provider_name=self.provider_name
            )
    
    async def _generate_single(self, request: TTSRequest, start_time: float) -> TTSResult:
        """Generate audio for a single text without chunking."""
        # Check if voice cloning should be used
        use_voice_cloning = True
        if request.provider_config and 'use_voice_cloning' in request.provider_config:
            use_voice_cloning = request.provider_config.get('use_voice_cloning', True)
        
        # Generate audio
        if use_voice_cloning and request.voice_sample_path and Path(request.voice_sample_path).exists():
            # Voice cloning mode
            logger.info(f"[Chatterbox] Generating with voice cloning: {request.voice_sample_path}")
            wav = self._model.generate(
                request.text,
                audio_prompt_path=request.voice_sample_path
            )
        else:
            # Default voice mode
            if not use_voice_cloning:
                logger.info("[Chatterbox] Voice cloning disabled by config, using default voice")
            else:
                logger.info("[Chatterbox] Generating with default voice (no cloning)")
            wav = self._model.generate(request.text)
        
        # Save audio file
        audio_filename = f"msg_{request.message_id}_audio.wav"
        audio_dir = Path("data/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / audio_filename
        
        # Use soundfile directly (TorchAudio 2.9+ requires torchcodec/FFmpeg)
        wav_numpy = wav.cpu().numpy().squeeze()
        sf.write(str(audio_path), wav_numpy, self._model.sr)
        
        generation_duration = time.time() - start_time
        
        # Calculate actual audio duration
        audio_duration = wav.shape[-1] / self._model.sr
        rtf = generation_duration / audio_duration if audio_duration > 0 else 0
        
        logger.info(
            f"[Chatterbox] TTS complete: {audio_duration:.2f}s audio "
            f"in {generation_duration:.2f}s (RTF: {rtf:.2f}x)"
        )
        
        return TTSResult(
            success=True,
            audio_filename=audio_filename,
            audio_path=audio_path,
            generation_duration=generation_duration,
            provider_name=self.provider_name,
            metadata={
                'audio_duration': audio_duration,
                'real_time_factor': rtf,
                'sample_rate': self._model.sr,
                'device': self._device,
                'voice_cloned': bool(request.voice_sample_path),
                'chunked': False
            }
        )
    
    async def _generate_chunked(self, request: TTSRequest, chunk_threshold: int, start_time: float) -> TTSResult:
        """Generate audio for long text by chunking and concatenating."""
        # Check if voice cloning should be used
        use_voice_cloning = True
        if request.provider_config and 'use_voice_cloning' in request.provider_config:
            use_voice_cloning = request.provider_config.get('use_voice_cloning', True)
        
        # Split text into chunks
        chunks = self._split_into_chunks(request.text, chunk_threshold)
        logger.info(f"[Chatterbox] Split text into {len(chunks)} chunks")
        
        audio_segments = []
        total_audio_duration = 0.0
        
        # Generate audio for each chunk
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"[Chatterbox] Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            if use_voice_cloning and request.voice_sample_path and Path(request.voice_sample_path).exists():
                wav = self._model.generate(chunk, audio_prompt_path=request.voice_sample_path)
            else:
                wav = self._model.generate(chunk)
            
            wav_numpy = wav.cpu().numpy().squeeze()
            audio_segments.append(wav_numpy)
            total_audio_duration += wav.shape[-1] / self._model.sr
        
        # Concatenate all audio segments
        logger.info(f"[Chatterbox] Concatenating {len(audio_segments)} audio segments...")
        combined_audio = np.concatenate(audio_segments)
        
        # Save combined audio file
        audio_filename = f"msg_{request.message_id}_audio.wav"
        audio_dir = Path("data/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / audio_filename
        
        sf.write(str(audio_path), combined_audio, self._model.sr)
        
        generation_duration = time.time() - start_time
        rtf = generation_duration / total_audio_duration if total_audio_duration > 0 else 0
        
        logger.info(
            f"[Chatterbox] TTS complete (chunked): {total_audio_duration:.2f}s audio "
            f"from {len(chunks)} chunks in {generation_duration:.2f}s (RTF: {rtf:.2f}x)"
        )
        
        return TTSResult(
            success=True,
            audio_filename=audio_filename,
            audio_path=audio_path,
            generation_duration=generation_duration,
            provider_name=self.provider_name,
            metadata={
                'audio_duration': total_audio_duration,
                'real_time_factor': rtf,
                'sample_rate': self._model.sr,
                'device': self._device,
                'voice_cloned': bool(request.voice_sample_path),
                'chunked': True,
                'num_chunks': len(chunks)
            }
        )
    
    def _split_into_chunks(self, text: str, threshold: int) -> List[str]:
        """
        Split text into chunks at sentence boundaries.
        
        Args:
            text: Text to split
            threshold: Target maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        # Split on sentence boundaries (., !, ?)
        sentence_endings = re.compile(r'([.!?]+\s+)')
        sentences = sentence_endings.split(text)
        
        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            combined_sentences.append(sentence.strip())
        
        # Handle last sentence if no punctuation at end
        if len(sentences) % 2 == 1:
            combined_sentences.append(sentences[-1].strip())
        
        # Group sentences into chunks below threshold
        chunks = []
        current_chunk = ""
        
        for sentence in combined_sentences:
            if not sentence:
                continue
            
            # If adding this sentence exceeds threshold, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > threshold:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            
            # If single sentence exceeds max chunk size, split it further
            if len(current_chunk) > self.MAX_CHUNK_SIZE:
                # Split on commas or just force-split
                if ',' in current_chunk:
                    parts = current_chunk.split(',')
                    for j, part in enumerate(parts[:-1]):
                        chunks.append((part + ',').strip())
                    current_chunk = parts[-1].strip()
                else:
                    # Force split at threshold
                    chunks.append(current_chunk[:self.MAX_CHUNK_SIZE].strip())
                    current_chunk = current_chunk[self.MAX_CHUNK_SIZE:].strip()
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Remove empty chunks
        chunks = [c for c in chunks if c]
        
        return chunks
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Check if Chatterbox is properly configured."""
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available"
        
        if not self._model:
            return False, "Chatterbox model not loaded"
        
        # Check CUDA availability (warning, not error)
        if not torch.cuda.is_available():
            logger.warning("[Chatterbox] Running on CPU (slow performance)")
        
        return True, None
    
    def get_estimated_duration(self, text: str) -> float:
        """
        Estimate Chatterbox generation time.
        
        Chatterbox Turbo is fast:
        - GPU: ~0.5-1.0x real-time (faster than audio duration)
        - CPU: ~5-10x real-time (slower)
        """
        audio_duration = self.preprocessor.estimate_duration(text)
        
        if self._device == "cuda":
            # GPU is fast, often sub-realtime
            return audio_duration * 0.7
        else:
            # CPU is slow
            return audio_duration * 7.0
    
    def is_available(self) -> bool:
        """Check if Chatterbox provider is available."""
        # Lazy loading: model loads on first request, so just check if PyTorch is available
        return TORCH_AVAILABLE
    
    def unload_model(self) -> None:
        """Unload Chatterbox model from VRAM."""
        if self._model is not None:
            logger.info("[Chatterbox] Unloading model to free VRAM...")
            del self._model
            self._model = None
            
            # Force garbage collection
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("[Chatterbox] Model unloaded")
    
    def reload_model(self) -> None:
        """Reload Chatterbox model into VRAM."""
        if self._model is None:
            logger.info("[Chatterbox] Reloading model...")
            try:
                self._initialize_model()
            except Exception as e:
                logger.error(f"[Chatterbox] Failed to reload model: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
