"""VRAM estimation and GPU detection for model loading recommendations."""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import pynvml for NVIDIA GPU detection
HAS_PYNVML = False
NVML_ERROR = None

try:
    # Add common NVIDIA driver paths to DLL search path (Windows)
    if sys.platform == 'win32':
        import ctypes
        
        # Try to find nvml.dll
        possible_dll_names = ['nvml.dll', 'nvml64.dll']
        possible_paths = [
            os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32'),
            os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'NVIDIA Corporation', 'NVSMI'),
            os.path.join(os.environ.get('ProgramW6432', 'C:\\Program Files'), 'NVIDIA Corporation', 'NVSMI'),
            'C:\\Windows\\System32',
            'C:\\Program Files\\NVIDIA Corporation\\NVSMI',
        ]
        
        # CRITICAL: Modify PATH environment variable so pynvml can find the DLL
        # This must happen BEFORE importing pynvml
        existing_path = os.environ.get('PATH', '')
        paths_to_add = []
        
        for path in possible_paths:
            if os.path.exists(path) and path not in existing_path:
                paths_to_add.append(path)
        
        if paths_to_add:
            os.environ['PATH'] = ';'.join(paths_to_add) + ';' + existing_path
            logger.info(f"Added {len(paths_to_add)} paths to PATH for NVIDIA DLL loading")
        
        # Try to add paths to DLL search (for Python 3.8+)
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(path)
                        logger.debug(f"Added DLL directory: {path}")
                except Exception as e:
                    logger.debug(f"Could not add DLL directory {path}: {e}")
        
        # Try to pre-load the DLL directly with ctypes (backup method)
        dll_loaded = False
        for path in possible_paths:
            for dll_name in possible_dll_names:
                dll_path = os.path.join(path, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                        logger.info(f"Successfully pre-loaded NVIDIA DLL from: {dll_path}")
                        dll_loaded = True
                        break
                    except Exception as e:
                        logger.debug(f"Could not load {dll_path}: {e}")
            if dll_loaded:
                break
        
        if not dll_loaded:
            logger.warning("Could not find or load NVIDIA DLL. Searched paths: " + ", ".join(possible_paths))
    
    import pynvml
    
    # CRITICAL FIX: Manually set the NVML library handle for embedded Python
    # pynvml's auto-detection fails in embedded Python environments
    if sys.platform == 'win32':
        import ctypes
        
        # Find and load nvml.dll manually
        dll_path = None
        for path in possible_paths:
            for dll_name in ['nvml.dll', 'nvml64.dll']:
                test_path = os.path.join(path, dll_name)
                if os.path.exists(test_path):
                    dll_path = test_path
                    break
            if dll_path:
                break
        
        if dll_path:
            try:
                # Load the DLL with ctypes
                nvml_lib = ctypes.CDLL(dll_path)
                
                # Monkey-patch pynvml's library handle BEFORE nvmlInit is called
                # This bypasses pynvml's broken DLL discovery in embedded Python
                if hasattr(pynvml, '_nvmlLib_refcount'):
                    # pynvml uses _nvmlLib as the library handle
                    pynvml._nvmlLib = nvml_lib
                    logger.info(f"Manually injected NVML library handle from: {dll_path}")
                
            except Exception as e:
                logger.warning(f"Failed to manually load NVML library: {e}")
    
    # Test if pynvml actually works by trying to initialize
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        HAS_PYNVML = True
        logger.info("pynvml imported and initialized successfully")
    except Exception as init_error:
        HAS_PYNVML = False
        NVML_ERROR = f"pynvml imported but nvmlInit() failed: {init_error}"
        logger.error(NVML_ERROR)
        logger.error("This usually means NVIDIA drivers are not properly installed or the embedded Python cannot access them.")
        logger.error("Workaround: Try running the application with system Python instead of embedded Python.")
    
except ImportError as e:
    NVML_ERROR = f"pynvml not installed: {e}"
    logger.warning(NVML_ERROR)
except Exception as e:
    NVML_ERROR = str(e)
    logger.warning(f"pynvml import failed: {e}. GPU detection will not be available.")


@dataclass
class GPUInfo:
    """Information about a GPU."""
    id: int
    name: str
    total_vram_mb: int
    available_vram_mb: int
    driver_version: str
    cuda_version: str


@dataclass
class VRAMEstimate:
    """VRAM usage estimate for a model configuration."""
    estimated_vram_mb: int
    breakdown: Dict[str, int]
    available_vram_mb: int
    will_fit: bool
    utilization_percentage: int
    recommendation: Optional[Dict[str, str]] = None


class VRAMEstimator:
    """
    Estimates VRAM requirements and recommends quantizations.
    
    Features:
    - Detect NVIDIA GPU VRAM (via pynvml)
    - Calculate model VRAM requirements
    - Recommend best quantization for hardware
    - Warning system for insufficient VRAM
    """
    
    # Quantization bits per parameter (effective, accounting for metadata)
    QUANT_BITS = {
        "Q2_K": 2.5,
        "Q3_K_S": 3.0,
        "Q3_K_M": 3.5,
        "Q3_K_L": 4.0,
        "Q4_0": 4.5,
        "Q4_K_S": 4.5,
        "Q4_K_M": 5.0,
        "Q4_1": 5.0,
        "Q5_0": 5.5,
        "Q5_K_S": 5.5,
        "Q5_K_M": 6.0,
        "Q5_1": 6.0,
        "Q6_K": 6.5,
        "Q8_0": 8.5,
        "F16": 16.0,
        "F32": 32.0,
    }
    
    # GPU VRAM tiers with recommendations
    GPU_VRAM_TIERS = {
        6144: {  # 6GB (GTX 1060, RTX 2060)
            "max_params": 7,
            "recommended_quant": "Q3_K_M",
            "context_limit": 8192
        },
        8192: {  # 8GB (RTX 3060, RTX 4060)
            "max_params": 7,
            "recommended_quant": "Q4_K_M",
            "context_limit": 16384
        },
        12288: {  # 12GB (RTX 3060 12GB, RTX 4060 Ti)
            "max_params": 13,
            "recommended_quant": "Q4_K_M",
            "context_limit": 32768
        },
        16384: {  # 16GB (RTX 4060 Ti 16GB, A4000)
            "max_params": 13,
            "recommended_quant": "Q5_K_M",
            "context_limit": 32768
        },
        24576: {  # 24GB (RTX 3090, RTX 4090)
            "max_params": 34,
            "recommended_quant": "Q4_K_M",
            "context_limit": 32768
        },
        32768: {  # 32GB (RTX 5090)
            "max_params": 70,
            "recommended_quant": "Q4_K_M",
            "context_limit": 65536
        },
        49152: {  # 48GB (RTX 6000 Ada, A6000)
            "max_params": 70,
            "recommended_quant": "Q5_K_M",
            "context_limit": 65536
        },
    }
    
    @staticmethod
    def detect_gpu_vram() -> List[GPUInfo]:
        """
        Detect available NVIDIA GPUs and their VRAM.
        
        Returns:
            List of GPUInfo objects, empty list if no GPUs or detection fails
        """
        if not HAS_PYNVML:
            if NVML_ERROR:
                logger.warning(f"pynvml not available: {NVML_ERROR}")
            else:
                logger.warning("pynvml not available, cannot detect GPU VRAM")
            return []
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Get VRAM info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_mb = mem_info.total // (1024 * 1024)
                available_vram_mb = mem_info.free // (1024 * 1024)
                
                # Get driver/CUDA version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')
                except:
                    driver_version = "unknown"
                
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    cuda_major = cuda_version // 1000
                    cuda_minor = (cuda_version % 1000) // 10
                    cuda_version_str = f"{cuda_major}.{cuda_minor}"
                except:
                    cuda_version_str = "unknown"
                
                gpus.append(GPUInfo(
                    id=i,
                    name=name,
                    total_vram_mb=total_vram_mb,
                    available_vram_mb=available_vram_mb,
                    driver_version=driver_version,
                    cuda_version=cuda_version_str
                ))
            
            pynvml.nvmlShutdown()
            return gpus
            
        except Exception as e:
            error_msg = str(e)
            if "Shared Library" in error_msg or "NVML" in error_msg:
                logger.error(
                    f"Failed to detect GPU: NVIDIA driver libraries not accessible. "
                    f"Error: {error_msg}. "
                    f"If you have an NVIDIA GPU, try restarting the application or check NVIDIA driver installation."
                )
            else:
                logger.error(f"Failed to detect GPU VRAM: {e}")
            return []
    
    @staticmethod
    def estimate_vram_usage(
        model_params_billions: float,
        quantization: str,
        context_window: int,
        n_gpu_layers: int = -1
    ) -> int:
        """
        Estimate VRAM usage in MB for a model configuration.
        
        Formula components:
        1. Model weights (params * bits_per_param / 8)
        2. KV cache (context-dependent)
        3. Activation memory (generation overhead)
        4. Runtime overhead (llama.cpp, CUDA kernels)
        
        Args:
            model_params_billions: Model size in billions of parameters
            quantization: Quantization type (Q4_K_M, Q5_K_M, etc.)
            context_window: Context window size
            n_gpu_layers: Number of layers on GPU (-1 = all, 0 = CPU only)
            
        Returns:
            Estimated VRAM usage in MB
        """
        # Get bits per parameter for this quantization
        bits_per_param = VRAMEstimator.QUANT_BITS.get(quantization, 5.0)
        
        # 1. Model weights in MB
        model_weights_mb = (model_params_billions * 1000 * bits_per_param) / 8
        
        # If not all layers on GPU, reduce proportionally
        if n_gpu_layers != -1 and n_gpu_layers >= 0:
            # Rough approximation: ~5 layers per billion parameters
            total_layers = int(model_params_billions * 5)
            layer_ratio = min(n_gpu_layers / total_layers, 1.0) if total_layers > 0 else 0
            model_weights_mb *= layer_ratio
        
        # 2. KV cache (context-dependent)
        # More accurate formula: context affects KV cache linearly
        # For Q8: ~1.5MB per 1000 tokens per billion params
        # For Q4/Q5: proportionally less
        quant_factor = bits_per_param / 8.5  # Normalize to Q8
        kv_cache_mb = (context_window / 1000) * model_params_billions * 1.5 * quant_factor
        
        # 3. Activation memory (during generation)
        # Scales with sqrt of params (not linear)
        activation_mb = (model_params_billions ** 0.7) * 50
        
        # 4. Runtime overhead (llama.cpp, CUDA kernels, buffers)
        overhead_mb = 500
        
        # Total VRAM
        total_vram_mb = model_weights_mb + kv_cache_mb + activation_mb + overhead_mb
        
        return int(total_vram_mb)
    
    @staticmethod
    def recommend_quantization(
        model_params_billions: float,
        available_vram_mb: int,
        context_window: int = 32768
    ) -> Dict[str, any]:
        """
        Recommend best quantization for available hardware.
        
        Args:
            model_params_billions: Model size in billions
            available_vram_mb: Available VRAM in MB
            context_window: Desired context window
            
        Returns:
            Dict with recommended quant, alternatives, and warnings
        """
        # Try quantizations from highest to lowest quality
        quant_priority = ["Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K"]
        
        recommended = None
        alternatives = []
        
        for quant in quant_priority:
            estimated_vram = VRAMEstimator.estimate_vram_usage(
                model_params_billions, quant, context_window, n_gpu_layers=-1
            )
            
            # Add 20% buffer for safety
            with_buffer = int(estimated_vram * 1.2)
            
            if with_buffer <= available_vram_mb:
                if recommended is None:
                    recommended = quant
                else:
                    alternatives.append(quant)
        
        # Build result
        if recommended:
            estimated_vram = VRAMEstimator.estimate_vram_usage(
                model_params_billions, recommended, context_window
            )
            
            return {
                "recommended": recommended,
                "alternatives": alternatives[:2],  # Top 2 alternatives
                "estimated_vram_mb": estimated_vram,
                "will_fit": True,
                "warning": None
            }
        else:
            # Even Q2_K doesn't fit
            estimated_q2k = VRAMEstimator.estimate_vram_usage(
                model_params_billions, "Q2_K", context_window
            )
            
            return {
                "recommended": "Q2_K",
                "alternatives": [],
                "estimated_vram_mb": estimated_q2k,
                "will_fit": False,
                "warning": f"Model requires at least {estimated_q2k}MB VRAM even with lowest quantization. "
                          f"You have {available_vram_mb}MB. Consider: 1) Smaller model, "
                          f"2) Reduce context window, 3) CPU-only mode (very slow)"
            }
    
    @staticmethod
    def estimate_with_recommendation(
        model_params_billions: float,
        quantization: str,
        context_window: int,
        n_gpu_layers: int = -1
    ) -> VRAMEstimate:
        """
        Estimate VRAM usage and provide recommendations.
        
        Args:
            model_params_billions: Model size in billions
            quantization: Quantization type
            context_window: Context window size
            n_gpu_layers: GPU layers (-1 = all)
            
        Returns:
            VRAMEstimate with detailed breakdown and recommendations
        """
        # Get bits per parameter
        bits_per_param = VRAMEstimator.QUANT_BITS.get(quantization, 5.0)
        
        # Calculate components
        model_weights_mb = int((model_params_billions * 1000 * bits_per_param) / 8)
        
        # Adjust for GPU layers
        if n_gpu_layers != -1 and n_gpu_layers >= 0:
            total_layers = int(model_params_billions * 5)
            layer_ratio = min(n_gpu_layers / total_layers, 1.0) if total_layers > 0 else 0
            model_weights_mb = int(model_weights_mb * layer_ratio)
        
        kv_cache_mb = int(context_window * model_params_billions * 0.0001)
        activation_mb = int(model_params_billions * 100)
        overhead_mb = 500
        
        total_vram_mb = model_weights_mb + kv_cache_mb + activation_mb + overhead_mb
        
        # Detect available VRAM
        gpus = VRAMEstimator.detect_gpu_vram()
        available_vram_mb = gpus[0].available_vram_mb if gpus else 0
        
        # Check if it will fit
        will_fit = total_vram_mb <= available_vram_mb if available_vram_mb > 0 else True
        utilization = int((total_vram_mb / available_vram_mb) * 100) if available_vram_mb > 0 else 0
        
        # Generate recommendation
        recommendation = None
        if available_vram_mb > 0:
            if utilization > 90:
                recommendation = {
                    "type": "warning",
                    "message": f"High VRAM usage ({utilization}%). Consider lower quantization or smaller context."
                }
            elif utilization < 50 and quantization in ["Q3_K_M", "Q4_K_S"]:
                recommendation = {
                    "type": "suggestion",
                    "message": f"You have plenty of VRAM. Consider Q5_K_M or Q6_K for better quality."
                }
        
        return VRAMEstimate(
            estimated_vram_mb=total_vram_mb,
            breakdown={
                "model_weights_mb": model_weights_mb,
                "kv_cache_mb": kv_cache_mb,
                "activation_mb": activation_mb,
                "overhead_mb": overhead_mb
            },
            available_vram_mb=available_vram_mb,
            will_fit=will_fit,
            utilization_percentage=utilization,
            recommendation=recommendation
        )
    
    @staticmethod
    def get_tier_recommendation(vram_mb: int) -> Dict[str, any]:
        """
        Get recommended settings for a GPU VRAM tier.
        
        Args:
            vram_mb: Total VRAM in MB
            
        Returns:
            Dict with max_params, recommended_quant, context_limit
        """
        # Find closest tier
        tiers = sorted(VRAMEstimator.GPU_VRAM_TIERS.keys())
        
        for tier_vram in tiers:
            if vram_mb <= tier_vram:
                return VRAMEstimator.GPU_VRAM_TIERS[tier_vram]
        
        # If larger than all tiers, return highest tier
        return VRAMEstimator.GPU_VRAM_TIERS[tiers[-1]]
