# KoboldCpp Setup Guide

This guide covers how to use KoboldCpp as your LLM backend with Chorus Engine.

## What is KoboldCpp?

KoboldCpp is a lightweight inference engine for running GGUF-format language models with:
- Excellent CPU support with partial GPU offloading
- Lower VRAM usage compared to full GPU inference
- Simple single-binary executable
- OpenAI-compatible API

## Installation

### Windows
1. Download latest release from: https://github.com/LostRuins/koboldcpp/releases
2. Extract `koboldcpp.exe` to a folder (e.g., `J:\Tools\koboldcpp\`)
3. Download GGUF models (see Model Sources below)

### Linux/Mac
```bash
git clone https://github.com/LostRuins/koboldcpp
cd koboldcpp
make
```

## Model Sources

GGUF models can be downloaded from:
- [Hugging Face](https://huggingface.co/models?library=gguf) - Search for GGUF models
- [TheBloke's Models](https://huggingface.co/TheBloke) - High-quality quantizations

**Recommended models for Chorus Engine**:
- Qwen2.5-14B-Instruct (Q5_K_M or Q4_K_M)
- Mistral-7B-Instruct (Q5_K_M)
- OpenHermes-2.5-Mistral-7B (Q5_K_M)

**Quantization guide**:
- `Q8_0` - Highest quality, largest size (~14GB for 14B model)
- `Q5_K_M` - Great balance of quality/size (~9GB for 14B model) **RECOMMENDED**
- `Q4_K_M` - Good quality, smaller size (~7GB for 14B model)
- `Q3_K_M` - Acceptable quality, very small (~5GB for 14B model)

## Starting KoboldCpp

### Basic Startup (GPU)
```bash
koboldcpp.exe --model "J:\Models\qwen2.5-14b-instruct-q5_k_m.gguf" --port 5001 --contextsize 8192 --gpulayers 99
```

### Explanation of flags:
- `--model` - Path to your GGUF model file
- `--port` - Port number (default 5001, must match Chorus config)
- `--contextsize` - Context window size (8192, 16384, or 32768)
- `--gpulayers` - Number of layers to offload to GPU
  - `99` = all layers (full GPU)
  - `35` = half the model on GPU (for 14B models)
  - `0` = CPU only

### CPU-Only Mode
```bash
koboldcpp.exe --model "J:\Models\qwen2.5-14b-instruct-q5_k_m.gguf" --port 5001 --contextsize 8192 --threads 8
```

### Partial GPU Offloading (Mixed CPU/GPU)
```bash
koboldcpp.exe --model "J:\Models\qwen2.5-14b-instruct-q5_k_m.gguf" --port 5001 --contextsize 8192 --gpulayers 35 --threads 8
```

## Chorus Engine Configuration

Edit `config/system.yaml`:

```yaml
llm:
  provider: koboldcpp
  base_url: http://localhost:5001
  model: "qwen-14b"  # Label for logging (actual model loaded at startup)
  context_window: 8192  # MUST match --contextsize from KoboldCpp
  max_response_tokens: 4096
  temperature: 0.7
  timeout_seconds: 120
  unload_during_image_generation: false  # Cannot unload KoboldCpp dynamically
```

**Important**: The `context_window` value in Chorus **must match** the `--contextsize` flag you used when starting KoboldCpp!

## Usage Workflow

1. **Start KoboldCpp** with your desired model and settings
2. **Wait for model to load** (30-90 seconds depending on model size)
3. **Start Chorus Engine** - it will connect to the running KoboldCpp instance
4. **Chat with characters** - all characters use the loaded model
5. **To switch models**: 
   - Stop KoboldCpp
   - Restart with different `--model` flag
   - Restart Chorus Engine (or just wait for reconnect)

## Important Limitations

### Single Model Only
- KoboldCpp loads **one model at startup**
- All characters share this model
- Character `preferred_llm_model` setting is **ignored**
- To switch models: restart KoboldCpp with different `--model` flag

### No Dynamic Model Management
- Cannot unload/reload models like LM Studio or Ollama
- Model stays in memory until KoboldCpp process is stopped
- `unload_during_image_generation` should be set to `false`

### VRAM Considerations
- KoboldCpp keeps model in VRAM/RAM while running
- This reduces VRAM available for ComfyUI image generation
- Solutions:
  - Use CPU mode for KoboldCpp (slower inference, frees GPU)
  - Use smaller quantized models (Q4_K_M instead of Q5_K_M)
  - Use partial GPU offloading (`--gpulayers 20`)
  - Manually stop KoboldCpp when doing heavy image work

## Per-Request Settings That Still Work

Even though all characters share one model, these settings still apply per-request:

- ✅ `temperature` - Sampling temperature (0.0-2.0)
- ✅ `max_tokens` - Maximum response length
- ✅ `system_prompt` - Character-specific prompts
- ✅ Context windows and memory retrieval

## Troubleshooting

### "Connection refused" error
- Ensure KoboldCpp is running and loaded
- Check port matches between KoboldCpp (`--port`) and `system.yaml` (`base_url`)
- Test manually: Open `http://localhost:5001/v1/models` in browser

### "Context overflow" error
- Your prompt is too large for the context window
- Increase `--contextsize` when starting KoboldCpp
- Update `context_window` in `system.yaml` to match

### Slow inference
- Try partial GPU offloading: `--gpulayers 20`
- Use smaller quantization: Q4_K_M instead of Q5_K_M
- Reduce context: `--contextsize 4096` instead of 8192

### Model won't fit in VRAM
- Use smaller quantization (Q4_K_M or Q3_K_M)
- Use CPU mode: Remove `--gpulayers` flag
- Use partial offloading: `--gpulayers 10` (adjust up/down)

## Performance Tips

### For Maximum Speed
- Full GPU offloading: `--gpulayers 99`
- Larger quantization: Q5_K_M or Q8_0
- Sufficient context: `--contextsize 8192`

### For Maximum VRAM Availability (ComfyUI)
- CPU mode: No `--gpulayers` flag
- Smaller model: 7B instead of 14B
- Lower quantization: Q4_K_M or Q3_K_M

### Balanced (Recommended)
- Partial GPU: `--gpulayers 20-35`
- Medium quantization: Q4_K_M
- Moderate context: `--contextsize 8192`

## Comparison to Other Backends

| Feature | Ollama | LM Studio | KoboldCpp |
|---------|--------|-----------|-----------|
| Multi-model support | ✅ Yes | ✅ Yes | ❌ No (one at a time) |
| Dynamic loading | ✅ Yes | ✅ Yes | ❌ No |
| GPU offloading | ✅ Full | ✅ Full | ✅ Partial/Full |
| CPU inference | ⚠️ Slow | ⚠️ Limited | ✅ Excellent |
| GGUF support | ✅ Yes | ✅ Yes | ✅ Native |
| VRAM unloading | ✅ Yes | ✅ Yes | ❌ No |
| Ease of setup | ✅ Easy | ✅ Easy | ⚠️ Manual |

## When to Use KoboldCpp

**Good fit**:
- You have limited VRAM (< 8GB)
- You want CPU inference with GPU acceleration
- You're okay with one model at a time
- You prefer lightweight tools

**Not ideal**:
- You want to switch models frequently
- You need multiple characters with different models
- You want automatic VRAM management
- You do lots of ComfyUI image generation

## Additional Resources

- [KoboldCpp GitHub](https://github.com/LostRuins/koboldcpp)
- [KoboldCpp Wiki](https://github.com/LostRuins/koboldcpp/wiki)
- [GGUF Model Search](https://huggingface.co/models?library=gguf)
- [Model Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
