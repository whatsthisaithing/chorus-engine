# Model Manager & Ollama Integration

Complete guide to Chorus Engine's integrated model management system with Ollama support.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Curated Models](#curated-models)
4. [HuggingFace Model Import](#huggingface-model-import)
5. [Model Selection in Character Editor](#model-selection-in-character-editor)
6. [VRAM Estimation](#vram-estimation)
7. [Database Tracking](#database-tracking)
8. [Using the Model Manager](#using-the-model-manager)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The **Model Manager** is Chorus Engine's integrated system for discovering, downloading, and managing LLM models with **Ollama** as the backend. It provides:

- **Curated Model Library**: Pre-vetted models tested with Chorus Engine
- **HuggingFace Import**: Direct import of GGUF models from HuggingFace
- **Automatic Chat Templates**: Extracts chat templates from GGUF metadata (no manual Modelfile creation)
- **VRAM Estimation**: GPU memory requirement predictions and fit analysis
- **Database Tracking**: Persistent model inventory across sessions
- **Character Editor Integration**: Easy model selection per character

### Why Ollama?

Ollama provides the infrastructure for model management:
- Automatic VRAM management and layer offloading
- Model caching and fast switching
- Native GGUF support with metadata extraction
- HuggingFace integration via `hf.co/` format
- Cross-platform compatibility

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│                  Model Manager UI                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │   Curated    │  │ HuggingFace  │  │ Downloaded ││
│  │    Models    │  │    Import    │  │   Models   ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌──────────────────────────────────┐
         │      FastAPI Backend (Python)     │
         │  ┌────────────┐  ┌──────────────┐│
         │  │   Model    │  │   Database   ││
         │  │  Library   │  │   Tracking   ││
         │  └────────────┘  └──────────────┘│
         └──────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │   Ollama Server   │
              │  (localhost:11434) │
              │                    │
              │  All models use    │
              │  hf.co/ format     │
              │  (unified storage) │
              └──────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  GGUF Model Files │
              │  + Chat Templates │
              │ (Ollama-managed)  │
              └──────────────────┘
```

### Unified Model Management

**Key Design**: Both curated and custom models use Ollama's `hf.co/` format:
- `hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:Q4_K_M` (curated)
- `hf.co/mradermacher/Llama-3.3-70B-Instruct-GGUF:Q4_K_M` (custom)

**Benefits**:
- No manual file management
- Automatic chat template extraction from GGUF metadata
- Ollama handles storage, deduplication, and cleanup
- Consistent behavior across all model types

### Database Schema

Models are tracked in the `downloaded_models` table:

```sql
CREATE TABLE downloaded_models (
    id INTEGER PRIMARY KEY,
    model_id VARCHAR(200) UNIQUE,      -- Identifier (file path or hf.co/ string)
    display_name VARCHAR(500),          -- Human-readable name
    repo_id VARCHAR(500),               -- HuggingFace repo (if applicable)
    filename VARCHAR(500),              -- GGUF filename (curated only)
    quantization VARCHAR(100),          -- Q4_K_M, Q5_K_M, etc.
    parameters FLOAT,                   -- Model size (7B, 14B, etc.)
    context_window INTEGER,             -- Token context (32K, 128K, etc.)
    file_size_mb FLOAT,                 -- Disk size
    file_path VARCHAR(1000),            -- Local path (curated only)
    ollama_model_name VARCHAR(500),     -- Ollama registry name (custom HF only)
    source VARCHAR(50),                 -- 'curated' or 'custom_hf'
    tags JSON,                          -- Model tags/categories
    downloaded_at DATETIME,             -- Installation timestamp
    last_used DATETIME                  -- Last activation timestamp
);
```

**Source Types**:
- **`curated`**: Pre-vetted models from Chorus Engine's library (pulled via Ollama)
- **`custom_hf`**: User-imported models from HuggingFace (pulled via Ollama)

**Note**: Both types use Ollama's `hf.co/` format and are managed identically by Ollama. The only difference is that curated models come with performance ratings and metadata in Chorus Engine's library.

---

## Curated Models

### What Are Curated Models?

Curated models are pre-vetted, tested models with known performance characteristics in Chorus Engine. Each model includes:

- **Performance Ratings**: Conversation quality, memory extraction, prompt following, creativity
- **Quantization Options**: Multiple GGUF quantizations with VRAM requirements
- **Metadata**: Parameter count, context window, recommended use cases
- **Category**: Balanced, Creative, Technical, Advanced
- **Tags**: Descriptive labels for filtering

### Model Categories

**🔹 Balanced**
- General-purpose conversation and task handling
- Good balance of performance and resource usage
- Recommended for most users

**🎨 Creative**
- Enhanced creative writing and roleplay
- More expressive language generation
- May require more VRAM

**🔧 Technical**
- Instruction following and structured tasks
- Code generation and analysis
- Precise, factual responses

**🚀 Advanced**
- Larger models with superior capabilities
- High VRAM requirements (16GB+)
- Best performance for complex scenarios

### Using Curated Models

1. **Open Model Manager**: Click the gear icon (⚙️) → **Model Management**
2. **Browse Tab**: View curated models with performance ratings
3. **Select Quantization**: Choose from dropdown based on your VRAM (badges indicate fit)
   - ✓ = Perfect fit (plenty of headroom)
   - ⚠ = Tight fit (will work but limited headroom)
   - ✗ = Won't fit (exceeds available VRAM)
4. **Download**: Click **Download** button
5. **Progress**: Modal shows download progress with percentage and ETA
6. **Completion**: Click **Done** to switch to Downloaded tab

### VRAM Estimates

Each quantization shows estimated VRAM requirements:

```
Q2_K   - 2-3GB   (lowest quality, smallest size)
Q3_K_M - 3-4GB   (low quality, small size)
Q4_K_M - 4-6GB   (good quality, standard quantization) ⭐ Recommended
Q5_K_M - 5-7GB   (high quality, larger size)
Q6_K   - 6-8GB   (very high quality, large size)
Q8_0   - 8-12GB  (near-original quality, very large)
```

**Recommendation**: Start with **Q4_K_M** for the best balance of quality and efficiency.

---

## HuggingFace Model Import

### Overview

The HuggingFace import feature allows you to import **any GGUF model** from HuggingFace directly into Ollama without manual Modelfile creation. This uses Ollama's native `hf.co/` integration which:

1. Automatically downloads the GGUF file from HuggingFace
2. Extracts the chat template from GGUF metadata
3. Registers the model in Ollama's model registry
4. Saves model info to Chorus Engine's database

### Why This Matters

**Before this feature**, importing custom GGUF models required:
1. Manual download of GGUF file
2. Creating a Modelfile with correct chat template
3. Running `ollama create` command
4. Hope you got the template format right

**With HuggingFace import**:
1. Paste HuggingFace URL
2. Select quantization
3. Click Pull
4. Done ✅ (chat template extracted automatically)

### Using HuggingFace Import

1. **Open Model Manager**: Click gear icon → **Model Management**
2. **Import Tab**: Click **Import from HuggingFace**
3. **Enter URL**: Paste the HuggingFace model repo URL
   - Example: `https://huggingface.co/mradermacher/Llama-3.3-70B-Instruct-GGUF`
4. **Load Quantizations**: Click **Load Available Quantizations**
   - The system queries the repo for available GGUF files
   - Shows file sizes and estimated VRAM requirements
5. **Select Quantization**: Choose from dropdown (fit indicators shown)
6. **Pull Model**: Click **Pull Model to Ollama**
7. **Progress**: Watch real-time download progress (MB/GB downloaded)
8. **Completion**: Click **Close** to switch to Downloaded tab

### Model Naming

Custom HF models are registered in Ollama with the format:
```
hf.co/<repo>:<quantization>

Example:
hf.co/mradermacher/Llama-3.3-70B-Instruct-GGUF:Q4_K_M
```

This format tells Ollama to:
- Pull from HuggingFace
- Use the specific quantization file
- Extract chat template from GGUF metadata

### Supported Repositories

Any HuggingFace repository containing GGUF files works, including:

**Popular Quantizers**:
- `bartowski/*` - High-quality requants with many options
- `mradermacher/*` - Extensive quantization matrix
- `QuantFactory/*` - Standard quantizations
- `unsloth/*` - Optimized models

**Model Types**:
- Instruct/Chat models (recommended)
- Base models (require prompt engineering)
- Specialized models (code, roleplay, etc.)

⚠️ **Important**: Ensure the model has a chat template in its GGUF metadata. Most modern instruct models include this automatically.

---

## Model Selection in Character Editor

### Overview

Each character can have their own preferred LLM model. The character editor provides two ways to specify a model:

1. **Select from Downloaded**: Choose from models in your library (dropdown)
2. **Custom Model Name**: Enter any Ollama model name manually

### Using Model Selection

**Access Character Editor**:
1. Click **Characters** tab
2. Select a character
3. Scroll to **LLM Settings** section

**Option 1: Select from Downloaded**
1. Select radio button: **"Select from Downloaded"**
2. Dropdown shows two groups:
   - **Curated Models**: Models from the curated library
   - **Custom HuggingFace Models**: Models imported from HuggingFace
3. Select your model
4. Save character

**Option 2: Custom Model Name**
1. Select radio button: **"Custom Model Name"**
2. Enter the full Ollama model name
   - Example: `llama3.2:3b-instruct-q4_K_M`
   - Example: `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M`
3. Save character

⚠️ **Note**: Custom model names must be valid Ollama model identifiers. The system will attempt to use this model when the character is active.

### Auto-Detection

When editing a character, the system automatically:
1. Checks if the current model exists in the downloaded models list
2. If found, selects "Select from Downloaded" mode and highlights it
3. If not found, selects "Custom Model Name" mode and shows the name

This means characters can seamlessly use models from either source.

---

## VRAM Estimation

### How It Works

The VRAM estimator predicts GPU memory requirements for each model/quantization combination using:

1. **Base Model Size**: Derived from parameter count and quantization
2. **Context Window**: Token context impacts KV cache size
3. **Overhead**: Fixed overhead for model inference (~500MB)

**Formula**:
```
estimated_vram_mb = (model_file_size_mb * 1.2) + (context_tokens / 1000) + 500
```

### GPU Detection

On page load, the Model Manager:
1. Queries `/api/system/gpu` for GPU information
2. Displays detected GPU(s) and total VRAM
3. Pre-filters VRAM dropdown to match your GPU
4. Shows fit indicators on all quantizations

**GPU Info Display**:
```
✅ GPU Detected
NVIDIA GeForce RTX 4090 - 24GB VRAM
Models are pre-filtered and recommended for your GPU
```

### Fit Indicators

Each quantization dropdown shows fit status:

- **✓ Perfect fit**: Model requires <90% of available VRAM (plenty of headroom)
- **⚠ Tight fit**: Model requires 90-99% of available VRAM (will work but limited)
- **✗ Won't fit**: Model requires >100% of available VRAM (will fail or be very slow)

**Example**:
```
Q4_K_M - 7GB (6GB VRAM) ✓ Recommended
Q5_K_M - 9GB (8GB VRAM) ⚠
Q6_K   - 12GB (11GB VRAM) ✗
```

### VRAM Tiers

The filter dropdown groups GPUs by VRAM capacity:

- **6GB**: GTX 1060, RTX 3050
- **8GB**: RTX 2070, RTX 3060
- **12GB**: RTX 3060 (12GB), RTX 3080 (12GB)
- **16GB**: RTX 4060 Ti (16GB)
- **24GB**: RTX 3090, RTX 4090
- **32GB+**: A100, H100, Multi-GPU setups

Models are filtered to show only those that fit your selected tier.

---

## Database Tracking

### Purpose

All downloaded models (curated and custom HF) are tracked in a SQLite database for:

1. **Persistence**: Model list survives restarts and browser changes
2. **Metadata Storage**: Display names, quantizations, download dates
3. **Usage Tracking**: Last used timestamp for each model
4. **Character Integration**: Dropdown population in character editor
5. **Duplicate Detection**: Prevents re-downloading existing models

### Auto-Save on Download

When a download completes, the system automatically:

1. **Curated Models**: Saves to database with full metadata
   - Source: `curated`
   - File path, size, VRAM requirements
   - Performance ratings and tags

2. **Custom HF Models**: Saves to database with available metadata
   - Source: `custom_hf`
   - Ollama model name (e.g., `hf.co/...`)
   - Repo ID, quantization, display name

### Model Lifecycle

```
Download → Database Entry → Available in Dropdowns → Used in Chat → last_used Updated
```

When you delete a model:
- **Curated**: File deleted from disk + database entry removed
- **Custom HF**: Stays in Ollama, only database entry removed (can re-add later)

---

## Using the Model Manager

### Accessing the Model Manager

**From Web Interface**:
1. Click the gear icon (⚙️) in the top-right navigation
2. Select **Model Management** from dropdown

**Modal Structure**:
- **Tab 1: Curated Models** - Browse and download curated models
- **Tab 2: Downloaded Models** - View your model library
- **Tab 3: Import from HuggingFace** - Import custom models

### Workflow: Downloading a Curated Model

```
1. Open Model Manager
2. Browse curated models (filtered by VRAM)
3. Review performance ratings and description
4. Select quantization from dropdown
5. Click "Download"
6. Ollama pulls model with hf.co/ format
7. Monitor progress in modal (streaming MB downloaded)
8. Click "Done" when complete
9. Automatically switches to "Downloaded Models" tab
10. Model appears in:
    - Downloaded Models tab (Curated section)
    - Character editor dropdowns (for selection)
    - Ollama registry (accessible via ollama list)
```

### Workflow: Importing from HuggingFace

```
1. Open Model Manager → Import tab
2. Paste HuggingFace repo URL
3. Click "Load Available Quantizations"
4. Review list of GGUF files with sizes
5. Select quantization from dropdown
6. Click "Pull Model to Ollama"
7. Monitor download progress (streaming)
8. Click "Close" when complete
9. Model appears in:
   - Downloaded Models tab (HuggingFace section)
   - Character editor dropdowns (custom HF group)
```

### Workflow: Switching Models

**Method 1: From Downloaded Models Tab**
1. Open Model Manager → Downloaded tab
2. Find the model you want to use
3. Click **"Switch To"** button
4. System updates config and restarts
5. Page reloads with new model active

**Method 2: From Character Editor**
1. Open Characters tab
2. Edit character
3. Select model from dropdown (LLM Settings)
4. Save character
5. Model will be used when this character is active

### Workflow: Deleting Models

**Both Curated and Custom HF Models** (unified behavior):
1. Open Model Manager → Downloaded tab
2. Find model in either section
3. Click **"Delete"** or **"Remove"** button (red trash icon)
4. Confirm deletion
5. Database entry removed
6. Model remains in Ollama (use `ollama rm <model>` to delete from Ollama)
7. Model removed from Chorus Engine dropdowns

**Note**: Deleting from Chorus Engine only removes tracking - the model stays in Ollama and can be manually re-added to the database if needed.

---

## Troubleshooting

### Model Not Appearing After Download

**Symptoms**: Downloaded model doesn't show in Downloaded tab or character editor dropdown

**Causes & Solutions**:

1. **Page Needs Refresh**:
   - Refresh the browser page
   - Model Manager caches on page load

2. **Database Not Updated**:
   - Check server logs for database errors
   - Verify `data/chorus.db` exists and is writable
   - Restart server

3. **Ollama Connection Issue**:
   - Check Ollama is running: `ollama list`
   - Verify model in Ollama registry
   - If in Ollama but not Chorus Engine, manually add to database (see [Manual Database Entry](#manual-database-entry))

### Download Stalled or Failed

**Symptoms**: Progress bar stuck, or download shows error

**Causes & Solutions**:

1. **Network Issue**:
   - Check internet connection
   - HuggingFace might be slow or down
   - Try again later

2. **Disk Space**:
   - Check available disk space
   - Large models (70B Q4_K_M) can be 40GB+

3. **Ollama Issue**:
   - Check Ollama logs: `ollama logs`
   - Restart Ollama service
   - Try pulling manually: `ollama pull hf.co/...`

### Chat Template Errors

**Symptoms**: "Garbage responses", formatting issues, or chat fails

**Causes & Solutions**:

1. **Using Custom HF Import**: ✅ Chat template extracted automatically
2. **Using Manual Modelfile**: ❌ You may have wrong template
   - **Solution**: Delete and re-import using HF import feature
   - The `hf.co/` format ensures correct template extraction

3. **Model Has No Template**:
   - Base models often lack chat templates
   - **Solution**: Use an instruct/chat variant instead

### VRAM Estimates Incorrect

**Symptoms**: Model fits/doesn't fit despite estimator prediction

**Causes**:

- Estimates are approximate and conservative
- Actual VRAM usage depends on:
  - Context length actively used
  - Batch size
  - GPU overhead
  - Background processes

**Solutions**:

- Treat indicators as guidelines, not guarantees
- Test models incrementally (start with smaller quantizations)
- Monitor GPU usage with `nvidia-smi`

### Model Performance Issues

**Symptoms**: Slow inference, quality issues, crashes

**Debugging Steps**:

1. **Check VRAM Fit**:
   - `nvidia-smi` to check actual usage
   - Model should use <90% of total VRAM

2. **Verify Context Window**:
   - Character config `context_window` should match model capability
   - Don't exceed model's trained context length

3. **Test with Small Context**:
   - Start new conversation (small context)
   - If fast → context overflow issue
   - If slow → model too large for GPU

4. **Try Different Quantization**:
   - Higher quant (Q6_K, Q8_0) = better quality, more VRAM
   - Lower quant (Q4_K_M, Q3_K_M) = worse quality, less VRAM

### Manual Database Entry

If a model is in Ollama but not showing in Chorus Engine:

**Using Utility Script**:
```bash
python utilities/add_curated_model.py --help
```

**Manual SQL** (advanced):
```sql
INSERT INTO downloaded_models (
    model_id, display_name, repo_id, quantization, 
    source, ollama_model_name, downloaded_at
) VALUES (
    'hf.co/bartowski/Model-Name:Q4_K_M',
    'Model Name Q4_K_M',
    'bartowski/Model-Name',
    'Q4_K_M',
    'custom_hf',
    'hf.co/bartowski/Model-Name:Q4_K_M',
    datetime('now')
);
```

Then refresh the page.

---

## Advanced Topics

### Custom Model Library

Developers can add models to the curated library:

1. Edit `chorus_engine/data/curated_models.json`
2. Add model entry with metadata:
   ```json
   {
     "id": "my-model",
     "name": "My Model",
     "description": "...",
     "repo_id": "author/model-name",
     "filename_template": "model-name-{quant}.gguf",
     "quantizations": [...],
     "performance": {...}
   }
   ```
3. Restart server
4. Model appears in curated tab

### Multiple Ollama Instances

Chorus Engine can connect to Ollama on different hosts:

**Edit `config/system.yaml`**:
```yaml
llm:
  provider: ollama
  base_url: http://other-machine:11434
```

This allows using a remote GPU server for models.

### Model Aliases

Create Ollama aliases for convenience:

```bash
ollama create my-alias -f Modelfile
```

Then use `my-alias` as the custom model name in character editor.

---

## Best Practices

### Model Selection

1. **Start Small**: Begin with 7-14B models at Q4_K_M
2. **Test Performance**: Evaluate conversation quality before committing to large downloads
3. **Match VRAM**: Use estimator to avoid out-of-memory errors
4. **Keep 1-2 Active**: Don't download every model—disk space adds up quickly

### Quantization Choice

- **Q4_K_M**: Best balance for most users (default recommendation)
- **Q5_K_M**: Slight quality improvement, ~25% more VRAM
- **Q6_K/Q8_0**: Minimal quality gains over Q5_K_M, only if VRAM to spare
- **Q3_K_M**: Emergency fallback for low VRAM (quality degrades noticeably)

### Database Maintenance

- Database self-maintains (no manual cleanup needed)
- Deleting models removes database entries automatically
- Backup `data/chorus.db` before major changes

### Performance Monitoring

Monitor model performance:
```bash
# GPU usage
nvidia-smi

# Ollama logs
ollama logs

# Chorus Engine logs
tail -f data/debug_logs/server/server_*.log
```

---

## Future Enhancements

Planned improvements to Model Manager:

- [ ] Model performance benchmarking (automatic quality testing)
- [ ] Multi-model comparison (A/B testing)
- [ ] Automatic model updates (notify when new versions available)
- [ ] Cloud model registry (community curated lists)
- [ ] Model pruning (remove unused models automatically)
- [ ] GGUF quantization conversion (requant models locally)

---

## See Also

- [GETTING_STARTED.md](../GETTING_STARTED.md) - Initial setup and configuration
- [USER_GUIDE.md](USER_GUIDE.md) - General usage instructions
- [CHARACTER_CONFIGURATION.md](CHARACTER_CONFIGURATION.md) - Character-specific model settings
- [Ollama Documentation](https://ollama.ai/docs) - Official Ollama docs
- [HuggingFace GGUF Models](https://huggingface.co/models?library=gguf) - Browse available models
