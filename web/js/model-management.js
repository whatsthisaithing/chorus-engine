/**
 * Model Management UI (Phase 10)
 * Handles integrated LLM model selection, download, and switching
 */

class ModelManager {
    constructor() {
        this.modal = null;
        this.downloadModal = null;
        this.curatedModels = [];
        this.downloadedModels = [];
        this.customModels = [];  // Track HF custom models
        this.gpuInfo = null;  // Store GPU info for badge rendering
        this.downloadJobId = null;
        this.downloadPollInterval = null;
    }

    init() {
        // Model selection modal
        const modalEl = document.getElementById('modelSelectionModal');
        if (modalEl) {
            this.modal = new bootstrap.Modal(modalEl);
        }

        // Download progress modal
        const downloadModalEl = document.getElementById('modelDownloadModal');
        if (downloadModalEl) {
            this.downloadModal = new bootstrap.Modal(downloadModalEl, {
                backdrop: 'static',
                keyboard: false
            });
        }

        // Search functionality
        const searchInput = document.getElementById('modelSearchInput');
        if (searchInput) {
            searchInput.addEventListener('input', () => this.filterModels());
        }

        // Category filter
        const categoryFilter = document.getElementById('modelCategoryFilter');
        if (categoryFilter) {
            categoryFilter.addEventListener('change', () => this.filterModels());
        }

        // VRAM filter
        const vramFilter = document.getElementById('modelVramFilter');
        if (vramFilter) {
            vramFilter.addEventListener('change', () => this.filterModels());
        }
    }

    async show() {
        try {
            // Load models and config
            await this.loadCuratedModels();
            await this.loadDownloadedModels();
            await this.loadSystemConfig();
            await this.loadGPUInfo();

            // Render initial content
            this.renderCuratedModels();
            this.renderDownloadedModels();

            // Show modal
            if (this.modal) {
                this.modal.show();
            }
        } catch (error) {
            console.error('Failed to load model data:', error);
            UI.showToast('Failed to load model data', 'danger');
        }
    }

    async loadCuratedModels() {
        try {
            const response = await fetch('/api/models/curated');
            if (!response.ok) throw new Error('Failed to load curated models');
            this.curatedModels = await response.json();
        } catch (error) {
            console.error('Failed to load curated models:', error);
            throw error;
        }
    }

    async loadDownloadedModels() {
        try {
            const response = await fetch('/api/models/downloaded');
            if (!response.ok) throw new Error('Failed to load downloaded models');
            const allModels = await response.json();
            
            // Separate by source
            this.downloadedModels = allModels.filter(m => m.source === 'curated');
            this.customModels = allModels.filter(m => m.source === 'custom_hf');
        } catch (error) {
            console.error('Failed to load downloaded models:', error);
            throw error;
        }
    }
    
    async loadSystemConfig() {
        try {
            const response = await fetch('/system/config');
            if (!response.ok) return;
            const config = await response.json();
            this.activeModel = config.llm?.model || null;
        } catch (error) {
            console.error('Failed to load system config:', error);
            this.activeModel = null;
        }
    }

    async loadGPUInfo() {
        try {
            const response = await fetch('/api/system/gpu');
            if (!response.ok) return;
            
            const gpuInfo = await response.json();
            this.gpuInfo = gpuInfo;  // Store for badge rendering
            
            // Display GPU info
            const gpuInfoEl = document.getElementById('gpuInfo');
            if (gpuInfoEl) {
                if (gpuInfo.cuda_available && gpuInfo.gpus.length > 0) {
                    const totalVRAM = gpuInfo.total_vram_mb;
                    const gpuList = gpuInfo.gpus.map(gpu => 
                        `<strong>${gpu.name}</strong> - ${Math.round(gpu.vram_mb / 1024)}GB VRAM`
                    ).join('<br>');
                    
                    gpuInfoEl.innerHTML = `
                        <div class="alert alert-success">
                            <div class="d-flex align-items-center">
                                <i class="bi bi-gpu-card me-2 fs-4"></i>
                                <div>
                                    <div class="mb-1"><strong>GPU Detected</strong></div>
                                    ${gpuList}
                                    <div class="small mt-1 text-secondary">Models are pre-filtered and recommended for your GPU</div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Pre-select VRAM filter based on GPU (Phase 10)
                    const vramFilter = document.getElementById('modelVramFilter');
                    if (vramFilter) {
                        const tier = this.getRecommendedVRAMTier(totalVRAM);
                        vramFilter.value = tier;
                        // Trigger filter update
                        this.filterModels();
                    }
                } else {
                    gpuInfoEl.innerHTML = `
                        <div class="alert alert-warning">
                            <i class="bi bi-exclamation-triangle me-2"></i>
                            <strong>No GPU detected</strong> - Models will run on CPU (very slow)
                        </div>
                    `;
                }
            }
        } catch (error) {
            console.error('Failed to load GPU info:', error);
        }
    }
    
    // ========== Custom Model Management ==========
    
    getRecommendedVRAMTier(vramMB) {
        // Select tier that matches or is slightly below available VRAM
        // This ensures we show models that will actually fit
        if (vramMB >= 48000) return '48gb';
        if (vramMB >= 32000) return '32gb';
        if (vramMB >= 24000) return '24gb';
        if (vramMB >= 16000) return '16gb';
        if (vramMB >= 12000) return '12gb';
        if (vramMB >= 8000) return '8gb';
        if (vramMB >= 6000) return '6gb';
        return 'all';  // No GPU or very low VRAM
    }

    filterModels() {
        const searchTerm = document.getElementById('modelSearchInput')?.value.toLowerCase() || '';
        const category = document.getElementById('modelCategoryFilter')?.value || 'all';
        const vramTier = document.getElementById('modelVramFilter')?.value || 'all';

        let filtered = this.curatedModels;

        // Filter by search
        if (searchTerm) {
            filtered = filtered.filter(model => 
                model.name.toLowerCase().includes(searchTerm) ||
                model.description.toLowerCase().includes(searchTerm) ||
                model.tags.some(tag => tag.toLowerCase().includes(searchTerm))
            );
        }

        // Filter by category
        if (category !== 'all') {
            filtered = filtered.filter(model => model.category === category);
        }

        // Filter by VRAM
        if (vramTier !== 'all') {
            const vramMB = this.getVRAMTierMB(vramTier);
            filtered = filtered.filter(model => 
                model.quantizations.some(q => q.min_vram_mb <= vramMB)
            );
        }

        this.renderCuratedModels(filtered);
    }

    getVRAMTierMB(tier) {
        const tiers = {
            '6gb': 6000,
            '8gb': 8000,
            '12gb': 12000,
            '16gb': 16000,
            '24gb': 24000,
            '32gb': 32000,
            '48gb': 48000
        };
        return tiers[tier] || 999999;
    }

    renderCuratedModels(models = null) {
        const container = document.getElementById('curatedModelsContainer');
        if (!container) return;

        const modelsToRender = models || this.curatedModels;

        if (modelsToRender.length === 0) {
            container.innerHTML = `
                <div class="text-center text-secondary py-5">
                    <i class="bi bi-inbox fs-1"></i>
                    <p class="mt-3">No models found matching your criteria</p>
                </div>
            `;
            return;
        }

        container.innerHTML = modelsToRender.map(model => this.renderModelCard(model)).join('');
    }

    renderModelCard(model) {
        const defaultBadge = model.default ? '<span class="badge bg-success ms-2">Default</span>' : '';
        const testedBadge = model.tested ? '<span class="badge bg-info ms-2">Tested</span>' : '';
        const categoryIcon = this.getCategoryIcon(model.category);
        const warningBadge = model.warning ? `<span class="badge bg-warning text-dark ms-2">${model.warning}</span>` : '';

        // Check which quantizations are downloaded
        const downloadedQuants = this.downloadedModels
            .filter(dm => dm.repo_id === model.repo_id)
            .map(dm => dm.quantization);
        
        const downloadedBadge = downloadedQuants.length > 0 
            ? `<span class="badge bg-secondary ms-2" title="Downloaded: ${downloadedQuants.join(', ')}">
                 <i class="bi bi-check-circle me-1"></i>${downloadedQuants.length} Downloaded
               </span>` 
            : '';

        // Find best quantization for user's GPU (Phase 10)
        let bestQuantIndex = 0;
        if (this.gpuInfo && this.gpuInfo.cuda_available && this.gpuInfo.total_vram_mb) {
            const availableVRAM = this.gpuInfo.total_vram_mb;
            // Find largest quantization that fits in 90% of available VRAM
            for (let i = model.quantizations.length - 1; i >= 0; i--) {
                if (model.quantizations[i].min_vram_mb < availableVRAM * 0.9) {
                    bestQuantIndex = i;
                    break;
                }
            }
        }
        
        const quantOptions = model.quantizations.map((q, index) => {
            // Add VRAM fit badge if GPU detected (Phase 10)
            let badge = '';
            let recommended = '';
            if (this.gpuInfo && this.gpuInfo.cuda_available && this.gpuInfo.total_vram_mb) {
                const availableVRAM = this.gpuInfo.total_vram_mb;
                const requiredVRAM = q.min_vram_mb;
                
                if (index === bestQuantIndex) {
                    recommended = ' 🌟 Recommended';
                }
                
                if (requiredVRAM < availableVRAM * 0.9) {
                    badge = ' ✓';  // Perfect fit
                } else if (requiredVRAM < availableVRAM) {
                    badge = ' ⚠';  // Tight fit
                } else {
                    badge = ' ✗';  // Won't fit
                }
            }
            
            return `
                <option value="${q.quant}" data-size="${q.file_size_mb}" data-vram="${q.min_vram_mb}" ${index === bestQuantIndex ? 'selected' : ''}>
                    ${q.quant} - ${Math.round(q.file_size_mb / 1024)}GB (${Math.round(q.min_vram_mb / 1024)}GB VRAM)${badge}${recommended}
                </option>
            `;
        }).join('');

        return `
            <div class="card bg-dark border-secondary mb-3 model-card text-light" data-model-id="${model.id}">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h6 class="card-title mb-0 text-light">
                            ${categoryIcon} ${model.name}
                            ${defaultBadge}${testedBadge}${downloadedBadge}${warningBadge}
                        </h6>
                        <span class="badge bg-secondary">${model.parameters}B params</span>
                    </div>
                    
                    <p class="card-text text-secondary small mb-2">${model.description}</p>
                    
                    <div class="mb-2">
                        <span class="badge bg-dark border border-secondary text-light me-1">${model.context_window.toLocaleString()} tokens</span>
                        ${model.tags.map(tag => `<span class="badge bg-dark border border-secondary text-light me-1">${tag}</span>`).join('')}
                    </div>
                    
                    <!-- Performance ratings -->
                    <div class="mb-3 small">
                        <div class="row g-2">
                            <div class="col-6">
                                <span class="text-secondary">Conversation:</span> ${this.renderRating(model.performance.conversation)}
                            </div>
                            <div class="col-6">
                                <span class="text-secondary">Memory:</span> ${this.renderRating(model.performance.memory_extraction)}
                            </div>
                            <div class="col-6">
                                <span class="text-secondary">Prompts:</span> ${this.renderRating(model.performance.prompt_following)}
                            </div>
                            <div class="col-6">
                                <span class="text-secondary">Creativity:</span> ${this.renderRating(model.performance.creativity)}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Download section -->
                    <div class="d-flex gap-2">
                        <select class="form-select form-select-sm bg-dark text-light border-secondary flex-grow-1" 
                                id="quant-${model.id}">
                            ${quantOptions}
                        </select>
                        <button class="btn btn-primary btn-sm" onclick="modelManager.downloadModel('${model.id}')">
                            <i class="bi bi-download me-1"></i>Download
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    getCategoryIcon(category) {
        const icons = {
            'balanced': '⚖️',
            'creative': '🎨',
            'technical': '🔧',
            'advanced': '🚀'
        };
        return icons[category] || '📦';
    }

    renderRating(rating) {
        const colors = {
            'excellent': 'success',
            'very_good': 'info',
            'good': 'primary',
            'fair': 'warning',
            'poor': 'danger'
        };
        const color = colors[rating] || 'secondary';
        const label = rating.replace('_', ' ');
        return `<span class="badge bg-${color}">${label}</span>`;
    }

    renderDownloadedModels() {
        const container = document.getElementById('downloadedModelsContainer');
        if (!container) return;

        const hasLocalModels = this.downloadedModels.length > 0;
        const hasCustomModels = this.customModels.length > 0;

        if (!hasLocalModels && !hasCustomModels) {
            container.innerHTML = `
                <div class="text-center text-secondary py-5">
                    <i class="bi bi-inbox fs-1"></i>
                    <p class="mt-3">No models downloaded yet</p>
                    <button class="btn btn-primary" data-bs-toggle="tab" data-bs-target="#curatedTab">
                        Browse Models
                    </button>
                </div>
            `;
            return;
        }

        let html = '';
        
        // Custom HF Models Section (source = 'custom_hf')
        if (hasCustomModels) {
            html += `
                <h6 class="text-light mb-3">
                    <i class="bi bi-cloud-download me-2"></i>HuggingFace Models
                </h6>
            `;
            
            html += this.customModels.map(model => {
                const escapedId = model.model_id.replace(/'/g, "\\'");
                const repoName = model.repo_id.split('/')[1] || model.repo_id;
                const isActive = this.activeModel && (model.model_id === this.activeModel || model.ollama_model_name === this.activeModel);
                const activeBadge = isActive ? '<span class="badge bg-success ms-2"><i class="bi bi-star-fill me-1"></i>Active</span>' : '';
                const cardClass = isActive ? 'border-success' : 'border-secondary';
                
                return `
                <div class="card bg-dark ${cardClass} mb-3 text-light">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h6 class="card-title mb-0 text-light">${repoName}${activeBadge}</h6>
                            <span class="badge bg-info">${model.quantization}</span>
                        </div>
                        
                        <p class="card-text text-secondary small mb-2">
                            <code class="text-info">${model.model_id}</code>
                        </p>
                        
                        <div class="mb-2 small">
                            <div class="text-secondary">
                                <i class="bi bi-calendar me-1"></i>Added: ${new Date(model.downloaded_at).toLocaleDateString()}
                            </div>
                        </div>
                        
                        <div class="d-flex gap-2">
                            <button class="btn btn-success btn-sm" onclick="modelManager.switchToCustomModel('${escapedId}')">
                                <i class="bi bi-arrow-repeat me-1"></i>Switch To
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="modelManager.copyModelName('${escapedId}')">
                                <i class="bi bi-clipboard me-1"></i>Copy Name
                            </button>
                            <button class="btn btn-danger btn-sm ms-auto" onclick="modelManager.deleteCustomModel('${escapedId}')">
                                <i class="bi bi-trash me-1"></i>Remove
                            </button>
                        </div>
                    </div>
                </div>
                `;
            }).join('');
            
            if (hasLocalModels) {
                html += '<hr class="border-secondary my-4">';
            }
        }
        
        // Curated Downloaded Models Section (source = 'curated')
        if (hasLocalModels) {
            html += `
                <h6 class="text-light mb-3">
                    <i class="bi bi-hdd me-2"></i>Curated Models
                </h6>
            `;
            
            html += this.downloadedModels.map(model => {
                const escapedId = model.model_id.replace(/'/g, "\\'");
                const escapedPath = (model.file_path || "").replace(/\\/g, '\\\\');
                const isActive = this.activeModel && (model.model_id === this.activeModel || model.ollama_model_name === this.activeModel);
                const activeBadge = isActive ? '<span class="badge bg-success ms-2"><i class="bi bi-star-fill me-1"></i>Active</span>' : '';
                const cardClass = isActive ? 'border-success' : 'border-secondary';
                
                // Format model info line (handle null values for custom HF models)
                let modelInfoLine = '';
                if (model.parameters && model.context_window && model.file_size_mb) {
                    modelInfoLine = `${model.parameters}B parameters • ${model.context_window.toLocaleString()} context • ${Math.round(model.file_size_mb / 1024)}GB`;
                } else {
                    modelInfoLine = `<code class="text-info">${model.model_id}</code>`;
                }
                
                return `
                <div class="card bg-dark ${cardClass} mb-3 text-light">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h6 class="card-title mb-0 text-light">${model.display_name}${activeBadge}</h6>
                            <span class="badge bg-secondary">${model.quantization}</span>
                        </div>
                        
                        <p class="card-text text-secondary small mb-2">
                            ${modelInfoLine}
                        </p>
                        
                        <div class="mb-2 small">
                            <div class="text-secondary">
                                <i class="bi bi-calendar me-1"></i>Downloaded: ${new Date(model.downloaded_at).toLocaleDateString()}
                            </div>
                            ${model.last_used ? `
                                <div class="text-secondary">
                                    <i class="bi bi-clock me-1"></i>Last used: ${new Date(model.last_used).toLocaleDateString()}
                                </div>
                            ` : ''}
                        </div>
                        
                        <div class="d-flex gap-2">
                            <button class="btn btn-success btn-sm" onclick="modelManager.switchToModel('${escapedPath}')">
                                <i class="bi bi-arrow-repeat me-1"></i>Switch To
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="modelManager.copyPath('${escapedPath}')">
                                <i class="bi bi-clipboard me-1"></i>Copy Path
                            </button>
                            <button class="btn btn-danger btn-sm ms-auto" onclick="modelManager.deleteModel('${escapedId}')">
                                <i class="bi bi-trash me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
                `;
            }).join('');
        }

        container.innerHTML = html;
    }

    async downloadModel(modelId) {
        const quantSelect = document.getElementById(`quant-${modelId}`);
        if (!quantSelect) return;

        const quantization = quantSelect.value;
        
        try {
            // Estimate VRAM first
            const estimate = await this.estimateVRAM(modelId, quantization);
            
            if (!estimate.will_fit) {
                const confirmed = confirm(
                    `⚠️ This model may not fit in your available VRAM!\n\n` +
                    `Required: ${Math.round(estimate.total_vram_mb / 1024)}GB\n` +
                    `Available: ${Math.round(estimate.available_vram_mb / 1024)}GB\n\n` +
                    (estimate.recommended_quantization ? 
                        `Recommendation: Try ${estimate.recommended_quantization} instead.\n\n` : '') +
                    `Download anyway?`
                );
                
                if (!confirmed) return;
            }

            // Start download
            const response = await fetch('/api/models/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId, quantization })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Download failed');
            }

            const result = await response.json();
            
            if (result.status === 'completed') {
                UI.showToast('Model already downloaded!', 'info');
                await this.loadDownloadedModels();
                this.renderDownloadedModels();
                return;
            }

            // Show download progress modal
            this.downloadJobId = result.job_id;
            this.showDownloadProgress();
            this.startDownloadPolling();

        } catch (error) {
            console.error('Download failed:', error);
            UI.showToast(`Download failed: ${error.message}`, 'danger');
        }
    }

    async estimateVRAM(modelId, quantization) {
        try {
            const response = await fetch('/api/models/estimate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId, quantization })
            });

            if (!response.ok) throw new Error('VRAM estimation failed');
            return await response.json();
        } catch (error) {
            console.error('VRAM estimation failed:', error);
            return { will_fit: true }; // Assume OK if estimation fails
        }
    }

    showDownloadProgress() {
        // Update modal content
        document.getElementById('downloadModelName').textContent = 'Downloading...';
        document.getElementById('downloadProgress').style.width = '0%';
        document.getElementById('downloadProgress').textContent = '0%';
        document.getElementById('downloadSize').textContent = '0 MB / ? MB';
        document.getElementById('downloadStatus').textContent = 'Initializing...';

        // Show modal
        if (this.downloadModal) {
            this.downloadModal.show();
        }
    }

    startDownloadPolling() {
        if (this.downloadPollInterval) {
            clearInterval(this.downloadPollInterval);
        }

        this.downloadPollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/models/download/${this.downloadJobId}`);
                if (!response.ok) {
                    this.stopDownloadPolling();
                    return;
                }

                const status = await response.json();
                this.updateDownloadProgress(status);

                if (status.status === 'completed' || status.status === 'failed') {
                    this.stopDownloadPolling();
                    
                    if (status.status === 'completed') {
                        // Reload the downloaded models list
                        await this.loadDownloadedModels();
                        this.renderDownloadedModels();
                        
                        // Check if this was an HF model pull (has model_name starting with hf.co/)
                        const isHFModel = status.model_name && status.model_name.startsWith('hf.co/');
                        
                        if (isHFModel) {
                            // Show "Use This Model" button for HF models
                            this.showModelReadyButton(status.model_name);
                            UI.showToast('Model ready! Click "Use This Model" to activate it.', 'success');
                        } else {
                            // Show done button for curated models
                            const statusEl = document.getElementById('downloadStatus');
                            if (statusEl) {
                                statusEl.innerHTML = `
                                    <div class="text-success">
                                        <i class="bi bi-check-circle me-2"></i>Download complete!
                                    </div>
                                    <button class="btn btn-success mt-3" onclick="modelManager.closeDownloadModal()">
                                        <i class="bi bi-check me-1"></i>Done
                                    </button>
                                `;
                            }
                            UI.showToast('Model downloaded successfully!', 'success');
                        }
                    } else {
                        UI.showToast(`Download failed: ${status.error}`, 'danger');
                    }
                }
            } catch (error) {
                console.error('Failed to check download status:', error);
                this.stopDownloadPolling();
            }
        }, 1000); // Poll every second
    }

    stopDownloadPolling() {
        if (this.downloadPollInterval) {
            clearInterval(this.downloadPollInterval);
            this.downloadPollInterval = null;
        }
    }

    updateDownloadProgress(status) {
        const progress = status.progress || 0;
        const progressBar = document.getElementById('downloadProgress');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${Math.round(progress)}%`;
        }

        if (status.current_size_mb && status.total_size_mb) {
            document.getElementById('downloadSize').textContent = 
                `${Math.round(status.current_size_mb)} MB / ${Math.round(status.total_size_mb)} MB`;
        }

        const statusText = {
            'pending': 'Waiting to start...',
            'downloading': 'Downloading from HuggingFace...',
            'pulling': 'Pulling model via Ollama...',
            'importing': 'Importing to Ollama...',
            'completed': 'Download complete!',
            'failed': `Failed: ${status.error || 'Unknown error'}`
        }[status.status] || status.status;

        document.getElementById('downloadStatus').textContent = statusText;
    }

    async switchToModel(modelPath) {
        // Find model by path
        const model = this.downloadedModels.find(m => m.file_path === modelPath);
        if (!model) {
            UI.showToast('Model not found', 'danger');
            return;
        }

        // Check if model has ollama_model_name, if not, generate it
        let ollamaModelName = model.ollama_model_name;
        if (!ollamaModelName) {
            // Generate Ollama model name from model_id and quantization
            ollamaModelName = `${model.model_id.replace(/\./g, '-')}-${model.quantization.toLowerCase().replace(/_/g, '-')}`;
            
            // Try to import to Ollama
            UI.showToast('Importing model to Ollama...', 'info');
            try {
                const importResponse = await fetch(`/api/models/${model.model_id}/import-ollama`, {
                    method: 'POST'
                });
                
                if (!importResponse.ok) {
                    throw new Error('Failed to import model to Ollama');
                }
                
                const importResult = await importResponse.json();
                ollamaModelName = importResult.ollama_model_name;
                UI.showToast('Model imported to Ollama successfully', 'success');
            } catch (error) {
                console.error('Failed to import to Ollama:', error);
                UI.showToast('Failed to import model to Ollama. Make sure Ollama is running.', 'danger');
                return;
            }
        }

        // Get current config
        let currentConfig;
        try {
            const configResponse = await fetch('/system/config');
            if (!configResponse.ok) {
                throw new Error('Failed to load system config');
            }
            currentConfig = await configResponse.json();
        } catch (error) {
            console.error('Failed to load config:', error);
            UI.showToast('Failed to load system configuration', 'danger');
            return;
        }
        
        // Show confirmation with restart warning
        if (!confirm(
            `Set this as your default model?\n\n` +
            `This will:\n` +
            `1. Update your system configuration\n` +
            `2. Restart the server with the new model\n` +
            `3. Reload the page\n\n` +
            `The server will be back online in a few seconds.`
        )) {
            return;
        }

        try {
            // Update config with Ollama model name
            currentConfig.llm.model = ollamaModelName;
            
            // Save config (this triggers server restart)
            const response = await fetch('/system/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentConfig)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update configuration');
            }

            // Show success message
            UI.showToast('Model set as default. Server restarting... (this may take 15-20 seconds)', 'success');
            
            // Close modal
            if (this.modal) {
                this.modal.hide();
            }
            
            // Wait for server restart and reload page
            setTimeout(() => {
                UI.showToast('Server should be ready. Reloading page...', 'info');
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            }, 15000); // Wait 15 seconds before showing reload message, then 2 more seconds
            
        } catch (error) {
            console.error('Failed to switch model:', error);
            UI.showToast(`Failed to switch model: ${error.message}`, 'danger');
        }
    }

    async deleteModel(modelId) {
        if (!confirm(`Delete this model?\n\nThis will permanently remove the model files from your system.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/models?model_id=${encodeURIComponent(modelId)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete model');
            }

            UI.showToast('Model deleted successfully', 'success');
            await this.loadDownloadedModels();
            this.renderDownloadedModels();
        } catch (error) {
            console.error('Failed to delete model:', error);
            UI.showToast(`Failed to delete model: ${error.message}`, 'danger');
        }
    }
    
    async deleteCustomModel(modelId) {
        if (!confirm(`Remove this model from your library?\n\nThis will remove the model from tracking but won't delete it from Ollama.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/models/custom?model_id=${encodeURIComponent(modelId)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to remove model');
            }

            UI.showToast('Model removed successfully', 'success');
            await this.loadDownloadedModels();
            this.renderDownloadedModels();
        } catch (error) {
            console.error('Failed to remove model:', error);
            UI.showToast(`Failed to remove model: ${error.message}`, 'danger');
        }
    }

    copyPath(path) {
        navigator.clipboard.writeText(path).then(() => {
            UI.showToast('Path copied to clipboard', 'success');
        }).catch(err => {
            console.error('Failed to copy path:', err);
            UI.showToast('Failed to copy path', 'danger');
        });
    }

    async downloadCustomModel() {
        const repoId = document.getElementById('customRepoId').value.trim();
        const filename = document.getElementById('customFilename').value.trim();
        const displayName = document.getElementById('customDisplayName').value.trim();

        if (!repoId || !filename) {
            UI.showToast('Please fill in repository ID and filename', 'warning');
            return;
        }

        // TODO: Implement custom model download
        UI.showToast('Custom model download not yet implemented', 'info');
    }

    // Show/hide model management menu item based on provider
    updateMenuVisibility(provider) {
        const menuItem = document.getElementById('modelManagementMenuItem');
        if (menuItem) {
            if (provider === 'ollama') {
                menuItem.style.display = '';
            } else {
                menuItem.style.display = 'none';
            }
        }
    }
    
    // ========== HuggingFace Model Import ==========
    
    async loadHFQuantizations() {
        const urlInput = document.getElementById('hfRepoUrl');
        const quantSelect = document.getElementById('hfQuantSelect');
        const pullBtn = document.getElementById('hfPullBtn');
        const loadingStatus = document.getElementById('hfLoadingStatus');
        const modelPreview = document.getElementById('hfModelPreview');
        
        const url = urlInput.value.trim();
        if (!url) {
            quantSelect.innerHTML = '<option value="">Enter repository URL first...</option>';
            quantSelect.disabled = true;
            pullBtn.disabled = true;
            modelPreview.style.display = 'none';
            return;
        }
        
        try {
            loadingStatus.style.display = 'block';
            quantSelect.disabled = true;
            pullBtn.disabled = true;
            
            const response = await fetch(`/api/models/hf-quantizations?hf_url=${encodeURIComponent(url)}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to load quantizations');
            }
            
            const data = await response.json();
            
            // Populate quantization dropdown
            quantSelect.innerHTML = '';
            for (const quant of data.quantizations) {
                const option = document.createElement('option');
                option.value = quant.quant;
                option.textContent = quant.quant;
                quantSelect.appendChild(option);
            }
            
            quantSelect.disabled = false;
            pullBtn.disabled = false;
            loadingStatus.style.display = 'none';
            
            // Show model name preview
            this.updateHFModelPreview();
            
            // Add change handler for quantization
            quantSelect.addEventListener('change', () => this.updateHFModelPreview());
            
            UI.showToast(`Found ${data.quantizations.length} quantizations`, 'success');
            
        } catch (error) {
            console.error('Failed to load quantizations:', error);
            UI.showToast(`Failed to load quantizations: ${error.message}`, 'danger');
            quantSelect.innerHTML = '<option value="">Error loading quantizations</option>';
            quantSelect.disabled = true;
            pullBtn.disabled = true;
            loadingStatus.style.display = 'none';
        }
    }
    
    updateHFModelPreview() {
        const urlInput = document.getElementById('hfRepoUrl');
        const quantSelect = document.getElementById('hfQuantSelect');
        const modelPreview = document.getElementById('hfModelPreview');
        const modelNameEl = document.getElementById('hfModelName');
        
        const url = urlInput.value.trim();
        const quant = quantSelect.value;
        
        if (!url || !quant) {
            modelPreview.style.display = 'none';
            return;
        }
        
        // Parse repo_id from URL
        let repo_id = url;
        if (url.startsWith('https://huggingface.co/')) {
            repo_id = url.replace('https://huggingface.co/', '');
        } else if (url.startsWith('hf.co/')) {
            repo_id = url.replace('hf.co/', '');
        }
        repo_id = repo_id.split('?')[0].replace(/\/$/, '');
        
        // Generate Ollama model name
        const modelName = `hf.co/${repo_id}:${quant}`;
        modelNameEl.textContent = modelName;
        modelPreview.style.display = 'block';
    }
    
    async pullHFModel() {
        const urlInput = document.getElementById('hfRepoUrl');
        const quantSelect = document.getElementById('hfQuantSelect');
        
        const url = urlInput.value.trim();
        const quant = quantSelect.value;
        
        if (!url || !quant) {
            UI.showToast('Please enter a repository URL and select a quantization', 'warning');
            return;
        }
        
        try {
            // Start pull
            const response = await fetch('/api/models/pull-hf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    hf_url: url,
                    quantization: quant
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Pull failed');
            }
            
            const result = await response.json();
            
            // Parse repo_id for saving
            let repo_id = url;
            if (url.startsWith('https://huggingface.co/')) {
                repo_id = url.replace('https://huggingface.co/', '');
            } else if (url.startsWith('hf.co/')) {
                repo_id = url.replace('hf.co/', '');
            }
            repo_id = repo_id.split('?')[0].replace(/\/$/, '');
            
            // Store model info for after completion
            this.pendingCustomModel = {
                model_name: result.model_name,
                repo_id: repo_id,
                quantization: quant
            };
            
            // Show progress modal
            this.downloadJobId = result.job_id;
            document.getElementById('downloadModelName').textContent = result.model_name;
            this.showDownloadProgress();
            this.startDownloadPolling();
            
            // Keep model selection modal open (don't close it)
            
        } catch (error) {
            console.error('Pull failed:', error);
            UI.showToast(`Pull failed: ${error.message}`, 'danger');
        }
    }
    
    showModelReadyButton(modelName) {
        // Add "Use This Model" button to the download modal
        const statusEl = document.getElementById('downloadStatus');
        if (statusEl) {
            statusEl.innerHTML = `
                <div class="text-success">
                    <i class="bi bi-check-circle me-2"></i>Model ready!
                </div>
                <button class="btn btn-success mt-3" onclick="modelManager.activateReadyModel('${modelName.replace(/'/g, "\\'")}')">
                    <i class="bi bi-arrow-repeat me-1"></i>Use This Model
                </button>
                <button class="btn btn-outline-secondary mt-3 ms-2" onclick="modelManager.closeDownloadModal()">
                    Close
                </button>
            `;
        }
    }
    
    async activateReadyModel(modelName) {
        // Backend auto-saves, just switch to it
        await this.switchToCustomModel(modelName);
        this.closeDownloadModal();
    }
    
    closeDownloadModal() {
        if (this.downloadModal) {
            this.downloadModal.hide();
        }
        // Refresh both lists
        this.loadDownloadedModels().then(() => {
            this.renderDownloadedModels();
            
            // Switch to Downloaded tab in the main model selection modal
            const downloadedTab = document.getElementById('downloaded-tab');
            if (downloadedTab) {
                const tab = new bootstrap.Tab(downloadedTab);
                tab.show();
            }
        });
    }
    
    async switchToCustomModel(modelName) {
        try {
            // Update system config
            const response = await fetch('/system/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    llm: {
                        model: modelName
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to update config');
            }
            
            UI.showToast('Model activated! Restarting server...', 'success');
            
            // Close model selection modal
            if (this.modal) {
                this.modal.hide();
            }
            
            // Wait and reload
            await new Promise(resolve => setTimeout(resolve, 15000));
            UI.showToast('Reloading page...', 'info');
            await new Promise(resolve => setTimeout(resolve, 2000));
            window.location.reload();
            
        } catch (error) {
            console.error('Failed to switch model:', error);
            UI.showToast(`Failed to switch model: ${error.message}`, 'danger');
        }
    }
    
    copyModelName(modelName) {
        navigator.clipboard.writeText(modelName).then(() => {
            UI.showToast('Model name copied to clipboard', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
            UI.showToast('Failed to copy to clipboard', 'danger');
        });
    }
}

// Initialize model manager
const modelManager = new ModelManager();

// Add menu item to settings dropdown
document.addEventListener('DOMContentLoaded', () => {
    modelManager.init();

    // Add click handler for menu item (already in HTML)
    const menuItem = document.getElementById('modelManagementMenuItem');
    if (menuItem) {
        menuItem.addEventListener('click', (e) => {
            e.preventDefault();
            modelManager.show();
        });
    }
    
    // Add click handler for banner button
    const bannerBtn = document.getElementById('openModelManagerBtn');
    if (bannerBtn) {
        bannerBtn.addEventListener('click', (e) => {
            e.preventDefault();
            modelManager.show();
        });
    }
});
