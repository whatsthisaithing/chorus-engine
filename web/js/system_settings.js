/**
 * System Settings Management
 * Handles loading, editing, and saving system.yaml configuration
 */

class SystemSettingsManager {
    constructor() {
        this.modal = null;
        this.form = null;
    }

    init() {
        const modalEl = document.getElementById('systemSettingsModal');
        if (!modalEl) {
            console.error('System settings modal not found');
            return;
        }
        
        this.modal = new bootstrap.Modal(modalEl);
        this.form = document.getElementById('systemSettingsForm');
        
        // Toggle visibility based on checkboxes
        const comfyuiToggle = document.getElementById('comfyui_enabled');
        if (comfyuiToggle) {
            comfyuiToggle.addEventListener('change', (e) => {
                const settings = document.getElementById('comfyui_settings');
                if (settings) settings.style.display = e.target.checked ? 'block' : 'none';
            });
        }
        
        const docAnalysisToggle = document.getElementById('document_analysis_enabled');
        if (docAnalysisToggle) {
            docAnalysisToggle.addEventListener('change', (e) => {
                const settings = document.getElementById('document_analysis_settings');
                if (settings) settings.style.display = e.target.checked ? 'block' : 'none';
            });
        }
        
        // Task 1.9 Phase 1 Expansion: Vision configuration toggle
        const visionToggle = document.getElementById('vision_enabled');
        if (visionToggle) {
            visionToggle.addEventListener('change', (e) => {
                const settings = document.getElementById('vision_settings');
                if (settings) settings.style.display = e.target.checked ? 'block' : 'none';
            });
        }
        
        // Heartbeat configuration toggle
        const heartbeatToggle = document.getElementById('heartbeat_enabled');
        if (heartbeatToggle) {
            heartbeatToggle.addEventListener('change', (e) => {
                const settings = document.getElementById('heartbeat_settings');
                if (settings) settings.style.display = e.target.checked ? 'block' : 'none';
            });
        }
        
        // Save button
        const saveBtn = document.getElementById('saveSystemSettingsBtn');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveSettings());
        }
        
        // Theme preview button
        const previewBtn = document.getElementById('previewThemeBtn');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => {
                const selectedTheme = document.getElementById('ui_color_scheme').value;
                ThemeManager.applyTheme(selectedTheme);
                UI.showToast(`Preview: ${THEMES[selectedTheme].displayName}`, 'info');
            });
        }
    }

    async show() {
        try {
            // Load current settings
            const config = await this.loadSystemConfig();
            this.populateForm(config);
            this.modal.show();
        } catch (error) {
            console.error('Failed to load system configuration:', error);
            UI.showToast('Failed to load system configuration', 'danger');
        }
    }

    async loadSystemConfig() {
        const response = await fetch('/system/config');
        if (!response.ok) {
            throw new Error(`Failed to load config: ${response.statusText}`);
        }
        return await response.json();
    }

    populateForm(config) {
        // LLM Configuration
        document.getElementById('llm_provider').value = config.llm.provider || 'ollama';
        document.getElementById('llm_base_url').value = config.llm.base_url || '';
        document.getElementById('llm_model').value = config.llm.model || '';
        document.getElementById('llm_archivist_model').value = config.llm.archivist_model || '';
        document.getElementById('llm_analysis_max_tokens_summary').value = config.llm.analysis_max_tokens_summary || 4096;
        document.getElementById('llm_analysis_max_tokens_memories').value = config.llm.analysis_max_tokens_memories || 4096;
        document.getElementById('llm_analysis_min_tokens_summary').value = config.llm.analysis_min_tokens_summary ?? 500;
        document.getElementById('llm_analysis_min_tokens_memories').value = config.llm.analysis_min_tokens_memories ?? 0;
        document.getElementById('llm_context_window').value = config.llm.context_window || 32768;
        document.getElementById('llm_max_response_tokens').value = config.llm.max_response_tokens || 4096;
        document.getElementById('llm_temperature').value = config.llm.temperature || 0.7;
        document.getElementById('llm_timeout_seconds').value = config.llm.timeout_seconds || 120;
        document.getElementById('llm_unload_during_image_generation').checked = config.llm.unload_during_image_generation || false;

        // Add provider change listener
        document.getElementById('llm_provider').addEventListener('change', (e) => {
            // Update model manager menu visibility
            if (typeof modelManager !== 'undefined') {
                modelManager.updateMenuVisibility(e.target.value);
            }
        });

        // Update model manager menu visibility on page load
        if (typeof modelManager !== 'undefined') {
            modelManager.updateMenuVisibility(config.llm.provider);
        }

        // Memory Configuration
        document.getElementById('memory_embedding_model').value = config.memory.embedding_model || 'all-MiniLM-L6-v2';
        document.getElementById('memory_vector_store').value = config.memory.vector_store || 'chroma';
        document.getElementById('memory_default_budget_tokens').value = config.memory.default_budget_tokens || 1000;
        document.getElementById('memory_explicit_minimum').value = config.memory.similarity_thresholds.explicit_minimum || 0.70;
        document.getElementById('memory_implicit_minimum').value = config.memory.similarity_thresholds.implicit_minimum || 0.75;
        document.getElementById('memory_search_api_minimum').value = config.memory.similarity_thresholds.search_api_minimum || 0.65;

        // ComfyUI Configuration
        const comfyuiEnabled = config.comfyui.enabled || false;
        document.getElementById('comfyui_enabled').checked = comfyuiEnabled;
        document.getElementById('comfyui_settings').style.display = comfyuiEnabled ? 'block' : 'none';
        document.getElementById('comfyui_url').value = config.comfyui.url || 'http://localhost:8188';
        document.getElementById('comfyui_timeout_seconds').value = config.comfyui.timeout_seconds || 300;
        document.getElementById('comfyui_video_timeout_seconds').value = config.comfyui.video_timeout_seconds || 600;
        document.getElementById('comfyui_polling_interval_seconds').value = config.comfyui.polling_interval_seconds || 2.0;
        document.getElementById('comfyui_max_concurrent_jobs').value = config.comfyui.max_concurrent_jobs || 2;

        // Document Analysis Configuration
        const docEnabled = config.document_analysis.enabled || false;
        document.getElementById('document_analysis_enabled').checked = docEnabled;
        document.getElementById('document_analysis_settings').style.display = docEnabled ? 'block' : 'none';
        document.getElementById('document_analysis_default_max_chunks').value = config.document_analysis.default_max_chunks || 3;
        document.getElementById('document_analysis_max_chunks_cap').value = config.document_analysis.max_chunks_cap || 25;
        document.getElementById('document_analysis_chunk_token_estimate').value = config.document_analysis.chunk_token_estimate || 512;
        document.getElementById('document_analysis_document_budget_ratio').value = config.document_analysis.document_budget_ratio || 0.15;

        // Task 1.9 Phase 1 Expansion: Vision Configuration
        const vision = config.vision || {};
        const visionEnabled = vision.enabled || false;
        document.getElementById('vision_enabled').checked = visionEnabled;
        document.getElementById('vision_settings').style.display = visionEnabled ? 'block' : 'none';
        
        // Model settings (backend is derived from llm.provider, not configurable separately)
        document.getElementById('vision_model_name').value = vision.model?.name || 'qwen3-vl:4b';
        document.getElementById('vision_model_load_timeout_seconds').value = vision.model?.load_timeout_seconds || 60;
        
        // Processing settings
        document.getElementById('vision_processing_timeout_seconds').value = vision.processing?.timeout_seconds || 30;
        document.getElementById('vision_processing_max_retries').value = vision.processing?.max_retries || 2;
        document.getElementById('vision_processing_resize_target').value = vision.processing?.resize_target || 1024;
        document.getElementById('vision_processing_max_file_size_mb').value = vision.processing?.max_file_size_mb || 10;
        
        // Output settings
        document.getElementById('vision_output_format').value = vision.output?.format || 'structured';
        document.getElementById('vision_output_include_confidence').checked = vision.output?.include_confidence !== false;
        
        // Memory settings
        document.getElementById('vision_memory_auto_create').checked = vision.memory?.auto_create !== false;
        document.getElementById('vision_memory_min_confidence').value = vision.memory?.min_confidence || 0.6;
        document.getElementById('vision_memory_category').value = vision.memory?.category || 'visual';
        document.getElementById('vision_memory_default_priority').value = vision.memory?.default_priority || 70;
        
        // Intent settings
        document.getElementById('vision_intent_web_ui_always_analyze').checked = vision.intent?.web_ui_always_analyze !== false;
        document.getElementById('vision_intent_bridge_always_analyze').checked = vision.intent?.bridge_always_analyze || false;
        document.getElementById('vision_intent_bridge_never_analyze').checked = vision.intent?.bridge_never_analyze || false;
        
        // Cache settings
        document.getElementById('vision_cache_enabled').checked = vision.cache?.enabled !== false;
        document.getElementById('vision_cache_allow_reanalysis').checked = vision.cache?.allow_reanalysis !== false;

        // Heartbeat Configuration
        const heartbeat = config.heartbeat || {};
        const heartbeatEnabled = heartbeat.enabled !== false;  // Default true
        document.getElementById('heartbeat_enabled').checked = heartbeatEnabled;
        document.getElementById('heartbeat_settings').style.display = heartbeatEnabled ? 'block' : 'none';
        document.getElementById('heartbeat_interval_seconds').value = heartbeat.interval_seconds || 60;
        document.getElementById('heartbeat_idle_threshold_minutes').value = heartbeat.idle_threshold_minutes || 5;
        document.getElementById('heartbeat_resume_grace_seconds').value = heartbeat.resume_grace_seconds || 2;
        document.getElementById('heartbeat_analysis_summary_stale_hours').value = heartbeat.analysis_summary_stale_hours || 24;
        document.getElementById('heartbeat_analysis_summary_min_messages').value = heartbeat.analysis_summary_min_messages || 10;
        document.getElementById('heartbeat_analysis_summary_batch_size').value = heartbeat.analysis_summary_batch_size || 3;
        document.getElementById('heartbeat_analysis_memories_stale_hours').value = heartbeat.analysis_memories_stale_hours || 24;
        document.getElementById('heartbeat_analysis_memories_min_messages').value = heartbeat.analysis_memories_min_messages || 10;
        document.getElementById('heartbeat_analysis_memories_batch_size').value = heartbeat.analysis_memories_batch_size || 3;
        document.getElementById('heartbeat_gpu_check_enabled').checked = heartbeat.gpu_check_enabled || false;
        document.getElementById('heartbeat_gpu_max_utilization_percent').value = heartbeat.gpu_max_utilization_percent || 15;
        
        // Time Context Configuration
        const timeContext = config.time_context || {};
        const timeEnabled = timeContext.enabled !== false;
        document.getElementById('time_context_enabled').checked = timeEnabled;
        document.getElementById('time_context_timezone').value = timeContext.timezone || '';
        document.getElementById('time_context_format').value = timeContext.format || 'iso';

        // General Configuration
        document.getElementById('api_host').value = config.api_host || 'localhost';
        document.getElementById('api_port').value = config.api_port || 8080;
        document.getElementById('debug').checked = config.debug || false;
        
        // UI Configuration
        document.getElementById('ui_color_scheme').value = config.ui?.color_scheme || 'stage-night';
    }

    collectFormData() {
        const data = {
            llm: {
                provider: document.getElementById('llm_provider').value,
                base_url: document.getElementById('llm_base_url').value,
                model: document.getElementById('llm_model').value,
                archivist_model: document.getElementById('llm_archivist_model').value,
                analysis_max_tokens_summary: parseInt(document.getElementById('llm_analysis_max_tokens_summary').value),
                analysis_max_tokens_memories: parseInt(document.getElementById('llm_analysis_max_tokens_memories').value),
                analysis_min_tokens_summary: parseInt(document.getElementById('llm_analysis_min_tokens_summary').value),
                analysis_min_tokens_memories: parseInt(document.getElementById('llm_analysis_min_tokens_memories').value),
                context_window: parseInt(document.getElementById('llm_context_window').value),
                max_response_tokens: parseInt(document.getElementById('llm_max_response_tokens').value),
                temperature: parseFloat(document.getElementById('llm_temperature').value),
                timeout_seconds: parseInt(document.getElementById('llm_timeout_seconds').value),
                unload_during_image_generation: document.getElementById('llm_unload_during_image_generation').checked
            },
            memory: {
                embedding_model: document.getElementById('memory_embedding_model').value,
                vector_store: document.getElementById('memory_vector_store').value,
                default_budget_tokens: parseInt(document.getElementById('memory_default_budget_tokens').value),
                similarity_thresholds: {
                    explicit_minimum: parseFloat(document.getElementById('memory_explicit_minimum').value),
                    implicit_minimum: parseFloat(document.getElementById('memory_implicit_minimum').value),
                    search_api_minimum: parseFloat(document.getElementById('memory_search_api_minimum').value)
                }
            },
            comfyui: {
                enabled: document.getElementById('comfyui_enabled').checked,
                url: document.getElementById('comfyui_url').value,
                timeout_seconds: parseInt(document.getElementById('comfyui_timeout_seconds').value),
                video_timeout_seconds: parseInt(document.getElementById('comfyui_video_timeout_seconds').value),
                polling_interval_seconds: parseFloat(document.getElementById('comfyui_polling_interval_seconds').value),
                max_concurrent_jobs: parseInt(document.getElementById('comfyui_max_concurrent_jobs').value)
            },
            document_analysis: {
                enabled: document.getElementById('document_analysis_enabled').checked,
                default_max_chunks: parseInt(document.getElementById('document_analysis_default_max_chunks').value),
                max_chunks_cap: parseInt(document.getElementById('document_analysis_max_chunks_cap').value),
                chunk_token_estimate: parseInt(document.getElementById('document_analysis_chunk_token_estimate').value),
                document_budget_ratio: parseFloat(document.getElementById('document_analysis_document_budget_ratio').value)
            },
            vision: {
                enabled: document.getElementById('vision_enabled').checked,
                model: {
                    name: document.getElementById('vision_model_name').value,
                    load_timeout_seconds: parseInt(document.getElementById('vision_model_load_timeout_seconds').value)
                },
                processing: {
                    timeout_seconds: parseInt(document.getElementById('vision_processing_timeout_seconds').value),
                    max_retries: parseInt(document.getElementById('vision_processing_max_retries').value),
                    resize_target: parseInt(document.getElementById('vision_processing_resize_target').value),
                    max_file_size_mb: parseInt(document.getElementById('vision_processing_max_file_size_mb').value)
                },
                output: {
                    format: document.getElementById('vision_output_format').value,
                    include_confidence: document.getElementById('vision_output_include_confidence').checked
                },
                memory: {
                    auto_create: document.getElementById('vision_memory_auto_create').checked,
                    min_confidence: parseFloat(document.getElementById('vision_memory_min_confidence').value),
                    category: document.getElementById('vision_memory_category').value,
                    default_priority: parseInt(document.getElementById('vision_memory_default_priority').value)
                },
                intent: {
                    web_ui_always_analyze: document.getElementById('vision_intent_web_ui_always_analyze').checked,
                    bridge_always_analyze: document.getElementById('vision_intent_bridge_always_analyze').checked,
                    bridge_never_analyze: document.getElementById('vision_intent_bridge_never_analyze').checked
                },
                cache: {
                    enabled: document.getElementById('vision_cache_enabled').checked,
                    allow_reanalysis: document.getElementById('vision_cache_allow_reanalysis').checked
                }
            },
            heartbeat: {
                enabled: document.getElementById('heartbeat_enabled').checked,
                interval_seconds: parseFloat(document.getElementById('heartbeat_interval_seconds').value),
                idle_threshold_minutes: parseFloat(document.getElementById('heartbeat_idle_threshold_minutes').value),
                resume_grace_seconds: parseFloat(document.getElementById('heartbeat_resume_grace_seconds').value),
                analysis_summary_stale_hours: parseFloat(document.getElementById('heartbeat_analysis_summary_stale_hours').value),
                analysis_summary_min_messages: parseInt(document.getElementById('heartbeat_analysis_summary_min_messages').value),
                analysis_summary_batch_size: parseInt(document.getElementById('heartbeat_analysis_summary_batch_size').value),
                analysis_memories_stale_hours: parseFloat(document.getElementById('heartbeat_analysis_memories_stale_hours').value),
                analysis_memories_min_messages: parseInt(document.getElementById('heartbeat_analysis_memories_min_messages').value),
                analysis_memories_batch_size: parseInt(document.getElementById('heartbeat_analysis_memories_batch_size').value),
                gpu_check_enabled: document.getElementById('heartbeat_gpu_check_enabled').checked,
                gpu_max_utilization_percent: parseInt(document.getElementById('heartbeat_gpu_max_utilization_percent').value)
            },
            time_context: {
                enabled: document.getElementById('time_context_enabled').checked,
                timezone: document.getElementById('time_context_timezone').value.trim(),
                format: document.getElementById('time_context_format').value
            },
            ui: {
                color_scheme: document.getElementById('ui_color_scheme').value
            },
            api_host: document.getElementById('api_host').value,
            api_port: parseInt(document.getElementById('api_port').value),
            debug: document.getElementById('debug').checked
        };
        
        return data;
    }

    async saveSettings() {
        if (!this.form.checkValidity()) {
            this.form.reportValidity();
            return;
        }

        // Confirm restart
        if (!confirm('⚠️ Saving will restart the server. All active connections will be closed.\n\nContinue?')) {
            return;
        }

        const config = this.collectFormData();
        const saveBtn = document.getElementById('saveSystemSettingsBtn');
        const originalText = saveBtn.innerHTML;

        try {
            saveBtn.disabled = true;
            saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Saving & Restarting...';

            const response = await fetch('/system/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save configuration');
            }

            UI.showToast('Configuration saved. Server is restarting... (this may take 15-20 seconds)', 'success');
            this.modal.hide();

            // Wait for server to restart (typically takes 10-15 seconds with lazy TTS loading)
            setTimeout(() => {
                UI.showToast('Server should be ready. Reloading page...', 'info');
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            }, 15000); // Wait 15 seconds before showing reload message, then 2 more seconds

        } catch (error) {
            console.error('Failed to save configuration:', error);
            UI.showToast(`Failed to save configuration: ${error.message}`, 'danger');
            saveBtn.disabled = false;
            saveBtn.innerHTML = originalText;
        }
    }
}

// Initialize system settings manager
const systemSettingsManager = new SystemSettingsManager();

// Add menu item to settings dropdown
document.addEventListener('DOMContentLoaded', () => {
    // Initialize manager
    systemSettingsManager.init();

    // Add menu item
    const settingsMenu = document.querySelector('#settingsMenuBtn + .dropdown-menu');
    if (settingsMenu) {
        const systemSettingsItem = document.createElement('li');
        systemSettingsItem.innerHTML = `
            <a class="dropdown-item" href="#" id="systemSettingsMenuItem">
                <i class="bi bi-gear me-2"></i>System Settings
            </a>
        `;
        // Insert after the header
        const header = settingsMenu.querySelector('.dropdown-header');
        header.parentNode.insertBefore(systemSettingsItem, header.nextSibling);

        // Add click handler
        document.getElementById('systemSettingsMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            systemSettingsManager.show();
        });
    }
});
