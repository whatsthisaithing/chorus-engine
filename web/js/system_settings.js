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
