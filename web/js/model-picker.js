/**
 * Provider-based model picker and installer.
 * Replaces model-manager-centric selection UX for system + character model fields.
 */

class ProviderModelPicker {
    constructor() {
        this.pickerModal = null;
        this.installerModal = null;
        this.targetInputId = null;
        this.currentProvider = null;
        this.currentBaseUrl = '';
        this.models = [];
        this.installJob = null;
        this.installPollTimer = null;
        this.quantLoadTimer = null;
    }

    init() {
        const pickerEl = document.getElementById('providerModelPickerModal');
        const installerEl = document.getElementById('providerInstallerModal');
        if (!pickerEl || !installerEl) return;

        this.pickerModal = new bootstrap.Modal(pickerEl);
        this.installerModal = new bootstrap.Modal(installerEl);

        document.getElementById('openSystemModelPickerBtn')?.addEventListener('click', () => this.openPickerForSystem());
        document.getElementById('openCharacterModelPickerBtn')?.addEventListener('click', () => this.openPickerForCharacter());
        document.getElementById('openArchivistModelPickerBtn')?.addEventListener('click', () => this.openPickerForArchivist());
        document.getElementById('openVisionModelPickerBtn')?.addEventListener('click', () => this.openPickerForVision());
        document.getElementById('openModelManagerBtn')?.addEventListener('click', () => this.openInstallerForSystem());
        document.getElementById('modelManagementMenuItem')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.openInstallerForSystem();
        });
        document.getElementById('refreshProviderModelsBtn')?.addEventListener('click', () => this.loadInstalledModels(true));
        document.getElementById('providerModelSearchInput')?.addEventListener('input', () => this.renderModelList());

        document.getElementById('openSystemModelInstallerBtn')?.addEventListener('click', () => this.openInstallerForSystem());
        document.getElementById('openCharacterModelInstallerBtn')?.addEventListener('click', () => this.openInstallerForCharacter());
        document.getElementById('openArchivistModelInstallerBtn')?.addEventListener('click', () => this.openInstallerForArchivist());
        document.getElementById('openVisionModelInstallerBtn')?.addEventListener('click', () => this.openInstallerForVision());
        document.getElementById('startProviderInstallBtn')?.addEventListener('click', () => this.startInstall());
        document.getElementById('providerInstallerHfUrl')?.addEventListener('input', () => this.queueQuantizationLoad());
        document.getElementById('providerInstallerHfUrl')?.addEventListener('blur', () => this.loadHfQuantizations());

        const providerEl = document.getElementById('llm_provider');
        providerEl?.addEventListener('change', (e) => this.updateEntryPointVisibility(String(e.target.value || '').toLowerCase()));
        this.resolveProvider().then((provider) => this.updateEntryPointVisibility(provider));
    }

    async resolveProvider() {
        const cfg = await this.resolveLlmRuntimeConfig();
        return String(cfg.provider || 'ollama').toLowerCase();
    }

    isSystemSettingsModalOpen() {
        const modalEl = document.getElementById('systemSettingsModal');
        return !!(modalEl && modalEl.classList.contains('show'));
    }

    async resolveLlmRuntimeConfig() {
        const providerFromForm = document.getElementById('llm_provider')?.value;
        const baseUrlFromForm = (document.getElementById('llm_base_url')?.value || '').trim();

        if (this.isSystemSettingsModalOpen() && providerFromForm) {
            return {
                provider: String(providerFromForm || 'ollama').toLowerCase(),
                base_url: baseUrlFromForm,
            };
        }

        try {
            const resp = await fetch('/system/config');
            if (resp.ok) {
                const cfg = await resp.json();
                return {
                    provider: String(cfg?.llm?.provider || 'ollama').toLowerCase(),
                    base_url: String(cfg?.llm?.base_url || '').trim(),
                };
            }
        } catch (e) {
            console.warn('Failed loading provider from system config:', e);
        }

        return {
            provider: String(providerFromForm || 'ollama').toLowerCase(),
            base_url: baseUrlFromForm,
        };
    }

    updateEntryPointVisibility(provider) {
        const show = provider === 'ollama' || provider === 'lmstudio';
        const menu = document.getElementById('modelManagementMenuItem');
        const bannerBtn = document.getElementById('openModelManagerBtn');
        if (menu?.parentElement) menu.parentElement.style.display = show ? '' : 'none';
        if (bannerBtn) bannerBtn.style.display = show ? '' : 'none';
    }

    async openPicker(targetInputId) {
        this.targetInputId = targetInputId;
        const cfg = await this.resolveLlmRuntimeConfig();
        this.currentProvider = String(cfg.provider || 'ollama').toLowerCase();
        this.currentBaseUrl = String(cfg.base_url || '').trim();
        document.getElementById('providerModelPickerMeta').textContent = `Provider: ${this.currentProvider}`;
        document.getElementById('providerModelSearchInput').value = '';
        this.pickerModal.show();
        await this.loadInstalledModels(false);
    }

    openPickerForSystem() {
        this.openPicker('llm_model');
    }

    openPickerForCharacter() {
        this.openPicker('charLlmModel');
    }

    openPickerForArchivist() {
        this.openPicker('llm_archivist_model');
    }

    openPickerForVision() {
        this.openPicker('vision_model_name');
    }

    async loadInstalledModels(showToastOnFail) {
        const listEl = document.getElementById('providerModelList');
        listEl.innerHTML = '<div class="text-secondary">Loading installed models...</div>';
        try {
            if (this.isSystemSettingsModalOpen()) {
                const providerFromForm = document.getElementById('llm_provider')?.value;
                if (providerFromForm) {
                    this.currentProvider = String(providerFromForm).toLowerCase();
                    document.getElementById('providerModelPickerMeta').textContent = `Provider: ${this.currentProvider}`;
                }
                this.currentBaseUrl = (document.getElementById('llm_base_url')?.value || '').trim();
            }
            const baseUrl = this.normalizeBaseUrl(this.currentProvider, this.currentBaseUrl);
            const params = new URLSearchParams({
                provider: this.currentProvider || '',
            });
            if (baseUrl) params.set('base_url', baseUrl);
            const resp = await fetch(`/api/models/installed?${params.toString()}`);
            if (!resp.ok) throw new Error(`Failed to load models: ${resp.status}`);
            this.models = await resp.json();
            this.renderModelList();
        } catch (e) {
            console.error(e);
            listEl.innerHTML = `<div class="text-danger">Failed to load models: ${e.message}</div>`;
            if (showToastOnFail && typeof UI !== 'undefined') UI.showToast('Failed to load installed models', 'danger');
        }
    }

    normalizeBaseUrl(provider, rawUrl) {
        let url = String(rawUrl || '').trim().replace(/\/+$/, '');
        const p = String(provider || '').toLowerCase();
        if (!url) return url;
        if (p === 'lmstudio') {
            if (url.endsWith('/api/v1')) url = url.slice(0, -7);
            else if (url.endsWith('/api/v0')) url = url.slice(0, -7);
            else if (url.endsWith('/v1')) url = url.slice(0, -3);
        } else if (p === 'ollama') {
            if (url.endsWith('/v1')) url = url.slice(0, -3);
        }
        return url.replace(/\/+$/, '');
    }

    renderModelList() {
        const listEl = document.getElementById('providerModelList');
        const search = (document.getElementById('providerModelSearchInput')?.value || '').toLowerCase().trim();
        const filtered = this.models.filter((m) => {
            const hay = `${m.name || ''} ${m.id || ''} ${m.family || ''} ${m.quantization || ''}`.toLowerCase();
            return !search || hay.includes(search);
        });

        if (!filtered.length) {
            const providerMeta = this.escapeHtml(this.currentProvider || 'unknown');
            listEl.innerHTML = `<div class="text-secondary">No installed models found for provider: ${providerMeta}.</div>`;
            return;
        }

        listEl.innerHTML = filtered.map((m) => {
            const meta = [];
            if (m.quantization) meta.push(m.quantization);
            if (m.family) meta.push(m.family);
            if (m.size_bytes) meta.push(this.formatBytes(m.size_bytes));
            return `
                <button type="button" class="list-group-item list-group-item-action bg-dark text-light border-secondary provider-model-row" data-model-id="${this.escapeAttr(m.id)}">
                    <div class="d-flex justify-content-between">
                        <strong>${this.escapeHtml(m.name || m.id)}</strong>
                        <small class="text-secondary">${this.escapeHtml(meta.join(' | '))}</small>
                    </div>
                    <small class="text-secondary">${this.escapeHtml(m.id)}</small>
                </button>
            `;
        }).join('');

        listEl.querySelectorAll('.provider-model-row').forEach((el) => {
            el.addEventListener('click', () => {
                const modelId = el.getAttribute('data-model-id') || '';
                const target = document.getElementById(this.targetInputId);
                if (target) target.value = modelId;
                this.pickerModal.hide();
            });
        });
    }

    async openInstaller(targetInputId) {
        this.targetInputId = targetInputId;
        const cfg = await this.resolveLlmRuntimeConfig();
        this.currentProvider = String(cfg.provider || 'ollama').toLowerCase();
        this.currentBaseUrl = String(cfg.base_url || '').trim();
        document.getElementById('providerInstallerProvider').value = this.currentProvider;
        const hfUrlInput = document.getElementById('providerInstallerHfUrl');
        const modelIdInput = document.getElementById('providerInstallerModelId');
        if (hfUrlInput) hfUrlInput.value = '';
        const quantSelect = document.getElementById('providerInstallerQuantSelect');
        if (quantSelect) {
            quantSelect.innerHTML = '<option value="">Enter HuggingFace URL first...</option>';
            quantSelect.disabled = true;
        }
        if (modelIdInput) modelIdInput.value = '';

        this.configureInstallerForProvider(this.currentProvider);
        this.installerModal.show();
    }

    openInstallerForSystem() {
        this.openInstaller('llm_model');
    }

    openInstallerForCharacter() {
        this.openInstaller('charLlmModel');
    }

    openInstallerForArchivist() {
        this.openInstaller('llm_archivist_model');
    }

    openInstallerForVision() {
        this.openInstaller('vision_model_name');
    }

    async startInstall() {
        const provider = this.currentProvider;
        if (provider !== 'ollama' && provider !== 'lmstudio') {
            if (typeof UI !== 'undefined') UI.showToast('Installer is available only for Ollama and LM Studio', 'warning');
            return;
        }
        if (provider === 'lmstudio') {
            const msg = 'LM Studio installation is currently unavailable in Chorus Engine. Install in LM Studio, then select in Chorus.';
            this.setInstallStatus(msg, null);
            if (typeof UI !== 'undefined') UI.showToast(msg, 'warning');
            return;
        }

        const hfUrl = (document.getElementById('providerInstallerHfUrl').value || '').trim();
        const quant = (document.getElementById('providerInstallerQuantSelect')?.value || '').trim();

        try {
            this.setInstallStatus('Starting install...', 0);
            let jobId = null;
            if (provider === 'ollama') {
                if (!hfUrl) throw new Error('HuggingFace URL is required for Ollama installs.');
                if (!quant) throw new Error('Quantization is required for Ollama installs.');
                const resp = await fetch('/api/models/pull-hf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hf_url: hfUrl, quantization: quant }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data?.detail || 'Failed to start Ollama install');
                jobId = data.job_id;
            }
            this.installJob = { provider, jobId };
            this.pollInstallStatus();
        } catch (e) {
            console.error(e);
            this.setInstallStatus(`Install failed to start: ${e.message}`, null);
            if (typeof UI !== 'undefined') UI.showToast(`Install failed: ${e.message}`, 'danger');
        }
    }

    queueQuantizationLoad() {
        clearTimeout(this.quantLoadTimer);
        this.quantLoadTimer = setTimeout(() => this.loadHfQuantizations(), 450);
    }

    async loadHfQuantizations() {
        if (this.currentProvider === 'lmstudio') return;

        const url = (document.getElementById('providerInstallerHfUrl')?.value || '').trim();
        const quantSelect = document.getElementById('providerInstallerQuantSelect');
        if (!quantSelect) return;

        if (!url) {
            quantSelect.innerHTML = '<option value="">Enter HuggingFace URL first...</option>';
            quantSelect.disabled = true;
            return;
        }

        try {
            quantSelect.disabled = true;
            quantSelect.innerHTML = '<option value="">Loading quantizations...</option>';
            const resp = await fetch(`/api/models/hf-quantizations?hf_url=${encodeURIComponent(url)}`);
            const data = await resp.json();
            if (!resp.ok) throw new Error(data?.detail || 'Failed to load quantizations');
            const quants = Array.isArray(data.quantizations) ? data.quantizations : [];
            if (!quants.length) {
                quantSelect.innerHTML = '<option value="">No GGUF quantizations found</option>';
                quantSelect.disabled = true;
                const warning = data?.warning || 'No GGUF quantizations found for this repository URL.';
                this.setInstallStatus(warning, null);
                return;
            }

            quantSelect.innerHTML = quants.map((q, idx) => {
                const quant = this.escapeAttr(q.quant || '');
                const label = this.escapeHtml(q.quant || `quant-${idx + 1}`);
                return `<option value="${quant}">${label}</option>`;
            }).join('');
            quantSelect.disabled = false;
        } catch (e) {
            quantSelect.innerHTML = '<option value="">Unable to load quantizations (use ID fallback)</option>';
            quantSelect.disabled = true;
            this.setInstallStatus(`Quantization lookup failed: ${e.message}`, null);
        }
    }

    configureInstallerForProvider(provider) {
        const p = String(provider || '').toLowerCase();
        const isLmStudio = p === 'lmstudio';
        const hfUrlInput = document.getElementById('providerInstallerHfUrl');
        const quantSelect = document.getElementById('providerInstallerQuantSelect');
        const modelIdInput = document.getElementById('providerInstallerModelId');
        const startBtn = document.getElementById('startProviderInstallBtn');

        if (isLmStudio) {
            if (hfUrlInput) {
                hfUrlInput.disabled = true;
                hfUrlInput.placeholder = 'Install in LM Studio app';
            }
            if (quantSelect) {
                quantSelect.disabled = true;
                quantSelect.innerHTML = '<option value="">Unavailable for LM Studio in Chorus</option>';
            }
            if (modelIdInput) {
                modelIdInput.disabled = true;
                modelIdInput.placeholder = 'Unavailable for LM Studio in Chorus';
            }
            if (startBtn) startBtn.disabled = true;
            this.setInstallStatus(
                'LM Studio installation is currently unavailable in Chorus Engine. Install models in LM Studio, then select them in Chorus.',
                null
            );
            return;
        }

        if (hfUrlInput) {
            hfUrlInput.disabled = false;
            hfUrlInput.placeholder = 'https://huggingface.co/username/repo';
        }
        if (quantSelect) {
            quantSelect.disabled = true;
            quantSelect.innerHTML = '<option value="">Enter HuggingFace URL first...</option>';
        }
        if (modelIdInput) {
            modelIdInput.disabled = false;
            modelIdInput.placeholder = 'qwen/qwen2.5-coder-14b@q4_k_m';
        }
        if (startBtn) startBtn.disabled = false;
        this.setInstallStatus('Idle', null);
    }

    async pollInstallStatus() {
        if (!this.installJob) return;
        clearInterval(this.installPollTimer);
        this.installPollTimer = setInterval(async () => {
            const { provider, jobId } = this.installJob;
            try {
                const url = provider === 'ollama'
                    ? `/api/models/download/${jobId}`
                    : `/api/models/lmstudio/install/${jobId}`;
                const resp = await fetch(url);
                const data = await resp.json();
                if (!resp.ok) throw new Error(data?.detail || 'Status poll failed');

                const progress = typeof data.progress === 'number' ? data.progress : null;
                const text = data.status_text || data.last_status || data.status || 'Installing...';
                this.setInstallStatus(text, progress);

                if (data.status === 'completed') {
                    clearInterval(this.installPollTimer);
                    this.setInstallStatus('Install completed.', 100);
                    if (typeof UI !== 'undefined') UI.showToast('Model install completed', 'success');
                    await this.loadInstalledModels(false);
                } else if (data.status === 'failed') {
                    clearInterval(this.installPollTimer);
                    const msg = data.error || 'Install failed';
                    this.setInstallStatus(`Failed: ${msg}`, progress);
                    if (typeof UI !== 'undefined') UI.showToast(`Install failed: ${msg}`, 'danger');
                }
            } catch (e) {
                console.error(e);
            }
        }, 1500);
    }

    setInstallStatus(text, progressPercent) {
        const statusEl = document.getElementById('providerInstallerStatus');
        const bar = document.getElementById('providerInstallerProgress');
        if (statusEl) statusEl.textContent = text;
        if (bar) {
            if (typeof progressPercent === 'number' && isFinite(progressPercent)) {
                const pct = Math.max(0, Math.min(100, Math.round(progressPercent)));
                bar.style.width = `${pct}%`;
                bar.textContent = `${pct}%`;
            } else {
                bar.style.width = '100%';
                bar.textContent = '...';
            }
        }
    }

    formatBytes(bytes) {
        const n = Number(bytes || 0);
        if (!n) return '';
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = n;
        let idx = 0;
        while (size >= 1024 && idx < units.length - 1) {
            size /= 1024;
            idx += 1;
        }
        return `${size.toFixed(idx > 1 ? 1 : 0)} ${units[idx]}`;
    }

    escapeHtml(str) {
        return String(str || '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    escapeAttr(str) {
        return this.escapeHtml(str).replaceAll('"', '&quot;');
    }
}

window.providerModelPicker = new ProviderModelPicker();
document.addEventListener('DOMContentLoaded', () => window.providerModelPicker.init());
