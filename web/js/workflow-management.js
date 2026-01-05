/**
 * Workflow Management
 * Handles workflow upload, listing, deletion, renaming, and default selection
 */

window.WorkflowManagement = {
    currentCharacterId: null,
    workflows: [],
    defaultWorkflow: null,
    modal: null,
    selectedWorkflowType: 'image', // Phase 6.5: Track selected workflow type
    
    /**
     * Initialize workflow management
     */
    init() {
        this.modal = new bootstrap.Modal(document.getElementById('workflowManagementModal'));
        this.setupEventListeners();
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Open modal button
        document.getElementById('manageWorkflowsBtn').addEventListener('click', () => {
            this.openModal();
        });
        
        // Upload workflow button
        document.getElementById('uploadWorkflowBtn').addEventListener('click', () => {
            this.uploadWorkflow();
        });
        
        // Phase 6.5: Workflow type selector
        document.querySelectorAll('input[name="workflowType"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.selectedWorkflowType = e.target.value;
                this.updatePlaceholderDocs();
                this.renderWorkflowsList(); // Re-filter workflows
            });
        });
    },
    
    /**
     * Update placeholder documentation based on workflow type
     */
    updatePlaceholderDocs() {
        const placeholderDiv = document.getElementById('workflowPlaceholders');
        if (!placeholderDiv) return;
        
        if (this.selectedWorkflowType === 'image') {
            placeholderDiv.innerHTML = `
                <strong>Image Placeholders:</strong>
                <code>__CHORUS_PROMPT__</code>, <code>__CHORUS_NEGATIVE__</code>, <code>__CHORUS_SEED__</code>
            `;
        } else if (this.selectedWorkflowType === 'audio') {
            placeholderDiv.innerHTML = `
                <strong>TTS Placeholders:</strong>
                <code>__CHORUS_TEXT__</code> (text to speak),
                <code>__CHORUS_VOICE_SAMPLE__</code> (voice sample path),
                <code>__CHORUS_VOICE_TRANSCRIPT__</code> (sample transcript)
            `;
        } else if (this.selectedWorkflowType === 'video') {
            placeholderDiv.innerHTML = `
                <strong>Video Placeholders:</strong> Coming soon
            `;
        }
    },
    
    /**
     * Open workflow management modal
     */
    async openModal() {
        const characterId = window.App?.state?.selectedCharacterId;
        
        if (!characterId) {
            UI.showToast('Please select a character first');
            return;
        }
        
        this.currentCharacterId = characterId;
        
        // Update modal title
        document.getElementById('workflowCharacterName').textContent = this.currentCharacterId;
        
        // Update placeholder docs
        this.updatePlaceholderDocs();
        
        // Load workflows
        await this.loadWorkflows();
        
        // Show modal
        this.modal.show();
    },
    
    /**
     * Load workflows for current character
     */
    async loadWorkflows() {
        try {
            const response = await API.listWorkflows(this.currentCharacterId);
            this.workflows = response.workflows || [];
            this.defaultWorkflow = response.default_workflow;
            
            this.renderWorkflowsList();
        } catch (error) {
            console.error('Failed to load workflows:', error);
            UI.showToast('Failed to load workflows: ' + error.message, 'danger');
        }
    },
    
    /**
     * Render workflows list
     */
    renderWorkflowsList() {
        const container = document.getElementById('workflowsList');
        
        // Phase 6.5: Filter workflows by selected type
        const filteredWorkflows = this.workflows.filter(w => {
            // For now, assume image type if not specified (legacy workflows)
            const workflowType = w.workflow_type || 'image';
            return workflowType === this.selectedWorkflowType;
        });
        
        if (filteredWorkflows.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-3">
                    <small>No ${this.selectedWorkflowType} workflows found. Upload one to get started.</small>
                </div>
            `;
            return;
        }
        
        container.innerHTML = filteredWorkflows.map(workflow => {
            const isDefault = workflow.is_default;
            
            return `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div class="form-check flex-grow-1">
                            <input class="form-check-input" type="radio" name="defaultWorkflow" 
                                   id="workflow_${workflow.id}" value="${workflow.name}" 
                                   ${isDefault ? 'checked' : ''}
                                   onchange="WorkflowManagement.setDefaultWorkflow('${workflow.name}')">
                            <label class="form-check-label ms-2" for="workflow_${workflow.id}">
                                <strong>${workflow.name}</strong>
                                ${isDefault ? '<span class="badge bg-primary ms-2">Default</span>' : ''}
                            </label>
                        </div>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary" onclick="WorkflowManagement.editConfig(${workflow.id})" title="Edit Config">
                                <i class="bi bi-gear"></i>
                            </button>
                            <button class="btn btn-outline-secondary" onclick="WorkflowManagement.renameWorkflow('${workflow.name}')" title="Rename">
                                <i class="bi bi-pencil"></i>
                            </button>
                            <button class="btn btn-outline-danger" onclick="WorkflowManagement.deleteWorkflow('${workflow.name}')" title="Delete">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </div>
                    ${workflow.trigger_word || workflow.default_style ? `
                        <div class="text-muted small mt-1">
                            ${workflow.trigger_word ? `<div><strong>Trigger:</strong> ${workflow.trigger_word}</div>` : ''}
                            ${workflow.default_style ? `<div><strong>Style:</strong> ${workflow.default_style.substring(0, 50)}${workflow.default_style.length > 50 ? '...' : ''}</div>` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
    },
    
    /**
     * Upload a new workflow
     */
    async uploadWorkflow() {
        const nameInput = document.getElementById('newWorkflowName');
        const fileInput = document.getElementById('workflowFileInput');
        
        const workflowName = nameInput.value.trim();
        const file = fileInput.files[0];
        
        if (!workflowName) {
            UI.showToast('Please enter a workflow name', 'warning');
            return;
        }
        
        if (!file) {
            UI.showToast('Please select a workflow file', 'warning');
            return;
        }
        
        try {
            // Read file as JSON
            const fileContent = await file.text();
            const workflowData = JSON.parse(fileContent);
            
            // Upload to backend (Phase 6.5: include workflow type)
            await API.uploadWorkflow(this.currentCharacterId, workflowName, workflowData, this.selectedWorkflowType);
            
            UI.showToast(`${this.selectedWorkflowType.charAt(0).toUpperCase() + this.selectedWorkflowType.slice(1)} workflow "${workflowName}" uploaded successfully!`, 'success');
            
            // Clear inputs
            nameInput.value = '';
            fileInput.value = '';
            
            // Reload workflows
            await this.loadWorkflows();
            
        } catch (error) {
            console.error('Failed to upload workflow:', error);
            UI.showToast('Failed to upload workflow: ' + error.message, 'danger');
        }
    },
    
    /**
     * Delete a workflow
     */
    async deleteWorkflow(workflowName) {
        if (!confirm(`Delete workflow "${workflowName}"? This cannot be undone.`)) {
            return;
        }
        
        try {
            await API.deleteWorkflow(this.currentCharacterId, workflowName);
            UI.showToast(`Workflow "${workflowName}" deleted`, 'success');
            
            // Reload workflows
            await this.loadWorkflows();
            
        } catch (error) {
            console.error('Failed to delete workflow:', error);
            UI.showToast('Failed to delete workflow: ' + error.message, 'danger');
        }
    },
    
    /**
     * Rename a workflow
     */
    async renameWorkflow(oldName) {
        const newName = prompt(`Rename workflow "${oldName}" to:`, oldName);
        
        if (!newName || newName === oldName) {
            return;
        }
        
        try {
            await API.renameWorkflow(this.currentCharacterId, oldName, newName);
            UI.showToast(`Workflow renamed to "${newName}"`, 'success');
            
            // Reload workflows
            await this.loadWorkflows();
            
        } catch (error) {
            console.error('Failed to rename workflow:', error);
            UI.showToast('Failed to rename workflow: ' + error.message, 'danger');
        }
    },
    
    /**
     * Set default workflow
     */
    async setDefaultWorkflow(workflowName) {
        try {
            await API.setDefaultWorkflow(this.currentCharacterId, workflowName);
            UI.showToast(`Default workflow set to "${workflowName}"`, 'success');
            
            // Reload workflows to update UI
            await this.loadWorkflows();
            
        } catch (error) {
            console.error('Failed to set default workflow:', error);
            UI.showToast('Failed to set default workflow: ' + error.message, 'danger');
            
            // Reload workflows to reset radio buttons
            await this.loadWorkflows();
        }
    },
    
    /**
     * Edit workflow configuration
     */
    async editConfig(workflowId) {
        // Find the workflow
        const workflow = this.workflows.find(w => w.id === workflowId);
        if (!workflow) return;
        
        // Create modal content
        const modalHtml = `
            <div class="modal fade" id="editConfigModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Edit Workflow Config: ${workflow.name}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <label for="editTriggerWord" class="form-label">Trigger Word</label>
                                <input type="text" class="form-control" id="editTriggerWord" 
                                       value="${workflow.trigger_word || ''}" 
                                       placeholder="e.g., nova">
                                <small class="text-muted">Word or phrase to trigger image generation</small>
                            </div>
                            <div class="mb-3">
                                <label for="editDefaultStyle" class="form-label">Default Style</label>
                                <textarea class="form-control" id="editDefaultStyle" rows="2" 
                                          placeholder="e.g., photorealistic portrait, dramatic lighting">${workflow.default_style || ''}</textarea>
                                <small class="text-muted">Base style applied to all image prompts</small>
                            </div>
                            <div class="mb-3">
                                <label for="editNegativePrompt" class="form-label">Negative Prompt</label>
                                <textarea class="form-control" id="editNegativePrompt" rows="2" 
                                          placeholder="e.g., cartoon, anime, illustration, 3d render">${workflow.negative_prompt || ''}</textarea>
                                <small class="text-muted">Things to avoid in generated images</small>
                            </div>
                            <div class="mb-3">
                                <label for="editSelfDescription" class="form-label">Self Description</label>
                                <textarea class="form-control" id="editSelfDescription" rows="3" 
                                          placeholder="e.g., 25 year old woman with auburn hair and green eyes">${workflow.self_description || ''}</textarea>
                                <small class="text-muted">Physical description for character appearances</small>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveConfigBtn">Save Changes</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        const existing = document.getElementById('editConfigModal');
        if (existing) existing.remove();
        
        // Add to DOM
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Show modal
        const editModal = new bootstrap.Modal(document.getElementById('editConfigModal'));
        editModal.show();
        
        // Handle save
        document.getElementById('saveConfigBtn').onclick = async () => {
            const triggerWord = document.getElementById('editTriggerWord').value.trim();
            const defaultStyle = document.getElementById('editDefaultStyle').value.trim();
            const negativePrompt = document.getElementById('editNegativePrompt').value.trim();
            const selfDescription = document.getElementById('editSelfDescription').value.trim();
            
            try {
                await API.updateWorkflowConfig(workflowId, {
                    trigger_word: triggerWord || null,
                    default_style: defaultStyle || null,
                    negative_prompt: negativePrompt || null,
                    self_description: selfDescription || null
                });
                
                UI.showToast('Workflow configuration updated', 'success');
                editModal.hide();
                
                // Reload workflows
                await this.loadWorkflows();
                
            } catch (error) {
                console.error('Failed to update config:', error);
                UI.showToast('Failed to update config: ' + error.message, 'danger');
            }
        };
        
        // Clean up modal on hide
        document.getElementById('editConfigModal').addEventListener('hidden.bs.modal', () => {
            document.getElementById('editConfigModal').remove();
        });
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    WorkflowManagement.init();
});
