/**
 * Character Management
 * Handles character CRUD operations and UI
 */

const IMMUTABLE_CHARACTERS = ['nova', 'alex', 'aria', 'marcus'];

window.CharacterManagement = {
    currentCharacter: null,
    isEditMode: false,
    
    /**
     * Initialize character management
     */
    init() {
        this.setupEventListeners();
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Open modal button (gear icon next to character dropdown)
        document.getElementById('openCharacterConfig')?.addEventListener('click', () => {
            // Open with current character pre-selected
            const currentCharacterId = App?.state?.selectedCharacterId || document.getElementById('characterSelect')?.value;
            this.openModal(currentCharacterId);
        });
        
        // New character button
        document.getElementById('newCharacterBtn').addEventListener('click', () => {
            this.createNewCharacter();
        });
        
        // Character form submit
        document.getElementById('characterForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveCharacter();
        });
        
        // Clone button
        document.getElementById('cloneCharacterBtn').addEventListener('click', () => {
            this.cloneCurrentCharacter();
        });
        
        // Delete button
        document.getElementById('deleteCharacterBtn').addEventListener('click', () => {
            this.deleteCurrentCharacter();
        });
        
        // Add core memory button
        document.getElementById('addCoreMemoryBtn').addEventListener('click', () => {
            this.addCoreMemoryRow();
        });
        
        // Custom system prompt checkbox - toggle immersion warning
        document.getElementById('charCustomSystemPrompt').addEventListener('change', (e) => {
            const warning = document.getElementById('immersionDisabledWarning');
            warning.style.display = e.target.checked ? 'block' : 'none';
            
            // Disable/enable immersion controls
            const immersionControls = [
                'charImmersionLevel',
                'charAllowPreferences',
                'charAllowOpinions',
                'charAllowExperiences',
                'charAllowPhysicalMetaphor',
                'charAllowPhysicalSensation',
                'charDisclaimerBehavior'
            ];
            immersionControls.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.disabled = e.target.checked;
            });
        });
        
        // TTS Provider change - show/hide provider-specific settings
        document.getElementById('charTtsProvider').addEventListener('change', (e) => {
            const isComfyui = e.target.value === 'comfyui';
            document.getElementById('ttsComfyuiSettings').style.display = isComfyui ? 'block' : 'none';
            document.getElementById('ttsChatterboxSettings').style.display = isComfyui ? 'none' : 'block';
        });
    },
    
    /**
     * Open character management modal
     * @param {string} characterId - Optional character ID to pre-load
     */
    async openModal(characterId = null) {
        await this.loadCharacterList();
        
        // If characterId provided, load that character
        if (characterId) {
            try {
                const character = await API.getCharacter(characterId);
                this.selectCharacter(character);
            } catch (error) {
                console.error('Failed to load character:', error);
            }
        }
        
        const modal = new bootstrap.Modal(document.getElementById('characterManagementModal'));
        modal.show();
    },
    
    /**
     * Load and display character list
     */
    async loadCharacterList() {
        try {
            const data = await API.listCharacters();
            this.renderCharacterList(data.characters);
        } catch (error) {
            console.error('Failed to load characters:', error);
            UI.showToast('Failed to load characters', 'error');
        }
    },
    
    /**
     * Render character list in sidebar
     */
    renderCharacterList(characters) {
        const list = document.getElementById('characterListPanel');
        
        if (!characters || characters.length === 0) {
            list.innerHTML = '<div class="text-muted text-center py-3">No characters found</div>';
            return;
        }
        
        list.innerHTML = characters.map(char => {
            const isImmutable = IMMUTABLE_CHARACTERS.includes(char.id);
            return `
                <button type="button" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center" 
                        data-character-id="${char.id}">
                    <span>
                        <i class="bi bi-person-circle me-2"></i>
                        ${char.name}
                    </span>
                    ${isImmutable ? '<span class="badge bg-secondary">Default</span>' : ''}
                </button>
            `;
        }).join('');
        
        // Add click handlers
        list.querySelectorAll('[data-character-id]').forEach(btn => {
            btn.addEventListener('click', () => {
                const characterId = btn.dataset.characterId;
                const character = characters.find(c => c.id === characterId);
                this.selectCharacter(character);
            });
        });
    },
    
    /**
     * Select a character for editing
     */
    async selectCharacter(character) {
        // Fetch full character details (list endpoint doesn't include system_prompt)
        try {
            const fullCharacter = await API.getCharacter(character.id);
            this.currentCharacter = fullCharacter;
            this.isEditMode = true;
            this.populateForm(fullCharacter);
        } catch (error) {
            console.error('Failed to load character details:', error);
            UI.showToast('Failed to load character details', 'error');
            return;
        }
        
        // Update UI
        document.getElementById('characterEditorPlaceholder').style.display = 'none';
        document.getElementById('characterForm').style.display = 'block';
        
        // Disable ID field for existing characters
        document.getElementById('charId').disabled = true;
        
        // Show/hide buttons based on character type
        const isImmutable = IMMUTABLE_CHARACTERS.includes(character.id);
        document.getElementById('deleteCharacterBtn').style.display = isImmutable ? 'none' : 'inline-block';
        
        // Update form controls state
        const formElements = document.getElementById('characterForm').elements;
        for (let element of formElements) {
            if (element.id !== 'charId' && element.type !== 'submit' && element.type !== 'button') {
                element.disabled = isImmutable;
            }
        }
        
        if (isImmutable) {
            this.showFormStatus('This is an immutable default character. Clone it to make changes.', 'info');
        } else {
            this.hideFormStatus();
        }
    },
    
    /**
     * Create new character
     */
    createNewCharacter() {
        this.currentCharacter = null;
        this.isEditMode = false;
        this.clearForm();
        
        // Update UI
        document.getElementById('characterEditorPlaceholder').style.display = 'none';
        document.getElementById('characterForm').style.display = 'block';
        
        // Enable ID field for new characters
        document.getElementById('charId').disabled = false;
        
        // Enable all form fields
        const formElements = document.getElementById('characterForm').elements;
        for (let element of formElements) {
            element.disabled = false;
        }
        
        // Hide delete button for new characters
        document.getElementById('deleteCharacterBtn').style.display = 'none';
        
        this.hideFormStatus();
    },
    
    /**
     * Populate form with character data
     */
    populateForm(character) {
        // Debug: log the character data to see structure
        console.log('Populating form with character:', character);
        console.log('Immersion settings:', character.immersion_settings);
        console.log('Core memories:', character.core_memories);
        
        // BASIC TAB
        document.getElementById('charId').value = character.id || '';
        document.getElementById('charName').value = character.name || '';
        document.getElementById('charRole').value = character.role || '';
        document.getElementById('charRoleType').value = character.role_type || 'assistant';
        document.getElementById('charSystemPrompt').value = character.system_prompt || '';
        document.getElementById('charCustomSystemPrompt').checked = character.custom_system_prompt || false;
        document.getElementById('charTraits').value = character.personality_traits ? character.personality_traits.join(', ') : '';
        document.getElementById('charProfileImage').value = character.profile_image || '';
        
        // Preferred LLM
        document.getElementById('charLlmModel').value = character.preferred_llm?.model || '';
        document.getElementById('charLlmTemperature').value = character.preferred_llm?.temperature !== undefined ? character.preferred_llm.temperature : '';
        document.getElementById('charLlmMaxTokens').value = character.preferred_llm?.max_tokens || '';
        document.getElementById('charLlmContextWindow').value = character.preferred_llm?.context_window || '';
        
        // IMMERSION TAB
        document.getElementById('charImmersionLevel').value = character.immersion_level || 'balanced';
        
        // Immersion settings
        const immersionSettings = character.immersion_settings || {};
        document.getElementById('charAllowPreferences').checked = Boolean(immersionSettings.allow_preferences);
        document.getElementById('charAllowOpinions').checked = Boolean(immersionSettings.allow_opinions);
        document.getElementById('charAllowExperiences').checked = Boolean(immersionSettings.allow_experiences);
        document.getElementById('charAllowPhysicalMetaphor').checked = Boolean(immersionSettings.allow_physical_metaphor);
        document.getElementById('charAllowPhysicalSensation').checked = Boolean(immersionSettings.allow_physical_sensation);
        document.getElementById('charDisclaimerBehavior').value = immersionSettings.disclaimer_behavior || 'only_when_asked';
        
        // Emotional range
        const emotionalRange = character.emotional_range || {};
        document.getElementById('charEmotionalBaseline').value = emotionalRange.baseline || 'positive';
        document.getElementById('charEmotionalAllowed').value = emotionalRange.allowed ? emotionalRange.allowed.join(', ') : 'positive, curious';
        
        // MEMORY TAB
        const memory = character.memory || {};
        document.getElementById('charMemoryScope').value = memory.scope || 'character';
        
        // Core memories
        this.populateCoreMemories(character.core_memories || []);
        
        // Memory profile
        const memoryProfile = character.memory_profile || {};
        document.getElementById('charExtractFacts').checked = Boolean(memoryProfile.extract_facts);
        document.getElementById('charExtractProjects').checked = Boolean(memoryProfile.extract_projects);
        document.getElementById('charExtractExperiences').checked = Boolean(memoryProfile.extract_experiences);
        document.getElementById('charExtractStories').checked = Boolean(memoryProfile.extract_stories);
        document.getElementById('charExtractRelationship').checked = Boolean(memoryProfile.extract_relationship);
        document.getElementById('charTrackEmotionalWeight').checked = Boolean(memoryProfile.track_emotional_weight);
        document.getElementById('charTrackParticipants').checked = Boolean(memoryProfile.track_participants);
        
        // FEATURES TAB
        const imageGen = character.image_generation || {};
        const videoGen = character.video_generation || {};
        const docAnalysis = character.document_analysis || {};
        const codeExec = character.code_execution || {};
        
        document.getElementById('charImageGenEnabled').checked = Boolean(imageGen.enabled);
        document.getElementById('charVideoGenEnabled').checked = Boolean(videoGen.enabled);
        document.getElementById('charDocAnalysisEnabled').checked = Boolean(docAnalysis.enabled);
        document.getElementById('charDocMaxDocuments').value = docAnalysis.max_documents || '';
        document.getElementById('charDocAllowedTypes').value = docAnalysis.allowed_document_types ? docAnalysis.allowed_document_types.join(', ') : '';
        document.getElementById('charDocMaxChunks').value = docAnalysis.max_chunks || '';
        document.getElementById('charDocBudgetRatio').value = docAnalysis.document_budget_ratio || '';
        document.getElementById('charDocMaxChunks').value = docAnalysis.max_chunks || '';
        document.getElementById('charDocBudgetRatio').value = docAnalysis.document_budget_ratio || '';
        document.getElementById('charCodeExecEnabled').checked = Boolean(codeExec.enabled);
        document.getElementById('charCodeMaxTime').value = codeExec.max_execution_time || 30;
        document.getElementById('charCodeAllowedLibs').value = codeExec.allowed_libraries ? codeExec.allowed_libraries.join(', ') : '';
        
        // VOICE/TTS TAB
        const voice = character.voice || {};
        const ttsProvider = voice.tts_provider || {};
        
        document.getElementById('charVoiceEnabled').checked = Boolean(voice.enabled);
        document.getElementById('charVoiceAlwaysOn').checked = Boolean(voice.always_on);
        document.getElementById('charTtsProvider').value = ttsProvider.provider || 'comfyui';
        
        // Trigger provider change to show correct settings
        const providerChangeEvent = new Event('change');
        document.getElementById('charTtsProvider').dispatchEvent(providerChangeEvent);
        
        // ComfyUI settings
        const comfyui = ttsProvider.comfyui || {};
        document.getElementById('charTtsComfyWorkflow').value = comfyui.workflow_name || '';
        
        // Chatterbox settings
        const chatterbox = ttsProvider.chatterbox || {};
        document.getElementById('charTtsChatterboxTemp').value = chatterbox.temperature || 0.8;
        document.getElementById('charTtsChatterboxChunkThreshold').value = chatterbox.chunk_threshold || 200;
        document.getElementById('charTtsChatterboxCloning').checked = Boolean(chatterbox.use_voice_cloning);
        
        // Trigger custom system prompt warning
        const customPromptEvent = new Event('change');
        document.getElementById('charCustomSystemPrompt').dispatchEvent(customPromptEvent);
    },
    
    /**
     * Populate core memories list
     */
    populateCoreMemories(coreMemories) {
        const container = document.getElementById('coreMemoriesList');
        container.innerHTML = '';
        
        coreMemories.forEach((memory, index) => {
            this.addCoreMemoryRow(memory);
        });
        
        // Add one empty row if none exist
        if (coreMemories.length === 0) {
            this.addCoreMemoryRow();
        }
    },
    
    /**
     * Add a core memory row
     */
    addCoreMemoryRow(memory = null) {
        const container = document.getElementById('coreMemoriesList');
        const index = container.children.length;
        
        const row = document.createElement('div');
        row.className = 'core-memory-item border rounded p-3 mb-2 bg-dark';
        row.innerHTML = `
            <div class="d-flex justify-content-between align-items-start mb-2">
                <strong>Memory #${index + 1}</strong>
                <button type="button" class="btn btn-sm btn-danger remove-core-memory">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
            <div class="mb-2">
                <label class="form-label">Content</label>
                <textarea class="form-control form-control-sm core-memory-content" rows="3" 
                          placeholder="A key fact about the character...">${memory?.content || ''}</textarea>
            </div>
            <div class="row">
                <div class="col-md-8 mb-2">
                    <label class="form-label">Tags</label>
                    <input type="text" class="form-control form-control-sm core-memory-tags" 
                           placeholder="e.g., background, personality" 
                           value="${memory?.tags ? memory.tags.join(', ') : ''}">
                </div>
                <div class="col-md-4 mb-2">
                    <label class="form-label">Priority</label>
                    <select class="form-select form-select-sm core-memory-priority">
                        <option value="low" ${memory?.embedding_priority === 'low' ? 'selected' : ''}>Low</option>
                        <option value="medium" ${memory?.embedding_priority === 'medium' || !memory ? 'selected' : ''}>Medium</option>
                        <option value="high" ${memory?.embedding_priority === 'high' ? 'selected' : ''}>High</option>
                    </select>
                </div>
            </div>
        `;
        
        // Add remove button handler
        row.querySelector('.remove-core-memory').addEventListener('click', () => {
            row.remove();
            // Renumber remaining memories
            this.renumberCoreMemories();
        });
        
        container.appendChild(row);
    },
    
    /**
     * Renumber core memory rows
     */
    renumberCoreMemories() {
        const container = document.getElementById('coreMemoriesList');
        Array.from(container.children).forEach((row, index) => {
            row.querySelector('strong').textContent = `Memory #${index + 1}`;
        });
    },
    
    /**
     * Collect core memories from form
     */
    getCoreMemoriesFromForm() {
        const container = document.getElementById('coreMemoriesList');
        const memories = [];
        
        Array.from(container.children).forEach(row => {
            const content = row.querySelector('.core-memory-content').value.trim();
            if (content) {
                const tagsStr = row.querySelector('.core-memory-tags').value.trim();
                const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(t => t) : [];
                const priority = row.querySelector('.core-memory-priority').value;
                
                memories.push({
                    content: content,
                    tags: tags,
                    embedding_priority: priority
                });
            }
        });
        
        return memories;
    },
    
    /**
     * Clear form
     */
    clearForm() {
        document.getElementById('characterForm').reset();
        
        // Set defaults
        document.getElementById('charRoleType').value = 'assistant';
        document.getElementById('charImmersionLevel').value = 'balanced';
        document.getElementById('charDisclaimerBehavior').value = 'only_when_asked';
        document.getElementById('charEmotionalBaseline').value = 'positive';
        document.getElementById('charMemoryScope').value = 'character';
        document.getElementById('charTtsProvider').value = 'comfyui';
        document.getElementById('charCodeMaxTime').value = '30';
        document.getElementById('charTtsChatterboxTemp').value = '0.8';
        document.getElementById('charTtsChatterboxChunkThreshold').value = '200';
        
        // Clear core memories and add one empty row
        document.getElementById('coreMemoriesList').innerHTML = '';
        this.addCoreMemoryRow();
        
        // Trigger provider change
        const event = new Event('change');
        document.getElementById('charTtsProvider').dispatchEvent(event);
    },
    
    /**
     * Get form data as character object
     */
    getFormData() {
        const traits = document.getElementById('charTraits').value
            .split(',')
            .map(t => t.trim())
            .filter(t => t.length > 0);
        
        const emotionalAllowed = document.getElementById('charEmotionalAllowed').value
            .split(',')
            .map(e => e.trim())
            .filter(e => e.length > 0);
        
        const characterData = {
            id: document.getElementById('charId').value.trim().toLowerCase(),
            name: document.getElementById('charName').value.trim(),
            role: document.getElementById('charRole').value.trim(),
            role_type: document.getElementById('charRoleType').value,
            system_prompt: document.getElementById('charSystemPrompt').value.trim(),
            custom_system_prompt: document.getElementById('charCustomSystemPrompt').checked,
            personality_traits: traits,
            immersion_level: document.getElementById('charImmersionLevel').value,
            
            // Immersion settings
            immersion_settings: {
                allow_preferences: document.getElementById('charAllowPreferences').checked,
                allow_opinions: document.getElementById('charAllowOpinions').checked,
                allow_experiences: document.getElementById('charAllowExperiences').checked,
                allow_physical_metaphor: document.getElementById('charAllowPhysicalMetaphor').checked,
                allow_physical_sensation: document.getElementById('charAllowPhysicalSensation').checked,
                disclaimer_behavior: document.getElementById('charDisclaimerBehavior').value
            },
            
            // Emotional range
            emotional_range: {
                baseline: document.getElementById('charEmotionalBaseline').value,
                allowed: emotionalAllowed
            },
            
            // Memory
            memory: {
                scope: document.getElementById('charMemoryScope').value,
                vector_store: 'chroma_default'
            },
            
            // Core memories
            core_memories: this.getCoreMemoriesFromForm(),
            
            // Memory profile
            memory_profile: {
                extract_facts: document.getElementById('charExtractFacts').checked,
                extract_projects: document.getElementById('charExtractProjects').checked,
                extract_experiences: document.getElementById('charExtractExperiences').checked || null,
                extract_stories: document.getElementById('charExtractStories').checked || null,
                extract_relationship: document.getElementById('charExtractRelationship').checked || null,
                track_emotional_weight: document.getElementById('charTrackEmotionalWeight').checked,
                track_participants: document.getElementById('charTrackParticipants').checked
            },
            
            // Features
            image_generation: {
                enabled: document.getElementById('charImageGenEnabled').checked
            },
            video_generation: {
                enabled: document.getElementById('charVideoGenEnabled').checked
            },
            document_analysis: {
                enabled: document.getElementById('charDocAnalysisEnabled').checked
            },
            code_execution: {
                enabled: document.getElementById('charCodeExecEnabled').checked
            }
        };
        
        // Profile image
        const profileImage = document.getElementById('charProfileImage').value.trim();
        if (profileImage) {
            characterData.profile_image = profileImage;
        }
        
        // Document analysis advanced options
        const docMaxDocs = document.getElementById('charDocMaxDocuments').value.trim();
        if (docMaxDocs) {
            characterData.document_analysis.max_documents = parseInt(docMaxDocs);
        }
        
        const docAllowedTypes = document.getElementById('charDocAllowedTypes').value
            .split(',')
            .map(t => t.trim())
            .filter(t => t.length > 0);
        if (docAllowedTypes.length > 0) {
            characterData.document_analysis.allowed_document_types = docAllowedTypes;
        }
        
        const docMaxChunks = document.getElementById('charDocMaxChunks').value.trim();
        if (docMaxChunks) {
            characterData.document_analysis.max_chunks = parseInt(docMaxChunks);
        }
        
        const docBudgetRatio = document.getElementById('charDocBudgetRatio').value.trim();
        if (docBudgetRatio) {
            characterData.document_analysis.document_budget_ratio = parseFloat(docBudgetRatio);
        }
        
        // Code execution advanced options
        const codeMaxTime = document.getElementById('charCodeMaxTime').value.trim();
        if (codeMaxTime) {
            characterData.code_execution.max_execution_time = parseInt(codeMaxTime);
        }
        
        const codeAllowedLibs = document.getElementById('charCodeAllowedLibs').value
            .split(',')
            .map(l => l.trim())
            .filter(l => l.length > 0);
        if (codeAllowedLibs.length > 0) {
            characterData.code_execution.allowed_libraries = codeAllowedLibs;
        }
        
        // Voice/TTS
        const voiceEnabled = document.getElementById('charVoiceEnabled').checked;
        if (voiceEnabled) {
            const ttsProvider = document.getElementById('charTtsProvider').value;
            characterData.voice = {
                enabled: true,
                always_on: document.getElementById('charVoiceAlwaysOn').checked,
                tts_provider: {
                    provider: ttsProvider
                }
            };
            
            if (ttsProvider === 'comfyui') {
                const workflowName = document.getElementById('charTtsComfyWorkflow').value.trim();
                characterData.voice.tts_provider.comfyui = {
                    workflow_name: workflowName || null
                };
            } else if (ttsProvider === 'chatterbox') {
                characterData.voice.tts_provider.chatterbox = {
                    temperature: parseFloat(document.getElementById('charTtsChatterboxTemp').value) || 0.8,
                    use_voice_cloning: document.getElementById('charTtsChatterboxCloning').checked,
                    chunk_threshold: parseInt(document.getElementById('charTtsChatterboxChunkThreshold').value) || 200
                };
            }
        }
        
        // Preferred LLM
        const llmModel = document.getElementById('charLlmModel').value.trim();
        const llmTemp = document.getElementById('charLlmTemperature').value.trim();
        const llmMaxTokens = document.getElementById('charLlmMaxTokens').value.trim();
        const llmContextWindow = document.getElementById('charLlmContextWindow').value.trim();
        
        if (llmModel || llmTemp || llmMaxTokens || llmContextWindow) {
            characterData.preferred_llm = {};
            if (llmModel) characterData.preferred_llm.model = llmModel;
            if (llmTemp) characterData.preferred_llm.temperature = parseFloat(llmTemp);
            if (llmMaxTokens) characterData.preferred_llm.max_tokens = parseInt(llmMaxTokens);
            if (llmContextWindow) characterData.preferred_llm.context_window = parseInt(llmContextWindow);
        }
        
        return characterData;
    },
    
    /**
     * Save character (create or update)
     */
    async saveCharacter() {
        try {
            const character = this.getFormData();
            
            // Validation
            if (!character.id || !character.name || !character.role || !character.system_prompt) {
                this.showFormStatus('Please fill in all required fields', 'warning');
                return;
            }
            
            // Check if trying to modify immutable character
            if (this.isEditMode && IMMUTABLE_CHARACTERS.includes(character.id)) {
                this.showFormStatus('Cannot modify immutable character. Clone it instead.', 'danger');
                return;
            }
            
            this.showFormStatus('Saving...', 'info');
            
            if (this.isEditMode) {
                // Update existing character
                await API.updateCharacter(character.id, character);
                this.showFormStatus('Character updated successfully!', 'success');
            } else {
                // Create new character
                await API.createCharacter(character);
                this.showFormStatus('Character created successfully!', 'success');
            }
            
            // Reload character list
            await this.loadCharacterList();
            
            // Reload characters in main app
            await App.loadCharacters();
            
        } catch (error) {
            console.error('Failed to save character:', error);
            this.showFormStatus(`Error: ${error.message}`, 'danger');
        }
    },
    
    /**
     * Clone current character
     */
    async cloneCurrentCharacter() {
        if (!this.currentCharacter) {
            UI.showToast('No character selected to clone', 'error');
            return;
        }
        
        // Prompt for new ID
        const newId = prompt(`Clone "${this.currentCharacter.name}" as:\n\nEnter new character ID (lowercase, letters/numbers/underscores/hyphens only):`);
        
        if (!newId) return;
        
        // Validate ID format
        if (!/^[a-z0-9_-]+$/.test(newId)) {
            this.showFormStatus('Invalid character ID format', 'danger');
            return;
        }
        
        try {
            this.showFormStatus('Cloning...', 'info');
            
            await API.cloneCharacter(this.currentCharacter.id, newId);
            
            this.showFormStatus('Character cloned successfully!', 'success');
            
            // Reload lists
            await this.loadCharacterList();
            await App.loadCharacters();
            
            // Switch to the newly cloned character
            const clonedCharacter = await API.getCharacter(newId);
            this.selectCharacter(clonedCharacter);
            
            UI.showToast(`Cloned "${this.currentCharacter.name}" successfully`, 'success');
            
        } catch (error) {
            console.error('Failed to clone character:', error);
            this.showFormStatus(`Error: ${error.message}`, 'danger');
        }
    },
    
    /**
     * Delete current character
     */
    async deleteCurrentCharacter() {
        if (!this.currentCharacter) {
            UI.showToast('No character selected', 'error');
            return;
        }
        
        if (IMMUTABLE_CHARACTERS.includes(this.currentCharacter.id)) {
            this.showFormStatus('Cannot delete immutable character', 'danger');
            return;
        }
        
        const confirmed = confirm(`Are you sure you want to delete "${this.currentCharacter.name}"?\n\nThis action cannot be undone.`);
        
        if (!confirmed) return;
        
        try {
            this.showFormStatus('Deleting...', 'info');
            
            await API.deleteCharacter(this.currentCharacter.id);
            
            this.showFormStatus('Character deleted successfully!', 'success');
            
            // Reload lists
            await this.loadCharacterList();
            await App.loadCharacters();
            
            // Clear form
            this.createNewCharacter();
            
            UI.showToast('Character deleted', 'success');
            
        } catch (error) {
            console.error('Failed to delete character:', error);
            this.showFormStatus(`Error: ${error.message}`, 'danger');
        }
    },
    
    /**
     * Show form status message
     */
    showFormStatus(message, type) {
        const statusEl = document.getElementById('characterFormStatus');
        statusEl.className = `alert alert-${type}`;
        statusEl.textContent = message;
        statusEl.classList.remove('d-none');
    },
    
    /**
     * Hide form status message
     */
    hideFormStatus() {
        const statusEl = document.getElementById('characterFormStatus');
        statusEl.classList.add('d-none');
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => CharacterManagement.init());
} else {
    CharacterManagement.init();
}
