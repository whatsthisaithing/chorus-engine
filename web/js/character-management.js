/**
 * Character Management
 * Handles character CRUD operations and UI
 */

const IMMUTABLE_CHARACTERS = ['nova', 'alex'];

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
        // Open modal button
        document.getElementById('manageCharactersBtn').addEventListener('click', () => {
            this.openModal();
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
    },
    
    /**
     * Open character management modal
     */
    async openModal() {
        await this.loadCharacterList();
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
        document.getElementById('charId').value = character.id || '';
        document.getElementById('charName').value = character.name || '';
        document.getElementById('charRole').value = character.role || '';
        document.getElementById('charSystemPrompt').value = character.system_prompt || '';
        document.getElementById('charImmersionLevel').value = character.immersion_level || 'balanced';
        document.getElementById('charTraits').value = character.personality_traits ? character.personality_traits.join(', ') : '';
        
        // LLM preferences
        document.getElementById('charLlmModel').value = character.preferred_llm?.model || '';
        document.getElementById('charLlmTemperature').value = character.preferred_llm?.temperature !== undefined ? character.preferred_llm.temperature : '';
    },
    
    /**
     * Clear form
     */
    clearForm() {
        document.getElementById('characterForm').reset();
        document.getElementById('charImmersionLevel').value = 'balanced';
        document.getElementById('charLlmModel').value = '';
        document.getElementById('charLlmTemperature').value = '';
    },
    
    /**
     * Get form data as character object
     */
    getFormData() {
        const traits = document.getElementById('charTraits').value
            .split(',')
            .map(t => t.trim())
            .filter(t => t.length > 0);
        
        const llmModel = document.getElementById('charLlmModel').value.trim();
        const llmTemperature = document.getElementById('charLlmTemperature').value.trim();
        
        const characterData = {
            id: document.getElementById('charId').value.trim().toLowerCase(),
            name: document.getElementById('charName').value.trim(),
            role: document.getElementById('charRole').value.trim(),
            system_prompt: document.getElementById('charSystemPrompt').value.trim(),
            immersion_level: document.getElementById('charImmersionLevel').value,
            personality_traits: traits
        };
        
        // Add LLM preferences if specified
        if (llmModel || llmTemperature) {
            characterData.preferred_llm = {};
            if (llmModel) {
                characterData.preferred_llm.model = llmModel;
            }
            if (llmTemperature) {
                characterData.preferred_llm.temperature = parseFloat(llmTemperature);
            }
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
            
            // Clear form after a delay
            setTimeout(() => {
                this.createNewCharacter();
            }, 1500);
            
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
