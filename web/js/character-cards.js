/**
 * Character Card Management
 * Handles import/export of character cards (Chorus Engine and SillyTavern formats)
 */

window.CharacterCards = {
    currentPreview: null,
    
    /**
     * Initialize character card management
     */
    init() {
        this.setupEventListeners();
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Import Character Card menu item
        document.getElementById('importCharacterCardMenuItem')?.addEventListener('click', () => {
            this.openImportModal();
        });
        
        // Export Character Card button in character editor
        document.getElementById('exportCharacterCardBtn')?.addEventListener('click', () => {
            this.openExportModal();
        });
        
        // Regenerate Vectors button in character editor
        document.getElementById('regenerateVectorsModalBtn')?.addEventListener('click', async () => {
            if (window.App && typeof window.App.showRegenerateVectorsModal === 'function') {
                await window.App.showRegenerateVectorsModal();
            }
        });
        
        // Confirm export button
        document.getElementById('confirmExportCardBtn')?.addEventListener('click', async () => {
            await this.confirmExport();
        });
        
        // Export modal upload button
        document.getElementById('exportModalUploadBtn')?.addEventListener('click', () => {
            document.getElementById('exportModalImageInput').click();
        });
        
        // Export modal image input
        document.getElementById('exportModalImageInput')?.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await this.handleExportModalImageUpload(e.target.files[0]);
            }
        });
        
        // Upload profile image button
        document.getElementById('uploadProfileImageBtn')?.addEventListener('click', () => {
            document.getElementById('profileImageFileInput').click();
        });
        
        // Profile image file input change
        document.getElementById('profileImageFileInput')?.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await this.handleProfileImageUpload(e.target.files[0]);
            }
        });
        
        // Select file button
        document.getElementById('selectCardFileBtn')?.addEventListener('click', () => {
            document.getElementById('cardFileInput').click();
        });
        
        // File input change
        document.getElementById('cardFileInput')?.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await this.handleFileSelected(e.target.files[0]);
            }
        });
        
        // Back to upload button
        document.getElementById('backToUploadBtn')?.addEventListener('click', () => {
            this.resetToUploadView();
        });
        
        // Confirm import button
        document.getElementById('confirmImportCardBtn')?.addEventListener('click', async () => {
            await this.confirmImport();
        });
        
        // Reset when modal is closed
        const modal = document.getElementById('importCharacterCardModal');
        modal?.addEventListener('hidden.bs.modal', () => {
            this.resetModal();
        });
        
        // Drag and drop for card upload
        this.setupDragAndDrop();
    },
    
    /**
     * Setup drag and drop for card upload
     */
    setupDragAndDrop() {
        const dropZone = document.getElementById('cardDropZone');
        if (!dropZone) return;
        
        // Prevent default drag behaviors on the document
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        // Highlight drop zone when dragging over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('border-primary');
                dropZone.style.backgroundColor = 'rgba(13, 110, 253, 0.05)';
                const icon = document.getElementById('cardUploadIcon');
                if (icon) {
                    icon.style.color = 'var(--bs-primary)';
                    icon.style.transform = 'scale(1.1)';
                }
            });
        });
        
        // Remove highlight when dragging leaves
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('border-primary');
                dropZone.style.backgroundColor = '';
                const icon = document.getElementById('cardUploadIcon');
                if (icon) {
                    icon.style.color = 'var(--bs-secondary)';
                    icon.style.transform = 'scale(1)';
                }
            });
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', async (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                await this.handleFileSelected(files[0]);
            }
        });
        
        // Make the entire drop zone clickable
        dropZone.addEventListener('click', (e) => {
            // Don't trigger if clicking the button itself
            if (e.target.id !== 'selectCardFileBtn' && !e.target.closest('#selectCardFileBtn')) {
                document.getElementById('cardFileInput').click();
            }
        });
        dropZone.style.cursor = 'pointer';
    },
    
    /**
     * Open the import modal
     */
    openImportModal() {
        this.resetModal();
        const modal = new bootstrap.Modal(document.getElementById('importCharacterCardModal'));
        modal.show();
    },
    
    /**
     * Reset the modal to initial state
     */
    resetModal() {
        this.currentPreview = null;
        document.getElementById('cardFileInput').value = '';
        document.getElementById('customCharacterName').value = '';
        this.resetToUploadView();
    },
    
    /**
     * Reset to upload view
     */
    resetToUploadView() {
        document.getElementById('cardUploadSection').style.display = 'block';
        document.getElementById('cardPreviewSection').style.display = 'none';
        document.getElementById('cardImportProgress').style.display = 'none';
        document.getElementById('confirmImportCardBtn').style.display = 'none';
    },
    
    /**
     * Handle file selected for import
     */
    async handleFileSelected(file) {
        if (!file) return;
        
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.png')) {
            UI.showToast('Invalid file type. Please select a PNG file.', 'error');
            return;
        }
        
        // Show progress
        document.getElementById('cardUploadSection').style.display = 'none';
        document.getElementById('cardImportProgress').style.display = 'block';
        
        try {
            // Upload and preview the card
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/characters/cards/import/preview', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to preview character card');
            }
            
            const result = await response.json();
            this.currentPreview = result;
            
            // Display preview
            this.displayPreview(file, result);
            
        } catch (error) {
            console.error('Error previewing character card:', error);
            UI.showToast(`Failed to preview card: ${error.message}`, 'error');
            this.resetToUploadView();
        }
    },
    
    /**
     * Display the character card preview
     */
    displayPreview(file, previewData) {
        // Debug logging
        console.log('Preview data received:', previewData);
        console.log('Character data:', previewData?.character_data);
        
        // Show preview section
        document.getElementById('cardImportProgress').style.display = 'none';
        document.getElementById('cardPreviewSection').style.display = 'block';
        document.getElementById('confirmImportCardBtn').style.display = 'inline-block';
        
        // Display image
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('cardPreviewImage').src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Display format
        const formatMap = {
            'CHORUS_V1': 'Chorus Engine V1',
            'SILLYTAVERN_V2': 'SillyTavern V2',
            'SILLYTAVERN_V3': 'SillyTavern V3',
            'UNKNOWN': 'Unknown Format'
        };
        document.getElementById('cardPreviewFormat').textContent = formatMap[previewData.format] || previewData.format;
        
        // Get character data with fallback
        const charData = previewData?.character_data || {};
        
        // Display basic info
        document.getElementById('cardPreviewName').textContent = charData.name || 'Unnamed';
        // Show full role (no truncation)
        const roleText = charData.role || charData.scenario || 'No role/scenario defined';
        document.getElementById('cardPreviewDescription').textContent = roleText;
        
        // Display personality (Chorus format)
        let personalityHtml = '';
        
        // Show full system prompt with scrollable container
        if (charData.system_prompt) {
            personalityHtml += `<div class="mb-3"><strong>System Prompt:</strong><div class="mt-1 small" style="max-height: 200px; overflow-y: auto; white-space: pre-wrap; border: 1px solid var(--bs-border-color); border-radius: 4px; padding: 8px;">${this.escapeHtml(charData.system_prompt)}</div></div>`;
        }
        
        // Show personality traits if present
        if (charData.personality_traits?.length > 0) {
            personalityHtml += `<div class="mt-2"><strong>Personality Traits:</strong> ${charData.personality_traits.map(t => this.escapeHtml(t)).join(', ')}</div>`;
        }
        
        // Show core memories count
        if (charData.core_memories?.length > 0) {
            personalityHtml += `<div class="mt-2\"><strong>Core Memories:</strong> ${charData.core_memories.length} memories</div>`;
        }
        
        // Show immersion level if set
        if (charData.immersion_level) {
            const immersionLabels = {
                'minimal': 'Minimal (Assistant mode)',
                'balanced': 'Balanced',
                'full': 'Full Immersion',
                'unbounded': 'Unbounded (Roleplay)'
            };
            personalityHtml += `<div class="mt-2"><strong>Immersion Level:</strong> ${immersionLabels[charData.immersion_level] || charData.immersion_level}</div>`;
        }
        
        // Show role type if set
        if (charData.role_type) {
            personalityHtml += `<div class="mt-2"><strong>Type:</strong> ${charData.role_type}</div>`;
        }
        
        // Show SillyTavern import indicator if present
        if (charData.extensions?.sillytavern_import) {
            personalityHtml += `<div class="mt-3 alert alert-info small"><i class="bi bi-info-circle me-2"></i><strong>SillyTavern Import:</strong> Character data has been automatically adapted to Chorus Engine format. Original fields preserved in extensions.</div>`;
            personalityHtml += `<div class="mt-2 alert alert-secondary small"><i class="bi bi-code-square me-2"></i><strong>Macro Processing:</strong> Any SillyTavern macros ({{char}}, {{user}}, etc.) will be processed when the character is loaded, allowing names and other details to stay up-to-date with changes.</div>`;
        }
        
        // Legacy traits support (if present)
        if (charData.traits) {
            const traits = charData.traits;
            personalityHtml += '<div class="mt-2"><strong>Legacy Traits:</strong></div>';
            if (traits.quirks?.length > 0) {
                personalityHtml += `<div class="small"><strong>Quirks:</strong> ${traits.quirks.map(q => this.escapeHtml(q)).join(', ')}</div>`;
            }
            if (traits.interests?.length > 0) {
                personalityHtml += `<div class="small"><strong>Interests:</strong> ${traits.interests.map(i => this.escapeHtml(i)).join(', ')}</div>`;
            }
            if (traits.dislikes?.length > 0) {
                personalityHtml += `<div class="small"><strong>Dislikes:</strong> ${traits.dislikes.map(d => this.escapeHtml(d)).join(', ')}</div>`;
            }
        }
        document.getElementById('cardPreviewPersonality').innerHTML = personalityHtml || '<p class="text-secondary">No personality information</p>';
        
        // Display voice config
        let voiceHtml = '';
        if (charData.voice) {
            const voice = charData.voice;
            if (voice.provider) {
                voiceHtml += `<div><strong>Provider:</strong> ${this.escapeHtml(voice.provider)}</div>`;
            }
            if (voice.voice_id) {
                voiceHtml += `<div><strong>Voice ID:</strong> ${this.escapeHtml(voice.voice_id)}</div>`;
            }
            if (voice.stability !== undefined) {
                voiceHtml += `<div><strong>Stability:</strong> ${voice.stability}</div>`;
            }
            if (voice.similarity_boost !== undefined) {
                voiceHtml += `<div><strong>Similarity Boost:</strong> ${voice.similarity_boost}</div>`;
            }
        }
        document.getElementById('cardPreviewVoice').innerHTML = voiceHtml || '<p class="text-secondary">No voice configuration</p>';
    },
    
    /**
     * Confirm and import the character
     */
    async confirmImport() {
        if (!this.currentPreview) {
            UI.showToast('No character preview available', 'error');
            return;
        }
        
        // Show progress
        document.getElementById('cardPreviewSection').style.display = 'none';
        document.getElementById('cardImportProgress').style.display = 'block';
        document.getElementById('confirmImportCardBtn').style.display = 'none';
        
        try {
            const customName = document.getElementById('customCharacterName').value.trim();
            
            const response = await fetch('/characters/cards/import/confirm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    preview_id: this.currentPreview.preview_id,
                    custom_name: customName || undefined
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to import character card');
            }
            
            const result = await response.json();
            
            UI.showToast(`Character "${result.character_name}" imported successfully!`, 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('importCharacterCardModal'));
            modal.hide();
            
            // Reload characters
            if (window.App && typeof window.App.loadCharacters === 'function') {
                await window.App.loadCharacters();
                
                // Select the newly imported character if possible
                const characterSelect = document.getElementById('characterSelect');
                if (characterSelect) {
                    // The character ID is the sanitized filename
                    const characterId = result.character_id;
                    const option = Array.from(characterSelect.options).find(opt => opt.value === characterId);
                    if (option) {
                        characterSelect.value = characterId;
                        if (window.App && typeof window.App.selectCharacter === 'function') {
                            await window.App.selectCharacter(characterId);
                        }
                    }
                }
            }
            
        } catch (error) {
            console.error('Error importing character card:', error);
            UI.showToast(`Failed to import card: ${error.message}`, 'error');
            this.resetToUploadView();
            document.getElementById('confirmImportCardBtn').style.display = 'inline-block';
        }
    },
    
    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    /**
     * Open the export modal
     */
    openExportModal() {
        // Get current character from CharacterManagement
        const currentCharacter = window.CharacterManagement?.currentCharacter;
        if (!currentCharacter) {
            UI.showToast('No character selected to export', 'error');
            return;
        }
        
        // Set character name in modal
        document.getElementById('exportCardCharacterName').textContent = currentCharacter.name || currentCharacter.id;
        
        // Check if character has a real profile image
        const hasProfileImage = this.checkHasProfileImage(currentCharacter);
        
        // Update image preview
        this.updateExportImagePreview(currentCharacter, hasProfileImage);
        
        // Show/hide warning and enable/disable export button
        const warningEl = document.getElementById('exportNoImageWarning');
        const exportBtn = document.getElementById('confirmExportCardBtn');
        const optionsSection = document.getElementById('exportOptionsSection');
        
        if (!hasProfileImage) {
            warningEl.style.display = 'block';
            exportBtn.disabled = true;
            optionsSection.style.opacity = '0.5';
            optionsSection.style.pointerEvents = 'none';
        } else {
            warningEl.style.display = 'none';
            exportBtn.disabled = false;
            optionsSection.style.opacity = '1';
            optionsSection.style.pointerEvents = 'auto';
        }
        
        // Reset form
        document.getElementById('exportIncludeVoice').checked = true;
        document.getElementById('exportIncludeWorkflows').checked = true;
        document.getElementById('exportVoiceSampleUrl').value = '';
        document.getElementById('exportCardProgress').style.display = 'none';
        document.getElementById('confirmExportCardBtn').style.display = 'inline-block';
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('exportCharacterCardModal'));
        modal.show();
    },
    
    /**
     * Check if character has a real profile image (not default)
     */
    checkHasProfileImage(character) {
        const profileImage = character.profile_image;
        
        // No profile image set
        if (!profileImage) {
            return false;
        }
        
        // Check if it's a default image
        if (profileImage.includes('default.') || profileImage === '') {
            return false;
        }
        
        return true;
    },
    
    /**
     * Update the export image preview
     */
    updateExportImagePreview(character, hasProfileImage) {
        const previewImg = document.getElementById('exportCardPreviewImage');
        const statusText = document.getElementById('exportCardImageStatus');
        
        if (hasProfileImage && character.profile_image) {
            previewImg.src = `/character_images/${character.profile_image}`;
            statusText.textContent = 'Profile image ready';
            statusText.className = 'text-muted';
        } else {
            previewImg.src = '/character_images/default.svg';
            statusText.textContent = 'Using default image (not recommended)';
            statusText.className = 'text-warning';
        }
    },
    
    /**
     * Handle image upload from export modal
     */
    async handleExportModalImageUpload(file) {
        const currentCharacter = window.CharacterManagement?.currentCharacter;
        if (!currentCharacter) {
            UI.showToast('No character selected', 'error');
            return;
        }
        
        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            UI.showToast('Invalid file type. Please select a PNG, JPG, or WEBP image.', 'error');
            return;
        }
        
        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            UI.showToast('File too large. Maximum size is 10MB.', 'error');
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`/characters/${currentCharacter.id}/upload-profile-image`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to upload profile image');
            }
            
            const result = await response.json();
            
            // Update character in memory
            currentCharacter.profile_image = result.filename;
            
            // Update the character form if it's open
            const profileImageInput = document.getElementById('charProfileImage');
            if (profileImageInput) {
                profileImageInput.value = result.filename;
            }
            
            // Update preview and enable export
            this.updateExportImagePreview(currentCharacter, true);
            document.getElementById('exportNoImageWarning').style.display = 'none';
            document.getElementById('confirmExportCardBtn').disabled = false;
            document.getElementById('exportOptionsSection').style.opacity = '1';
            document.getElementById('exportOptionsSection').style.pointerEvents = 'auto';
            
            UI.showToast('Profile image uploaded successfully!', 'success');
            
        } catch (error) {
            console.error('Error uploading profile image:', error);
            UI.showToast(`Failed to upload image: ${error.message}`, 'error');
        } finally {
            // Reset file input
            document.getElementById('exportModalImageInput').value = '';
        }
    },
    
    /**
     * Confirm and export the character card
     */
    async confirmExport() {
        const currentCharacter = window.CharacterManagement?.currentCharacter;
        if (!currentCharacter) {
            UI.showToast('No character selected', 'error');
            return;
        }
        
        // Show progress
        document.getElementById('confirmExportCardBtn').style.display = 'none';
        document.getElementById('exportCardProgress').style.display = 'block';
        
        try {
            const includeVoice = document.getElementById('exportIncludeVoice').checked;
            const includeWorkflows = document.getElementById('exportIncludeWorkflows').checked;
            const voiceSampleUrl = document.getElementById('exportVoiceSampleUrl').value.trim();
            
            // Build form data (backend expects Form parameters, not JSON)
            const formData = new FormData();
            formData.append('character_name', currentCharacter.id);
            formData.append('include_voice', includeVoice.toString());
            formData.append('include_workflows', includeWorkflows.toString());
            if (voiceSampleUrl) {
                formData.append('voice_sample_url', voiceSampleUrl);
            }
            
            const response = await fetch('/characters/cards/export', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                let errorMsg = 'Failed to export character card';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || errorMsg;
                } catch (e) {
                    errorMsg = `HTTP ${response.status}: ${response.statusText}`;
                }
                throw new Error(errorMsg);
            }
            
            // Get the PNG blob
            const blob = await response.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentCharacter.id}_card.png`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            UI.showToast(`Character card exported: ${a.download}`, 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('exportCharacterCardModal'));
            modal.hide();
            
        } catch (error) {
            console.error('Error exporting character card:', error);
            UI.showToast(`Failed to export card: ${error.message}`, 'error');
            document.getElementById('confirmExportCardBtn').style.display = 'inline-block';
            document.getElementById('exportCardProgress').style.display = 'none';
        }
    },
    
    /**
     * Handle profile image upload
     */
    async handleProfileImageUpload(file) {
        const currentCharacter = window.CharacterManagement?.currentCharacter;
        if (!currentCharacter) {
            UI.showToast('No character selected', 'error');
            return;
        }
        
        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            UI.showToast('Invalid file type. Please select a PNG, JPG, or WEBP image.', 'error');
            return;
        }
        
        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            UI.showToast('File too large. Maximum size is 10MB.', 'error');
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`/characters/${currentCharacter.id}/upload-profile-image`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to upload profile image');
            }
            
            const result = await response.json();
            
            // Update the profile image field with just the filename (character_id.png)
            const filename = `${currentCharacter.id}.png`;
            document.getElementById('charProfileImage').value = filename;
            
            UI.showToast(`Profile image uploaded: ${filename}`, 'success');
            
        } catch (error) {
            console.error('Error uploading profile image:', error);
            UI.showToast(`Failed to upload image: ${error.message}`, 'error');
        } finally {
            // Reset file input
            document.getElementById('profileImageFileInput').value = '';
        }
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        CharacterCards.init();
    });
} else {
    CharacterCards.init();
}
