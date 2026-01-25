/**
 * Character Backup Restore
 * Handles restoring characters from .zip backup files
 */

window.CharacterRestore = {
    selectedFile: null,
    
    /**
     * Initialize character restore
     */
    init() {
        this.setupEventListeners();
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Restore menu item
        document.getElementById('restoreCharacterMenuItem')?.addEventListener('click', () => {
            this.openRestoreModal();
        });
        
        // File input
        const fileInput = document.getElementById('restoreFileInput');
        fileInput?.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelected(e.target.files[0]);
            }
        });
        
        // Select file button
        document.getElementById('selectRestoreFileBtn')?.addEventListener('click', () => {
            fileInput?.click();
        });
        
        // Drag and drop
        const dropZone = document.getElementById('restoreDropZone');
        
        dropZone?.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = 'var(--bs-primary)';
            dropZone.style.backgroundColor = 'rgba(var(--bs-primary-rgb), 0.05)';
            document.getElementById('restoreUploadIcon').style.color = 'var(--bs-primary)';
        });
        
        dropZone?.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = '';
            dropZone.style.backgroundColor = '';
            document.getElementById('restoreUploadIcon').style.color = '';
        });
        
        dropZone?.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = '';
            dropZone.style.backgroundColor = '';
            document.getElementById('restoreUploadIcon').style.color = '';
            
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.endsWith('.zip') || file.name.endsWith('.cbak')) {
                    this.handleFileSelected(file);
                } else {
                    UI.showToast('Please upload a .zip backup file', 'error');
                }
            }
        });
        
        // Confirm restore button
        document.getElementById('confirmRestoreBtn')?.addEventListener('click', async () => {
            await this.confirmRestore();
        });
        
        // Modal close handler
        const modal = document.getElementById('restoreCharacterModal');
        modal?.addEventListener('hidden.bs.modal', () => {
            this.resetModal();
        });
    },
    
    /**
     * Open restore modal
     */
    openRestoreModal() {
        const modal = new bootstrap.Modal(document.getElementById('restoreCharacterModal'));
        this.resetModal();
        modal.show();
    },
    
    /**
     * Handle file selected
     */
    handleFileSelected(file) {
        if (!file.name.endsWith('.zip') && !file.name.endsWith('.cbak')) {
            UI.showToast('Please select a .zip backup file', 'error');
            return;
        }
        
        this.selectedFile = file;
        
        // Update UI to show file selected
        const dropZone = document.getElementById('restoreDropZone');
        dropZone.classList.add('border-success');
        dropZone.querySelector('.card-title').textContent = `Selected: ${file.name}`;
        dropZone.querySelector('.card-text').textContent = `File size: ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        
        UI.showToast(`Backup file selected: ${file.name}`, 'success');
    },
    
    /**
     * Confirm restore
     */
    async confirmRestore() {
        if (!this.selectedFile) {
            UI.showToast('Please select a backup file first', 'error');
            return;
        }
        
        const customId = document.getElementById('restoreCharacterName').value.trim();
        const autoRename = document.getElementById('restoreAutoRename').checked;
        
        // Validate custom ID if provided
        if (customId) {
            if (!/^[a-zA-Z0-9_-]+$/.test(customId)) {
                UI.showToast('Character ID can only contain letters, numbers, underscores, and hyphens', 'error');
                return;
            }
        }
        
        // Show progress
        document.getElementById('restoreUploadSection').style.display = 'none';
        document.getElementById('restoreProgress').style.display = 'block';
        document.getElementById('confirmRestoreBtn').disabled = true;
        document.getElementById('restoreCancelBtn').disabled = true;
        
        try {
            // Restore character
            const result = await API.restoreCharacter(
                this.selectedFile,
                customId || null,
                autoRename,
                false // cleanup_orphans - initially false
            );
            
            // Show success
            document.getElementById('restoreProgress').style.display = 'none';
            document.getElementById('restoreSuccess').style.display = 'block';
            
            // Build summary
            const summary = this.buildRestoreSummary(result);
            document.getElementById('restoreSummary').innerHTML = summary;
            
            // Change cancel button to close
            const cancelBtn = document.getElementById('restoreCancelBtn');
            cancelBtn.textContent = 'Close';
            cancelBtn.disabled = false;
            
            // Hide confirm button
            document.getElementById('confirmRestoreBtn').style.display = 'none';
            
            UI.showToast(`Character restored successfully: ${result.character_id}`, 'success');
            
            // Reload character list to show the restored character
            if (window.App && typeof window.App.loadCharacters === 'function') {
                await window.App.loadCharacters();
                
                // Select the restored character
                const characterSelect = document.getElementById('characterSelect');
                if (characterSelect) {
                    characterSelect.value = result.character_id;
                    // Trigger change event to load the character
                    characterSelect.dispatchEvent(new Event('change'));
                }
            }
            
        } catch (error) {
            console.error('Restore failed:', error);
            
            // Check if error is about orphaned data
            if (error.message && error.message.includes('orphaned data')) {
                // Show confirmation dialog for cleanup
                const shouldCleanup = confirm(
                    '⚠️ Orphaned Data Detected\n\n' +
                    error.message + '\n\n' +
                    'Would you like to clean up the orphaned data and try again?\n\n' +
                    '(This will permanently delete the orphaned conversations, memories, and vectors)'
                );
                
                if (shouldCleanup) {
                    try {
                        // Retry with cleanup_orphans=true
                        document.getElementById('restoreProgressText').textContent = 'Cleaning up orphaned data and restoring...';
                        
                        const result = await API.restoreCharacter(
                            this.selectedFile,
                            customId || null,
                            autoRename,
                            true // cleanup_orphans - now true
                        );
                        
                        // Show success (same as above)
                        document.getElementById('restoreProgress').style.display = 'none';
                        document.getElementById('restoreSuccess').style.display = 'block';
                        document.getElementById('restoreSummary').innerHTML = this.buildRestoreSummary(result);
                        document.getElementById('restoreCancelBtn').textContent = 'Close';
                        document.getElementById('restoreCancelBtn').disabled = false;
                        document.getElementById('confirmRestoreBtn').style.display = 'none';
                        
                        UI.showToast(`Character restored successfully after cleanup: ${result.character_id}`, 'success');
                        
                        if (window.App && typeof window.App.loadCharacters === 'function') {
                            await window.App.loadCharacters();
                            const characterSelect = document.getElementById('characterSelect');
                            if (characterSelect) {
                                characterSelect.value = result.character_id;
                                characterSelect.dispatchEvent(new Event('change'));
                            }
                        }
                        return; // Exit successfully
                    } catch (cleanupError) {
                        console.error('Restore with cleanup failed:', cleanupError);
                        UI.showToast(`Restore failed even after cleanup: ${cleanupError.message}`, 'error');
                    }
                }
            } else {
                UI.showToast(`Restore failed: ${error.message}`, 'error');
            }
            
            // Reset UI
            document.getElementById('restoreProgress').style.display = 'none';
            document.getElementById('restoreUploadSection').style.display = 'block';
            document.getElementById('confirmRestoreBtn').disabled = false;
            document.getElementById('restoreCancelBtn').disabled = false;
        }
    },
    
    /**
     * Build restore summary HTML
     */
    buildRestoreSummary(result) {
        const counts = result.restored_counts;
        
        let html = `
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Restoration Summary</h6>
                    <table class="table table-sm">
                        <tbody>
                            <tr>
                                <td><strong>Character ID:</strong></td>
                                <td><code>${result.character_id}</code></td>
                            </tr>
        `;
        
        if (result.renamed) {
            html += `
                            <tr>
                                <td><strong>Original ID:</strong></td>
                                <td><code>${result.original_id}</code></td>
                            </tr>
                            <tr>
                                <td colspan="2" class="text-warning">
                                    <i class="bi bi-info-circle me-1"></i>
                                    Character was renamed during restore
                                </td>
                            </tr>
            `;
        }
        
        if (result.backup_date) {
            html += `
                            <tr>
                                <td><strong>Backup Date:</strong></td>
                                <td>${new Date(result.backup_date).toLocaleString()}</td>
                            </tr>
            `;
        }
        
        html += `
                            <tr><td colspan="2"><hr class="my-2"></td></tr>
                            <tr>
                                <td><strong>Conversations:</strong></td>
                                <td>${counts.conversations || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Threads:</strong></td>
                                <td>${counts.threads || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Messages:</strong></td>
                                <td>${counts.messages || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Memories:</strong></td>
                                <td>${counts.memories || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Vectors:</strong></td>
                                <td>${counts.vectors || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Images:</strong></td>
                                <td>${counts.images || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Videos:</strong></td>
                                <td>${counts.videos || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Audio:</strong></td>
                                <td>${counts.audio || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Voice Samples:</strong></td>
                                <td>${counts.voice_samples || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Media Files:</strong></td>
                                <td>${counts.media_files || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Workflow Files:</strong></td>
                                <td>${counts.workflow_files || 0}</td>
                            </tr>
                            <tr>
                                <td><strong>Workflow Records:</strong></td>
                                <td>${counts.workflow_records || 0}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        return html;
    },
    
    /**
     * Reset modal to initial state
     */
    resetModal() {
        this.selectedFile = null;
        
        // Reset UI sections
        document.getElementById('restoreUploadSection').style.display = 'block';
        document.getElementById('restoreProgress').style.display = 'none';
        document.getElementById('restoreSuccess').style.display = 'none';
        
        // Reset form
        document.getElementById('restoreCharacterName').value = '';
        document.getElementById('restoreAutoRename').checked = true;
        
        // Reset drop zone
        const dropZone = document.getElementById('restoreDropZone');
        dropZone.classList.remove('border-success');
        dropZone.querySelector('.card-title').textContent = 'Upload Backup File';
        dropZone.querySelector('.card-text').textContent = 'Drag & drop a .zip backup file here, or click to browse';
        
        // Reset file input
        document.getElementById('restoreFileInput').value = '';
        
        // Reset buttons
        document.getElementById('confirmRestoreBtn').disabled = false;
        document.getElementById('confirmRestoreBtn').style.display = 'block';
        document.getElementById('restoreCancelBtn').disabled = false;
        document.getElementById('restoreCancelBtn').textContent = 'Cancel';
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    CharacterRestore.init();
});
