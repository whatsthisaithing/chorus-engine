/**
 * Document Library Module
 * Handles document upload, listing, and management
 */

window.DocumentLibrary = (function() {
    const API_BASE = window.location.origin;
    let pendingFile = null;
    let currentCharacterId = null;

    // Initialize on DOM load
    document.addEventListener('DOMContentLoaded', function() {
        setupUploadZone();
        setupModalHandlers();
    });

    function setupUploadZone() {
        const zone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('docFileInput');
        
        if (!zone || !fileInput) return;

        // Click to upload (but not when clicking dropdown)
        zone.addEventListener('click', function(e) {
            // Don't trigger file input if clicking on the select dropdown
            if (e.target.id === 'chunkMethod' || e.target.closest('#chunkMethod')) {
                return;
            }
            fileInput.click();
        });

        // Drag and drop
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.querySelector('.card-body').style.borderColor = '#0d6efd';
            zone.querySelector('.card-body').style.background = 'rgba(13, 110, 253, 0.1)';
        });

        zone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.querySelector('.card-body').style.borderColor = '#444';
            zone.querySelector('.card-body').style.background = '';
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.querySelector('.card-body').style.borderColor = '#444';
            zone.querySelector('.card-body').style.background = '';
            
            const files = Array.from(e.dataTransfer.files);
            if (files.length > 0) {
                handleFiles(files);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            if (files.length > 0) {
                handleFiles(files);
            }
            // Reset input so same file can be selected again
            e.target.value = '';
        });
    }

    function setupModalHandlers() {
        const confirmBtn = document.getElementById('confirmDocUpload');
        if (confirmBtn) {
            confirmBtn.addEventListener('click', uploadDocument);
        }
    }

    function initializeScopeSelector() {
        const scopeSelect = document.getElementById('docScope');
        const scopeHelp = document.getElementById('docScopeHelp');
        if (!scopeSelect || !scopeHelp) return;

        // Check if we're in a conversation context
        const hasConversation = window.App && window.App.state && window.App.state.selectedConversationId;

        if (hasConversation) {
            // In conversation: default to conversation scope, show all options
            scopeSelect.value = 'conversation';
            scopeHelp.textContent = 'Document only accessible in this conversation (recommended for privacy)';
            
            // Enable all options
            Array.from(scopeSelect.options).forEach(opt => opt.disabled = false);
        } else {
            // Not in conversation: default to character scope, disable conversation option
            scopeSelect.value = 'character';
            scopeHelp.textContent = 'No active conversation - document will be available to character in all conversations';
            
            // Disable conversation scope option
            const conversationOption = scopeSelect.querySelector('option[value="conversation"]');
            if (conversationOption) {
                conversationOption.disabled = true;
                conversationOption.textContent = '🔒 Conversation Only (requires active conversation)';
            }
        }

        // Update help text when scope changes
        scopeSelect.addEventListener('change', function() {
            switch(this.value) {
                case 'conversation':
                    scopeHelp.textContent = 'Document only accessible in this conversation (recommended for privacy)';
                    scopeHelp.className = 'form-text text-light d-block mt-1';
                    break;
                case 'character':
                    scopeHelp.textContent = 'Document accessible in all conversations with this character';
                    scopeHelp.className = 'form-text text-warning d-block mt-1';
                    break;
                case 'global':
                    scopeHelp.textContent = 'Document accessible system-wide (use with caution)';
                    scopeHelp.className = 'form-text text-danger d-block mt-1';
                    break;
            }
        });
    }

    function handleFiles(files) {
        if (files.length === 1) {
            // Show upload details modal for single file
            pendingFile = files[0];
            const title = files[0].name.replace(/\.[^/.]+$/, "");
            document.getElementById('docTitle').value = title;
            document.getElementById('docDescription').value = '';
            
            // Initialize scope selector based on conversation context
            initializeScopeSelector();
            
            const uploadModal = new bootstrap.Modal(document.getElementById('docUploadModal'));
            uploadModal.show();
        } else {
            // Batch upload multiple files without details modal
            files.forEach(file => uploadFileDirectly(file));
        }
    }

    async function uploadDocument() {
        if (!pendingFile) return;

        const title = document.getElementById('docTitle').value || pendingFile.name.replace(/\.[^/.]+$/, "");
        const description = document.getElementById('docDescription').value;
        const chunkMethod = document.getElementById('chunkMethod').value;
        const documentScope = document.getElementById('docScope').value;

        // Get current character ID
        const characterSelect = document.getElementById('characterSelect');
        currentCharacterId = characterSelect ? characterSelect.value : null;

        // Get conversation ID if available
        const conversationId = window.App && window.App.state && window.App.state.selectedConversationId;

        const progressDiv = document.getElementById('docUploadProgress');
        const confirmBtn = document.getElementById('confirmDocUpload');
        
        progressDiv.style.display = 'block';
        confirmBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', pendingFile);
            formData.append('title', title);
            if (description) formData.append('description', description);
            formData.append('chunk_method', chunkMethod);
            formData.append('document_scope', documentScope);
            if (currentCharacterId) formData.append('character_id', currentCharacterId);
            if (conversationId) formData.append('conversation_id', conversationId);

            const response = await fetch(`${API_BASE}/documents/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            console.log('Upload successful:', result);

            // Close modals and refresh
            bootstrap.Modal.getInstance(document.getElementById('docUploadModal')).hide();
            await loadDocuments();
            await loadStats();

            // Show success message
            showNotification('Document uploaded successfully!', 'success');

        } catch (error) {
            console.error('Upload error:', error);
            showNotification(`Upload failed: ${error.message}`, 'danger');
        } finally {
            progressDiv.style.display = 'none';
            confirmBtn.disabled = false;
            pendingFile = null;
        }
    }

    async function uploadFileDirectly(file) {
        const chunkMethod = document.getElementById('chunkMethod').value;
        const characterSelect = document.getElementById('characterSelect');
        currentCharacterId = characterSelect ? characterSelect.value : null;

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('title', file.name.replace(/\.[^/.]+$/, ""));
            formData.append('chunk_method', chunkMethod);
            if (currentCharacterId) formData.append('character_id', currentCharacterId);

            const response = await fetch(`${API_BASE}/documents/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            await loadDocuments();
            await loadStats();
            showNotification(`${file.name} uploaded successfully!`, 'success');

        } catch (error) {
            console.error('Upload error:', error);
            showNotification(`Failed to upload ${file.name}`, 'danger');
        }
    }

    async function loadStats() {
        try {
            const response = await fetch(`${API_BASE}/documents/stats`);
            if (!response.ok) throw new Error('Failed to load stats');
            
            const stats = await response.json();
            
            document.getElementById('statTotalDocs').textContent = stats.total_documents || 0;
            document.getElementById('statTotalChunks').textContent = stats.total_chunks || 0;
            document.getElementById('statStorageSize').textContent = formatBytes(stats.total_storage_bytes || 0);
            document.getElementById('statAvgChunks').textContent = stats.avg_chunks_per_document 
                ? stats.avg_chunks_per_document.toFixed(1) 
                : '0';

        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    async function loadDocuments() {
        const container = document.getElementById('documentsList');
        const tableContainer = document.getElementById('documentsTableContainer');
        const loading = document.getElementById('docLoadingSpinner');
        const empty = document.getElementById('documentsEmpty');

        loading.style.display = 'block';
        tableContainer.style.display = 'none';
        empty.style.display = 'none';

        try {
            // Get current character ID for filtering
            const characterSelect = document.getElementById('characterSelect');
            currentCharacterId = characterSelect ? characterSelect.value : null;

            // Get conversation ID if available (for scope-aware filtering)
            const conversationId = window.App && window.App.state && window.App.state.selectedConversationId;

            let url = `${API_BASE}/documents?limit=100`;
            if (currentCharacterId) {
                url += `&character_id=${currentCharacterId}`;
            }
            if (conversationId) {
                url += `&conversation_id=${conversationId}`;
            }

            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to load documents');
            
            const documents = await response.json();

            if (documents.length === 0) {
                loading.style.display = 'none';
                empty.style.display = 'block';
                return;
            }

            container.innerHTML = documents.map(doc => createDocumentRow(doc)).join('');
            tableContainer.style.display = 'block';
            loading.style.display = 'none';

            // Attach delete handlers
            documents.forEach(doc => {
                const deleteBtn = document.getElementById(`delete-${doc.id}`);
                if (deleteBtn) {
                    deleteBtn.addEventListener('click', () => deleteDocument(doc.id));
                }
            });

        } catch (error) {
            console.error('Failed to load documents:', error);
            loading.style.display = 'none';
            container.innerHTML = `<tr><td colspan="8" class="text-center text-danger">Failed to load documents: ${error.message}</td></tr>`;
            tableContainer.style.display = 'block';
        }
    }

    function createDocumentRow(doc) {
        // Scope badge configuration
        const scope = doc.document_scope || 'character';
        let scopeClass, scopeIcon, scopeLabel;
        
        if (scope === 'conversation') {
            scopeClass = 'primary';
            scopeIcon = 'lock';
            scopeLabel = 'Conversation';
        } else if (scope === 'character') {
            scopeClass = 'warning';
            scopeIcon = 'person';
            scopeLabel = 'Character';
        } else {
            scopeClass = 'info';
            scopeIcon = 'globe';
            scopeLabel = 'Global';
        }
        
        const date = new Date(doc.uploaded_at).toLocaleDateString();
        const reference = `#doc:${doc.filename}`;

        return `
            <tr>
                <td class="text-light" title="${doc.filename}">
                    <i class="bi bi-file-earmark-text text-primary"></i> ${doc.title}
                </td>
                <td class="text-light">${doc.file_type.toUpperCase()}</td>
                <td class="text-light">${formatBytes(doc.file_size_bytes)}</td>
                <td class="text-center text-light">${doc.chunk_count || 0}</td>
                <td>
                    <span class="badge bg-${scopeClass}">
                        <i class="bi bi-${scopeIcon}"></i> ${scopeLabel}
                    </span>
                </td>
                <td class="text-white-50" style="font-size: 0.9em;">${date}</td>
                <td>
                    <code class="text-info" style="font-size: 0.85em;">${reference}</code>
                </td>
                <td>
                    <button class="btn btn-sm btn-danger" id="delete-${doc.id}" title="Delete document">
                        <i class="bi bi-trash"></i>
                    </button>
                </td>
            </tr>
        `;
    }

    async function deleteDocument(docId) {
        if (!confirm('Are you sure you want to delete this document? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/documents/${docId}`, {
                method: 'DELETE'
            });

            if (!response.ok) throw new Error('Delete failed');

            await loadDocuments();
            await loadStats();
            showNotification('Document deleted successfully', 'success');

        } catch (error) {
            console.error('Delete error:', error);
            showNotification('Failed to delete document', 'danger');
        }
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function showNotification(message, type = 'info') {
        // Use Bootstrap toast or alert - for now, just console
        console.log(`[${type}] ${message}`);
        // Could integrate with existing notification system if available
    }

    // Public API
    return {
        loadStats,
        loadDocuments,
        deleteDocument
    };
})();
