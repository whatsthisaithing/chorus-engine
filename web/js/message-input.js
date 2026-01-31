/**
 * Enhanced Message Input Handler
 * Manages image upload, drag-and-drop, auto-resizing textarea, and attachment preview
 * Task 1.8: Web UI Enhanced Message Input
 */

window.MessageInput = class {
    constructor() {
        this.textarea = document.getElementById('messageInput');
        this.fileInput = document.getElementById('imageFileInput');
        this.attachBtn = document.getElementById('attachImageBtn');
        this.previewArea = document.getElementById('attachmentPreview');
        this.previewScroll = this.previewArea.querySelector('.attachment-preview-scroll');
        this.dropZone = document.getElementById('dropZoneOverlay');
        this.chatInput = document.querySelector('.chat-input');
        this.sendBtn = document.getElementById('sendBtn');
        
        this.attachments = []; // Multiple attachments support (Phase 3.1)
        this.dragCounter = 0; // Track nested drag events
        
        this.init();
    }
    
    init() {
        // Auto-expanding textarea
        this.textarea.addEventListener('input', () => this.autoResize());
        this.textarea.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Attach button
        this.attachBtn.addEventListener('click', () => this.openFilePicker());
        
        // File input
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop - entire page
        document.addEventListener('dragenter', (e) => this.handleDragEnter(e));
        document.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        document.addEventListener('dragover', (e) => this.handleDragOver(e));
        document.addEventListener('drop', (e) => this.handleDrop(e));
        
        console.log('Enhanced Message Input initialized');
    }
    
    /**
     * Auto-resize textarea based on content
     */
    autoResize() {
        // Reset height to recalculate
        this.textarea.style.height = 'auto';
        
        // Calculate new height (max 200px from CSS)
        const scrollHeight = this.textarea.scrollHeight;
        const newHeight = Math.min(scrollHeight, 200);
        
        this.textarea.style.height = newHeight + 'px';
    }
    
    /**
     * Handle keyboard shortcuts
     */
    handleKeyDown(e) {
        // Enter without Shift = send (handled by form submit)
        // Shift+Enter = newline (default behavior)
        
        // Note: The main send is handled by app.js messageForm submit event
        // This just ensures proper behavior with multiline
    }
    
    /**
     * Open file picker
     */
    openFilePicker() {
        this.fileInput.click();
    }
    
    /**
     * Handle file selection from file input
     */
    handleFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            this.handleFiles(Array.from(files));
        }
        // Reset file input
        this.fileInput.value = '';
    }
    
    /**
     * Process files (from picker or drag-and-drop)
     */
    handleFiles(files) {
        // Phase 3.1: Support multiple images
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
        const maxSizeMB = 10;
        const maxSizeBytes = maxSizeMB * 1024 * 1024;
        
        // Validate and add each file
        for (const file of files) {
            // Validate file type
            if (!validTypes.includes(file.type)) {
                alert(`${file.name}: Please select image files only (JPG, PNG, WebP, or GIF).`);
                continue;
            }
            
            // Validate file size
            if (file.size > maxSizeBytes) {
                alert(`${file.name}: Image too large. Maximum size is ${maxSizeMB}MB.`);
                continue;
            }
            
            // Add to attachments
            this.attachments.push(file);
        }
        
        this.renderPreview();
        return;
    }
    
    // Keep old single-file validation for backward compatibility (not used)
    handleFilesSingleOld(files) {
        const file = files[0];
        
        // Old validation moved to handleFiles - this path shouldn't be reached
    }
    
    /**
     * Add attachment (kept for potential future single-add use)
     */
    addAttachment(file) {
        this.attachments.push(file);
        this.renderPreview();
    }
    
    /**
     * Render attachment preview (supports multiple attachments)
     */
    renderPreview() {
        if (!this.attachments || this.attachments.length === 0) {
            this.previewArea.style.display = 'none';
            this.previewScroll.innerHTML = '';
            return;
        }
        
        // Show preview area
        this.previewArea.style.display = 'block';
        
        // Clear existing previews
        this.previewScroll.innerHTML = '';
        
        // Create preview for each attachment
        this.attachments.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const previewItem = document.createElement('div');
                previewItem.className = 'attachment-preview-item';
                previewItem.innerHTML = `
                    <img src="${e.target.result}" alt="Preview" class="preview-thumbnail">
                    <span class="preview-filename" title="${file.name}">${file.name}</span>
                    <button type="button" class="btn-remove-preview" title="Remove" data-index="${index}">&times;</button>
                `;
                
                // Add click handler for remove button
                const removeBtn = previewItem.querySelector('.btn-remove-preview');
                removeBtn.addEventListener('click', () => this.removeAttachment(index));
                
                this.previewScroll.appendChild(previewItem);
            };
            reader.readAsDataURL(file);
        });
    }
    
    /**
     * Remove attachment by index
     */
    removeAttachment(index) {
        this.attachments.splice(index, 1);
        this.renderPreview();
    }
    
    /**
     * Get current attachments (for app.js)
     * Returns array of files or empty array
     */
    getAttachment() {
        return this.attachments;
    }
    
    /**
     * Check if has attachments
     */
    hasAttachments() {
        return this.attachments && this.attachments.length > 0;
    }
    
    /**
     * Clear attachments after send
     */
    clearAttachment() {
        this.attachments = [];
        this.renderPreview();
    }
    
    /**
     * Drag enter handler
     */
    handleDragEnter(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Check if dragging files
        if (!this.hasDraggedFiles(e)) {
            return;
        }
        
        this.dragCounter++;
        if (this.dragCounter === 1) {
            this.dropZone.classList.add('active');
        }
    }
    
    /**
     * Drag leave handler
     */
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        
        this.dragCounter--;
        if (this.dragCounter === 0) {
            this.dropZone.classList.remove('active', 'drag-over');
        }
    }
    
    /**
     * Drag over handler
     */
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        
        if (this.hasDraggedFiles(e)) {
            e.dataTransfer.dropEffect = 'copy';
            this.dropZone.classList.add('drag-over');
        }
    }
    
    /**
     * Drop handler
     */
    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        
        this.dragCounter = 0;
        this.dropZone.classList.remove('active', 'drag-over');
        
        if (!this.hasDraggedFiles(e)) {
            return;
        }
        
        const files = Array.from(e.dataTransfer.files);
        this.handleFiles(files);
    }
    
    /**
     * Check if drag event has files
     */
    hasDraggedFiles(e) {
        if (!e.dataTransfer || !e.dataTransfer.types) {
            return false;
        }
        return e.dataTransfer.types.includes('Files');
    }
    
    /**
     * Upload all attachments to server (Step 1 of two-step process)
     * Returns array of attachment_ids or empty array if no attachments
     */
    async uploadAttachment() {
        if (!this.attachments || this.attachments.length === 0) {
            return [];
        }
        
        try {
            // Show uploading state
            const originalBtnText = this.sendBtn.innerHTML;
            const count = this.attachments.length;
            this.sendBtn.innerHTML = `<i class="bi bi-hourglass-split"></i> Uploading ${count} image${count > 1 ? 's' : ''}...`;
            this.sendBtn.disabled = true;
            
            const attachmentIds = [];
            
            // Upload each attachment
            for (let i = 0; i < this.attachments.length; i++) {
                const file = this.attachments[i];
                
                // Update progress
                if (this.attachments.length > 1) {
                    this.sendBtn.innerHTML = `<i class="bi bi-hourglass-split"></i> Uploading ${i + 1}/${count}...`;
                }
                
                // Create FormData
                const formData = new FormData();
                formData.append('file', file);
                
                // Upload to server
                const response = await fetch('/api/attachments/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(`Failed to upload ${file.name}: ${error.detail || 'Unknown error'}`);
                }
                
                const data = await response.json();
                attachmentIds.push(data.attachment_id);
            }
            
            // Restore button
            this.sendBtn.innerHTML = originalBtnText;
            this.sendBtn.disabled = false;
            
            return attachmentIds;
            
        } catch (error) {
            console.error('Image upload failed:', error);
            
            // Restore button
            this.sendBtn.innerHTML = '<i class="bi bi-send-fill"></i>';
            this.sendBtn.disabled = false;
            
            // Show error to user
            if (typeof UI !== 'undefined' && UI.showToast) {
                UI.showToast('Failed to upload images: ' + error.message, 'error');
            } else {
                alert('Failed to upload images: ' + error.message);
            }
            
            return [];
        }
    }
    
    /**
     * Reset textarea height
     */
    resetHeight() {
        this.textarea.style.height = 'auto';
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.messageInput = new MessageInput();
});
