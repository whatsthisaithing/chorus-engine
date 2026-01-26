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
        
        this.attachment = null; // Single attachment for Phase 1
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
        // Phase 1: Only support single image
        if (files.length > 1) {
            alert('Multiple images not supported yet. Only the first image will be used.');
        }
        
        const file = files[0];
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            alert('Please select an image file (JPG, PNG, WebP, or GIF).');
            return;
        }
        
        // Validate file size (10MB max from system.yaml config)
        const maxSizeMB = 10;
        const maxSizeBytes = maxSizeMB * 1024 * 1024;
        if (file.size > maxSizeBytes) {
            alert(`Image too large. Maximum size is ${maxSizeMB}MB.`);
            return;
        }
        
        // Set attachment and render preview
        this.setAttachment(file);
    }
    
    /**
     * Set attachment and show preview
     */
    setAttachment(file) {
        // Phase 1: Replace existing attachment
        this.attachment = file;
        this.renderPreview();
    }
    
    /**
     * Render attachment preview
     */
    renderPreview() {
        if (!this.attachment) {
            this.previewArea.style.display = 'none';
            this.previewScroll.innerHTML = '';
            return;
        }
        
        // Show preview area
        this.previewArea.style.display = 'block';
        
        // Create preview item
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewScroll.innerHTML = `
                <div class="attachment-preview-item">
                    <img src="${e.target.result}" alt="Preview" class="preview-thumbnail">
                    <span class="preview-filename" title="${this.attachment.name}">${this.attachment.name}</span>
                    <button type="button" class="btn-remove-preview" title="Remove" onclick="window.messageInput.removeAttachment()">&times;</button>
                </div>
            `;
        };
        reader.readAsDataURL(this.attachment);
    }
    
    /**
     * Remove attachment
     */
    removeAttachment() {
        this.attachment = null;
        this.renderPreview();
    }
    
    /**
     * Get current attachment (for app.js)
     */
    getAttachment() {
        return this.attachment;
    }
    
    /**
     * Clear attachment after send
     */
    clearAttachment() {
        this.attachment = null;
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
     * Upload attachment to server (Step 1 of two-step process)
     * Returns attachment_id or null if no attachment
     */
    async uploadAttachment() {
        if (!this.attachment) {
            return null;
        }
        
        try {
            // Show uploading state
            const originalBtnText = this.sendBtn.innerHTML;
            this.sendBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Uploading...';
            this.sendBtn.disabled = true;
            
            // Create FormData
            const formData = new FormData();
            formData.append('file', this.attachment);
            
            // Upload to server
            const response = await fetch('/api/attachments/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to upload image');
            }
            
            const data = await response.json();
            
            // Restore button
            this.sendBtn.innerHTML = originalBtnText;
            this.sendBtn.disabled = false;
            
            return data.attachment_id;
            
        } catch (error) {
            console.error('Image upload failed:', error);
            
            // Restore button
            this.sendBtn.innerHTML = '<i class="bi bi-send-fill"></i>';
            this.sendBtn.disabled = false;
            
            // Show error to user
            if (typeof UI !== 'undefined' && UI.showToast) {
                UI.showToast('Failed to upload image: ' + error.message, 'error');
            } else {
                alert('Failed to upload image: ' + error.message);
            }
            
            return null;
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
