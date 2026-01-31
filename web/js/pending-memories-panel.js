/**
 * Pending Memories Panel (Phase 4.1)
 * 
 * Manages the UI for reviewing and approving/rejecting pending implicit memories.
 */

class PendingMemoriesPanel {
    constructor() {
        this.memories = [];
        this.selectedCharacterId = null;
        this.selectedMemoryIds = new Set();
        
        this.initializeElements();
        this.attachEventListeners();
    }
    
    initializeElements() {
        this.panel = document.getElementById('pending-memories-panel');
        this.button = document.getElementById('pending-memories-btn');
        this.badge = document.getElementById('pending-count');
        this.list = document.getElementById('pending-memories-list');
        this.batchApproveBtn = document.getElementById('batch-approve-btn');
        this.batchRejectBtn = document.getElementById('batch-reject-btn');
        this.selectAllCheckbox = document.getElementById('select-all-pending');
    }
    
    attachEventListeners() {
        // Open panel
        if (this.button) {
            this.button.addEventListener('click', () => this.open());
        }
        
        // Initialize global reference
        window.pendingMemoriesPanel = this;
        
        // Initialize global reference
        window.pendingMemoriesPanel = this;
        
        // Batch actions
        if (this.batchApproveBtn) {
            this.batchApproveBtn.addEventListener('click', () => this.batchApprove());
        }
        
        if (this.batchRejectBtn) {
            this.batchRejectBtn.addEventListener('click', () => this.batchReject());
        }
        
        // Select all
        if (this.selectAllCheckbox) {
            this.selectAllCheckbox.addEventListener('change', (e) => {
                this.toggleSelectAll(e.target.checked);
            });
        }
    }
    
    async updateCount(characterId) {
        // Update badge with pending memory count
        try {
            const response = await fetch(`/characters/${characterId}/pending-memories`);
            if (response.ok) {
                const memories = await response.json();
                const count = memories.length;
                
                if (this.badge) {
                    this.badge.textContent = count;
                    this.badge.style.display = count > 0 ? '' : 'none';
                }
            }
        } catch (error) {
            console.error('Error updating pending count:', error);
        }
    }
    
    async open() {
        // Get current character from App state
        const characterId = window.App?.state?.currentCharacter;
        if (!characterId) {
            UI.showToast('Please select a character first', 'warning');
            return;
        }
        
        this.selectedCharacterId = characterId;
        await this.load();
        
        // Show panel
        const bsOffcanvas = new bootstrap.Offcanvas(this.panel);
        bsOffcanvas.show();
    }
    
    async load() {
        if (!this.selectedCharacterId) return;
        
        try {
            const response = await fetch(`/characters/${this.selectedCharacterId}/pending-memories`);
            if (!response.ok) throw new Error('Failed to load pending memories');
            
            this.memories = await response.json();
            this.selectedMemoryIds.clear();
            this.render();
            this.updateBadge();
            
        } catch (error) {
            console.error('Error loading pending memories:', error);
            UI.showToast('Failed to load pending memories', 'error');
        }
    }
    
    render() {
        if (!this.list) return;
        
        if (this.memories.length === 0) {
            this.list.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-check-circle" style="font-size: 3rem;"></i>
                    <p class="mt-2">No pending memories</p>
                    <small>All extracted memories have been reviewed</small>
                </div>
            `;
            this.updateBatchButtons();
            return;
        }
        
        this.list.innerHTML = this.memories.map(memory => this.renderMemoryCard(memory)).join('');
        this.updateBatchButtons();
    }
    
    renderMemoryCard(memory) {
        const isSelected = this.selectedMemoryIds.has(memory.id);
        const confidencePercent = Math.round(memory.confidence * 100);
        const confidenceClass = this.getConfidenceClass(memory.confidence);
        const categoryBadge = this.getCategoryBadge(memory.category);
        
        return `
            <div class="pending-memory-card ${isSelected ? 'selected' : ''}" data-memory-id="${memory.id}">
                <div class="form-check">
                    <input 
                        class="form-check-input memory-checkbox" 
                        type="checkbox" 
                        id="memory-${memory.id}"
                        ${isSelected ? 'checked' : ''}
                        onchange="pendingMemoriesPanel.toggleSelection('${memory.id}')"
                    >
                    <label class="form-check-label w-100" for="memory-${memory.id}">
                        <div class="memory-content">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div>
                                    ${categoryBadge}
                                    <span class="badge confidence-badge ${confidenceClass}">
                                        ${confidencePercent}% confident
                                    </span>
                                </div>
                                <small class="text-muted">
                                    ${new Date(memory.created_at).toLocaleString()}
                                </small>
                            </div>
                            
                            <p class="mb-2">${this.escapeHtml(memory.content)}</p>
                            
                            ${memory.metadata?.reasoning ? `
                                <small class="text-muted d-block mb-2">
                                    <i class="bi bi-lightbulb"></i> 
                                    ${this.escapeHtml(memory.metadata.reasoning)}
                                </small>
                            ` : ''}
                            
                            <div class="memory-actions">
                                <button 
                                    class="btn btn-sm btn-success"
                                    onclick="pendingMemoriesPanel.approve('${memory.id}')"
                                >
                                    <i class="bi bi-check-lg"></i> Approve
                                </button>
                                <button 
                                    class="btn btn-sm btn-danger"
                                    onclick="pendingMemoriesPanel.reject('${memory.id}')"
                                >
                                    <i class="bi bi-x-lg"></i> Reject
                                </button>
                            </div>
                        </div>
                    </label>
                </div>
            </div>
        `;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.9) return 'confidence-high';
        if (confidence >= 0.7) return 'confidence-medium';
        return 'confidence-low';
    }
    
    getCategoryBadge(category) {
        const categoryMap = {
            personal_info: { icon: 'person', label: 'Personal Info', class: 'bg-primary' },
            preference: { icon: 'heart', label: 'Preference', class: 'bg-success' },
            experience: { icon: 'calendar-event', label: 'Experience', class: 'bg-info' },
            relationship: { icon: 'people', label: 'Relationship', class: 'bg-warning' },
            goal: { icon: 'flag', label: 'Goal', class: 'bg-danger' },
            skill: { icon: 'tools', label: 'Skill', class: 'bg-secondary' }
        };
        
        const cat = categoryMap[category] || { icon: 'tag', label: 'Other', class: 'bg-dark' };
        
        return `
            <span class="badge ${cat.class} me-1">
                <i class="bi bi-${cat.icon}"></i> ${cat.label}
            </span>
        `;
    }
    
    toggleSelection(memoryId) {
        if (this.selectedMemoryIds.has(memoryId)) {
            this.selectedMemoryIds.delete(memoryId);
        } else {
            this.selectedMemoryIds.add(memoryId);
        }
        
        // Update card styling
        const card = document.querySelector(`[data-memory-id="${memoryId}"]`);
        if (card) {
            card.classList.toggle('selected', this.selectedMemoryIds.has(memoryId));
        }
        
        this.updateBatchButtons();
        this.updateSelectAllCheckbox();
    }
    
    toggleSelectAll(checked) {
        this.selectedMemoryIds.clear();
        
        if (checked) {
            this.memories.forEach(m => this.selectedMemoryIds.add(m.id));
        }
        
        // Update all checkboxes and cards
        document.querySelectorAll('.memory-checkbox').forEach(checkbox => {
            checkbox.checked = checked;
        });
        
        document.querySelectorAll('.pending-memory-card').forEach(card => {
            card.classList.toggle('selected', checked);
        });
        
        this.updateBatchButtons();
    }
    
    updateSelectAllCheckbox() {
        if (!this.selectAllCheckbox) return;
        
        const totalMemories = this.memories.length;
        const selectedCount = this.selectedMemoryIds.size;
        
        if (selectedCount === 0) {
            this.selectAllCheckbox.checked = false;
            this.selectAllCheckbox.indeterminate = false;
        } else if (selectedCount === totalMemories) {
            this.selectAllCheckbox.checked = true;
            this.selectAllCheckbox.indeterminate = false;
        } else {
            this.selectAllCheckbox.checked = false;
            this.selectAllCheckbox.indeterminate = true;
        }
    }
    
    updateBatchButtons() {
        const hasSelection = this.selectedMemoryIds.size > 0;
        
        if (this.batchApproveBtn) {
            this.batchApproveBtn.disabled = !hasSelection;
            this.batchApproveBtn.textContent = hasSelection 
                ? `Approve Selected (${this.selectedMemoryIds.size})`
                : 'Approve Selected';
        }
        
        if (this.batchRejectBtn) {
            this.batchRejectBtn.disabled = !hasSelection;
            this.batchRejectBtn.textContent = hasSelection
                ? `Reject Selected (${this.selectedMemoryIds.size})`
                : 'Reject Selected';
        }
    }
    
    async approve(memoryId) {
        try {
            const response = await fetch(`/memories/${memoryId}/approve`, {
                method: 'POST'
            });
            
            if (!response.ok) throw new Error('Failed to approve memory');
            
            UI.showToast('Memory approved', 'success');
            await this.load(); // Reload list
            this.updateCount(this.selectedCharacterId); // Update badge
            
        } catch (error) {
            console.error('Error approving memory:', error);
            UI.showToast('Failed to approve memory', 'error');
        }
    }
    
    async reject(memoryId) {
        try {
            const response = await fetch(`/memories/${memoryId}/reject`, {
                method: 'POST'
            });
            
            if (!response.ok) throw new Error('Failed to reject memory');
            
            UI.showToast('Memory rejected', 'success');
            await this.load(); // Reload list
            this.updateCount(this.selectedCharacterId); // Update badge
            
        } catch (error) {
            console.error('Error rejecting memory:', error);
            UI.showToast('Failed to reject memory', 'error');
        }
    }
    
    async batchApprove() {
        if (this.selectedMemoryIds.size === 0) return;
        
        try {
            const response = await fetch('/memories/batch-approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    memory_ids: Array.from(this.selectedMemoryIds)
                })
            });
            
            if (!response.ok) throw new Error('Failed to batch approve');
            
            const result = await response.json();
            UI.showToast(`Approved ${result.approved_count} memories`, 'success');
            await this.load();
            this.updateCount(this.selectedCharacterId); // Update badge
            
        } catch (error) {
            console.error('Error batch approving:', error);
            UI.showToast('Failed to batch approve', 'error');
        }
    }
    
    async batchReject() {
        if (this.selectedMemoryIds.size === 0) return;
        
        if (!confirm(`Are you sure you want to reject ${this.selectedMemoryIds.size} memories?`)) {
            return;
        }
        
        try {
            let successCount = 0;
            for (const memoryId of this.selectedMemoryIds) {
                const response = await fetch(`/memories/${memoryId}/reject`, {
                    method: 'POST'
                });
                if (response.ok) successCount++;
            }
            
            UI.showToast(`Rejected ${successCount} memories`, 'success');
            await this.load();
            this.updateCount(this.selectedCharacterId); // Update badge
            
        } catch (error) {
            console.error('Error batch rejecting:', error);
            UI.showToast('Failed to batch reject', 'error');
        }
    }
    
    updateBadge() {
        if (!this.badge) return;
        
        const count = this.memories.length;
        this.badge.textContent = count;
        this.badge.style.display = count > 0 ? 'inline' : 'none';
        
        // Remove sparkle if no pending memories
        if (count === 0 && this.button) {
            this.button.classList.remove('pending-sparkle');
        }
    }
    
    // Periodically check for new pending memories
    startPolling(intervalMs = 30000) { // Every 30 seconds
        this.stopPolling();
        this.pollInterval = setInterval(() => {
            if (this.selectedCharacterId) {
                this.updateBadge();
            }
        }, intervalMs);
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global instance
let pendingMemoriesPanel;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    pendingMemoriesPanel = new PendingMemoriesPanel();
    
    // Start polling for updates
    pendingMemoriesPanel.startPolling();
});

