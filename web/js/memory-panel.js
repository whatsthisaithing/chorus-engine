/**
 * Memory Panel
 * Handles memory display, filtering, and CRUD operations
 */

const IMMUTABLE_CHARACTERS_MEMORY = ['nova', 'alex'];

window.MemoryPanel = {
    currentCharacter: null,
    currentFilter: 'all',
    isSearchMode: false,
    searchResults: [],
    allMemories: [],
    offcanvas: null,
    currentMemoryType: 'explicit', // 'explicit' or 'core'
    
    /**
     * Initialize memory panel
     */
    init() {
        this.offcanvas = new bootstrap.Offcanvas(document.getElementById('memoryPanel'));
        this.setupEventListeners();
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Open panel button
        document.getElementById('memoryPanelBtn').addEventListener('click', () => {
            this.openPanel();
        });
        
        // Filter buttons
        document.querySelectorAll('input[name="memoryTypeFilter"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentFilter = e.target.value;
                this.renderMemories();
            });
        });
        
        // Search
        document.getElementById('memorySearchBtn').addEventListener('click', () => {
            this.performSearch();
        });
        
        document.getElementById('memorySearchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        
        document.getElementById('memoryClearSearchBtn').addEventListener('click', () => {
            this.clearSearch();
        });
        
        // Add memory buttons
        document.getElementById('addExplicitMemoryBtn').addEventListener('click', () => {
            this.showAddMemoryForm('explicit');
        });
        
        document.getElementById('addCoreMemoryBtn').addEventListener('click', () => {
            this.showAddMemoryForm('core');
        });
        
        // Memory form
        document.getElementById('memoryForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveMemory();
        });
        
        document.getElementById('cancelMemoryFormBtn').addEventListener('click', () => {
            this.hideAddMemoryForm();
        });
        
        document.getElementById('cancelMemoryFormBtn2').addEventListener('click', () => {
            this.hideAddMemoryForm();
        });
    },
    
    /**
     * Open memory panel for current character
     */
    async openPanel() {
        const characterId = window.App?.state?.currentCharacter;
        if (!characterId) {
            UI.showToast('Please select a character first', 'warning');
            return;
        }
        
        this.currentCharacter = characterId;
        this.updateCharacterInfo();
        this.offcanvas.show();
        
        // Show/hide core memory button based on character mutability
        const isImmutable = IMMUTABLE_CHARACTERS_MEMORY.includes(characterId);
        document.getElementById('addCoreMemoryBtn').style.display = isImmutable ? 'none' : 'inline-block';
        
        await this.loadMemories();
    },
    
    /**
     * Update character info display
     */
    updateCharacterInfo() {
        const character = window.App?.state?.characters?.find(c => c.id === this.currentCharacter);
        if (character) {
            document.getElementById('memoryCharacterName').textContent = character.name;
        }
    },
    
    /**
     * Load all memories for current character
     */
    async loadMemories() {
        if (!this.currentCharacter) return;
        
        try {
            this.showStatus('Loading memories...', 'info');
            const memories = await API.getCharacterMemories(this.currentCharacter);
            this.allMemories = memories;
            this.isSearchMode = false;
            this.renderMemories();
            this.hideStatus();
        } catch (error) {
            console.error('Failed to load memories:', error);
            this.showStatus('Failed to load memories', 'danger');
        }
    },
    
    /**
     * Render memories based on current filter
     */
    renderMemories() {
        const container = document.getElementById('memoryList');
        const memories = this.isSearchMode ? this.searchResults : this.allMemories;
        
        // Apply filter
        let filtered = memories;
        if (this.currentFilter !== 'all') {
            filtered = memories.filter(m => m.memory_type === this.currentFilter);
        }
        
        if (filtered.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-inbox" style="font-size: 2rem;"></i>
                    <p class="mt-2">${this.isSearchMode ? 'No matching memories found' : 'No memories found'}</p>
                </div>
            `;
            return;
        }
        
        // Sort by priority (higher first) then by created date (newer first)
        filtered.sort((a, b) => {
            const priorityDiff = (b.priority || 0) - (a.priority || 0);
            if (priorityDiff !== 0) return priorityDiff;
            return new Date(b.created_at) - new Date(a.created_at);
        });
        
        container.innerHTML = filtered.map(memory => this.renderMemoryCard(memory)).join('');
        
        // Add delete handlers
        container.querySelectorAll('[data-delete-memory]').forEach(btn => {
            btn.addEventListener('click', () => {
                const memoryId = btn.dataset.deleteMemory;
                const memoryType = btn.dataset.memoryType;
                this.deleteMemory(memoryId, memoryType);
            });
        });
    },
    
    /**
     * Render a single memory card
     */
    renderMemoryCard(memory) {
        const isImmutable = IMMUTABLE_CHARACTERS_MEMORY.includes(this.currentCharacter);
        const canDelete = !(memory.memory_type === 'core' && isImmutable);
        
        const badgeClass = {
            'core': 'bg-primary',
            'explicit': 'bg-success',
            'fact': 'bg-info',
            'project': 'bg-warning text-dark',
            'experience': 'bg-primary',
            'story': 'bg-secondary',
            'relationship': 'bg-success'
        }[memory.memory_type] || 'bg-secondary';
        
        // Show priority stars for core/explicit, but category badge for facts
        const priorityStars = (memory.memory_type !== 'fact' && memory.priority) 
            ? '⭐'.repeat(Math.min(memory.priority, 5)) 
            : '';
        
        const categoryBadge = (memory.memory_type === 'fact' && memory.category)
            ? `<span class="badge bg-secondary ms-1">${memory.category.replace(/_/g, ' ')}</span>`
            : '';
        
        const tagsHtml = memory.tags && memory.tags.length > 0
            ? `<small class="text-muted"><i class="bi bi-tags"></i> ${memory.tags.join(', ')}</small>`
            : '';
        
        const similarityBadge = memory.similarity !== undefined
            ? `<span class="badge bg-warning ms-2">${(memory.similarity * 100).toFixed(0)}% match</span>`
            : '';
        
        return `
            <div class="card mb-2">
                <div class="card-body py-2 px-3">
                    <div class="d-flex justify-content-between align-items-start mb-1">
                        <div>
                            <span class="badge ${badgeClass} me-1">${memory.memory_type}</span>
                            ${categoryBadge}
                            ${similarityBadge}
                            ${priorityStars ? `<span class="ms-1">${priorityStars}</span>` : ''}
                        </div>
                        ${canDelete ? `
                            <button class="btn btn-sm btn-outline-danger py-0 px-1" 
                                    data-delete-memory="${memory.id}"
                                    data-memory-type="${memory.memory_type}"
                                    title="Delete memory">
                                <i class="bi bi-trash"></i>
                            </button>
                        ` : ''}
                    </div>
                    <p class="mb-1">${this.escapeHtml(memory.content)}</p>
                    ${tagsHtml}
                    <small class="text-muted d-block">
                        <i class="bi bi-clock"></i> ${new Date(memory.created_at).toLocaleString()}
                    </small>
                </div>
            </div>
        `;
    },
    
    /**
     * Perform semantic search
     */
    async performSearch() {
        const query = document.getElementById('memorySearchInput').value.trim();
        if (!query) return;
        
        try {
            this.showStatus('Searching...', 'info');
            const results = await API.searchMemories({
                query: query,
                character_id: this.currentCharacter,
                limit: 20
            });
            
            // Transform results to include similarity
            this.searchResults = results.map(r => ({
                ...r.memory,
                similarity: r.similarity,
                rank_score: r.rank_score
            }));
            
            this.isSearchMode = true;
            document.getElementById('memoryClearSearchBtn').style.display = 'inline-block';
            this.renderMemories();
            this.hideStatus();
        } catch (error) {
            console.error('Search failed:', error);
            this.showStatus('Search failed', 'danger');
        }
    },
    
    /**
     * Clear search and show all memories
     */
    clearSearch() {
        document.getElementById('memorySearchInput').value = '';
        document.getElementById('memoryClearSearchBtn').style.display = 'none';
        this.isSearchMode = false;
        this.searchResults = [];
        this.renderMemories();
    },
    
    /**
     * Show add memory form
     */
    showAddMemoryForm(memoryType) {
        this.currentMemoryType = memoryType;
        document.getElementById('memoryFormTitle').textContent = 
            memoryType === 'core' ? 'Add Core Memory' : 'Add Explicit Memory';
        document.getElementById('addMemoryForm').style.display = 'block';
        document.getElementById('memoryContent').focus();
    },
    
    /**
     * Hide add memory form
     */
    hideAddMemoryForm() {
        document.getElementById('addMemoryForm').style.display = 'none';
        document.getElementById('memoryForm').reset();
    },
    
    /**
     * Save new memory
     */
    async saveMemory() {
        const content = document.getElementById('memoryContent').value.trim();
        if (!content) {
            this.showStatus('Please enter memory content', 'warning');
            return;
        }
        
        const tags = document.getElementById('memoryTags').value
            .split(',')
            .map(t => t.trim())
            .filter(t => t.length > 0);
        
        const priority = parseInt(document.getElementById('memoryPriority').value);
        
        try {
            this.showStatus('Saving memory...', 'info');
            
            if (this.currentMemoryType === 'core') {
                await API.createCoreMemory(this.currentCharacter, {
                    content: content,
                    tags: tags.length > 0 ? tags : null,
                    priority: priority
                });
            } else {
                // For explicit memories, we need a conversation context
                // If no conversation is active, show error
                if (!window.App?.state?.selectedConversationId) {
                    this.showStatus('Please open or create a conversation first', 'warning');
                    return;
                }
                
                await API.createMemory(window.App.state.selectedConversationId, {
                    content: content,
                    memory_type: 'explicit',
                    thread_id: window.App.state.selectedThreadId || null
                });
            }
            
            this.showStatus('Memory saved successfully', 'success');
            this.hideAddMemoryForm();
            await this.loadMemories();
            
            setTimeout(() => this.hideStatus(), 2000);
        } catch (error) {
            console.error('Failed to save memory:', error);
            this.showStatus(error.message || 'Failed to save memory', 'danger');
        }
    },
    
    /**
     * Delete a memory
     */
    async deleteMemory(memoryId, memoryType) {
        const isImmutable = IMMUTABLE_CHARACTERS_MEMORY.includes(this.currentCharacter);
        if (memoryType === 'core' && isImmutable) {
            this.showStatus('Cannot delete core memories for default characters', 'warning');
            return;
        }
        
        if (!confirm('Are you sure you want to delete this memory?')) {
            return;
        }
        
        try {
            this.showStatus('Deleting memory...', 'info');
            await API.deleteMemory(memoryId);
            this.showStatus('Memory deleted', 'success');
            await this.loadMemories();
            setTimeout(() => this.hideStatus(), 2000);
        } catch (error) {
            console.error('Failed to delete memory:', error);
            this.showStatus('Failed to delete memory', 'danger');
        }
    },
    
    /**
     * Show status message
     */
    showStatus(message, type) {
        const status = document.getElementById('memoryStatus');
        status.className = `alert alert-${type}`;
        status.textContent = message;
        status.classList.remove('d-none');
    },
    
    /**
     * Hide status message
     */
    hideStatus() {
        const status = document.getElementById('memoryStatus');
        status.classList.add('d-none');
    },
    
    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    MemoryPanel.init();
});
