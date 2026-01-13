/**
 * Main Application Logic
 * Manages state and coordinates between API and UI
 */

window.App = {
    // Application state
    state: {
        characters: [],
        selectedCharacterId: null,
        conversations: [],
        selectedConversationId: null,
        threads: [],
        selectedThreadId: null,
        messages: [],
        lastMemoryCount: {}, // Track memory counts per character for sparkle effect
        memoryPollTimer: null, // Timer for memory polling
        ttsEnabled: false, // Phase 6: TTS status for current conversation
        galleryImages: [], // Phase 9: Image gallery
        galleryVideos: [] // Video gallery
    },
    
    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing Chorus Engine...');
        
        try {
            // Initialize theme system
            await ThemeManager.initialize();
            
            // Check backend health
            const health = await API.getHealth();
            console.log('Backend health:', health);
            
            if (!health.llm_available) {
                UI.showToast('Warning: LLM engine is not available. Check your LLM server configuration.', 'warning');
            }
            
            // Check model status (Phase 10)
            await this.checkModelStatus();
            
            // Load characters
            await this.loadCharacters();
            
            // Setup event listeners
            this.setupEventListeners();
            
            console.log('Chorus Engine initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize:', error);
            UI.showToast('Failed to connect to backend. Make sure the server is running.', 'error');
        }
        
        // Initialize character management
        if (window.CharacterManagement) {
            CharacterManagement.init();
        }
    },
    
    /**
     * Load all characters
     */
    async loadCharacters() {
        const data = await API.listCharacters();
        this.state.characters = data.characters;
        UI.renderCharacters(this.state.characters);
    },
    
    /**
     * Check LLM model status and show warning if needed (Phase 10)
     */
    async checkModelStatus() {
        try {
            const response = await fetch('/system/config');
            const config = await response.json();
            
            // Update Model Manager visibility based on provider
            if (typeof modelManager !== 'undefined') {
                modelManager.updateMenuVisibility(config.llm.provider || 'ollama');
            }
            
            const banner = document.getElementById('modelMissingBanner');
            const messageInput = document.getElementById('messageInput');
            
            // Show warning if no model configured in system.yaml
            if (!config.llm.model || config.llm.model.trim() === '') {
                if (banner) {
                    banner.style.display = 'block';
                }
                // Disable chat input
                if (messageInput) {
                    messageInput.disabled = true;
                    messageInput.placeholder = 'Download a model to start chatting...';
                }
                
                console.warn('No model configured in system.yaml');
            } else {
                // Hide banner and enable input
                if (banner) {
                    banner.style.display = 'none';
                }
                if (messageInput && !this.state.currentConversation) {
                    // Re-enable if we have a conversation
                    messageInput.disabled = false;
                    messageInput.placeholder = 'Type your message...';
                }
            }
        } catch (error) {
            console.error('Failed to check model status:', error);
            // Don't show error to user - this is a non-critical check
        }
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Character selection
        document.getElementById('characterSelect').addEventListener('change', (e) => {
            this.selectCharacter(e.target.value);
        });
        
        // View Full Profile button
        document.getElementById('viewFullProfileBtn').addEventListener('click', () => {
            this.showCharacterProfileModal();
        });
        
        // Phase 6: Voice sample management button
        document.getElementById('manageVoiceSamplesBtn').addEventListener('click', () => {
            this.showVoiceManagementModal();
        });
        
        // Phase 6: Voice upload form
        document.getElementById('voiceUploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadVoiceSample();
        });
        
        // New conversation button
        document.getElementById('newConversationBtn').addEventListener('click', () => {
            this.createNewConversation();
        });
        
        // Message form
        document.getElementById('messageForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Export confirmation button
        document.getElementById('confirmExportBtn').addEventListener('click', () => {
            this.exportConversation();
        });
        
        // Save title button (for modal)
        document.getElementById('saveTitleBtn').addEventListener('click', () => {
            this.saveTitle();
        });
        
        // Actions menu items (dropdown)
        document.getElementById('analyzeMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.analyzeConversation();
        });
        
        document.getElementById('analysisHistoryMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showAnalysisHistory();
        });
        
        document.getElementById('exportMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showExportModal();
        });
        
        document.getElementById('editTitleMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showEditTitleModal();
        });
        
        document.getElementById('deleteMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.deleteCurrentConversation();
        });
        
        // Inline title editing
        const titleElement = document.getElementById('conversationTitle');
        titleElement.addEventListener('dblclick', () => {
            if (this.currentConversationId && !titleElement.getAttribute('contenteditable') || titleElement.getAttribute('contenteditable') === 'false') {
                titleElement.setAttribute('contenteditable', 'true');
                titleElement.focus();
                // Select all text
                const range = document.createRange();
                range.selectNodeContents(titleElement);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            }
        });
        
        titleElement.addEventListener('blur', () => {
            if (titleElement.getAttribute('contenteditable') === 'true') {
                titleElement.setAttribute('contenteditable', 'false');
                const newTitle = titleElement.textContent.trim();
                if (newTitle && newTitle !== this.currentConversation?.title) {
                    this.saveInlineTitle(newTitle);
                } else {
                    // Restore original title if empty or unchanged
                    titleElement.textContent = this.currentConversation?.title || 'Select a conversation';
                }
            }
        });
        
        titleElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                titleElement.blur();
            } else if (e.key === 'Escape') {
                titleElement.textContent = this.currentConversation?.title || 'Select a conversation';
                titleElement.blur();
            }
        });
        
        // Confirm delete conversation button
        document.getElementById('confirmDeleteConversation').addEventListener('click', () => {
            this.confirmDeleteConversation();
        });
        
        // Reset database button (now in settings menu)
        document.getElementById('resetDatabaseMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showResetDatabaseModal();
        });
        
        // Export/Import menu items
        document.getElementById('exportCharacterMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.exportCharacter();
        });
        
        document.getElementById('importCharacterMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showImportCharacterModal();
        });
        
        document.getElementById('exportSystemConfigMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.exportSystemConfig();
        });
        
        document.getElementById('importSystemConfigMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showImportSystemConfigModal();
        });
        
        // Log viewer menu items
        document.getElementById('viewServerLogsMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showServerLogsModal();
        });
        
        // Per-conversation log viewers
        document.getElementById('viewDebugLogMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showConversationDebugLog();
        });
        
        document.getElementById('viewExtractionLogMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showConversationExtractionLog();
        });
        
        document.getElementById('viewImageRequestLogMenuItem').addEventListener('click', (e) => {
            e.preventDefault();
            this.showConversationImageRequestLog();
        });
        
        // Import character button
        document.getElementById('importCharacterBtn').addEventListener('click', () => {
            this.importCharacter();
        });
        
        // Import system config button
        document.getElementById('importSystemConfigBtn').addEventListener('click', () => {
            this.importSystemConfig();
        });
        
        // Log refresh buttons
        document.getElementById('refreshServerLogsBtn').addEventListener('click', () => {
            this.loadServerLogs();
        });
        
        document.getElementById('refreshConversationLogsBtn').addEventListener('click', () => {
            if (this.state.selectedConversationId) {
                this.loadConversationLog(this.state.selectedConversationId);
            }
        });
        
        document.getElementById('refreshExtractionLogsBtn').addEventListener('click', () => {
            if (this.state.selectedConversationId) {
                this.loadExtractionLog(this.state.selectedConversationId);
            }
        });
        
        document.getElementById('refreshImageRequestLogsBtn').addEventListener('click', () => {
            if (this.state.selectedConversationId) {
                this.loadImageRequestLog(this.state.selectedConversationId);
            }
        });
        
        // Log download buttons
        document.getElementById('downloadServerLogsBtn').addEventListener('click', () => {
            this.downloadServerLogs();
        });
        
        document.getElementById('downloadConversationLogBtn').addEventListener('click', () => {
            if (this.state.selectedConversationId) {
                this.downloadConversationLog(this.state.selectedConversationId);
            }
        });
        
        document.getElementById('downloadExtractionLogsBtn').addEventListener('click', () => {
            if (this.state.selectedConversationId) {
                this.downloadExtractionLog(this.state.selectedConversationId);
            }
        });
        
        document.getElementById('downloadImageRequestLogsBtn').addEventListener('click', () => {
            if (this.state.selectedConversationId) {
                this.downloadImageRequestLog(this.state.selectedConversationId);
            }
        });
        
        // Conversation log selection (removed)
        
        // Privacy toggle
        document.getElementById('privacyToggle').addEventListener('change', (e) => {
            this.setConversationPrivacy(e.target.checked);
        });
        
        // Phase 6: TTS toggle
        document.getElementById('ttsToggle').addEventListener('change', (e) => {
            this.setConversationTTS(e.target.checked);
        });
        
        // Reset confirmation input
        document.getElementById('resetConfirmInput').addEventListener('input', (e) => {
            const confirmBtn = document.getElementById('confirmResetBtn');
            confirmBtn.disabled = e.target.value !== 'RESET';
        });
        
        // Confirm reset button
        document.getElementById('confirmResetBtn').addEventListener('click', () => {
            this.resetDatabase();
        });
        
        // Phase 5: Image generation confirmation
        document.getElementById('confirmGenerateImageBtn').addEventListener('click', () => {
            this.confirmImageGeneration();
        });
        
        // Phase 9: Scene capture button
        document.getElementById('sceneCaptureBtn').addEventListener('click', () => {
            this.captureScene();
        });
        
        document.getElementById('videoSceneCaptureBtn').addEventListener('click', () => {
            this.captureVideoScene();
        });
        
        // Enter key to send (without shift)
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    },
    
    /**
     * Select a character
     */
    async selectCharacter(characterId) {
        if (!characterId) return;
        
        this.state.selectedCharacterId = characterId;
        this.state.currentCharacter = characterId;
        
        // Apply character's theme (or system default)
        await ThemeManager.applyCharacterTheme(characterId);
        
        // IMPORTANT: Reset conversation state when switching characters
        // This prevents documents from being uploaded to wrong conversation
        this.state.selectedConversationId = null;
        this.state.selectedThreadId = null;
        this.state.currentThread = null;
        this.state.messages = [];
        this.state.threads = [];
        
        // Clear all conversation UI components
        UI.renderMessages([]);  // Clear messages properly
        UI.renderThreads([], null);  // Clear thread tabs
        UI.updateHeader('No conversation selected', '');  // Clear header
        
        // Enable memory and workflow buttons
        document.getElementById('memoryPanelBtn').disabled = false;
        document.getElementById('manageWorkflowsBtn').disabled = false;
        document.getElementById('manageVoiceSamplesBtn').disabled = false; // Phase 6
        
        // Update character profile card
        const character = this.state.characters.find(c => c.id === characterId);
        if (character) {
            await UI.updateProfileCard(character);
            
            // Phase 1: Show/hide document library button based on character capability
            const docLibBtn = document.getElementById('documentLibraryBtn');
            if (docLibBtn) {
                if (character.document_analysis?.enabled) {
                    docLibBtn.style.display = '';
                    docLibBtn.disabled = false;
                } else {
                    docLibBtn.style.display = 'none';
                }
            }
        }
        
        // Initialize memory count for this character (if not already tracked)
        if (!(characterId in this.state.lastMemoryCount)) {
            try {
                const stats = await API.getCharacterMemoryStats(characterId);
                this.state.lastMemoryCount[characterId] = stats.implicit_memory_count;
            } catch (error) {
                console.error('Failed to initialize memory count:', error);
                this.state.lastMemoryCount[characterId] = 0;
            }
        }
        
        // Check and show immersion notice if needed
        await this.checkImmersionNotice(characterId);
        
        // Load conversations for this character
        await this.loadConversations();
        
        // Phase 9: Update scene capture button visibility
        this.updateSceneCaptureButton();
    },
    
    /**
     * Check if character needs immersion notice and show it
     */
    async checkImmersionNotice(characterId) {
        try {
            // Check localStorage to see if user has already seen notice for this character
            const seenKey = `immersion_notice_seen_${characterId}`;
            const hasSeenNotice = localStorage.getItem(seenKey) === 'true';
            
            if (hasSeenNotice) {
                return; // User already dismissed this notice
            }
            
            // Fetch immersion notice from API
            const noticeData = await API.getCharacterImmersionNotice(characterId);
            
            if (noticeData.should_show_notice && noticeData.notice_text) {
                this.showImmersionNotice(characterId, noticeData.notice_text);
            }
        } catch (error) {
            console.error('Failed to check immersion notice:', error);
            // Don't show error to user - this is non-critical
        }
    },
    
    /**
     * Show immersion notice modal
     */
    showImmersionNotice(characterId, noticeText) {
        // Parse markdown in notice text
        const noticeHtml = marked.parse(noticeText);
        
        // Populate modal content
        const modalBody = document.getElementById('immersionNoticeText');
        modalBody.innerHTML = noticeHtml;
        
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('immersionNoticeModal'));
        modal.show();
        
        // Handle "Don't show again" checkbox when modal closes
        const modalElement = document.getElementById('immersionNoticeModal');
        const handleModalHide = () => {
            const dontShowAgain = document.getElementById('dontShowAgainCheck').checked;
            if (dontShowAgain) {
                const seenKey = `immersion_notice_seen_${characterId}`;
                localStorage.setItem(seenKey, 'true');
            }
            // Reset checkbox for next time
            document.getElementById('dontShowAgainCheck').checked = false;
            // Remove listener
            modalElement.removeEventListener('hidden.bs.modal', handleModalHide);
        };
        
        modalElement.addEventListener('hidden.bs.modal', handleModalHide);
    },
    
    /**
     * Load conversations for selected character
     */
    async loadConversations() {
        if (!this.state.selectedCharacterId) return;
        
        try {
            const conversations = await API.listConversations(this.state.selectedCharacterId);
            this.state.conversations = conversations;
            UI.renderConversations(conversations, this.state.selectedConversationId);
        } catch (error) {
            console.error('Failed to load conversations:', error);
            UI.showToast('Failed to load conversations', 'error');
        }
    },
    
    /**
     * Create a new conversation
     */
    async createNewConversation() {
        if (!this.state.selectedCharacterId) {
            UI.showToast('Please select a character first', 'error');
            return;
        }
        
        try {
            const character = this.state.characters.find(c => c.id === this.state.selectedCharacterId);
            const title = `Chat with ${character.name}`;
            
            const conversation = await API.createConversation(this.state.selectedCharacterId, title);
            
            // Reload conversations
            await this.loadConversations();
            
            // Select the new conversation
            await this.selectConversation(conversation.id);
            
            UI.showToast('New conversation created', 'success');
        } catch (error) {
            console.error('Failed to create conversation:', error);
            UI.showToast('Failed to create conversation', 'error');
        }
    },
    
    /**
     * Select a conversation
     */
    async selectConversation(conversationId) {
        this.state.selectedConversationId = conversationId;
        
        try {
            // Load threads for this conversation
            const threads = await API.listThreads(conversationId);
            this.state.threads = threads;
            
            // Select first thread by default
            if (threads.length > 0) {
                await this.selectThread(threads[0].id);
            } else {
                // No threads, clear messages
                UI.renderMessages([]);
            }
            
            // Update conversation list highlighting
            UI.renderConversations(this.state.conversations, conversationId);
            
            // Phase 6: Load TTS status for this conversation
            await this.loadConversationTTS();
            
            // Phase 9: Load image gallery for this conversation
            await this.loadImageGallery();
            
            // Enable input
            UI.setInputEnabled(true);
            
            // Phase 9: Update scene capture button visibility
            this.updateSceneCaptureButton();
            
        } catch (error) {
            console.error('Failed to select conversation:', error);
            UI.showToast('Failed to load conversation', 'error');
        }
    },
    
    /**
     * Select a thread
     */
    async selectThread(threadId) {
        this.state.selectedThreadId = threadId;
        
        try {
            // Load thread details
            const thread = await API.getThread(threadId);
            
            // Load messages
            const messages = await API.listMessages(threadId);
            this.state.messages = messages;
            
            // Update UI
            const conversation = await API.getConversation(this.state.selectedConversationId);
            const character = this.state.characters.find(c => c.id === conversation.character_id);
            
            UI.updateHeader(conversation.title, character ? character.name : '');
            UI.renderThreads(this.state.threads, threadId);
            UI.renderMessages(messages);
            
            // Load privacy status for conversation
            await this.loadConversationPrivacy();
            
            // Scroll to bottom after loading messages
            setTimeout(() => UI.scrollToBottom(), 100);
            
        } catch (error) {
            console.error('Failed to select thread:', error);
            UI.showToast('Failed to load thread', 'error');
        }
    },
    
    /**
     * Send a message
     */
    async sendMessage() {
        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        if (!this.state.selectedThreadId) {
            UI.showToast('Please select a conversation first', 'error');
            return;
        }
        
        // Clear input
        input.value = '';
        
        // Disable input temporarily
        input.disabled = true;
        document.getElementById('sendBtn').disabled = true;
        
        try {
            // Add user message immediately
            const userMsg = {
                role: 'user',
                content: message,
                created_at: new Date().toISOString()
            };
            UI.appendMessage(userMsg);
            this.state.messages.push(userMsg);
            
            // Show typing indicator
            UI.showTypingIndicator();
            
            // Use streaming API for real-time response
            let assistantContent = '';
            let assistantMessageElement = null;
            let imagePromptPreview = null;
            
            // Create callback for image requests
            const onChunk = (chunk) => {
                assistantContent += chunk;
                if (!assistantMessageElement) {
                    // Create message element on first chunk
                    const tempMsg = {
                        role: 'assistant',
                        content: assistantContent,
                        created_at: new Date().toISOString()
                    };
                    assistantMessageElement = UI.appendMessage(tempMsg);
                    UI.hideTypingIndicator();
                } else {
                    // Update existing message content
                    const contentDiv = assistantMessageElement.querySelector('.message-content');
                    if (contentDiv) {
                        // Re-render markdown for assistant messages
                        contentDiv.innerHTML = UI.renderMarkdown(assistantContent);
                    }
                }
                UI.scrollToBottom();
            };
            
            // Add image callback to onChunk function
            onChunk.imageCallback = (imageInfo) => {
                imagePromptPreview = imageInfo;
            };
            
            // Add video callback to onChunk function
            let videoPromptPreview = null;
            onChunk.videoCallback = (videoInfo) => {
                videoPromptPreview = videoInfo;
            };
            
            // Add title update callback
            onChunk.titleCallback = (newTitle) => {
                // Update conversation title in sidebar and header
                this.updateConversationTitle(this.state.selectedConversationId, newTitle);
            };
            
            await API.sendMessageStream(
                this.state.selectedThreadId,
                message,
                onChunk,
                // onComplete
                async (messageId) => {
                    const assistantMsg = {
                        role: 'assistant',
                        content: assistantContent,
                        created_at: new Date().toISOString(),
                        id: messageId
                    };
                    this.state.messages.push(assistantMsg);
                    
                    // Update the message element with the actual message ID
                    if (assistantMessageElement) {
                        assistantMessageElement.setAttribute('data-message-id', messageId);
                    }
                    
                    // Phase 6: Auto-generate audio if TTS is enabled
                    if (this.state.ttsEnabled) {
                        this.autoGenerateAudio(messageId);
                    }
                    
                    // Check for image request
                    if (imagePromptPreview) {
                        if (imagePromptPreview.needs_confirmation) {
                            await this.showImageConfirmDialog(imagePromptPreview);
                        } else {
                            this.autoGenerateImage(imagePromptPreview);
                        }
                    }
                    
                    // Check for video request
                    if (videoPromptPreview) {
                        if (videoPromptPreview.needs_confirmation) {
                            await this.showVideoConfirmDialog(videoPromptPreview);
                        } else {
                            this.autoGenerateVideo(videoPromptPreview);
                        }
                    }
                    
                    // Start polling for new implicit memories (they're extracted in background)
                    this.startMemoryPolling();
                },
                // onError
                (error) => {
                    console.error('Streaming error:', error);
                    UI.hideTypingIndicator();
                    UI.showToast('Failed to send message: ' + error.message, 'error');
                }
            );
            
        } catch (error) {
            console.error('Failed to send message:', error);
            UI.hideTypingIndicator();
            UI.showToast('Failed to send message: ' + error.message, 'error');
        } finally {
            // Re-enable input
            input.disabled = false;
            document.getElementById('sendBtn').disabled = false;
            input.focus();
        }
    },
    
    /**
     * Show edit title modal
     */
    showEditTitleModal() {
        if (!this.state.selectedConversationId) return;
        
        const conversation = this.state.conversations.find(c => c.id === this.state.selectedConversationId);
        if (!conversation) return;
        
        document.getElementById('newTitleInput').value = conversation.title;
        
        const modal = new bootstrap.Modal(document.getElementById('editTitleModal'));
        modal.show();
    },
    
    /**
     * Save new title
     */
    async saveTitle() {
        const newTitle = document.getElementById('newTitleInput').value.trim();
        
        if (!newTitle) {
            UI.showToast('Title cannot be empty', 'error');
            return;
        }
        
        try {
            await API.updateConversation(this.state.selectedConversationId, newTitle);
            
            // Reload conversations
            await this.loadConversations();
            
            // Update header
            UI.updateHeader(newTitle, document.getElementById('characterName').textContent);
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('editTitleModal'));
            modal.hide();
            
            UI.showToast('Title updated', 'success');
            
        } catch (error) {
            console.error('Failed to update title:', error);
            UI.showToast('Failed to update title', 'error');
        }
    },
    
    /**
     * Save title from inline editing
     */
    async saveInlineTitle(newTitle) {
        try {
            await API.updateConversation(this.state.selectedConversationId, newTitle);
            
            // Update current conversation object
            if (this.currentConversation) {
                this.currentConversation.title = newTitle;
            }
            
            // Reload conversations list
            await this.loadConversations();
            
            UI.showToast('Title updated', 'success');
            
        } catch (error) {
            console.error('Failed to update title:', error);
            UI.showToast('Failed to update title', 'error');
            // Restore original title
            document.getElementById('conversationTitle').textContent = this.currentConversation?.title || 'Select a conversation';
        }
    },
    
    /**
     * Delete current conversation
     */
    async deleteCurrentConversation() {
        if (!this.state.selectedConversationId) return;
        
        try {
            // Get conversation details for confirmation
            const conversation = this.state.conversations.find(
                c => c.id === this.state.selectedConversationId
            );
            
            if (!conversation) return;
            
            // Get threads to count messages
            const threads = await API.listThreads(this.state.selectedConversationId);
            let totalMessages = 0;
            
            // Count messages across all threads
            for (const thread of threads) {
                const messages = await API.listMessages(thread.id);
                // Filter out scene capture messages from count
                totalMessages += messages.filter(m => m.role !== 'scene_capture').length;
            }
            
            // Get memory count
            const memories = await API.getCharacterMemories(this.state.selectedCharacterId);
            const conversationMemories = memories.filter(
                m => m.conversation_id === this.state.selectedConversationId
            );
            
            // Populate modal
            document.getElementById('deleteConvMessageCount').textContent = totalMessages;
            
            const memoryInfo = document.getElementById('deleteConvMemoryInfo');
            if (conversationMemories.length > 0) {
                document.getElementById('deleteConvMemoryCount').textContent = conversationMemories.length;
                memoryInfo.style.display = 'block';
            } else {
                memoryInfo.style.display = 'none';
            }
            
            // Reset checkbox
            document.getElementById('deleteConvKeepMemories').checked = true;
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('deleteConversationModal'));
            modal.show();
            
        } catch (error) {
            console.error('Error preparing delete confirmation:', error);
            alert('Failed to load conversation details');
        }
    },
    
    async confirmDeleteConversation() {
        const conversationId = this.state.selectedConversationId;
        if (!conversationId) return;
        
        const keepMemories = document.getElementById('deleteConvKeepMemories').checked;
        const deleteMemories = !keepMemories;
        
        try {
            await API.deleteConversation(conversationId, deleteMemories);
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('deleteConversationModal'));
            modal.hide();
            
            // Clear selection
            this.state.selectedConversationId = null;
            this.state.selectedThreadId = null;
            this.state.messages = [];
            
            // Reload conversations
            await this.loadConversations();
            
            // Clear UI
            UI.updateHeader('Select a conversation', '');
            UI.renderThreads([], null);
            UI.renderMessages([]);
            UI.showEmptyState(true);
            UI.setInputEnabled(false);
            
            UI.showToast('Conversation deleted', 'success');
            
        } catch (error) {
            console.error('Failed to delete conversation:', error);
            UI.showToast('Failed to delete conversation', 'error');
        }
    },
    
    /**
     * Load privacy status for current conversation
     */
    async loadConversationPrivacy() {
        if (!this.state.selectedConversationId) return;
        
        try {
            const data = await API.getConversationPrivacy(this.state.selectedConversationId);
            const toggle = document.getElementById('privacyToggle');
            toggle.checked = data.is_private;
            toggle.disabled = false;
            
            // Update visual indicator
            const header = document.querySelector('.chat-header');
            if (data.is_private) {
                header.classList.add('privacy-active');
            } else {
                header.classList.remove('privacy-active');
            }
        } catch (error) {
            console.error('Failed to load privacy status:', error);
            // Don't show error to user - this is non-critical
        }
    },
    
    /**
     * Update conversation title (called when auto-generated title arrives)
     */
    updateConversationTitle(conversationId, newTitle) {
        // Update in state
        const conv = this.state.conversations.find(c => c.id === conversationId);
        if (conv) {
            conv.title = newTitle;
        }
        
        // Update sidebar conversation item
        const convItem = document.querySelector(`[data-conversation-id="${conversationId}"]`);
        if (convItem) {
            const titleSpan = convItem.querySelector('.conversation-title') || convItem.querySelector('.fw-semibold');
            if (titleSpan) {
                titleSpan.textContent = newTitle;
                // Add subtle fade animation
                titleSpan.style.transition = 'opacity 0.3s';
                titleSpan.style.opacity = '0.5';
                setTimeout(() => {
                    titleSpan.style.opacity = '1';
                }, 50);
            }
        }
        
        // Update header if this is the active conversation
        if (conversationId === this.state.selectedConversationId) {
            const titleElement = document.getElementById('conversationTitle');
            if (titleElement) {
                titleElement.textContent = newTitle;
                // Add subtle fade animation
                titleElement.style.transition = 'opacity 0.3s';
                titleElement.style.opacity = '0.5';
                setTimeout(() => {
                    titleElement.style.opacity = '1';
                }, 50);
            }
        }
        
        console.log(`[TITLE UPDATE] Conversation title updated: "${newTitle}"`);
    },
    
    /**
     * Phase 6: Load TTS status for current conversation
     */
    async loadConversationTTS() {
        if (!this.state.selectedConversationId) return;
        
        try {
            const data = await API.getConversationTTS(this.state.selectedConversationId);
            this.state.ttsEnabled = data.tts_enabled;
            
            const toggle = document.getElementById('ttsToggle');
            if (toggle) {
                toggle.checked = data.tts_enabled;
                toggle.disabled = false;
            }
            
            console.log(`TTS status loaded: ${data.tts_enabled ? 'enabled' : 'disabled'} (override: ${data.conversation_override}, default: ${data.character_default})`);
        } catch (error) {
            console.error('Failed to load TTS status:', error);
            // Don't show error to user - this is non-critical
        }
    },
    
    /**
     * Phase 6: Set TTS status for current conversation
     */
    async setConversationTTS(enabled) {
        if (!this.state.selectedConversationId) return;
        
        try {
            await API.updateConversationTTS(this.state.selectedConversationId, enabled);
            this.state.ttsEnabled = enabled;
            
            UI.showToast(
                enabled ? 'TTS enabled for this conversation' : 'TTS disabled for this conversation',
                'success'
            );
            
            // Reload messages to update UI (show/hide generate buttons)
            if (this.state.selectedThreadId) {
                const messages = await API.listMessages(this.state.selectedThreadId);
                this.state.messages = messages;
                UI.renderMessages(messages);
            }
        } catch (error) {
            console.error('Failed to set TTS:', error);
            UI.showToast('Failed to update TTS setting', 'error');
            
            // Revert toggle
            const toggle = document.getElementById('ttsToggle');
            if (toggle) {
                toggle.checked = !enabled;
            }
        }
    },
    
    /**
     * Set privacy status for current conversation
     */
    async setConversationPrivacy(isPrivate) {
        if (!this.state.selectedConversationId) return;
        
        try {
            await API.setConversationPrivacy(this.state.selectedConversationId, isPrivate);
            
            // Update visual indicator
            const header = document.querySelector('.chat-header');
            if (isPrivate) {
                header.classList.add('privacy-active');
                UI.showToast('Private mode enabled. Memory extraction disabled for this conversation.', 'success');
            } else {
                header.classList.remove('privacy-active');
                UI.showToast('Private mode disabled. Memory extraction re-enabled.', 'success');
            }
        } catch (error) {
            console.error('Failed to set privacy:', error);
            UI.showToast('Failed to update privacy mode', 'error');
        }
    },
    
    /**
     * Show export modal
     */
    showExportModal() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }
        
        const modal = new bootstrap.Modal(document.getElementById('exportModal'));
        modal.show();
    },
    
    /**
     * Show character profile modal
     */
    async showCharacterProfileModal() {
        if (!this.state.selectedCharacterId) {
            UI.showToast('No character selected', 'error');
            return;
        }
        
        const character = this.state.characters.find(c => c.id === this.state.selectedCharacterId);
        if (!character) {
            UI.showToast('Character not found', 'error');
            return;
        }
        
        // Update avatar
        const modalAvatar = document.getElementById('modalProfileAvatar');
        modalAvatar.src = character.profile_image_url || '/character_images/default.svg';
        
        // Apply focal point if set
        if (character.profile_image_focus && character.profile_image_focus.x !== undefined && character.profile_image_focus.y !== undefined) {
            modalAvatar.style.objectPosition = `${character.profile_image_focus.x}% ${character.profile_image_focus.y}%`;
        } else {
            modalAvatar.style.objectPosition = '50% 50%';
        }
        
        // Update basic info
        document.getElementById('modalProfileName').textContent = character.name;
        document.getElementById('modalProfileRole').textContent = character.role;
        
        // Fetch and update stats
        try {
            const stats = await API.getCharacterStats(character.id);
            document.getElementById('modalStatConversations').textContent = stats.conversation_count || 0;
            document.getElementById('modalStatMessages').textContent = stats.message_count || 0;
            document.getElementById('modalStatMemories').textContent = stats.memory_count || 0;
            document.getElementById('modalStatCore').textContent = stats.core_memory_count || 0;
        } catch (error) {
            console.error('Failed to fetch character stats:', error);
        }
        
        // Update capabilities
        const capabilitiesContainer = document.getElementById('modalCapabilities');
        capabilitiesContainer.innerHTML = '';
        
        const capabilities = [
            { 
                name: 'Image Generation', 
                icon: 'bi-image',
                enabled: character.capabilities?.image_generation || false 
            },
            { 
                name: 'Video Generation', 
                icon: 'bi-camera-video',
                enabled: character.capabilities?.video_generation || false 
            },
            { 
                name: 'Voice/Audio', 
                icon: 'bi-mic',
                enabled: character.capabilities?.audio_generation || false 
            }
        ];
        
        capabilities.forEach(cap => {
            const div = document.createElement('div');
            div.className = `capability-item ${cap.enabled ? 'enabled' : 'disabled'}`;
            div.innerHTML = `
                <i class="bi ${cap.icon}"></i>
                <span>${cap.name}</span>
                ${cap.enabled ? '<i class="bi bi-check-circle-fill ms-auto" style="color: #4ade80;"></i>' : '<i class="bi bi-x-circle-fill ms-auto" style="color: #ef4444;"></i>'}
            `;
            capabilitiesContainer.appendChild(div);
        });
        
        // Update personality traits
        const traitsContainer = document.getElementById('modalPersonalityTraits');
        traitsContainer.innerHTML = '';
        
        if (character.personality_traits && character.personality_traits.length > 0) {
            character.personality_traits.forEach(trait => {
                const badge = document.createElement('span');
                badge.className = 'personality-trait-badge';
                badge.textContent = trait;
                traitsContainer.appendChild(badge);
            });
        } else {
            traitsContainer.innerHTML = '<span class="text-muted">No traits defined</span>';
        }
        
        // Update system prompt
        document.getElementById('modalSystemPrompt').textContent = character.system_prompt || 'No system prompt defined';
        
        // Update visual identity (if available)
        const visualIdentityCard = document.getElementById('modalVisualIdentityCard');
        const visualIdentityElement = document.getElementById('modalVisualIdentity');
        
        if (character.visual_identity && character.visual_identity.prompt_context) {
            visualIdentityCard.style.display = 'block';
            visualIdentityElement.textContent = character.visual_identity.prompt_context;
        } else {
            visualIdentityCard.style.display = 'none';
        }
        
        // Update configuration
        document.getElementById('modalImmersionLevel').textContent = character.immersion_level || 'default';
        document.getElementById('modalPreferredModel').textContent = character.preferred_llm?.model || 'Default';
        document.getElementById('modalTemperature').textContent = character.preferred_llm?.temperature?.toFixed(2) || '0.70';
        document.getElementById('modalMemoryScope').textContent = character.memory?.scope || 'character';
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('characterProfileModal'));
        modal.show();
    },
    
    /**
     * Export conversation
     */
    async exportConversation() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }
        
        try {
            // Get export options
            const format = document.getElementById('exportFormat').value;
            const includeMetadata = document.getElementById('includeMetadata').checked;
            const includeSummary = document.getElementById('includeSummary').checked;
            const includeMemories = document.getElementById('includeMemories').checked;
            
            // Show loading state
            const exportBtn = document.getElementById('confirmExportBtn');
            const originalHtml = exportBtn.innerHTML;
            exportBtn.disabled = true;
            exportBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Exporting...';
            
            // Call API
            const { blob, filename } = await API.exportConversation(
                this.state.selectedConversationId,
                format,
                includeMetadata,
                includeSummary,
                includeMemories
            );
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            // Hide modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('exportModal'));
            modal.hide();
            
            // Restore button
            exportBtn.disabled = false;
            exportBtn.innerHTML = originalHtml;
            
            UI.showToast(`Conversation exported successfully as ${format === 'markdown' ? 'Markdown' : 'Plain Text'}!`, 'success');
            
        } catch (error) {
            console.error('Failed to export conversation:', error);
            UI.showToast('Failed to export conversation', 'error');
            
            // Restore button
            const exportBtn = document.getElementById('confirmExportBtn');
            exportBtn.disabled = false;
            exportBtn.innerHTML = '<i class="bi bi-download me-1"></i> Export';
        }
    },

    async analyzeConversation() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }

        try {
            // Get modal elements
            const modal = new bootstrap.Modal(document.getElementById('analysisResultsModal'));
            const loadingSpinner = document.getElementById('analysisLoadingSpinner');
            const resultsDiv = document.getElementById('analysisResults');
            const errorDiv = document.getElementById('analysisError');
            
            // Show modal with loading state
            modal.show();
            loadingSpinner.style.display = 'block';
            resultsDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            
            // Call API with force=true for manual analysis
            const result = await API.analyzeConversation(this.state.selectedConversationId, true);
            
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            // Populate summary
            document.getElementById('analysisSummary').textContent = result.summary || 'No summary available';
            
            // Populate themes
            const themesDiv = document.getElementById('analysisThemes');
            if (result.themes && result.themes.length > 0) {
                themesDiv.innerHTML = result.themes.map(theme => 
                    `<span class="badge bg-primary me-1">${theme}</span>`
                ).join('');
            } else {
                themesDiv.innerHTML = '<span class="text-muted">No themes identified</span>';
            }
            
            // Populate tone
            document.getElementById('analysisTone').textContent = result.tone || 'No tone analysis available';
            
            // Populate emotional arc
            const emotionalArcDiv = document.getElementById('analysisEmotionalArc');
            let emotionalArc = result.emotional_arc;
            // Parse if it's a string
            if (typeof emotionalArc === 'string') {
                try {
                    emotionalArc = JSON.parse(emotionalArc);
                } catch (e) {
                    emotionalArc = [];
                }
            }
            if (emotionalArc && emotionalArc.length > 0) {
                emotionalArcDiv.innerHTML = emotionalArc.map((stage, index) => 
                    `<div><strong>Stage ${index + 1}:</strong> ${stage}</div>`
                ).join('');
            } else {
                emotionalArcDiv.innerHTML = '<span class="text-muted">No emotional arc data</span>';
            }
            
            // Populate memory counts
            const countsDiv = document.getElementById('analysisMemoryCounts');
            const totalBadge = document.getElementById('analysisMemoryTotal');
            const memoryCounts = result.memory_counts || {};
            const total = Object.values(memoryCounts).reduce((sum, count) => sum + count, 0);
            
            if (total > 0) {
                const badges = Object.entries(memoryCounts)
                    .filter(([type, count]) => count > 0)
                    .map(([type, count]) => {
                        const badgeClass = {
                            'fact': 'info',
                            'relationship': 'success',
                            'project': 'warning',
                            'experience': 'primary',
                            'story': 'secondary'
                        }[type] || 'secondary';
                        return `<span class="badge bg-${badgeClass} me-1">${type}: ${count}</span>`;
                    }).join('');
                countsDiv.innerHTML = badges;
                totalBadge.textContent = total;
            } else {
                countsDiv.innerHTML = '<span class="text-muted">No memories extracted</span>';
                totalBadge.textContent = '0';
            }
            
            // Populate memories list
            const memoriesList = document.getElementById('analysisMemoriesList');
            if (result.memories && result.memories.length > 0) {
                memoriesList.innerHTML = result.memories.map(memory => {
                    const badgeClass = {
                        'fact': 'info',
                        'relationship': 'success',
                        'project': 'warning',
                        'experience': 'primary',
                        'story': 'secondary'
                    }[memory.type] || 'secondary';
                    
                    return `
                        <li class="list-group-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div class="flex-grow-1">
                                    <span class="badge bg-${badgeClass} me-2">${memory.type}</span>
                                    ${memory.emotional_weight ? `<span class="badge bg-light text-dark ms-2">Weight: ${memory.emotional_weight.toFixed(2)}</span>` : ''}
                                    ${memory.confidence ? `<span class="badge bg-warning ms-1">Confidence: ${memory.confidence.toFixed(2)}</span>` : ''}
                                </div>
                            </div>
                            <p class="mb-0 mt-1">${memory.content}</p>
                        </li>
                    `;
                }).join('');
            } else {
                memoriesList.innerHTML = '<li class="list-group-item text-muted">No memories extracted</li>';
            }
            
            // Show results
            resultsDiv.style.display = 'block';
            
            // Refresh memory panel to show new memories
            if (this.state.selectedConversationId === this.currentConversationId) {
                await this.loadMemories();
            }
            
            UI.showToast(`Analysis complete! Extracted ${total} memories`, 'success');
            
        } catch (error) {
            console.error('Failed to analyze conversation:', error);
            
            // Show error in modal
            const errorDiv = document.getElementById('analysisError');
            errorDiv.textContent = `Failed to analyze conversation: ${error.message}`;
            errorDiv.style.display = 'block';
            
            // Hide loading spinner
            document.getElementById('analysisLoadingSpinner').style.display = 'none';
            
            UI.showToast('Failed to analyze conversation', 'error');
        }
    },

    async showAnalysisHistory() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }

        try {
            // Get modal elements
            const modal = new bootstrap.Modal(document.getElementById('analysisHistoryModal'));
            const loadingSpinner = document.getElementById('historyLoadingSpinner');
            const historyList = document.getElementById('historyList');
            const historyItems = document.getElementById('historyItems');
            const historyEmpty = document.getElementById('historyEmpty');
            const errorDiv = document.getElementById('historyError');
            
            // Show modal with loading state
            modal.show();
            loadingSpinner.style.display = 'block';
            historyList.style.display = 'none';
            historyEmpty.style.display = 'none';
            errorDiv.style.display = 'none';
            
            // Call API to get analyses
            const result = await API.getConversationAnalyses(this.state.selectedConversationId, false);
            
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (result.analyses && result.analyses.length > 0) {
                // Populate history list
                historyItems.innerHTML = result.analyses.map((analysis, index) => {
                    const date = new Date(analysis.analyzed_at);
                    const dateStr = date.toLocaleDateString('en-US', { 
                        month: 'short', day: 'numeric', year: 'numeric', 
                        hour: 'numeric', minute: '2-digit' 
                    });
                    
                    const isManual = analysis.manual === true;
                    const manualBadge = isManual 
                        ? '<span class="badge bg-info me-2">Manual</span>' 
                        : '<span class="badge bg-secondary me-2">Auto</span>';
                    
                    const totalMemories = Object.values(analysis.memory_counts || {}).reduce((sum, count) => sum + count, 0);
                    
                    // Parse themes if it's a string (API returns 'themes' field)
                    let themes = analysis.themes;
                    if (typeof themes === 'string') {
                        try {
                            themes = JSON.parse(themes);
                        } catch (e) {
                            themes = [];
                        }
                    }
                    
                    return `
                        <a href="#" class="list-group-item list-group-item-action" data-analysis-index="${index}">
                            <div class="d-flex justify-content-between align-items-start">
                                <div class="flex-grow-1">
                                    <div class="mb-1">
                                        ${manualBadge}
                                        <span class="text-muted">${dateStr}</span>
                                    </div>
                                    <h6 class="mb-1">${analysis.summary ? analysis.summary.substring(0, 100) + '...' : 'No summary'}</h6>
                                    ${themes && themes.length > 0 ? `
                                        <div class="mt-1">
                                            ${themes.map(topic => 
                                                `<span class="badge bg-primary me-1">${topic}</span>`
                                            ).join('')}
                                        </div>
                                    ` : ''}
                                </div>
                                <div class="text-end ms-3">
                                    <span class="badge bg-success">${totalMemories} memories</span>
                                </div>
                            </div>
                        </a>
                    `;
                }).join('');
                
                // Add click handlers to view details
                historyItems.querySelectorAll('.list-group-item').forEach(item => {
                    item.addEventListener('click', async (e) => {
                        e.preventDefault();
                        const index = parseInt(item.dataset.analysisIndex);
                        const analysis = result.analyses[index];
                        
                        // Close history modal and wait for it to fully close
                        const historyModal = bootstrap.Modal.getInstance(document.getElementById('analysisHistoryModal'));
                        const historyElement = document.getElementById('analysisHistoryModal');
                        
                        // Wait for modal to be fully hidden before showing next modal
                        historyElement.addEventListener('hidden.bs.modal', async () => {
                            // Add small delay to ensure Bootstrap cleanup is complete
                            setTimeout(async () => {
                                await this.showAnalysisDetails(analysis);
                            }, 100);
                        }, { once: true });
                        
                        historyModal.hide();
                    });
                });
                
                historyList.style.display = 'block';
            } else {
                historyEmpty.style.display = 'block';
            }
            
        } catch (error) {
            console.error('Failed to load analysis history:', error);
            
            // Show error in modal
            const errorDiv = document.getElementById('historyError');
            errorDiv.textContent = `Failed to load history: ${error.message}`;
            errorDiv.style.display = 'block';
            
            // Hide loading spinner
            document.getElementById('historyLoadingSpinner').style.display = 'none';
            
            UI.showToast('Failed to load analysis history', 'error');
        }
    },

    async showAnalysisDetails(analysis) {
        try {
            const modalElement = document.getElementById('analysisResultsModal');
            const loadingSpinner = document.getElementById('analysisLoadingSpinner');
            const resultsDiv = document.getElementById('analysisResults');
            const errorDiv = document.getElementById('analysisError');
            
            // Hide all sections initially
            loadingSpinner.style.display = 'none';
            resultsDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            
            // We need to fetch the full details with memories
            const result = await API.getConversationAnalyses(this.state.selectedConversationId, true);
            const fullAnalysis = result.analyses.find(a => a.analyzed_at === analysis.analyzed_at);
            
            if (!fullAnalysis) {
                throw new Error('Could not find analysis details');
            }
            
            // Populate summary
            document.getElementById('analysisSummary').textContent = fullAnalysis.summary || 'No summary available';
            
            // Populate themes
            const themesDiv = document.getElementById('analysisThemes');
            let themes = fullAnalysis.themes;
            // Parse if it's a string
            if (typeof themes === 'string') {
                try {
                    themes = JSON.parse(themes);
                } catch (e) {
                    themes = [];
                }
            }
            if (themes && themes.length > 0) {
                themesDiv.innerHTML = themes.map(theme => 
                    `<span class="badge bg-primary me-1">${theme}</span>`
                ).join('');
            } else {
                themesDiv.innerHTML = '<span class="text-muted">No themes identified</span>';
            }
            
            // Populate tone
            document.getElementById('analysisTone').textContent = fullAnalysis.tone || 'No tone analysis available';
            
            // Populate emotional arc
            const emotionalArcDiv = document.getElementById('analysisEmotionalArc');
            let emotionalArc = fullAnalysis.emotional_arc;
            // Parse if it's a string
            if (typeof emotionalArc === 'string') {
                try {
                    emotionalArc = JSON.parse(emotionalArc);
                } catch (e) {
                    emotionalArc = null;
                }
            }
            if (emotionalArc && emotionalArc.length > 0) {
                emotionalArcDiv.innerHTML = emotionalArc.map((stage, index) => 
                    `<div><strong>Stage ${index + 1}:</strong> ${stage}</div>`
                ).join('');
            } else {
                emotionalArcDiv.innerHTML = '<span class="text-muted">No emotional arc data</span>';
            }
            
            // Populate memory counts
            const countsDiv = document.getElementById('analysisMemoryCounts');
            const totalBadge = document.getElementById('analysisMemoryTotal');
            const memoryCounts = fullAnalysis.memory_counts || {};
            const total = Object.values(memoryCounts).reduce((sum, count) => sum + count, 0);
            
            if (total > 0) {
                const badges = Object.entries(memoryCounts)
                    .filter(([type, count]) => count > 0)
                    .map(([type, count]) => {
                        const badgeClass = {
                            'fact': 'info',
                            'relationship': 'success',
                            'project': 'warning',
                            'experience': 'primary',
                            'story': 'secondary'
                        }[type] || 'secondary';
                        return `<span class="badge bg-${badgeClass} me-1">${type}: ${count}</span>`;
                    }).join('');
                countsDiv.innerHTML = badges;
                totalBadge.textContent = total;
            } else {
                countsDiv.innerHTML = '<span class="text-muted">No memories extracted</span>';
                totalBadge.textContent = '0';
            }
            
            // Populate memories list
            const memoriesList = document.getElementById('analysisMemoriesList');
            if (fullAnalysis.memories && fullAnalysis.memories.length > 0) {
                memoriesList.innerHTML = fullAnalysis.memories.map(memory => {
                    const memType = memory.type || 'unknown';
                    const category = memory.category || memType;
                    
                    // Badge color based on type
                    const badgeClass = {
                        'fact': 'info',
                        'relationship': 'success',
                        'project': 'warning',
                        'experience': 'primary',
                        'story': 'secondary'
                    }[memType] || 'secondary';
                    
                    // Display category (e.g., "personal_info", "preference") with type in parentheses
                    const displayText = category !== memType ? `${category} (${memType})` : memType;
                    
                    return `
                        <li class="list-group-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div class="flex-grow-1">
                                    <span class="badge bg-${badgeClass} me-2">${displayText}</span>
                                    ${memory.confidence ? `<span class="badge bg-light text-dark ms-1">Conf: ${Math.round(memory.confidence * 100)}%</span>` : ''}
                                    ${memory.emotional_weight ? `<span class="badge bg-light text-dark ms-1">Weight: ${memory.emotional_weight.toFixed(1)}</span>` : ''}
                                </div>
                            </div>
                            <p class="mb-0 mt-1">${memory.content}</p>
                        </li>
                    `;
                }).join('');
            } else {
                memoriesList.innerHTML = '<li class="list-group-item text-muted">No memories extracted</li>';
            }
            
            // Show results
            resultsDiv.style.display = 'block';
            
            // Show the modal using Bootstrap API
            const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
            modalInstance.show();
            
        } catch (error) {
            console.error('Failed to load analysis details:', error);
            
            // Show error in modal
            const modalElement = document.getElementById('analysisResultsModal');
            const errorDiv = document.getElementById('analysisError');
            const resultsDiv = document.getElementById('analysisResults');
            errorDiv.textContent = `Failed to load analysis details: ${error.message}`;
            errorDiv.style.display = 'block';
            resultsDiv.style.display = 'none';
            
            // Show modal - only get existing instance
            let modalInstance = bootstrap.Modal.getInstance(modalElement);
            if (!modalInstance) {
                modalElement.classList.add('show');
                modalElement.style.display = 'block';
                document.body.classList.add('modal-open');
                if (!document.querySelector('.modal-backdrop')) {
                    const backdrop = document.createElement('div');
                    backdrop.className = 'modal-backdrop fade show';
                    document.body.appendChild(backdrop);
                }
            } else {
                modalInstance.show();
            }
            
            UI.showToast('Failed to load analysis details', 'error');
        }
    },
    
    /**
     * Show reset database modal
     */
    showResetDatabaseModal() {
        const modal = new bootstrap.Modal(document.getElementById('resetDatabaseModal'));
        // Clear input when showing
        document.getElementById('resetConfirmInput').value = '';
        document.getElementById('confirmResetBtn').disabled = true;
        modal.show();
    },
    
    /**
     * Reset database (complete wipe)
     */
    async resetDatabase() {
        try {
            UI.showToast('Resetting database...', 'info');
            
            await API.resetDatabase();
            
            // Hide modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('resetDatabaseModal'));
            modal.hide();
            
            // Clear application state
            this.state.characters = [];
            this.state.selectedCharacterId = null;
            this.state.conversations = [];
            this.state.selectedConversationId = null;
            this.state.threads = [];
            this.state.selectedThreadId = null;
            this.state.messages = [];
            
            // Clear UI
            UI.renderMessages([]);
            document.getElementById('conversationList').innerHTML = '';
            document.getElementById('characterSelect').value = '';
            
            // Disable buttons
            document.getElementById('messageInput').disabled = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('memoryPanelBtn').disabled = true;
            document.getElementById('manageWorkflowsBtn').disabled = true;
            document.getElementById('privacyToggle').disabled = true;
            document.getElementById('actionsMenuBtn').disabled = true;
            
            UI.showToast('Database reset successfully! All data cleared.', 'success');
            
            // Reload characters
            await this.loadCharacters();
            
        } catch (error) {
            console.error('Failed to reset database:', error);
            UI.showToast('Failed to reset database', 'error');
        }
    },
    
    /**
     * Phase 5: Show image confirmation dialog
     */
    async showImageConfirmDialog(imagePromptPreview, isSceneCapture = false) {
        // Phase 9: Enhanced to support scene captures and workflow selection
        // Store for later use
        this.pendingImageRequest = imagePromptPreview;
        this.pendingImageRequest.isSceneCapture = isSceneCapture;
        
        // Populate modal
        document.getElementById('imagePromptPreview').value = imagePromptPreview.prompt;
        document.getElementById('imageNegativePromptPreview').value = imagePromptPreview.negative_prompt || '';
        
        // Show trigger info if needed
        const triggerInfo = document.getElementById('imageTriggerInfo');
        if (imagePromptPreview.needs_trigger) {
            triggerInfo.classList.remove('d-none');
        } else {
            triggerInfo.classList.add('d-none');
        }
        
        // Phase 9: Populate workflow selector
        await this.populateWorkflowSelector();
        
        // Phase 9: For scene captures, always show dialog (disable the checkbox)
        const confirmCheckbox = document.getElementById('disableImageConfirmation');
        if (isSceneCapture) {
            confirmCheckbox.checked = false;
            confirmCheckbox.disabled = true;
            confirmCheckbox.parentElement.style.opacity = '0.5';
        } else {
            confirmCheckbox.disabled = false;
            confirmCheckbox.parentElement.style.opacity = '1';
            confirmCheckbox.checked = false;
        }
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('imageConfirmModal'));
        modal.show();
    },
    
    /**
     * Phase 5: Confirm and generate image
     * Phase 9: Enhanced to support scene captures and workflow selection
     */
    async confirmImageGeneration() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('imageConfirmModal'));
        modal.hide();
        
        if (!this.pendingImageRequest || !this.state.selectedThreadId) return;
        
        try {
            const disableConfirmation = document.getElementById('disableImageConfirmation').checked;
            const selectedWorkflow = document.getElementById('imageWorkflowSelector').value;
            const workflowId = selectedWorkflow !== 'default' ? selectedWorkflow : null;
            
            // Read current (potentially edited) values from textareas
            const editedPrompt = document.getElementById('imagePromptPreview').value;
            const editedNegativePrompt = document.getElementById('imageNegativePromptPreview').value;
            
            // Show progress message
            const progressDiv = UI.appendImageProgress();
            
            try {
                // Phase 9: Use different endpoint for scene captures
                if (this.pendingImageRequest.isSceneCapture) {
                    const result = await API.captureScene(
                        this.state.selectedThreadId,
                        editedPrompt,
                        editedNegativePrompt,
                        workflowId
                    );
                    
                    // Remove progress indicator
                    if (progressDiv) {
                        progressDiv.remove();
                    }
                    
                    // Scene capture now returns same format as normal image generation
                    if (result.success) {
                        UI.appendGeneratedImage(result);
                        UI.showToast('Scene captured!', 'success');
                        await this.refreshGallery();
                    } else {
                        UI.showToast('Scene capture failed: ' + (result.error || 'Unknown error'), 'error');
                    }
                } else {
                    // Regular image generation
                    const result = await API.generateImage(
                        this.state.selectedThreadId,
                        editedPrompt,
                        editedNegativePrompt,
                        disableConfirmation,
                        workflowId
                    );
                    
                    // Remove progress indicator
                    if (progressDiv) {
                        progressDiv.remove();
                    }
                    
                    if (result.success) {
                        UI.appendGeneratedImage(result);
                        UI.showToast('Image generated successfully!', 'success');
                        
                        // Phase 9: Refresh gallery after image generation
                        await this.refreshGallery();
                    } else {
                        UI.showToast('Image generation failed: ' + (result.error || 'Unknown error'), 'error');
                    }
                }
                
                // Clear pending request
                this.pendingImageRequest = null;
                
            } catch (error) {
                // Remove progress indicator on error
                if (progressDiv) {
                    progressDiv.remove();
                }
                throw error;
            }
            
        } catch (error) {
            console.error('Failed to generate image:', error);
            UI.showToast('Failed to generate image: ' + error.message, 'error');
        }
    },
    
    /**
     * Phase 5: Auto-generate image (confirmation disabled)
     */
    /**
     * Phase 6: Auto-generate audio for a message
     */
    async autoGenerateAudio(messageId) {
        if (!this.state.selectedConversationId) return;
        
        try {
            console.log(`Auto-generating audio for message ${messageId}...`);
            
            // Generate audio
            const result = await API.generateMessageAudio(
                this.state.selectedConversationId,
                messageId
            );
            
            if (result.success) {
                console.log('Audio generated successfully:', result.audio_filename);
                
                // Update just this message with the audio URL instead of reloading everything
                const messageIndex = this.state.messages.findIndex(m => m.id === messageId);
                if (messageIndex !== -1) {
                    // Update the message in state with audio URL
                    this.state.messages[messageIndex].audio_url = `/audio/${result.audio_filename}`;
                    this.state.messages[messageIndex].has_audio = "true";
                    
                    // Find the message element by data-message-id attribute
                    const msgEl = document.querySelector(`.message[data-message-id="${messageId}"]`);
                    if (msgEl) {
                        console.log('Found message element, prepending audio player');
                        
                        // Prepend audio player
                        const audioHtml = `
                            <div class="audio-player-container mb-2" data-message-id="${messageId}">
                                <audio controls preload="metadata" class="w-100 audio-player" data-message-id="${messageId}">
                                    <source src="/audio/${result.audio_filename}" type="audio/wav">
                                    Your browser does not support audio playback.
                                </audio>
                            </div>`;
                        
                        msgEl.insertAdjacentHTML('afterbegin', audioHtml);
                        
                        // Auto-play the audio
                        setTimeout(() => {
                            const audioPlayer = msgEl.querySelector(`audio[data-message-id="${messageId}"]`);
                            if (audioPlayer) {
                                console.log('Attempting auto-play of newly generated audio');
                                audioPlayer.play().catch(err => {
                                    console.log('Auto-play prevented by browser (this is normal):', err.message);
                                });
                            }
                        }, 100);
                    } else {
                        console.error('Could not find message element for ID:', messageId);
                    }
                }
            }
        } catch (error) {
            console.error('Failed to auto-generate audio:', error);
            // Don't show error toast - this is background generation
        }
    },
    
    /**
     * Phase 5: Auto-generate image after message
     */
    async autoGenerateImage(imagePromptPreview) {
        if (!this.state.selectedThreadId) return;
        
        try {
            // Show progress message
            const progressDiv = UI.appendImageProgress();
            
            // Generate image
            const result = await API.generateImage(
                this.state.selectedThreadId,
                imagePromptPreview.prompt,
                imagePromptPreview.negative_prompt,
                false
            );
            
            // Remove progress and show image
            if (progressDiv) {
                progressDiv.remove();
            }
            
            if (result.success) {
                UI.appendGeneratedImage(result);
            } else {
                UI.showToast('Image generation failed: ' + (result.error || 'Unknown error'), 'error');
            }
            
        } catch (error) {
            console.error('Failed to auto-generate image:', error);
            UI.showToast('Failed to generate image: ' + error.message, 'error');
        }
    },
    
    /**
     * Show video confirmation dialog
     */
    async showVideoConfirmDialog(videoPromptPreview, isSceneCapture = false) {
        this.pendingVideoRequest = videoPromptPreview;
        this.pendingVideoRequest.isSceneCapture = isSceneCapture;
        
        // Populate the prompt textarea
        document.getElementById('videoPromptPreview').value = videoPromptPreview.prompt || '';
        
        // Populate workflow selector
        await this.populateVideoWorkflowSelector();
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('videoConfirmModal'));
        modal.show();
    },
    
    /**
     * Confirm and generate video
     */
    async confirmVideoGeneration() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('videoConfirmModal'));
        modal.hide();
        
        if (!this.pendingVideoRequest || !this.state.selectedThreadId) return;
        
        try {
            const disableConfirmation = document.getElementById('disableVideoConfirmation').checked;
            const selectedWorkflow = document.getElementById('videoWorkflowSelector').value;
            const workflowId = selectedWorkflow !== 'default' ? selectedWorkflow : null;
            
            // Get edited prompt from textarea
            const editedPrompt = document.getElementById('videoPromptPreview').value;
            
            // Show progress message
            const progressDiv = UI.appendVideoProgress();
            
            try {
                let result;
                
                if (this.pendingVideoRequest.isSceneCapture) {
                    // Scene capture path
                    result = await API.captureVideoScene(
                        this.state.selectedThreadId,
                        editedPrompt,
                        null, // No negative prompt for video
                        workflowId
                    );
                } else {
                    // Regular generation path
                    result = await API.generateVideo(
                        this.state.selectedThreadId,
                        editedPrompt,
                        null, // No negative prompt
                        disableConfirmation,
                        workflowId
                    );
                }
                
                // Remove progress indicator
                if (progressDiv) {
                    progressDiv.remove();
                }
                
                if (result.success) {
                    UI.appendGeneratedVideo(result);
                    UI.showToast(`Video generated! (${result.format}, ${result.duration_seconds}s)`, 'success');
                    await this.refreshGallery();
                } else {
                    UI.showToast('Video generation failed: ' + (result.error || 'Unknown error'), 'error');
                }
                
                this.pendingVideoRequest = null;
                
            } catch (error) {
                if (progressDiv) {
                    progressDiv.remove();
                }
                throw error;
            }
            
        } catch (error) {
            console.error('Failed to generate video:', error);
            UI.showToast('Failed to generate video: ' + error.message, 'error');
        }
    },
    
    /**
     * Auto-generate video (confirmation disabled)
     */
    async autoGenerateVideo(videoPromptPreview) {
        if (!this.state.selectedThreadId) return;
        
        try {
            // Show progress message
            const progressDiv = UI.appendVideoProgress();
            
            // Generate video using the pre-generated prompt
            const result = await API.generateVideo(
                this.state.selectedThreadId,
                videoPromptPreview.prompt,  // Use generated prompt
                null,  // No negative prompt
                false,  // Don't disable confirmations
                null   // Use default workflow
            );
            
            // Remove progress and show video
            if (progressDiv) {
                progressDiv.remove();
            }
            
            if (result.success) {
                UI.appendGeneratedVideo(result);
                UI.showToast(`Video generated! (${result.format})`, 'success');
            } else {
                UI.showToast('Video generation failed: ' + (result.error || 'Unknown error'), 'error');
            }
            
        } catch (error) {
            console.error('Failed to auto-generate video:', error);
            UI.showToast('Failed to generate video: ' + error.message, 'error');
        }
    },
    
    /**
     * Capture current scene as video (🎥 button)
     */
    async captureVideoScene() {
        if (!this.state.selectedThreadId) {
            UI.showToast('Please select a conversation first', 'error');
            return;
        }
        
        try {
            // Get scene capture prompt from backend
            const promptData = await API.captureVideoScenePrompt(this.state.selectedThreadId);
            
            // Show video dialog with scene capture flag
            await this.showVideoConfirmDialog(promptData, true);
            
        } catch (error) {
            console.error('Failed to capture video scene:', error);
            UI.showToast('Failed to generate video scene prompt: ' + error.message, 'error');
        }
    },
    
    /**
     * Start polling for new implicit memories after sending a message
     */
    startMemoryPolling() {
        // Clear any existing timer
        if (this.state.memoryPollTimer) {
            clearTimeout(this.state.memoryPollTimer);
        }
        
        // Poll every 2 seconds for up to 10 seconds (5 attempts)
        let attempts = 0;
        const maxAttempts = 5;
        
        const poll = async () => {
            attempts++;
            
            try {
                await this.checkForNewMemories();
                
                // Continue polling if we haven't reached max attempts
                if (attempts < maxAttempts) {
                    this.state.memoryPollTimer = setTimeout(poll, 2000);
                }
            } catch (error) {
                console.error('Memory polling error:', error);
            }
        };
        
        // Start first poll after 2 seconds (give extraction time to start)
        this.state.memoryPollTimer = setTimeout(poll, 2000);
    },
    
    /**
     * Check for new implicit memories and show sparkle effect if count increased
     */
    async checkForNewMemories() {
        if (!this.state.selectedCharacterId) return;
        
        try {
            const stats = await API.getCharacterMemoryStats(this.state.selectedCharacterId);
            
            const characterId = this.state.selectedCharacterId;
            const previousCount = this.state.lastMemoryCount[characterId] || 0;
            const currentCount = stats.implicit_memory_count;
            
            // Update stored count
            this.state.lastMemoryCount[characterId] = currentCount;
            
            // If count increased, show sparkle effect
            if (currentCount > previousCount) {
                this.showMemorySparkle();
                console.log(`New implicit memory detected! Count: ${previousCount} → ${currentCount}`);
            }
        } catch (error) {
            console.error('Failed to check memory stats:', error);
        }
    },
    
    /**
     * Show sparkle effect on memory button
     */
    showMemorySparkle() {
        const memoryBtn = document.getElementById('memoryPanelBtn');
        
        // Add sparkle class
        memoryBtn.classList.add('memory-sparkle');
        
        // Remove sparkle after 8 seconds
        setTimeout(() => {
            memoryBtn.classList.remove('memory-sparkle');
        }, 8000);
    },
    
    /**
     * Phase 6: Show voice sample management modal
     */
    async showVoiceManagementModal() {
        if (!this.state.selectedCharacterId) {
            UI.showToast('Please select a character first', 'warning');
            return;
        }
        
        const character = this.state.characters.find(c => c.id === this.state.selectedCharacterId);
        if (!character) return;
        
        // Update modal title
        document.getElementById('voiceCharacterName').textContent = character.name;
        
        // Load existing voice samples
        await this.loadVoiceSamples();
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('voiceManagementModal'));
        modal.show();
    },
    
    /**
     * Phase 6: Load voice samples for current character
     */
    async loadVoiceSamples() {
        if (!this.state.selectedCharacterId) return;
        
        try {
            const samples = await API.listVoiceSamples(this.state.selectedCharacterId);
            const container = document.getElementById('voicesList');
            
            if (samples.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="bi bi-mic-mute-fill fs-1"></i>
                        <p class="mt-2">No voice samples uploaded yet</p>
                    </div>`;
                return;
            }
            
            container.innerHTML = samples.map(sample => `
                <div class="card mb-2 voice-sample-card" data-sample-id="${sample.id}">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <h6 class="card-title mb-1">
                                    <i class="bi bi-file-earmark-music"></i> ${sample.filename}
                                    ${sample.is_default ? '<span class="badge bg-success ms-2">Default</span>' : ''}
                                </h6>
                                <p class="card-text text-muted small mb-2">
                                    <i class="bi bi-calendar"></i> ${new Date(sample.uploaded_at).toLocaleString()}
                                </p>
                                <div class="transcript-box bg-light p-2 rounded mb-2">
                                    <small><strong>Transcript:</strong></small>
                                    <p class="mb-0 small">"${sample.transcript}"</p>
                                </div>
                            </div>
                            <div class="btn-group-vertical ms-3">
                                ${!sample.is_default ? `
                                    <button class="btn btn-sm btn-outline-primary set-default-btn" 
                                            data-sample-id="${sample.id}"
                                            title="Set as default">
                                        <i class="bi bi-star"></i>
                                    </button>
                                ` : ''}
                                <button class="btn btn-sm btn-outline-danger delete-sample-btn" 
                                        data-sample-id="${sample.id}"
                                        title="Delete">
                                    <i class="bi bi-trash"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
            
            // Attach event listeners
            container.querySelectorAll('.set-default-btn').forEach(btn => {
                btn.addEventListener('click', () => this.setDefaultVoiceSample(btn.dataset.sampleId));
            });
            
            container.querySelectorAll('.delete-sample-btn').forEach(btn => {
                btn.addEventListener('click', () => this.deleteVoiceSample(btn.dataset.sampleId));
            });
            
        } catch (error) {
            console.error('Failed to load voice samples:', error);
            UI.showToast('Failed to load voice samples', 'error');
        }
    },
    
    /**
     * Phase 6: Upload voice sample
     */
    async uploadVoiceSample() {
        if (!this.state.selectedCharacterId) return;
        
        const fileInput = document.getElementById('voiceFileInput');
        const transcriptInput = document.getElementById('voiceTranscriptInput');
        const defaultCheck = document.getElementById('voiceDefaultCheck');
        const uploadBtn = document.getElementById('uploadVoiceBtn');
        
        if (!fileInput.files[0]) {
            UI.showToast('Please select an audio file', 'warning');
            return;
        }
        
        if (!transcriptInput.value.trim()) {
            UI.showToast('Please provide a transcript', 'warning');
            return;
        }
        
        const originalHTML = uploadBtn.innerHTML;
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Uploading...';
        
        try {
            await API.uploadVoiceSample(
                this.state.selectedCharacterId,
                fileInput.files[0],
                transcriptInput.value.trim(),
                defaultCheck.checked
            );
            
            UI.showToast('Voice sample uploaded successfully', 'success');
            
            // Reset form
            fileInput.value = '';
            transcriptInput.value = '';
            defaultCheck.checked = false;
            
            // Reload list
            await this.loadVoiceSamples();
            
        } catch (error) {
            console.error('Failed to upload voice sample:', error);
            UI.showToast(error.message || 'Failed to upload voice sample', 'error');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = originalHTML;
        }
    },
    
    /**
     * Phase 6: Set default voice sample
     */
    async setDefaultVoiceSample(sampleId) {
        if (!this.state.selectedCharacterId) return;
        
        try {
            await API.updateVoiceSample(
                this.state.selectedCharacterId,
                sampleId,
                { is_default: true }
            );
            
            UI.showToast('Default voice sample updated', 'success');
            await this.loadVoiceSamples();
            
        } catch (error) {
            console.error('Failed to set default voice sample:', error);
            UI.showToast('Failed to set default voice sample', 'error');
        }
    },
    
    /**
     * Phase 6: Delete voice sample
     */
    async deleteVoiceSample(sampleId) {
        if (!this.state.selectedCharacterId) return;
        
        if (!confirm('Delete this voice sample? This cannot be undone.')) return;
        
        try {
            await API.deleteVoiceSample(this.state.selectedCharacterId, sampleId);
            
            UI.showToast('Voice sample deleted', 'success');
            await this.loadVoiceSamples();
            
        } catch (error) {
            console.error('Failed to delete voice sample:', error);
            UI.showToast(error.message || 'Failed to delete voice sample', 'error');
        }
    },
    
    /**
     * Phase 9: Update scene capture button visibility
     */
    updateSceneCaptureButton() {
        const imageBtn = document.getElementById('sceneCaptureBtn');
        const videoBtn = document.getElementById('videoSceneCaptureBtn');
        const character = this.state.characters.find(c => c.id === this.state.selectedCharacterId);
        
        console.log('Scene capture button check:', {
            hasImageButton: !!imageBtn,
            hasVideoButton: !!videoBtn,
            characterId: this.state.selectedCharacterId,
            character: character,
            immersionLevel: character?.immersion_level,
            capabilities: character?.capabilities,
            threadId: this.state.selectedThreadId
        });
        
        // Show image button for unbounded characters with image generation enabled and active conversation
        if (imageBtn) {
            if (character && character.immersion_level === 'unbounded' && 
                character.capabilities?.image_generation && this.state.selectedThreadId) {
                imageBtn.style.display = 'inline-block';
                imageBtn.disabled = false;
                console.log('✓ Image scene capture button enabled');
            } else {
                imageBtn.style.display = 'none';
                imageBtn.disabled = true;
                console.log('✗ Image scene capture button hidden:', {
                    isUnbounded: character?.immersion_level === 'unbounded',
                    imageGen: character?.capabilities?.image_generation,
                    hasThread: !!this.state.selectedThreadId
                });
            }
        }
        
        // Show video button for unbounded characters with video generation enabled and active conversation
        if (videoBtn) {
            if (character && character.immersion_level === 'unbounded' && 
                character.capabilities?.video_generation && this.state.selectedThreadId) {
                videoBtn.style.display = 'inline-block';
                videoBtn.disabled = false;
                console.log('✓ Video scene capture button enabled');
            } else {
                videoBtn.style.display = 'none';
                videoBtn.disabled = true;
                console.log('✗ Video scene capture button hidden:', {
                    isUnbounded: character?.immersion_level === 'unbounded',
                    videoGen: character?.capabilities?.video_generation,
                    hasThread: !!this.state.selectedThreadId
                });
            }
        }
    },
    
    /**
     * Phase 9: Capture scene from observer perspective
     */
    async captureScene() {
        if (!this.state.selectedThreadId) {
            UI.showToast('No active conversation', 'error');
            return;
        }
        
        try {
            // Get scene capture prompt from backend (no /api prefix - direct route)
            const response = await fetch(`/threads/${this.state.selectedThreadId}/capture-scene-prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate scene prompt');
            }
            
            const promptData = await response.json();
            
            // Show image dialog with scene capture flag
            await this.showImageConfirmDialog(promptData, true);
            
        } catch (error) {
            console.error('Failed to capture scene:', error);
            UI.showToast('Failed to generate scene prompt: ' + error.message, 'error');
        }
    },
    
    /**
     * Populate video workflow selector dropdown
     */
    async populateVideoWorkflowSelector() {
        const selector = document.getElementById('videoWorkflowSelector');
        selector.innerHTML = '';
        
        // Get current character
        const characterId = this.state.selectedCharacterId;
        if (!characterId) {
            // No character selected, just show default
            const defaultOption = document.createElement('option');
            defaultOption.value = 'default';
            defaultOption.textContent = 'Default';
            defaultOption.selected = true;
            selector.appendChild(defaultOption);
            return;
        }
        
        try {
            // Fetch workflows from database
            const response = await API.listWorkflows(characterId);
            const allWorkflows = response.workflows || [];
            
            // Filter for video workflows only
            const videoWorkflows = allWorkflows.filter(w => w.workflow_type === 'video');
            
            // Add each workflow as an option
            videoWorkflows.forEach((wf, index) => {
                const option = document.createElement('option');
                option.value = wf.id;
                option.textContent = wf.name;
                
                // Select the default workflow
                if (wf.is_default) {
                    option.selected = true;
                }
                
                selector.appendChild(option);
            });
            
            // If no workflows found, show a placeholder
            if (videoWorkflows.length === 0) {
                const defaultOption = document.createElement('option');
                defaultOption.value = 'default';
                defaultOption.textContent = 'No workflows configured';
                defaultOption.selected = true;
                selector.appendChild(defaultOption);
            }
        } catch (error) {
            console.error('Failed to load video workflows:', error);
            // Fallback to default
            const defaultOption = document.createElement('option');
            defaultOption.value = 'default';
            defaultOption.textContent = 'Default';
            defaultOption.selected = true;
            selector.appendChild(defaultOption);
        }
    },

    /**
     * Phase 9: Populate workflow selector dropdown
     */
    async populateWorkflowSelector() {
        const selector = document.getElementById('imageWorkflowSelector');
        selector.innerHTML = '';
        
        // Get current character
        const characterId = this.state.selectedCharacterId;
        if (!characterId) {
            // No character selected, just show default
            const defaultOption = document.createElement('option');
            defaultOption.value = 'default';
            defaultOption.textContent = 'Default';
            defaultOption.selected = true;
            selector.appendChild(defaultOption);
            return;
        }
        
        try {
            // Fetch workflows from database
            const response = await API.listWorkflows(characterId);
            const allWorkflows = response.workflows || [];
            
            // Filter for image workflows only
            const imageWorkflows = allWorkflows.filter(w => w.workflow_type === 'image');
            
            // Add each workflow as an option
            imageWorkflows.forEach((wf, index) => {
                const option = document.createElement('option');
                option.value = wf.id;
                option.textContent = wf.name;
                
                // Select the default workflow
                if (wf.is_default) {
                    option.selected = true;
                }
                
                selector.appendChild(option);
            });
            
            // If no workflows found, show a placeholder
            if (imageWorkflows.length === 0) {
                const defaultOption = document.createElement('option');
                defaultOption.value = 'default';
                defaultOption.textContent = 'No workflows available';
                defaultOption.selected = true;
                selector.appendChild(defaultOption);
            }
        } catch (error) {
            console.error('Failed to load workflows:', error);
            // Fallback to default option
            const defaultOption = document.createElement('option');
            defaultOption.value = 'default';
            defaultOption.textContent = 'Default';
            defaultOption.selected = true;
            selector.appendChild(defaultOption);
        }
    },
    
    /**
     * Phase 9: Load image gallery for current conversation
     */
    async loadImageGallery() {
        if (!this.state.selectedConversationId) {
            this.renderGallery([], []);
            return;
        }
        
        try {
            // Load images
            const imageResponse = await fetch(`/conversations/${this.state.selectedConversationId}/images`);
            let images = [];
            if (imageResponse.ok) {
                const imageData = await imageResponse.json();
                images = imageData.images || [];
            }
            
            // Load videos
            const videoResponse = await fetch(`/conversations/${this.state.selectedConversationId}/videos`);
            let videos = [];
            if (videoResponse.ok) {
                const videoData = await videoResponse.json();
                videos = videoData.videos || [];
                console.log('Loaded videos:', videos.length, videos);
            } else {
                console.warn('Failed to load videos:', videoResponse.status, videoResponse.statusText);
            }
            
            this.state.galleryImages = images;
            this.state.galleryVideos = videos;
            this.renderGallery(images, videos);
            
        } catch (error) {
            console.error('Failed to load gallery:', error);
            this.state.galleryImages = [];
            this.state.galleryVideos = [];
            this.renderGallery([], []);
        }
    },
    
    /**
     * Phase 9: Render image and video gallery
     */
    renderGallery(images, videos) {
        console.log('renderGallery called with:', images?.length || 0, 'images and', videos?.length || 0, 'videos');
        const imageGrid = document.getElementById('galleryGrid');
        const videoGrid = document.getElementById('videoGalleryGrid');
        const totalCount = document.querySelector('.gallery-count');
        const imageCount = document.getElementById('imageCount');
        const videoCount = document.getElementById('videoCount');
        
        console.log('HTML elements:', { imageGrid: !!imageGrid, videoGrid: !!videoGrid, totalCount: !!totalCount, imageCount: !!imageCount, videoCount: !!videoCount });
        
        const totalMedia = (images?.length || 0) + (videos?.length || 0);
        totalCount.textContent = totalMedia;
        imageCount.textContent = images?.length || 0;
        videoCount.textContent = videos?.length || 0;
        
        // Render images
        if (!images || images.length === 0) {
            imageGrid.innerHTML = '<p class="text-muted text-center mt-2">No images yet</p>';
        } else {
            imageGrid.innerHTML = images.map((img, index) => `
                <div class="gallery-thumbnail" onclick="App.viewGalleryImage(${index})" title="${this.escapeHtml(img.prompt || 'Image')}">
                    <img src="${img.thumbnail_path || img.file_path}" alt="Image ${index + 1}">
                    ${img.is_scene_capture ? '<span class="scene-capture-badge"><i class="bi bi-camera-fill"></i> Scene</span>' : ''}
                </div>
            `).join('');
        }
        
        // Render videos
        if (!videos || videos.length === 0) {
            videoGrid.innerHTML = '<p class="text-muted text-center mt-2">No videos yet</p>';
        } else {
            videoGrid.innerHTML = videos.map((vid, index) => {
                const isImageFormat = vid.format === 'webp' || vid.format === '.webp' || vid.format === 'gif' || vid.format === '.gif';
                return `
                    <div class="gallery-thumbnail" onclick="App.viewGalleryVideo(${index})" title="${this.escapeHtml(vid.prompt || 'Video')}">
                        ${isImageFormat ? `
                            <img src="${vid.file_path}" alt="Video ${index + 1}" style="width: 100%; height: 100%; object-fit: cover;">
                        ` : `
                            <video muted style="width: 100%; height: 100%; object-fit: cover; pointer-events: none;">
                                <source src="${vid.file_path}" type="${UI.getVideoMimeType(vid.format)}">
                            </video>
                        `}
                        ${vid.is_scene_capture ? '<span class="scene-capture-badge"><i class="bi bi-camera-video-fill"></i> Scene</span>' : ''}
                        <span class="video-duration-badge">${vid.format.toUpperCase()}</span>
                    </div>
                `;
            }).join('');
        }
    },
    
    /**
     * Phase 9: View gallery image in modal
     */
    viewGalleryImage(index) {
        const img = this.state.galleryImages[index];
        if (!img) return;
        
        // Create and show modal with full image
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-xl modal-dialog-centered">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">
                            ${img.is_scene_capture ? '<i class="bi bi-camera-fill me-2"></i>Scene Capture' : '<i class="bi bi-image-fill me-2"></i>Generated Image'}
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img src="${img.file_path}" class="img-fluid rounded" alt="Full image" style="max-height: 70vh;">
                        ${img.prompt ? `<p class="modal-label mt-3 small"><strong>Prompt:</strong> ${this.escapeHtml(img.prompt)}</p>` : ''}
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-outline-danger btn-sm" onclick="App.deleteImage(${img.id}, this)">
                            <i class="bi bi-trash me-1"></i> Delete Image
                        </button>
                        <button type="button" class="btn btn-outline-warning btn-sm" onclick="App.setAsProfileImage('${img.file_path}')">
                            <i class="bi bi-person-circle me-1"></i> Set as Character Profile
                        </button>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Remove modal from DOM when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    },
    
    /**
     * View gallery video in modal
     */
    viewGalleryVideo(index) {
        const vid = this.state.galleryVideos[index];
        if (!vid) return;
        
        const isImageFormat = vid.format === 'webp' || vid.format === '.webp' || vid.format === 'gif' || vid.format === '.gif';
        
        // Create and show modal with full video
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-xl modal-dialog-centered">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">
                            ${vid.is_scene_capture ? '<i class="bi bi-camera-video-fill me-2"></i>Scene Capture' : '<i class="bi bi-camera-video me-2"></i>Generated Video'}
                            <span class="badge bg-secondary ms-2">${vid.format.toUpperCase()}</span>
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        ${isImageFormat ? `
                            <img src="${vid.file_path}" class="img-fluid rounded" alt="Full video" style="max-height: 70vh;">
                        ` : `
                            <video controls preload="metadata" class="rounded" style="max-width: 100%; max-height: 70vh;">
                                <source src="${vid.file_path}" type="${UI.getVideoMimeType(vid.format)}">
                                Your browser does not support video playback.
                            </video>
                        `}
                        ${vid.prompt ? `<p class="modal-label mt-3 small"><strong>Prompt:</strong> ${this.escapeHtml(vid.prompt)}</p>` : ''}
                        ${vid.duration_seconds ? `<p class="modal-label small"><strong>Duration:</strong> ${vid.duration_seconds.toFixed(1)}s</p>` : ''}
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-outline-danger btn-sm" onclick="App.deleteVideo(${vid.id}, this)">
                            <i class="bi bi-trash me-1"></i> Delete Video
                        </button>
                        <a href="${vid.file_path}" download class="btn btn-outline-primary btn-sm">
                            <i class="bi bi-download me-1"></i> Download
                        </a>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Remove modal from DOM when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    },
    
    /**
     * Delete a video from the conversation
     */
    async deleteVideo(videoId, buttonElement) {
        // Confirm deletion
        if (!confirm('Are you sure you want to delete this video? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await API.deleteVideo(videoId);
            
            if (response.success) {
                UI.showToast('Video deleted successfully', 'success');
                
                // Close the modal
                const modal = buttonElement.closest('.modal');
                if (modal) {
                    const bsModal = bootstrap.Modal.getInstance(modal);
                    if (bsModal) {
                        bsModal.hide();
                    }
                }
                
                // Refresh gallery and conversation messages
                this.refreshGallery();
                if (this.state.selectedConversationId) {
                    await this.selectConversation(this.state.selectedConversationId);
                }
            } else {
                UI.showToast(response.message || 'Failed to delete video', 'error');
            }
        } catch (error) {
            console.error('Failed to delete video:', error);
            UI.showToast('Failed to delete video', 'error');
        }
    },
    
    /**
     * Delete an image from the conversation
     */
    async deleteImage(imageId, buttonElement) {
        // Confirm deletion
        if (!confirm('Are you sure you want to delete this image? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await API.deleteImage(imageId);
            
            if (response.success) {
                UI.showToast('Image deleted successfully', 'success');
                
                // Close the modal
                const modal = buttonElement.closest('.modal');
                if (modal) {
                    const bsModal = bootstrap.Modal.getInstance(modal);
                    if (bsModal) {
                        bsModal.hide();
                    }
                }
                
                // Refresh gallery and conversation messages
                this.refreshGallery();
                if (this.state.selectedConversationId) {
                    await this.selectConversation(this.state.selectedConversationId);
                }
            } else {
                UI.showToast('Failed to delete image', 'error');
            }
        } catch (error) {
            console.error('Error deleting image:', error);
            UI.showToast('Error deleting image: ' + error.message, 'error');
        }
    },
    
    /**
     * Phase 9: Toggle image gallery panel
     */
    toggleImageGallery() {
        const panel = document.getElementById('imageGallery');
        panel.classList.toggle('collapsed');
    },
    
    /**
     * Phase 9: Refresh gallery after new image is generated
     */
    refreshGallery() {
        if (this.state.selectedConversationId) {
            this.loadImageGallery();
        }
    },
    
    /**
     * Set an image as the character's profile picture
     */
    async setAsProfileImage(imagePath) {
        if (!this.state.selectedCharacterId) {
            UI.showToast('No character selected', 'error');
            return;
        }
        
        try {
            // Extract filename from path (e.g., /images/conv-id/filename.png -> filename.png)
            const filename = imagePath.split('/').pop();
            
            // Copy image to character_images directory and update character
            const response = await API.setCharacterProfileImage(this.state.selectedCharacterId, filename);
            
            if (response.success) {
                UI.showToast('Profile image updated!', 'success');
                
                // Reload characters to get updated profile image
                await this.loadCharacters();
                
                // Update profile card if character is still selected
                if (this.state.selectedCharacterId) {
                    const character = this.state.characters.find(c => c.id === this.state.selectedCharacterId);
                    if (character) {
                        await UI.updateProfileCard(character);
                    }
                }
                
                // Close any open modals
                const modals = document.querySelectorAll('.modal.show');
                modals.forEach(modal => {
                    const bsModal = bootstrap.Modal.getInstance(modal);
                    if (bsModal) bsModal.hide();
                });
            }
        } catch (error) {
            console.error('Failed to set profile image:', error);
            UI.showToast('Failed to update profile image', 'error');
        }
    },
    
    /**
     * Helper: Escape HTML for safe rendering
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    // === Export/Import Functions ===
    
    /**
     * Export current character configuration
     */
    async exportCharacter() {
        if (!this.state.selectedCharacterId) {
            UI.showToast('Please select a character first', 'error');
            return;
        }
        
        try {
            const blob = await API.exportCharacter(this.state.selectedCharacterId);
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.state.selectedCharacterId}.yaml`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            UI.showToast('Character exported successfully', 'success');
        } catch (error) {
            console.error('Failed to export character:', error);
            UI.showToast('Failed to export character', 'error');
        }
    },
    
    /**
     * Show import character modal
     */
    showImportCharacterModal() {
        const modal = new bootstrap.Modal(document.getElementById('importCharacterModal'));
        modal.show();
    },
    
    /**
     * Import character from file
     */
    async importCharacter() {
        const fileInput = document.getElementById('characterFileInput');
        const file = fileInput.files[0];
        
        if (!file) {
            UI.showToast('Please select a file', 'error');
            return;
        }
        
        try {
            const result = await API.importCharacter(file);
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('importCharacterModal'));
            modal.hide();
            
            // Reset form
            document.getElementById('importCharacterForm').reset();
            
            // Reload characters
            await this.loadCharacters();
            
            UI.showToast(`Character '${result.name}' imported successfully!`, 'success');
        } catch (error) {
            console.error('Failed to import character:', error);
            UI.showToast(`Failed to import character: ${error.message}`, 'error');
        }
    },
    
    /**
     * Export system configuration
     */
    async exportSystemConfig() {
        try {
            const blob = await API.exportSystemConfig();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'system.yaml';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            UI.showToast('System config exported successfully', 'success');
        } catch (error) {
            console.error('Failed to export system config:', error);
            UI.showToast('Failed to export system config', 'error');
        }
    },
    
    /**
     * Show import system config modal
     */
    showImportSystemConfigModal() {
        const modal = new bootstrap.Modal(document.getElementById('importSystemConfigModal'));
        modal.show();
    },
    
    /**
     * Import system configuration from file
     */
    async importSystemConfig() {
        const fileInput = document.getElementById('systemConfigFileInput');
        const file = fileInput.files[0];
        
        if (!file) {
            UI.showToast('Please select a file', 'error');
            return;
        }
        
        try {
            const result = await API.importSystemConfig(file);
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('importSystemConfigModal'));
            modal.hide();
            
            // Reset form
            document.getElementById('importSystemConfigForm').reset();
            
            if (result.restart_recommended) {
                UI.showToast('System config imported! Please restart the server.', 'warning');
            } else {
                UI.showToast('System config imported successfully!', 'success');
            }
        } catch (error) {
            console.error('Failed to import system config:', error);
            UI.showToast(`Failed to import system config: ${error.message}`, 'error');
        }
    },
    
    // === Log Viewer Functions ===
    
    /**
     * Show server logs modal
     */
    async showServerLogsModal() {
        const modal = new bootstrap.Modal(document.getElementById('serverLogsModal'));
        modal.show();
        await this.loadServerLogs();
    },
    
    /**
     * Load server logs
     */
    async loadServerLogs() {
        const content = document.getElementById('serverLogsContent');
        const info = document.getElementById('serverLogInfo');
        
        try {
            content.textContent = 'Loading...';
            info.textContent = '';
            
            const result = await API.getServerLogs(500);
            
            if (result.logs) {
                content.textContent = result.logs;
                
                if (result.file) {
                    info.textContent = `File: ${result.file} | Showing ${result.lines_returned} of ${result.total_lines} lines`;
                } else {
                    info.textContent = result.logs.includes('Debug mode') ? 'Debug mode disabled' : 'No log file';
                }
                
                // Scroll to bottom
                content.scrollTop = content.scrollHeight;
            } else {
                content.textContent = 'No logs available';
            }
        } catch (error) {
            console.error('Failed to load server logs:', error);
            content.textContent = `Error loading logs: ${error.message}`;
        }
    },
    
    /**
     * Show debug log for current conversation
     */
    async showConversationDebugLog() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }
        
        const modal = new bootstrap.Modal(document.getElementById('conversationLogsModal'));
        modal.show();
        
        // Set title
        const conversation = this.currentConversation;
        document.getElementById('debugLogConversationTitle').textContent = 
            conversation ? conversation.title : this.state.selectedConversationId;
        
        await this.loadConversationLog(this.state.selectedConversationId);
    },
    
    /**
     * Show extraction log for current conversation
     */
    async showConversationExtractionLog() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }
        
        const modal = new bootstrap.Modal(document.getElementById('extractionLogsModal'));
        modal.show();
        
        // Set title
        const conversation = this.currentConversation;
        document.getElementById('extractionLogConversationTitle').textContent = 
            conversation ? conversation.title : this.state.selectedConversationId;
        
        await this.loadExtractionLog(this.state.selectedConversationId);
    },
    
    /**
     * Load specific conversation log
     */
    async loadConversationLog(conversationId) {
        const content = document.getElementById('conversationLogsContent');
        const info = document.getElementById('conversationLogInfo');
        
        try {
            content.textContent = 'Loading...';
            info.textContent = '';
            
            const result = await API.getConversationLog(conversationId);
            
            if (result.interactions && result.interactions.length > 0) {
                // Pretty print the JSON
                content.textContent = JSON.stringify(result.interactions, null, 2);
                info.textContent = `${result.count} interactions`;
                
                // Scroll to bottom
                content.scrollTop = content.scrollHeight;
            } else {
                content.textContent = 'No interactions logged for this conversation';
            }
        } catch (error) {
            console.error('Failed to load conversation log:', error);
            if (error.message.includes('404')) {
                content.textContent = 'No debug log found for this conversation. Debug logging may be disabled or no LLM calls have been made yet.';
            } else {
                content.textContent = `Error loading log: ${error.message}`;
            }
        }
    },
    
    /**
     * Load extraction log for a conversation
     */
    async loadExtractionLog(conversationId) {
        const content = document.getElementById('extractionLogsContent');
        const info = document.getElementById('extractionLogInfo');
        
        try {
            content.textContent = 'Loading...';
            info.textContent = '';
            
            // Get the extraction log via API
            const result = await API.getExtractionLog(conversationId);
            
            if (result.extractions && result.extractions.length > 0) {
                // Pretty print the JSON
                content.textContent = JSON.stringify(result.extractions, null, 2);
                info.textContent = `${result.extractions.length} extraction${result.extractions.length !== 1 ? 's' : ''}`;
                
                // Scroll to bottom
                content.scrollTop = content.scrollHeight;
            } else {
                content.textContent = 'No extractions logged for this conversation';
            }
        } catch (error) {
            console.error('Failed to load extraction log:', error);
            if (error.message.includes('404')) {
                content.textContent = 'No extraction log found for this conversation. No extractions have been performed yet.';
            } else {
                content.textContent = `Error loading log: ${error.message}`;
            }
        }
    },
    
    /**
     * Show conversation image request log modal
     */
    async showConversationImageRequestLog() {
        if (!this.state.selectedConversationId) {
            UI.showToast('No conversation selected', 'error');
            return;
        }
        
        const modal = new bootstrap.Modal(document.getElementById('imageRequestLogsModal'));
        modal.show();
        
        // Set title
        const conversation = this.currentConversation;
        document.getElementById('imageRequestLogConversationTitle').textContent = 
            conversation ? conversation.title : this.state.selectedConversationId;
        
        await this.loadImageRequestLog(this.state.selectedConversationId);
    },
    
    /**
     * Load image request log for a conversation
     */
    async loadImageRequestLog(conversationId) {
        const content = document.getElementById('imageRequestLogsContent');
        const info = document.getElementById('imageRequestLogInfo');
        
        try {
            content.textContent = 'Loading...';
            info.textContent = '';
            
            // Get the image request log via API
            const result = await API.getImageRequestLog(conversationId);
            
            if (result.image_requests && result.image_requests.length > 0) {
                // Pretty print the JSON
                content.textContent = JSON.stringify(result.image_requests, null, 2);
                info.textContent = `${result.image_requests.length} image request${result.image_requests.length !== 1 ? 's' : ''}`;
                
                // Scroll to bottom
                content.scrollTop = content.scrollHeight;
            } else {
                content.textContent = 'No image requests logged for this conversation';
            }
        } catch (error) {
            if (error.message.includes('404')) {
                // Silently handle 404 - it's expected for conversations without image generations
                content.textContent = 'No image request log found for this conversation. No images have been generated yet.';
            } else {
                console.error('Failed to load image request log:', error);
                content.textContent = `Error loading log: ${error.message}`;
            }
        }
    },
    
    /**
     * Show intent detection logs modal (removed - now per conversation)
     */
    async showIntentLogsModal() {
        // This function is no longer used but kept for backward compatibility
        UI.showToast('Please select a conversation and use "View Intent Log" from the actions menu', 'info');
    },
    
    /**
     * Load intent detection logs (legacy global version)
     */
    async loadIntentLogs() {
        const content = document.getElementById('intentLogsContent');
        const info = document.getElementById('intentLogInfo');
        
        try {
            content.textContent = 'Loading...';
            info.textContent = '';
            
            const result = await API.getIntentDetectionLogs(100);
            
            if (result.interactions && result.interactions.length > 0) {
                // Pretty print the JSON
                content.textContent = JSON.stringify(result.interactions, null, 2);
                info.textContent = `Showing ${result.count} of ${result.total} interactions`;
                
                // Scroll to bottom
                content.scrollTop = content.scrollHeight;
            } else {
                content.textContent = result.message || 'No intent detection logs found';
            }
        } catch (error) {
            console.error('Failed to load intent logs:', error);
            content.textContent = `Error loading logs: ${error.message}`;
        }
    },
    
    // === Log Download Functions ===
    
    /**
     * Download server logs
     */
    async downloadServerLogs() {
        try {
            const result = await API.getServerLogs(10000); // Get more lines for download
            
            if (!result.logs || result.logs.includes('Debug mode')) {
                UI.showToast('No server log file available to download', 'error');
                return;
            }
            
            const blob = new Blob([result.logs], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = result.file || 'server.log';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            UI.showToast('Server log downloaded', 'success');
        } catch (error) {
            console.error('Failed to download server logs:', error);
            UI.showToast('Failed to download server logs', 'error');
        }
    },
    
    /**
     * Download conversation debug log
     */
    async downloadConversationLog(conversationId) {
        try {
            // Download raw JSONL
            const url = `${window.location.protocol}//${window.location.host}/logs/conversations/${conversationId}?prettify=false`;
            const a = document.createElement('a');
            a.href = url;
            a.download = `${conversationId}_debug.jsonl`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            UI.showToast('Conversation log downloaded', 'success');
        } catch (error) {
            console.error('Failed to download conversation log:', error);
            UI.showToast('Failed to download conversation log', 'error');
        }
    },
    
    /**
     * Download extraction log for a conversation
     */
    async downloadExtractionLog(conversationId) {
        try {
            // Download the extractions.jsonl file via API (raw mode)
            const url = `/logs/extractions/${conversationId}?prettify=false`;
            const response = await fetch(url);
            
            if (!response.ok) {
                if (response.status === 404) {
                    UI.showToast('No extraction log found for this conversation', 'error');
                    return;
                }
                throw new Error(`HTTP ${response.status}`);
            }
            
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `${conversationId}_extractions.jsonl`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(downloadUrl);
            
            UI.showToast('Extraction log downloaded', 'success');
        } catch (error) {
            console.error('Failed to download extraction log:', error);
            UI.showToast('Failed to download extraction log', 'error');
        }
    },
    
    /**
     * Download image request log for a conversation
     */
    async downloadImageRequestLog(conversationId) {
        try {
            // Download the image_prompts.jsonl file via API (raw mode)
            const url = `/logs/image-prompts/${conversationId}?prettify=false`;
            const response = await fetch(url);
            
            if (!response.ok) {
                if (response.status === 404) {
                    UI.showToast('No image request log found for this conversation', 'error');
                    return;
                }
                throw new Error(`HTTP ${response.status}`);
            }
            
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `${conversationId}_image_prompts.jsonl`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(downloadUrl);
            
            UI.showToast('Image request log downloaded', 'success');
        } catch (error) {
            console.error('Failed to download image request log:', error);
            UI.showToast('Failed to download image request log', 'error');
        }
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
