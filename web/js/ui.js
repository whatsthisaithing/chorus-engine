/**
 * UI Helper Functions
 * Handles DOM manipulation and UI updates
 */

const UI = {
    /**
     * Show a toast notification
     * @param {string} message - Message to display (supports newlines)
     * @param {string} type - Type: 'info', 'success', 'error', 'warning'
     * @param {number} duration - Auto-hide delay in ms (default 3000)
     */
    showToast(message, type = 'info', duration = 3000) {
        const toastEl = document.getElementById('notificationToast');
        const toastBody = document.getElementById('toastMessage');
        
        // Support multiline messages by converting \n to <br>
        toastBody.innerHTML = message.replace(/\n/g, '<br>');
        
        // Reset and add color class
        toastEl.className = 'toast';
        if (type === 'error') {
            toastEl.classList.add('bg-danger', 'text-white');
        } else if (type === 'success') {
            toastEl.classList.add('bg-success', 'text-white');
        } else if (type === 'warning') {
            toastEl.classList.add('bg-warning', 'text-dark');
        } else if (type === 'info') {
            toastEl.classList.add('bg-info', 'text-dark');
        }
        
        // Configure delay
        toastEl.setAttribute('data-bs-delay', duration);
        
        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    },
    
    /**
     * Show/hide loading modal
     */
    setLoading(loading) {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        if (loading) {
            modal.show();
        } else {
            modal.hide();
        }
    },
    
    /**
     * Render character options in select
     */
    renderCharacters(characters) {
        const select = document.getElementById('characterSelect');
        select.innerHTML = '<option value="">Select a character...</option>';
        
        characters.forEach(char => {
            const option = document.createElement('option');
            option.value = char.id;
            option.textContent = `${char.name} - ${char.role}`;
            select.appendChild(option);
        });
    },
    
    /**
     * Update the character profile card
     */
    async updateProfileCard(character) {
        const card = document.getElementById('characterProfileCard');
        if (!card || !character) {
            if (card) card.style.display = 'none';
            return;
        }

        // Show the card
        card.style.display = 'block';

        // Update avatar
        const avatar = document.getElementById('profileAvatar');
        avatar.src = character.profile_image_url || '/character_images/default.svg';
        avatar.alt = `${character.name} Avatar`;
        
        // Apply focal point if set
        if (character.profile_image_focus && character.profile_image_focus.x !== undefined && character.profile_image_focus.y !== undefined) {
            avatar.style.objectPosition = `${character.profile_image_focus.x}% ${character.profile_image_focus.y}%`;
        } else {
            avatar.style.objectPosition = '50% 50%'; // Default center
        }

        // Update name and role
        document.getElementById('profileName').textContent = character.name;
        document.getElementById('profileRole').textContent = character.role;

        // Fetch and update stats
        try {
            const stats = await API.getCharacterStats(character.id);
            document.getElementById('statConversations').textContent = stats.conversation_count || 0;
            document.getElementById('statMessages').textContent = stats.message_count || 0;
            document.getElementById('statMemories').textContent = stats.memory_count || 0;
            document.getElementById('statCore').textContent = stats.core_memory_count || 0;
        } catch (error) {
            console.error('Failed to fetch character stats:', error);
            // Show zeros on error
            document.getElementById('statConversations').textContent = '0';
            document.getElementById('statMessages').textContent = '0';
            document.getElementById('statMemories').textContent = '0';
            document.getElementById('statCore').textContent = '0';
        }

        // Update capabilities
        const capabilitiesContainer = document.getElementById('profileCapabilities');
        capabilitiesContainer.innerHTML = '';

        if (character.capabilities) {
            if (character.capabilities.image_generation) {
                const badge = document.createElement('span');
                badge.className = 'capability-badge';
                badge.innerHTML = '<i class="bi bi-image"></i> Images';
                capabilitiesContainer.appendChild(badge);
            }
            if (character.capabilities.video_generation) {
                const badge = document.createElement('span');
                badge.className = 'capability-badge';
                badge.innerHTML = '<i class="bi bi-camera-video"></i> Videos';
                capabilitiesContainer.appendChild(badge);
            }
            if (character.capabilities.audio_generation) {
                const badge = document.createElement('span');
                badge.className = 'capability-badge';
                badge.innerHTML = '<i class="bi bi-mic"></i> Voice';
                capabilitiesContainer.appendChild(badge);
            }
        }
    },
    
    /**
     * Render conversation list
     */
    renderConversations(conversations, activeId = null) {
        const container = document.getElementById('conversationList');
        
        if (conversations.length === 0) {
            container.innerHTML = '<p class="text-muted small">No conversations yet</p>';
            return;
        }
        
        container.innerHTML = '';
        
        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            item.setAttribute('data-conversation-id', conv.id);
            if (conv.id === activeId) {
                item.classList.add('active');
            }
            
            item.innerHTML = `
                <div class="conversation-title">${this.escapeHtml(conv.title)}</div>
                <div class="conversation-date">${this.formatDate(conv.updated_at)}</div>
            `;
            
            item.addEventListener('click', () => {
                window.App.selectConversation(conv.id);
            });
            
            container.appendChild(item);
        });
    },
    
    /**
     * Render thread tabs
     */
    renderThreads(threads, activeId = null) {
        const container = document.getElementById('threadTabs');
        container.innerHTML = '';
        
        threads.forEach(thread => {
            const li = document.createElement('li');
            li.className = 'nav-item';
            
            const link = document.createElement('a');
            link.className = 'nav-link';
            if (thread.id === activeId) {
                link.classList.add('active');
            }
            link.href = '#';
            link.textContent = thread.title;
            
            link.addEventListener('click', (e) => {
                e.preventDefault();
                window.App.selectThread(thread.id);
            });
            
            li.appendChild(link);
            container.appendChild(li);
        });
    },
    
    /**
     * Render messages
     */
    renderMessages(messages) {
        const container = document.getElementById('messagesContainer');
        
        // Clear only message divs, not the empty state
        const messageElements = container.querySelectorAll('.message');
        messageElements.forEach(el => el.remove());
        
        // Show/hide empty state based on message count
        this.showEmptyState(messages.length === 0);
        
        messages.forEach(msg => {
            this.appendMessage(msg);
        });
        
        this.scrollToBottom();
    },
    
    /**
     * Append a single message
     * Phase 9: Enhanced to handle scene capture messages
     */
    appendMessage(message) {
        const container = document.getElementById('messagesContainer');
        
        // Hide empty state when appending messages
        this.showEmptyState(false);
        
        // Phase 9: For scene captures, only render the image/video, not the message bubble
        if (message.role === 'scene_capture') {
            // Only render if generation completed successfully
            if (message.metadata && message.metadata.status === 'completed') {
                if (message.metadata.image_id) {
                    this.appendSceneCaptureImage(message);
                } else if (message.metadata.video_id) {
                    this.appendSceneCaptureVideo(message);
                }
            }
            return;
        }
        
        const messageDiv = document.createElement('div');
        // Add private-message class if message is private
        const privateClass = message.is_private === 'true' ? ' private-message' : '';
        messageDiv.className = `message ${message.role}-message${privateClass}`;
        messageDiv.setAttribute('data-message-id', message.id);
        
        // Build message content
        let contentHtml = '';
        
        // Phase 6: Add TTS audio player if audio exists
        if (message.audio_url && message.role === 'assistant') {
            contentHtml += `
                <div class="audio-player-container mb-2" data-message-id="${message.id}">
                    <audio controls preload="metadata" class="w-100 audio-player" data-message-id="${message.id}">
                        <source src="${message.audio_url}" type="audio/wav">
                        Your browser does not support audio playback.
                    </audio>
                </div>`;
        } 
        // Phase 6: Auto-generation enabled - no manual button needed
        
        // Add text content if present
        const content = (message.content || '').trim();
        if (content) {
            // For assistant messages, convert username references to @ format
            // LLM generates <username> which gets treated as HTML tags
            let processedContent = content;
            if (message.role === 'assistant') {
                // Replace <username> with @username (both Discord mentions and LLM-generated references)
                processedContent = processedContent.replace(/<@(\d+)>/g, '@$1'); // Discord mentions <@123456>
                processedContent = processedContent.replace(/<([A-Za-z0-9_ ]+)>/g, '@$1'); // LLM references <Username> (with spaces)
            }
            
            const formattedContent = message.role === 'assistant' 
                ? this.renderMarkdown(processedContent)
                : this.escapeHtml(processedContent);
            contentHtml += `<div class="message-content">${formattedContent}</div>`;
        }
        
        // Add username badge for non-web multi-user conversations (Discord, etc.)
        // Only show if platform is set and not 'web'
        if (message.role === 'user' && message.metadata && message.metadata.username) {
            const platform = message.metadata.platform ? message.metadata.platform.toLowerCase() : '';
            if (platform && platform !== 'web') {
                const username = this.escapeHtml(message.metadata.username);
                contentHtml += `<div class="message-username">
                    <span class="username-label">${username}</span>
                </div>`;
            }
        }
        
        // Task 1.9: Display image attachments for USER messages
        if (message.attachments && message.attachments.length > 0) {
            contentHtml += '<div class="message-attachments">';
            
            for (const attachment of message.attachments) {
                const imageUrl = `/api/attachments/${attachment.id}/file`;
                const hasVision = attachment.vision_processed === 'true';
                const confidence = attachment.vision_confidence ? (attachment.vision_confidence * 100).toFixed(0) : null;
                
                contentHtml += `
                    <div class="image-attachment ${hasVision ? 'has-vision' : ''}" 
                         data-attachment-id="${attachment.id}"
                         onclick="UI.showImageModal('${attachment.id}')">
                        <img src="${imageUrl}" 
                             alt="${this.escapeHtml(attachment.original_filename || 'Uploaded image')}" 
                             loading="lazy"
                             class="attachment-thumbnail">
                        ${hasVision && confidence ? `
                            <div class="vision-badge" title="Vision analyzed with ${confidence}% confidence">
                                <i class="bi bi-eye-fill"></i> ${confidence}%
                            </div>
                        ` : ''}
                    </div>`;
            }
            
            contentHtml += '</div>';
        }
        
        // Add timestamp
        contentHtml += `<div class="message-timestamp">${this.formatTime(message.created_at)}</div>`;
        
        messageDiv.innerHTML = contentHtml;
        
        // Highlight code blocks if present (for text content)
        if (message.role === 'assistant' && content) {
            setTimeout(() => {
                messageDiv.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }, 0);
        }
        
        // Append the text message
        container.appendChild(messageDiv);
        
        // Add image as a SEPARATE message bubble if metadata includes image_id
        if (message.role === 'assistant' && message.metadata && message.metadata.image_id) {
            const imageDiv = document.createElement('div');
            imageDiv.className = 'message assistant-message image-only-message mt-3';
            imageDiv.innerHTML = `
                <div class="message-content">
                    <div class="generated-image-container">
                        <img src="${message.metadata.thumbnail_path || message.metadata.image_path}" 
                             alt="Generated image" 
                             class="generated-image"
                             loading="lazy">
                        <div class="image-actions">
                            <button class="btn btn-sm btn-outline-secondary" 
                                    onclick="UI.viewInlineImage('${message.metadata.image_path}', '${this.escapeHtml(message.metadata.prompt || '').replace(/'/g, '\\&apos;')}', false)"
                                    title="View full size">
                                <i class="bi bi-arrows-fullscreen"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-warning" 
                                    onclick="App.setAsProfileImage('${message.metadata.image_path}')"
                                    title="Set as character profile">
                                <i class="bi bi-person-circle"></i>
                            </button>
                            <a href="${message.metadata.image_path}" 
                               download 
                               class="btn btn-sm btn-outline-secondary"
                               title="Download">
                                <i class="bi bi-download"></i>
                            </a>
                            ${message.metadata.prompt ? `
                            <button class="btn btn-sm btn-outline-secondary" 
                                    onclick="UI.showImagePrompt('${this.escapeHtml(message.metadata.prompt).replace(/'/g, '\\&apos;')}')"
                                    title="View prompt">
                                <i class="bi bi-chat-left-text"></i>
                            </button>
                            ` : ''}
                        </div>
                        ${message.metadata.generation_time ? `
                            <small class="text-muted d-block mt-2">
                                Generated in ${message.metadata.generation_time.toFixed(1)}s
                            </small>
                        ` : ''}
                    </div>
                </div>
            `;
            container.appendChild(imageDiv);
        }
        
        // Add video as a SEPARATE message bubble if metadata includes video_id
        if (message.role === 'assistant' && message.metadata && message.metadata.video_id) {
            const videoDiv = document.createElement('div');
            videoDiv.className = 'message assistant-message video-only-message mt-3';
            const isImageFormat = this.isImageVideoFormat(message.metadata.format);
            videoDiv.innerHTML = `
                <div class="message-content">
                    <div class="generated-video-container">
                        ${isImageFormat ? `
                            <img src="${message.metadata.video_path}" 
                                 alt="Generated animation" 
                                 class="generated-video"
                                 style="max-width: 100%; max-height: 600px; border-radius: 8px;"
                                 loading="lazy">
                        ` : `
                            <video controls preload="metadata" class="generated-video" style="max-width: 100%; max-height: 600px; border-radius: 8px;">
                                <source src="${message.metadata.video_path}" type="${this.getVideoMimeType(message.metadata.format)}">
                                Your browser does not support the video tag.
                            </video>
                        `}
                        <div class="video-actions">
                            <button class="btn btn-sm btn-outline-secondary" 
                                    onclick="UI.viewInlineVideo('${message.metadata.video_path}', '${this.escapeHtml(message.metadata.prompt || '').replace(/'/g, '\\&apos;')}', false, '${message.metadata.format || 'mp4'}')"
                                    title="View fullscreen">
                                <i class="bi bi-arrows-fullscreen"></i>
                            </button>
                            <a href="${message.metadata.video_path}" 
                               download 
                               class="btn btn-sm btn-outline-secondary"
                               title="Download">
                                <i class="bi bi-download"></i>
                            </a>
                            ${message.metadata.prompt ? `
                            <button class="btn btn-sm btn-outline-secondary" 
                                    onclick="UI.showVideoPrompt('${this.escapeHtml(message.metadata.prompt).replace(/'/g, '\\&apos;')}')"
                                    title="View prompt">
                                <i class="bi bi-chat-left-text"></i>
                            </button>
                            ` : ''}
                        </div>
                        ${message.metadata.generation_time ? `
                            <small class="text-muted d-block mt-2">
                                Generated in ${message.metadata.generation_time.toFixed(1)}s
                            </small>
                        ` : ''}
                        ${message.metadata.duration ? `
                            <small class="text-muted d-block">
                                Duration: ${message.metadata.duration.toFixed(1)}s • ${(message.metadata.format || 'video').toUpperCase()}
                            </small>
                        ` : ''}
                    </div>
                </div>
            `;
            container.appendChild(videoDiv);
        }
        
        return messageDiv;
    },
    
    /**
     * Phase 9: Append scene capture image (no message bubble)
     */
    appendSceneCaptureImage(message) {
        const container = document.getElementById('messagesContainer');
        
        const imageDiv = document.createElement('div');
        imageDiv.className = 'message assistant-message image-only-message mt-3';
        imageDiv.setAttribute('data-message-id', message.id);
        
        imageDiv.innerHTML = `
            <div class="message-content">
                <div class="generated-image-container">
                    <img src="${message.metadata.thumbnail_path || message.metadata.image_path}" 
                         alt="Scene capture" 
                         class="generated-image"
                         loading="lazy">
                    <span class="badge bg-info scene-capture-badge">
                        <i class="bi bi-camera-fill"></i> Scene Capture
                    </span>
                    <div class="image-actions">
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.viewInlineImage('${message.metadata.image_path}', '${this.escapeHtml(message.metadata.prompt || '').replace(/'/g, '\\&apos;')}', true)"
                                title="View full size">
                            <i class="bi bi-arrows-fullscreen"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-warning" 
                                onclick="App.setAsProfileImage('${message.metadata.image_path}')"
                                title="Set as character profile">
                            <i class="bi bi-person-circle"></i>
                        </button>
                        <a href="${message.metadata.image_path}" 
                           download 
                           class="btn btn-sm btn-outline-secondary"
                           title="Download">
                            <i class="bi bi-download"></i>
                        </a>
                        ${message.metadata.prompt ? `
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.showImagePrompt('${this.escapeHtml(message.metadata.prompt).replace(/'/g, '\\&apos;')}')"
                                title="View prompt">
                            <i class="bi bi-chat-left-text"></i>
                        </button>
                        ` : ''}
                    </div>
                    ${message.metadata.generation_time ? `
                        <small class="text-muted d-block mt-2">
                            Generated in ${message.metadata.generation_time.toFixed(1)}s
                        </small>
                    ` : ''}
                </div>
            </div>
        `;
        
        container.appendChild(imageDiv);
    },
    
    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        const container = document.getElementById('messagesContainer');
        
        const indicator = document.createElement('div');
        indicator.className = 'message assistant-message';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        container.appendChild(indicator);
        this.scrollToBottom();
    },
    
    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    },
    
    /**
     * Append a streaming message placeholder
     */
    appendStreamingMessage() {
        const container = document.getElementById('messagesContainer');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message streaming';
        messageDiv.id = 'streaming-message';
        
        messageDiv.innerHTML = `
            <div class="message-content"></div>
        `;
        
        container.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    },
    
    /**
     * Update streaming message content
     */
    updateStreamingMessage(messageDiv, content) {
        const contentDiv = messageDiv.querySelector('.message-content');
        const renderedContent = this.renderMarkdown(content);
        contentDiv.innerHTML = renderedContent;
        
        // Highlight code blocks
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        this.scrollToBottom();
    },
    
    /**
     * Finalize streaming message
     */
    finalizeStreamingMessage(messageDiv) {
        messageDiv.classList.remove('streaming');
        messageDiv.id = '';
        
        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = this.formatTime(new Date().toISOString());
        messageDiv.appendChild(timestamp);
    },
    
    /**
     * Update conversation header
     */
    updateHeader(title, characterName) {
        document.getElementById('conversationTitle').textContent = title;
        document.getElementById('characterName').textContent = characterName;
    },
    
    /**
     * Show/hide empty state
     */
    showEmptyState(show) {
        const emptyState = document.getElementById('emptyState');
        if (emptyState) {
            emptyState.style.display = show ? 'block' : 'none';
        }
    },
    
    /**
     * Enable/disable input controls
     */
    setInputEnabled(enabled) {
        document.getElementById('messageInput').disabled = !enabled;
        document.getElementById('sendBtn').disabled = !enabled;
        document.getElementById('actionsMenuBtn').disabled = !enabled;
        // Task 1.8: Enable/disable attach image button
        const attachBtn = document.getElementById('attachImageBtn');
        if (attachBtn) {
            attachBtn.disabled = !enabled;
        }
    },
    
    /**
     * Scroll messages to bottom
     */
    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        container.scrollTop = container.scrollHeight;
    },
    
    /**
     * Format datetime for display
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;
        
        // Less than 1 day
        if (diff < 86400000) {
            return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
        }
        
        // Less than 7 days
        if (diff < 604800000) {
            return date.toLocaleDateString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' });
        }
        
        // Older
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    },
    
    /**
     * Format timestamp with relative time and absolute tooltip
     * Returns HTML string with tooltip
     */
    formatTimestamp(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        // Relative time for display
        let relative;
        if (diffMins < 1) relative = 'Just now';
        else if (diffMins < 60) relative = `${diffMins}m ago`;
        else if (diffMins < 1440) relative = `${Math.floor(diffMins/60)}h ago`;
        else if (diffMins < 10080) relative = `${Math.floor(diffMins/1440)}d ago`;
        else relative = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        
        // Absolute time for tooltip
        const absolute = date.toLocaleString(undefined, {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        });
        
        return `<span title="${absolute}">${relative}</span>`;
    },
    
    /**
     * Format time for message timestamp
     */
    formatTime(dateString) {
        // Use the new formatTimestamp function with relative time and tooltip
        return this.formatTimestamp(dateString);
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
     * Render markdown to HTML
     */
    renderMarkdown(text) {
        // Clean up excessive newlines (more than 2 in a row)
        text = text.replace(/\n{3,}/g, '\n\n');
        
        // Configure marked options
        marked.setOptions({
            breaks: false,  // Don't convert single \n to <br>
            gfm: true,      // GitHub Flavored Markdown
            headerIds: false,
            mangle: false
        });
        
        // Parse and return markdown as HTML
        return marked.parse(text);
    },
    
    /**
     * Phase 5: Append image generation progress indicator
     */
    appendImageProgress() {
        const container = document.getElementById('messagesContainer');
        
        const progressDiv = document.createElement('div');
        progressDiv.className = 'message assistant-message mt-3';
        progressDiv.innerHTML = `
            <div class="message-content">
                <div class="image-progress">
                    <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
                        <span class="visually-hidden">Generating...</span>
                    </div>
                    <span>Generating image...</span>
                </div>
            </div>
        `;
        
        container.appendChild(progressDiv);
        this.scrollToBottom();
        
        return progressDiv;
    },
    
    /**
     * Phase 6: Handle generate audio button click
     */
    async handleGenerateAudio(messageId, button) {
        if (!App.state.selectedConversationId) return;
        
        const originalHTML = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Generating...';
        
        try {
            const result = await API.generateMessageAudio(
                App.state.selectedConversationId,
                messageId
            );
            
            this.showToast('Audio generated successfully', 'success');
            
            // Reload messages to show the new audio player
            await App.loadMessages();
            
        } catch (error) {
            console.error('Failed to generate audio:', error);
            this.showToast(error.message || 'Failed to generate audio', 'error');
            button.disabled = false;
            button.innerHTML = originalHTML;
        }
    },
    
    /**
     * Phase 6: Handle regenerate audio button click
     */
    async handleRegenerateAudio(messageId, button) {
        if (!App.state.selectedConversationId) return;
        
        if (!confirm('Delete existing audio and generate new?')) return;
        
        const originalHTML = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
        
        try {
            // Delete existing audio
            await API.deleteMessageAudio(
                App.state.selectedConversationId,
                messageId
            );
            
            // Generate new audio
            await API.generateMessageAudio(
                App.state.selectedConversationId,
                messageId
            );
            
            this.showToast('Audio regenerated successfully', 'success');
            
            // Reload messages to show the new audio player
            await App.loadMessages();
            
        } catch (error) {
            console.error('Failed to regenerate audio:', error);
            this.showToast(error.message || 'Failed to regenerate audio', 'error');
            button.disabled = false;
            button.innerHTML = originalHTML;
        }
    },
    
    /**
     * Phase 6: Handle delete audio button click
     */
    async handleDeleteAudio(messageId, button) {
        if (!App.state.selectedConversationId) return;
        
        if (!confirm('Delete this audio?')) return;
        
        const originalHTML = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
        
        try {
            await API.deleteMessageAudio(
                App.state.selectedConversationId,
                messageId
            );
            
            this.showToast('Audio deleted', 'success');
            
            // Reload messages to remove the audio player
            await App.loadMessages();
            
        } catch (error) {
            console.error('Failed to delete audio:', error);
            this.showToast(error.message || 'Failed to delete audio', 'error');
            button.disabled = false;
            button.innerHTML = originalHTML;
        }
    },
    
    /**
     * Phase 5: Append generated image to chat
     */
    appendGeneratedImage(result) {
        const container = document.getElementById('messagesContainer');
        
        const imageDiv = document.createElement('div');
        imageDiv.className = 'message assistant-message image-only-message mt-3';
        imageDiv.innerHTML = `
            <div class="message-content">
                <div class="generated-image-container">
                    <img src="${result.thumbnail_path || result.file_path}" 
                         alt="Generated image" 
                         class="generated-image"
                         loading="lazy">
                    <div class="image-actions">
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.viewInlineImage('${result.file_path}', '${this.escapeHtml(result.prompt || '').replace(/'/g, '\\&apos;')}', false)"
                                title="View full size">
                            <i class="bi bi-arrows-fullscreen"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-warning" 
                                onclick="App.setAsProfileImage('${result.file_path}')"
                                title="Set as character profile">
                            <i class="bi bi-person-circle"></i>
                        </button>
                        <a href="${result.file_path}" 
                           download 
                           class="btn btn-sm btn-outline-secondary"
                           title="Download">
                            <i class="bi bi-download"></i>
                        </a>
                        ${result.prompt ? `
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.showImagePrompt('${this.escapeHtml(result.prompt).replace(/'/g, '\\&apos;')}')"
                                title="View prompt">
                            <i class="bi bi-chat-left-text"></i>
                        </button>
                        ` : ''}
                    </div>
                    ${result.generation_time ? `
                        <small class="text-muted d-block mt-2">
                            Generated in ${result.generation_time.toFixed(1)}s
                        </small>
                    ` : ''}
                </div>
            </div>
        `;
        
        container.appendChild(imageDiv);
        
        // Wait for image to load before scrolling
        const img = imageDiv.querySelector('img');
        if (img) {
            img.onload = () => this.scrollToBottom();
        }
        // Fallback scroll in case image loads from cache
        setTimeout(() => this.scrollToBottom(), 100);
    },
    
    /**
     * Append video generation progress indicator
     */
    appendVideoProgress() {
        const container = document.getElementById('messagesContainer');
        
        const progressDiv = document.createElement('div');
        progressDiv.className = 'message assistant-message mt-3';
        progressDiv.innerHTML = `
            <div class="message-content">
                <div class="video-progress">
                    <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
                        <span class="visually-hidden">Generating...</span>
                    </div>
                    <span>Generating video... (this may take several minutes)</span>
                </div>
            </div>
        `;
        
        container.appendChild(progressDiv);
        this.scrollToBottom();
        
        return progressDiv;
    },
    
    /**
     * Append generated video to chat
     */
    appendGeneratedVideo(result) {
        const container = document.getElementById('messagesContainer');
        
        const videoDiv = document.createElement('div');
        videoDiv.className = 'message assistant-message video-only-message mt-3';
        
        const durationText = result.duration_seconds 
            ? `${result.duration_seconds.toFixed(1)}s` 
            : 'unknown duration';
        const formatText = result.format || 'video';
        const isImageFormat = this.isImageVideoFormat(result.format);
        
        videoDiv.innerHTML = `
            <div class="message-content">
                <div class="generated-video-container">
                    ${isImageFormat ? `
                        <img src="${result.file_path}" 
                             alt="Generated animation" 
                             class="generated-video"
                             style="max-width: 100%; max-height: 600px; border-radius: 8px;"
                             loading="lazy">
                    ` : `
                        <video controls preload="metadata" class="generated-video" style="max-width: 100%; max-height: 600px; border-radius: 8px;">
                            <source src="${result.file_path}" type="${this.getVideoMimeType(result.format)}">
                            Your browser does not support video playback.
                        </video>
                    `}
                    <div class="video-actions mt-2">
                        <a href="${result.file_path}" 
                           download 
                           class="btn btn-sm btn-outline-secondary"
                           title="Download">
                            <i class="bi bi-download"></i> Download
                        </a>
                        ${result.prompt ? `
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.showVideoPrompt('${this.escapeHtml(result.prompt).replace(/'/g, '\\&apos;')}')"
                                title="View prompt">
                            <i class="bi bi-chat-left-text"></i> Prompt
                        </button>
                        ` : ''}
                    </div>
                    <small class="text-muted d-block mt-2">
                        ${formatText.toUpperCase()} • ${durationText}
                        ${result.generation_time ? ` • Generated in ${result.generation_time.toFixed(1)}s` : ''}
                    </small>
                </div>
            </div>
        `;
        
        container.appendChild(videoDiv);
        this.scrollToBottom();
    },
    
    showVideoPrompt(prompt) {
        // Decode HTML entities
        const textarea = document.createElement('textarea');
        textarea.innerHTML = prompt;
        const decodedPrompt = textarea.value;
        
        // Show modal with prompt
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog-centered modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Video Generation Prompt</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-info mb-3">
                            <i class="bi bi-info-circle me-2"></i>
                            This is the motion-focused prompt that was used to create the video.
                        </div>
                        <div class="form-group">
                            <textarea class="form-control" rows="10" readonly>${this.escapeHtml(decodedPrompt)}</textarea>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    },

    appendSceneCaptureVideo(message) {
        const container = document.getElementById('messagesContainer');
        
        const videoDiv = document.createElement('div');
        videoDiv.className = 'message assistant-message video-only-message mt-3';
        videoDiv.setAttribute('data-message-id', message.id);
        const isImageFormat = this.isImageVideoFormat(message.metadata.format);
        
        videoDiv.innerHTML = `
            <div class="message-content">
                <div class="generated-video-container">
                    ${isImageFormat ? `
                        <img src="${message.metadata.video_path}" 
                             alt="Scene capture animation" 
                             class="generated-video"
                             style="max-width: 100%; max-height: 600px; border-radius: 8px;"
                             loading="lazy">
                    ` : `
                        <video controls preload="metadata" class="generated-video" style="max-width: 100%; max-height: 600px; border-radius: 8px;">
                            <source src="${message.metadata.video_path}" type="${this.getVideoMimeType(message.metadata.format)}">
                            Your browser does not support the video tag.
                        </video>
                    `}
                    <span class="badge bg-info scene-capture-badge">
                        <i class="bi bi-camera-video-fill"></i> Scene Capture
                    </span>
                    <div class="video-actions">
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.viewInlineVideo('${message.metadata.video_path}', '${this.escapeHtml(message.metadata.prompt || '').replace(/'/g, '\\&apos;')}', true, '${message.metadata.format || 'mp4'}')"
                                title="View fullscreen">
                            <i class="bi bi-arrows-fullscreen"></i>
                        </button>
                        <a href="${message.metadata.video_path}" 
                           download 
                           class="btn btn-sm btn-outline-secondary"
                           title="Download">
                            <i class="bi bi-download"></i>
                        </a>
                        ${message.metadata.prompt ? `
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="UI.showVideoPrompt('${this.escapeHtml(message.metadata.prompt).replace(/'/g, '\\&apos;')}')"
                                title="View prompt">
                            <i class="bi bi-chat-left-text"></i>
                        </button>
                        ` : ''}
                    </div>
                    ${message.metadata.generation_time ? `
                        <small class="text-muted d-block mt-2">
                            Generated in ${message.metadata.generation_time.toFixed(1)}s
                        </small>
                    ` : ''}
                    ${message.metadata.duration ? `
                        <small class="text-muted d-block">
                            Duration: ${message.metadata.duration.toFixed(1)}s • ${(message.metadata.format || 'video').toUpperCase()}
                        </small>
                    ` : ''}
                </div>
            </div>
        `;
        
        container.appendChild(videoDiv);
        this.scrollToBottom();
    },

    showImagePrompt(prompt) {
        // Decode HTML entities
        const textarea = document.createElement('textarea');
        textarea.innerHTML = prompt;
        const decodedPrompt = textarea.value;
        
        // Show modal with prompt
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog-centered modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Image Generation Prompt</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-info mb-3">
                            <i class="bi bi-info-circle me-2"></i>
                            This is the AI-generated prompt that was used to create the image.
                        </div>
                        <div class="form-group">
                            <textarea class="form-control" rows="10" readonly>${this.escapeHtml(decodedPrompt)}</textarea>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="navigator.clipboard.writeText(this.closest('.modal').querySelector('textarea').value); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy to Clipboard', 2000);">Copy to Clipboard</button>
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
    
    isImageVideoFormat(format) {
        // Determine if format is an image-based animation (should use img tag, not video tag)
        const formatLower = (format || '').toLowerCase().replace('.', '');
        return formatLower === 'webp' || formatLower === 'gif';
    },

    getVideoMimeType(format) {
        // Handle file formats and return proper MIME types
        const formatLower = (format || 'mp4').toLowerCase().replace('.', '');
        const mimeTypes = {
            'webp': 'image/webp',  // Animated WebP
            'gif': 'image/gif',
            'mp4': 'video/mp4',
            'webm': 'video/webm',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime',
            'mkv': 'video/x-matroska'
        };
        return mimeTypes[formatLower] || `video/${formatLower}`;
    },
    
    showVideoPrompt(prompt) {
        // Decode HTML entities
        const textarea = document.createElement('textarea');
        textarea.innerHTML = prompt;
        const decodedPrompt = textarea.value;
        
        // Show modal with prompt
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog-centered modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Video Generation Prompt</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-info mb-3">
                            <i class="bi bi-info-circle me-2"></i>
                            This is the AI-generated prompt that was used to create the video.
                        </div>
                        <div class="form-group">
                            <textarea class="form-control" rows="10" readonly>${this.escapeHtml(decodedPrompt)}</textarea>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="navigator.clipboard.writeText(this.closest('.modal').querySelector('textarea').value); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy to Clipboard', 2000);">Copy to Clipboard</button>
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
     * View inline image in modal (same as gallery)
     */
    viewInlineImage(imagePath, prompt = '', isSceneCapture = false) {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-xl modal-dialog-centered">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">
                            ${isSceneCapture ? '<i class="bi bi-camera-fill me-2"></i>Scene Capture' : '<i class="bi bi-image-fill me-2"></i>Generated Image'}
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img src="${imagePath}" class="img-fluid rounded" alt="Full image" style="max-height: 70vh;">
                        ${prompt ? `<p class="modal-label mt-3 small"><strong>Prompt:</strong> ${prompt}</p>` : ''}
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-outline-warning btn-sm" onclick="App.setAsProfileImage('${imagePath}')">
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

    viewInlineVideo(videoPath, prompt = '', isSceneCapture = false, format = 'mp4') {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-xl modal-dialog-centered">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">
                            ${isSceneCapture ? '<i class="bi bi-camera-video-fill me-2"></i>Scene Capture' : '<i class="bi bi-film me-2"></i>Generated Video'}
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        ${this.isImageVideoFormat(format) ? `
                            <img src="${videoPath}" class="rounded" style="max-width: 100%; max-height: 70vh;" alt="Generated animation">
                        ` : `
                            <video controls autoplay class="rounded" style="max-width: 100%; max-height: 70vh;">
                                <source src="${videoPath}" type="${this.getVideoMimeType(format)}">
                                Your browser does not support the video tag.
                            </video>
                        `}
                        ${prompt ? `<p class="modal-label mt-3 small"><strong>Prompt:</strong> ${prompt}</p>` : ''}
                    </div>
                    <div class="modal-footer border-secondary">
                        <a href="${videoPath}" download class="btn btn-outline-secondary btn-sm">
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
     * Task 1.9: Show image with vision details in modal
     */
    async showImageModal(attachmentId) {
        try {
            // Fetch attachment details
            const response = await fetch(`/api/attachments/${attachmentId}`);
            if (!response.ok) {
                throw new Error('Failed to load image details');
            }
            
            const attachment = await response.json();
            
            // Show modal
            const modal = document.getElementById('imageModal');
            modal.classList.add('active');
            
            // Set image
            const modalImage = document.getElementById('modalImage');
            modalImage.src = `/api/attachments/${attachmentId}/file`;
            
            // Set title
            document.getElementById('imageModalTitle').textContent = attachment.original_filename || 'Image';
            
            // Set observation
            const observation = attachment.vision_observation || 'No vision analysis available';
            document.getElementById('modalObservation').textContent = observation;
            
            // Set confidence
            const confidence = attachment.vision_confidence;
            if (confidence !== null && confidence !== undefined) {
                const confidencePercent = (confidence * 100).toFixed(1);
                document.getElementById('modalConfidence').textContent = `${confidencePercent}%`;
                document.getElementById('modalConfidenceBar').style.width = `${confidencePercent}%`;
            } else {
                document.getElementById('modalConfidence').textContent = 'N/A';
                document.getElementById('modalConfidenceBar').style.width = '0%';
            }
            
            // Set model and backend
            document.getElementById('modalModel').textContent = attachment.vision_model || 'N/A';
            document.getElementById('modalBackend').textContent = attachment.vision_backend || 'N/A';
            
            // Set processing time
            const processingTime = attachment.vision_processing_time_ms;
            if (processingTime) {
                const seconds = (processingTime / 1000).toFixed(2);
                document.getElementById('modalProcessingTime').textContent = `${seconds}s`;
            } else {
                document.getElementById('modalProcessingTime').textContent = 'N/A';
            }
            
            // Set tags
            const tagsContainer = document.getElementById('modalTags');
            const tagsGroup = document.getElementById('modalTagsGroup');
            if (attachment.vision_tags) {
                const tags = attachment.vision_tags.split(',').filter(t => t.trim());
                if (tags.length > 0) {
                    tagsContainer.innerHTML = tags.map(tag => 
                        `<span class="vision-tag">${this.escapeHtml(tag.trim())}</span>`
                    ).join('');
                    tagsGroup.style.display = 'block';
                } else {
                    tagsGroup.style.display = 'none';
                }
            } else {
                tagsGroup.style.display = 'none';
            }
            
            // Set file info
            document.getElementById('modalFilename').textContent = attachment.original_filename || 'Unknown';
            
            // Format file size
            if (attachment.file_size) {
                const size = attachment.file_size;
                const formatted = size < 1024 ? `${size} B` :
                    size < 1024 * 1024 ? `${(size / 1024).toFixed(1)} KB` :
                    `${(size / (1024 * 1024)).toFixed(2)} MB`;
                document.getElementById('modalFileSize').textContent = formatted;
            } else {
                document.getElementById('modalFileSize').textContent = 'Unknown';
            }
            
            // Set dimensions
            if (attachment.width && attachment.height) {
                document.getElementById('modalDimensions').textContent = 
                    `${attachment.width} × ${attachment.height}`;
            } else {
                document.getElementById('modalDimensions').textContent = 'Unknown';
            }
            
        } catch (error) {
            console.error('Error showing image modal:', error);
            this.showToast('Failed to load image details', 'error');
        }
    },
    
    /**
     * Task 1.9: Close image modal
     */
    closeImageModal() {
        const modal = document.getElementById('imageModal');
        modal.classList.remove('active');
    }
};

// Task 1.9: Close modal with Escape key or click outside
document.addEventListener('DOMContentLoaded', () => {
    const imageModal = document.getElementById('imageModal');
    
    if (imageModal) {
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && imageModal.classList.contains('active')) {
                UI.closeImageModal();
            }
        });
        
        // Close on click outside modal content
        imageModal.addEventListener('click', (e) => {
            if (e.target === imageModal) {
                UI.closeImageModal();
            }
        });
    }
});