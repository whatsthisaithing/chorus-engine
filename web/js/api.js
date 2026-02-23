/**
 * API Client for Chorus Engine
 * Handles all HTTP requests to the backend
 */

const API_BASE_URL = window.location.origin;

class API {
    static _detailToMessage(detail, fallback = null) {
        if (detail == null) return fallback || 'Unknown error';
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) {
            const parts = detail
                .map((item) => this._detailToMessage(item, ''))
                .filter((item) => typeof item === 'string' && item.trim().length > 0);
            return parts.join('; ') || (fallback || 'Unknown error');
        }
        if (typeof detail === 'object') {
            const stopReason = (detail.stop_reason || '').toString().trim();
            const errorCode = (detail.error || '').toString().trim();
            if (errorCode && stopReason) return `${errorCode}: ${stopReason}`;
            if (detail.message && typeof detail.message === 'string') return detail.message;
            if (stopReason) return stopReason;
            if (errorCode) return errorCode;
            try {
                return JSON.stringify(detail);
            } catch (_) {
                return fallback || 'Unknown error';
            }
        }
        return String(detail);
    }

    static _isConfigMutation(endpoint, method) {
        const m = (method || 'GET').toUpperCase();
        if (!['POST', 'PUT', 'PATCH', 'DELETE'].includes(m)) return false;
        return (
            endpoint.startsWith('/system/config') ||
            endpoint.startsWith('/system/user-identity') ||
            endpoint.startsWith('/config/system/import') ||
            endpoint.startsWith('/characters') ||
            endpoint.includes('/privacy') ||
            endpoint.includes('/media-offers') ||
            endpoint.includes('/tts') ||
            endpoint.includes('/workflows')
        );
    }

    /**
     * Make a fetch request with error handling
     */
    static async request(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
                ...options,
            });
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                const detailMessage = this._detailToMessage(
                    error.detail,
                    `HTTP ${response.status}: ${response.statusText}`
                );
                throw new Error(detailMessage);
            }
            
            const data = await response.json();
            if (this._isConfigMutation(endpoint, options.method) && window.App && typeof window.App.onPotentialConfigMutation === 'function') {
                try {
                    window.App.onPotentialConfigMutation(endpoint, options.method || 'GET', data);
                } catch (_) {}
            }
            return data;
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }
    
    // === Health & Characters ===
    
    static async getHealth() {
        return this.request('/health');
    }
    
    static async listCharacters() {
        return this.request('/characters');
    }
    
    static async getCharacter(characterId) {
        return this.request(`/characters/${characterId}`);
    }
    
    static async getCharacterStats(characterId) {
        return this.request(`/characters/${characterId}/stats`);
    }
    
    static async setCharacterProfileImage(characterId, imageFilename) {
        return this.request(`/characters/${characterId}/profile-image`, {
            method: 'POST',
            body: JSON.stringify({ image_filename: imageFilename }),
        });
    }
    
    static async getCharacterImmersionNotice(characterId) {
        return this.request(`/characters/${characterId}/immersion-notice`);
    }
    
    static async createCharacter(characterData) {
        return this.request('/characters', {
            method: 'POST',
            body: JSON.stringify(characterData),
        });
    }
    
    static async updateCharacter(characterId, updates) {
        return this.request(`/characters/${characterId}`, {
            method: 'PATCH',
            body: JSON.stringify(updates),
        });
    }
    
    static async deleteCharacter(characterId) {
        return this.request(`/characters/${characterId}`, {
            method: 'DELETE',
        });
    }
    
    static async cloneCharacter(sourceId, newId) {
        return this.request(`/characters/${sourceId}/clone?new_id=${encodeURIComponent(newId)}`, {
            method: 'POST',
        });
    }
    
    // === Conversations ===
    
    static async createConversation(characterId, title = null, source = 'web', primaryUser = null) {
        return this.request('/conversations', {
            method: 'POST',
            body: JSON.stringify({
                character_id: characterId,
                title: title,
                source: source,
                primary_user: primaryUser,
                conversation_kind: 'standard',
            }),
        });
    }

    static async resolveGeneralChat(characterId) {
        return this.request(`/characters/${characterId}/general-chat`, {
            method: 'POST',
        });
    }

    static async getContinuityPreview(characterId, conversationId) {
        return this.request(`/continuity/preview?character_id=${characterId}&conversation_id=${conversationId}`);
    }

    static async setContinuityChoice(conversationId, mode, rememberChoice = false) {
        return this.request('/continuity/choice', {
            method: 'POST',
            body: JSON.stringify({
                conversation_id: conversationId,
                mode: mode,
                remember_choice: rememberChoice
            }),
        });
    }

    static async refreshContinuity(characterId, force = false) {
        return this.request('/continuity/refresh', {
            method: 'POST',
            body: JSON.stringify({
                character_id: characterId,
                force: force
            }),
        });
    }
    
    static async listConversations(characterId = null, skip = 0, limit = 100, source = 'web', conversationKind = 'standard') {
        const params = new URLSearchParams();
        if (characterId) params.append('character_id', characterId);
        if (skip) params.append('skip', skip);
        if (limit) params.append('limit', limit);
        if (source) params.append('source', source);  // Filter by source (web, discord, all)
        if (conversationKind) params.append('conversation_kind', conversationKind);
        
        return this.request(`/conversations?${params}`);
    }
    
    static async searchConversations(characterId, query, limit = 10, source = null) {
        const params = new URLSearchParams({
            character_id: characterId,
            query: query,
            limit: limit
        });
        if (source) params.append('source', source);
        
        return this.request(`/conversations/search?${params}`);
    }
    
    static async getConversation(conversationId) {
        return this.request(`/conversations/${conversationId}`);
    }

    static async getInteractiveNarrativeSession(conversationId) {
        return this.request(`/conversations/${conversationId}/interactive-narrative/session`);
    }

    static async createInteractiveNarrativeSession(conversationId, payload = {}) {
        return this.request(`/conversations/${conversationId}/interactive-narrative/session`, {
            method: 'POST',
            body: JSON.stringify(payload || {}),
        });
    }

    static async pauseInteractiveNarrative(loopId) {
        return this.request(`/interactive-narrative/${loopId}/pause`, {
            method: 'POST',
        });
    }

    static async resumeInteractiveNarrative(loopId) {
        return this.request(`/interactive-narrative/${loopId}/resume`, {
            method: 'POST',
        });
    }

    static async tickInteractiveNarrative(loopId) {
        return this.request(`/interactive-narrative/${loopId}/tick`, {
            method: 'POST',
        });
    }

    static async listConversationSegments(conversationId, skip = 0, limit = 500) {
        const params = new URLSearchParams();
        if (skip) params.append('skip', skip);
        if (limit) params.append('limit', limit);
        return this.request(`/conversations/${conversationId}/segments?${params}`);
    }
    
    static async updateConversation(conversationId, title) {
        return this.request(`/conversations/${conversationId}`, {
            method: 'PUT',
            body: JSON.stringify({ title }),
        });
    }
    
    static async deleteConversation(conversationId, deleteMemories = false) {
        const params = new URLSearchParams({ delete_memories: deleteMemories });
        return this.request(`/conversations/${conversationId}?${params}`, {
            method: 'DELETE',
        });
    }
    
    static async getConversationPrivacy(conversationId) {
        return this.request(`/conversations/${conversationId}/privacy`);
    }
    
    static async setConversationPrivacy(conversationId, isPrivate) {
        return this.request(`/conversations/${conversationId}/privacy`, {
            method: 'PUT',
            body: JSON.stringify({ is_private: isPrivate }),
        });
    }

    static async getConversationMediaOffers(conversationId) {
        return this.request(`/conversations/${conversationId}/media-offers`);
    }

    static async updateConversationMediaOffers(conversationId, updates) {
        return this.request(`/conversations/${conversationId}/media-offers`, {
            method: 'PATCH',
            body: JSON.stringify(updates),
        });
    }
    
    static async exportConversation(conversationId, format = 'markdown', includeMetadata = true, includeSummary = true, includeMemories = false) {
        // Build query parameters
        const params = new URLSearchParams({
            format,
            include_metadata: includeMetadata,
            include_summary: includeSummary,
            include_memories: includeMemories
        });
        
        // Fetch the file
        const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/export?${params}`, {
            method: 'GET',
        });
        
        if (!response.ok) {
            throw new Error(`Export failed: ${response.statusText}`);
        }
        
        // Get filename from Content-Disposition header
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `conversation_${conversationId}.${format === 'markdown' ? 'md' : 'txt'}`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+?)"?$/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }
        
        // Get the blob
        const blob = await response.blob();
        
        return { blob, filename };
    }

    static async analyzeConversation(conversationId, force = false) {
        return this.request(`/conversations/${conversationId}/analyze`, {
            method: 'POST',
            body: JSON.stringify({ force }),
        });
    }

    static async getConversationAnalyses(conversationId, includeMemories = false) {
        const params = includeMemories ? '?include_memories=true' : '';
        return this.request(`/conversations/${conversationId}/analyses${params}`);
    }
    
    // === Database Management ===
    
    static async resetDatabase() {
        return this.request('/reset', {
            method: 'POST',
        });
    }
    
    // === Threads ===
    
    static async createThread(conversationId, title = 'New Thread') {
        return this.request(`/conversations/${conversationId}/threads`, {
            method: 'POST',
            body: JSON.stringify({ title }),
        });
    }
    
    static async listThreads(conversationId) {
        return this.request(`/conversations/${conversationId}/threads`);
    }
    
    static async getThread(threadId) {
        return this.request(`/threads/${threadId}`);
    }
    
    static async updateThread(threadId, title) {
        return this.request(`/threads/${threadId}`, {
            method: 'PATCH',
            body: JSON.stringify({ title }),
        });
    }
    
    static async deleteThread(threadId) {
        return this.request(`/threads/${threadId}`, {
            method: 'DELETE',
        });
    }
    
    // === Messages ===
    
    static async listMessages(threadId, skip = 0, limit = 1000) {
        return this.request(`/threads/${threadId}/messages?skip=${skip}&limit=${limit}`);
    }
    
    static async softDeleteMessages(threadId, messageIds) {
        return this.request(`/threads/${threadId}/messages/soft-delete`, {
            method: 'POST',
            body: JSON.stringify({ message_ids: messageIds }),
        });
    }

    static createClientMessageId() {
        if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
            return crypto.randomUUID();
        }
        return `cmid_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
    }

    static _buildMessageMetadata(extraMetadata = null) {
        const metadata = {};
        if (typeof userManager !== 'undefined') {
            Object.assign(metadata, userManager.getUserMetadata());
        }
        if (extraMetadata && typeof extraMetadata === 'object') {
            Object.assign(metadata, extraMetadata);
        }
        return metadata;
    }

    static buildNonStreamMessagePayload(message, attachmentId = null, options = {}) {
        const payload = { message };
        if (attachmentId) {
            payload.image_attachment_ids = Array.isArray(attachmentId) ? attachmentId : [attachmentId];
        }

        const metadata = this._buildMessageMetadata(options.metadata || null);
        if (options.clientMessageId) {
            metadata.client_message_id = options.clientMessageId;
        }
        if (Object.keys(metadata).length > 0) {
            payload.metadata = metadata;
        }

        if (typeof userManager !== 'undefined') {
            payload.primary_user = userManager.getUsername();
            payload.conversation_source = 'web';
        }

        return payload;
    }

    static buildNonStreamMessageRequest(threadId, payload) {
        return {
            url: `/threads/${threadId}/messages`,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.parse(JSON.stringify(payload)),
            client_message_id: (payload.metadata || {}).client_message_id || null,
        };
    }

    static async sendMessageWithPayload(threadId, payload) {
        const requestSnapshot = this.buildNonStreamMessageRequest(threadId, payload);
        const response = await this.request(requestSnapshot.url, {
            method: requestSnapshot.method,
            headers: requestSnapshot.headers,
            body: JSON.stringify(requestSnapshot.body),
        });
        return { response, requestSnapshot };
    }

    static async replayNonStreamAttempt(requestSnapshot) {
        if (!requestSnapshot || !requestSnapshot.url || !requestSnapshot.method) {
            throw new Error('No stored send attempt available to replay');
        }
        const response = await this.request(requestSnapshot.url, {
            method: requestSnapshot.method,
            headers: requestSnapshot.headers || { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestSnapshot.body || {}),
        });
        return response;
    }

    static async sendMessage(threadId, message, attachmentId = null) {
        const payload = this.buildNonStreamMessagePayload(message, attachmentId);
        const result = await this.sendMessageWithPayload(threadId, payload);
        return result.response;
    }
    
    /**
     * Send a message and stream the response
     * @param {string} threadId - Thread ID
     * @param {string} message - Message content
     * @param {function} onChunk - Callback for each content chunk
     * @param {function} onComplete - Callback when streaming completes
     * @param {function} onError - Callback for errors
     */
    static async sendMessageStream(threadId, message, onChunk, onComplete, onError, attachmentId = null) {
        const url = `${API_BASE_URL}/threads/${threadId}/messages/stream`;
        
        // Build payload with user metadata
        const payload = this.buildNonStreamMessagePayload(message, attachmentId);
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'user_message') {
                                // Backend echoes user message with attachment data - update if needed
                                if (onChunk.userMessageCallback) {
                                    onChunk.userMessageCallback(data);
                                }
                            } else if (data.type === 'content') {
                                onChunk(data.content);
                            } else if (data.type === 'tool_calls') {
                                if (onChunk.toolCallsCallback) {
                                    onChunk.toolCallsCallback(data.tool_calls || []);
                                }
                            } else if (data.type === 'title_updated') {
                                // New title was auto-generated
                                if (onChunk.titleCallback) {
                                    onChunk.titleCallback(data.title);
                                }
                            } else if (data.type === 'done') {
                                // Also check for title update in done message
                                if (data.conversation_title_updated && onChunk.titleCallback) {
                                    onChunk.titleCallback(data.conversation_title_updated);
                                }
                                onComplete(data.message_id, data.normalized_content);
                                return;
                            } else if (data.type === 'error') {
                                onError(new Error(data.error));
                                return;
                            }
                        } catch (e) {
                            console.error('Failed to parse SSE data:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            onError(error);
        }
    }
    
    // === Memories ===
    
    static async createMemory(conversationId, memoryData) {
        return this.request(`/conversations/${conversationId}/memories`, {
            method: 'POST',
            body: JSON.stringify(memoryData),
        });
    }
    
    static async listMemories(conversationId) {
        return this.request(`/conversations/${conversationId}/memories`);
    }

    // === Moment Pins ===

    static async branchConversation(conversationId, selectedMessageIds) {
        return this.request(`/conversations/${conversationId}/branch`, {
            method: 'POST',
            body: JSON.stringify({ selected_message_ids: selectedMessageIds }),
        });
    }

    static async createMomentPin(conversationId, selectedMessageIds) {
        return this.request(`/conversations/${conversationId}/moment-pins`, {
            method: 'POST',
            body: JSON.stringify({ selected_message_ids: selectedMessageIds }),
        });
    }

    static async listMomentPins(conversationId) {
        return this.request(`/conversations/${conversationId}/moment-pins`);
    }

    static async listCharacterMomentPins(characterId, options = {}) {
        const params = new URLSearchParams();
        if (options.conversation_id) {
            params.set('conversation_id', options.conversation_id);
        }
        if (typeof options.include_archived === 'boolean') {
            params.set('include_archived', String(options.include_archived));
        }
        const query = params.toString();
        return this.request(`/characters/${characterId}/moment-pins${query ? `?${query}` : ''}`);
    }

    static async getMomentPin(pinId) {
        return this.request(`/moment-pins/${pinId}`);
    }

    static async updateMomentPin(pinId, updates) {
        return this.request(`/moment-pins/${pinId}`, {
            method: 'PATCH',
            body: JSON.stringify(updates),
        });
    }

    static async deleteMomentPin(pinId) {
        return this.request(`/moment-pins/${pinId}`, {
            method: 'DELETE',
        });
    }
    
    static async deleteMemory(memoryId) {
        return this.request(`/memories/${memoryId}`, {
            method: 'DELETE',
        });
    }

    static async updateMemory(memoryId, updates) {
        return this.request(`/memories/${memoryId}`, {
            method: 'PATCH',
            body: JSON.stringify(updates),
        });
    }
    
    static async getCharacterMemories(characterId, memoryType = null, source = null) {
        const params = new URLSearchParams();
        if (memoryType) params.append('memory_type', memoryType);
        if (source) params.append('source', source);
        const queryString = params.toString() ? `?${params.toString()}` : '';
        return this.request(`/characters/${characterId}/memories${queryString}`);
    }
    
    static async createCoreMemory(characterId, memoryData) {
        return this.request(`/characters/${characterId}/core-memories`, {
            method: 'POST',
            body: JSON.stringify(memoryData),
        });
    }
    
    static async searchMemories(searchData) {
        return this.request('/memories/search', {
            method: 'POST',
            body: JSON.stringify(searchData),
        });
    }
    
    static async getCharacterMemoryStats(characterId) {
        return this.request(`/characters/${characterId}/memory-stats`);
    }
    
    // === Phase 5: Image Generation ===
    
    static async generateImage(threadId, prompt, negativePrompt = null, disableConfirmation = false, workflowId = null, toolCallId = null) {
        const body = {
            prompt: prompt,
            negative_prompt: negativePrompt,
            disable_future_confirmations: disableConfirmation
        };
        if (toolCallId) {
            body.tool_call_id = toolCallId;
        }
        
        if (workflowId) {
            body.workflow_id = workflowId;
        }
        
        return this.request(`/threads/${threadId}/generate-image`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
    }
    
    // Phase 9: Scene Capture
    static async captureScene(threadId, prompt, negativePrompt = null, workflowId = null, toolCallId = null) {
        const body = {
            prompt: prompt,
            negative_prompt: negativePrompt
        };
        if (toolCallId) {
            body.tool_call_id = toolCallId;
        }
        
        if (workflowId) {
            body.workflow_id = workflowId;
        }
        
        return this.request(`/threads/${threadId}/capture-scene`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
    }
    
    static async getConversationImages(conversationId) {
        return this.request(`/conversations/${conversationId}/images`);
    }
    
    static async deleteImage(imageId) {
        return this.request(`/images/${imageId}`, {
            method: 'DELETE',
        });
    }
    
    static async deleteVideo(videoId) {
        return this.request(`/videos/${videoId}`, {
            method: 'DELETE',
        });
    }
    
    // === Workflow Management ===
    
    static async listWorkflows(characterId) {
        return this.request(`/characters/${characterId}/workflows`);
    }
    
    static async uploadWorkflow(characterId, workflowName, workflowData, workflowType = 'image') {
        return this.request(`/characters/${characterId}/workflows/${workflowName}?workflow_type=${workflowType}`, {
            method: 'POST',
            body: JSON.stringify(workflowData),
        });
    }
    
    static async deleteWorkflow(characterId, workflowName) {
        return this.request(`/characters/${characterId}/workflows/${workflowName}`, {
            method: 'DELETE',
        });
    }
    
    static async renameWorkflow(characterId, oldName, newName) {
        return this.request(`/characters/${characterId}/workflows/${encodeURIComponent(oldName)}/rename?new_name=${encodeURIComponent(newName)}`, {
            method: 'PUT',
        });
    }
    
    static async setDefaultWorkflow(characterId, workflowName) {
        return this.request(`/characters/${characterId}/default-workflow?workflow_name=${encodeURIComponent(workflowName)}`, {
            method: 'PUT',
        });
    }
    
    static async updateWorkflowConfig(workflowId, config) {
        return this.request(`/workflows/${workflowId}/config`, {
            method: 'PUT',
            body: JSON.stringify(config),
        });
    }
    
    // === Phase 6: Audio Generation ===
    
    static async generateMessageAudio(conversationId, messageId, workflowName = null) {
        return this.request(`/conversations/${conversationId}/messages/${messageId}/audio`, {
            method: 'POST',
            body: JSON.stringify({ workflow_name: workflowName }),
        });
    }
    
    static async getMessageAudio(conversationId, messageId) {
        return this.request(`/conversations/${conversationId}/messages/${messageId}/audio`);
    }
    
    static async deleteMessageAudio(conversationId, messageId) {
        return this.request(`/conversations/${conversationId}/messages/${messageId}/audio`, {
            method: 'DELETE',
        });
    }
    
    static async updateConversationTTS(conversationId, enabled) {
        return this.request(`/conversations/${conversationId}/tts`, {
            method: 'PATCH',
            body: JSON.stringify({ enabled }),
        });
    }
    
    static async getConversationTTS(conversationId) {
        return this.request(`/conversations/${conversationId}/tts`);
    }
    
    static async uploadVoiceSample(characterId, file, transcript, isDefault = false) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('transcript', transcript);
        formData.append('is_default', isDefault.toString());
        
        return fetch(`${API_BASE_URL}/characters/${characterId}/voice-samples`, {
            method: 'POST',
            body: formData,
        }).then(async response => {
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            return response.json();
        });
    }
    
    static async listVoiceSamples(characterId) {
        return this.request(`/characters/${characterId}/voice-samples`);
    }
    
    static async updateVoiceSample(characterId, sampleId, data) {
        return this.request(`/characters/${characterId}/voice-samples/${sampleId}`, {
            method: 'PATCH',
            body: JSON.stringify(data),
        });
    }
    
    static async deleteVoiceSample(characterId, sampleId) {
        return this.request(`/characters/${characterId}/voice-samples/${sampleId}`, {
            method: 'DELETE',
        });
    }
    
    // === Export/Import ===
    
    static async backupCharacter(characterId, includeWorkflows = true, notes = null) {
        const params = new URLSearchParams({ include_workflows: includeWorkflows });
        if (notes) {
            params.append('notes', notes);
        }
        
        const url = `${API_BASE_URL}/characters/${characterId}/backup?${params}`;
        const response = await fetch(url, { method: 'POST' });
        
        if (!response.ok) {
            throw new Error(`Failed to backup character: ${response.statusText}`);
        }
        
        // Return blob and headers for metadata
        const blob = await response.blob();
        const size = response.headers.get('X-Backup-Size');
        const sizeMB = response.headers.get('X-Backup-Size-MB');
        
        return { blob, size, sizeMB };
    }
    
    /**
     * Restore character from backup file
     */
    static async restoreCharacter(file, newCharacterId = null, renameIfExists = false, cleanupOrphans = false, overwrite = false) {
        const formData = new FormData();
        formData.append('file', file);
        
        const params = new URLSearchParams({ 
            rename_if_exists: renameIfExists,
            cleanup_orphans: cleanupOrphans,
            overwrite: overwrite
        });
        if (newCharacterId) {
            params.append('new_character_id', newCharacterId);
        }
        
        const url = `${API_BASE_URL}/characters/restore?${params}`;
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `Failed to restore character: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    static async exportCharacter(characterId) {
        const url = `${API_BASE_URL}/characters/${characterId}/export`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to export character: ${response.statusText}`);
        }
        return response.blob();
    }
    
    static async importCharacter(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/characters/import`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        if (window.App && typeof window.App.onPotentialConfigMutation === 'function') {
            try {
                window.App.onPotentialConfigMutation('/characters/import', 'POST', data);
            } catch (_) {}
        }
        return data;
    }
    
    static async exportSystemConfig() {
        const url = `${API_BASE_URL}/config/system/export`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to export system config: ${response.statusText}`);
        }
        return response.blob();
    }
    
    static async importSystemConfig(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/config/system/import`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        if (window.App && typeof window.App.onPotentialConfigMutation === 'function') {
            try {
                window.App.onPotentialConfigMutation('/config/system/import', 'POST', data);
            } catch (_) {}
        }
        return data;
    }

    // === User Identity ===
    
    static async getUserIdentity() {
        return this.request('/system/user-identity');
    }
    
    static async updateUserIdentity(identity) {
        return this.request('/system/user-identity', {
            method: 'PUT',
            body: JSON.stringify(identity),
        });
    }

    static async getConfigDrift() {
        return this.request('/config/drift');
    }

    static async reloadSystemConfig() {
        return this.request('/system/config/reload', { method: 'POST' });
    }

    static async reloadCharacters() {
        return this.request('/characters/reload', { method: 'POST' });
    }
    
    // === Logs ===
    
    static async getServerLogs(lines = 500) {
        return this.request(`/logs/server?lines=${lines}`);
    }
    
    static async listConversationLogs() {
        return this.request('/logs/conversations');
    }
    
    static async getConversationLog(conversationId, options = {}) {
        const params = new URLSearchParams();
        params.append('prettify', 'true');
        if (options.file) params.append('file', options.file);
        if (options.date) params.append('date', options.date);
        return this.request(`/logs/conversations/${conversationId}?${params.toString()}`);
    }
    
    static async getExtractionLog(conversationId) {
        return this.request(`/logs/extractions/${conversationId}?prettify=true`);
    }
    
    static async getImageRequestLog(conversationId) {
        return this.request(`/logs/image-prompts/${conversationId}?prettify=true`);
    }
    
    static async deleteImage(imageId) {
        return this.request(`/images/${imageId}`, {
            method: 'DELETE'
        });
    }
    
    // === Video Generation ===
    
    static async generateVideo(threadId, prompt, negativePrompt, disableFutureConfirmations, workflowId = null, toolCallId = null) {
        const body = {
            prompt,
            negative_prompt: negativePrompt,
            disable_future_confirmations: disableFutureConfirmations,
            workflow_id: workflowId
        };
        if (toolCallId) {
            body.tool_call_id = toolCallId;
        }
        return this.request(`/threads/${threadId}/generate-video`, {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }
    
    static async captureVideoScenePrompt(threadId) {
        return this.request(`/threads/${threadId}/capture-video-scene-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
    }
    
    static async captureVideoScene(threadId, prompt, negativePrompt = null, workflowId = null, toolCallId = null) {
        const body = {
            prompt: prompt,
            negative_prompt: negativePrompt
        };
        if (toolCallId) {
            body.tool_call_id = toolCallId;
        }
        
        if (workflowId) {
            body.workflow_id = workflowId;
        }
        
        return this.request(`/threads/${threadId}/capture-video-scene`, {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }
    
    static async getConversationVideos(conversationId) {
        return this.request(`/conversations/${conversationId}/videos`);
    }
    
    static async deleteVideo(videoId) {
        return this.request(`/videos/${videoId}`, {
            method: 'DELETE'
        });
    }
    
    static async getIntentDetectionLogs(lines = 100) {
        return this.request(`/logs/intent-detection?lines=${lines}`);
    }}
