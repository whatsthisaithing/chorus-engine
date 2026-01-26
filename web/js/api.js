/**
 * API Client for Chorus Engine
 * Handles all HTTP requests to the backend
 */

const API_BASE_URL = window.location.origin;

class API {
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
                throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
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
    
    static async createConversation(characterId, title = null) {
        return this.request('/conversations', {
            method: 'POST',
            body: JSON.stringify({ character_id: characterId, title }),
        });
    }
    
    static async listConversations(characterId = null, skip = 0, limit = 100, source = 'web') {
        const params = new URLSearchParams();
        if (characterId) params.append('character_id', characterId);
        if (skip) params.append('skip', skip);
        if (limit) params.append('limit', limit);
        if (source) params.append('source', source);  // Filter by source (web, discord, all)
        
        return this.request(`/conversations?${params}`);
    }
    
    static async getConversation(conversationId) {
        return this.request(`/conversations/${conversationId}`);
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
    
    static async sendMessage(threadId, message) {
        // Get user metadata if UserManager is available
        const payload = { message };
        
        if (typeof userManager !== 'undefined') {
            payload.metadata = userManager.getUserMetadata();
            payload.primary_user = userManager.getUsername();
            payload.conversation_source = 'web';
        }
        
        return this.request(`/threads/${threadId}/messages`, {
            method: 'POST',
            body: JSON.stringify(payload),
        });
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
        const payload = { message };
        
        // Task 1.8: Add attachment_id if present
        if (attachmentId) {
            payload.image_attachment_id = attachmentId;
        }
        
        if (typeof userManager !== 'undefined') {
            payload.metadata = userManager.getUserMetadata();
            payload.primary_user = userManager.getUsername();
            payload.conversation_source = 'web';
        }
        
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
                            } else if (data.type === 'image_request') {
                                // Store image info for later processing
                                if (onChunk.imageCallback) {
                                    onChunk.imageCallback(data.image_info);
                                }
                            } else if (data.type === 'video_request') {
                                // Store video info for later processing
                                if (onChunk.videoCallback) {
                                    onChunk.videoCallback(data.video_info);
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
                                onComplete(data.message_id);
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
    
    static async deleteMemory(memoryId) {
        return this.request(`/memories/${memoryId}`, {
            method: 'DELETE',
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
    
    static async generateImage(threadId, prompt, negativePrompt = null, disableConfirmation = false, workflowId = null) {
        const body = {
            prompt: prompt,
            negative_prompt: negativePrompt,
            disable_future_confirmations: disableConfirmation
        };
        
        if (workflowId) {
            body.workflow_id = workflowId;
        }
        
        return this.request(`/threads/${threadId}/generate-image`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
    }
    
    // Phase 9: Scene Capture
    static async captureScene(threadId, prompt, negativePrompt = null, workflowId = null) {
        const body = {
            prompt: prompt,
            negative_prompt: negativePrompt
        };
        
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
        return this.request(`/characters/${characterId}/workflows/${oldName}/rename?new_name=${encodeURIComponent(newName)}`, {
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
        
        return response.json();
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
        
        return response.json();
    }
    
    // === Logs ===
    
    static async getServerLogs(lines = 500) {
        return this.request(`/logs/server?lines=${lines}`);
    }
    
    static async listConversationLogs() {
        return this.request('/logs/conversations');
    }
    
    static async getConversationLog(conversationId) {
        return this.request(`/logs/conversations/${conversationId}?prettify=true`);
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
    
    static async generateVideo(threadId, prompt, negativePrompt, disableFutureConfirmations, workflowId = null) {
        return this.request(`/threads/${threadId}/generate-video`, {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                negative_prompt: negativePrompt,
                disable_future_confirmations: disableFutureConfirmations,
                workflow_id: workflowId
            })
        });
    }
    
    static async captureVideoScenePrompt(threadId) {
        return this.request(`/threads/${threadId}/capture-video-scene-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
    }
    
    static async captureVideoScene(threadId, prompt, negativePrompt = null, workflowId = null) {
        const body = {
            prompt: prompt,
            negative_prompt: negativePrompt
        };
        
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