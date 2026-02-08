/**
 * User Management for Web UI
 * Handles user identification and preferences
 */

class UserManager {
    constructor() {
        this.userId = null;
        this.username = null;
        this.aliases = [];
        this.init();
    }
    
    /**
     * Initialize user from localStorage or create new
     */
    init() {
        // Get or create user ID
        this.userId = localStorage.getItem('chorus_web_user_id');
        if (!this.userId) {
            this.userId = this.generateUserId();
            localStorage.setItem('chorus_web_user_id', this.userId);
        }
        
        // Default username (will be overridden by system config)
        this.username = 'User';
        
        // Load identity from server (async)
        this.loadIdentity();
    }
    
    /**
     * Generate a unique user ID
     */
    generateUserId() {
        return 'web_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Set identity (display name + aliases)
     */
    setIdentity(displayName, aliases = []) {
        this.username = displayName && displayName.trim() ? displayName.trim() : 'User';
        this.aliases = Array.isArray(aliases) ? aliases : [];
    }
    
    /**
     * Get current username
     */
    getUsername() {
        return this.username;
    }

    /**
     * Get aliases
     */
    getAliases() {
        return this.aliases || [];
    }
    
    /**
     * Get user ID
     */
    getUserId() {
        return this.userId;
    }
    
    /**
     * Get user metadata to include with messages
     */
    getUserMetadata() {
        return {
            user_id: this.userId,
            username: this.username,
            platform: 'web'
        };
    }
    
    /**
     * Clear user data (for testing/reset)
     */
    clearUserData() {
        localStorage.removeItem('chorus_web_user_id');
        this.init();
    }

    /**
     * Load identity from server config
     */
    async loadIdentity() {
        try {
            const response = await fetch('/system/user-identity');
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            const displayName = data.display_name || '';
            const aliases = Array.isArray(data.aliases) ? data.aliases : [];
            this.setIdentity(displayName, aliases);
        } catch (e) {
            // Keep defaults if fetch fails
        }
    }
}

// Create global instance
const userManager = new UserManager();
