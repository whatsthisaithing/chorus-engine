/**
 * User Management for Web UI
 * Handles user identification and preferences
 */

class UserManager {
    constructor() {
        this.userId = null;
        this.username = null;
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
        
        // Get username (default to "User")
        this.username = localStorage.getItem('chorus_username') || 'User';
    }
    
    /**
     * Generate a unique user ID
     */
    generateUserId() {
        return 'web_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Set username
     */
    setUsername(username) {
        this.username = username || 'User';
        localStorage.setItem('chorus_username', this.username);
    }
    
    /**
     * Get current username
     */
    getUsername() {
        return this.username;
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
        localStorage.removeItem('chorus_username');
        this.init();
    }
}

// Create global instance
const userManager = new UserManager();
