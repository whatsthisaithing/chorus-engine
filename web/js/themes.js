/**
 * Theme System for Chorus Engine
 * Manages color schemes with instant CSS variable switching
 */

const THEMES = {
    'stage-night': {
        name: 'stage-night',
        displayName: 'Stage Night',
        description: 'Theatrical deep purple with gold spotlight accents',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#1a1625',
            '--bg-secondary': '#241d32',
            '--bg-tertiary': '#2d2440',
            '--bg-elevated': '#362b4d',
            '--accent-primary': '#d4af37',
            '--accent-secondary': '#b8860b',
            '--accent-tertiary': '#ffd700',
            '--text-primary': '#f5f3f7',
            '--text-secondary': '#c4b8d4',
            '--text-tertiary': '#8a7aa0',
            '--text-muted': '#635672',
            '--success': '#4caf50',
            '--warning': '#ff9800',
            '--error': '#f44336',
            '--info': '#6a7fdb',
            '--border-color': '#3d3350',
            '--border-accent': '#4a3f5e',
            '--shadow': 'rgba(0, 0, 0, 0.4)',
            '--msg-user-bg': '#2d3e50',
            '--msg-assistant-bg': '#362b4d',
            '--msg-private-bg': '#3d2945',
            '--sidebar-bg': '#1a1625',
            '--sidebar-hover': '#2d2440',
            '--sidebar-active': '#362b4d',
            '--header-bg': '#1a1625',
            '--header-border': '#3d3350',
            '--profile-gradient-end': '#1a1625'
        }
    },
    
    'spotlight-bright': {
        name: 'spotlight-bright',
        displayName: 'Spotlight Bright',
        description: 'High contrast light theme with theatrical gold accents',
        category: 'light',
        cssVariables: {
            '--bg-primary': '#fafafa',
            '--bg-secondary': '#f0f0f0',
            '--bg-tertiary': '#e8e8e8',
            '--bg-elevated': '#ffffff',
            '--accent-primary': '#d4af37',
            '--accent-secondary': '#b8860b',
            '--accent-tertiary': '#ffd700',
            '--text-primary': '#1a1a1a',
            '--text-secondary': '#4a4a4a',
            '--text-tertiary': '#6a6a6a',
            '--text-muted': '#707070',
            '--success': '#2e7d32',
            '--warning': '#f57c00',
            '--error': '#c62828',
            '--info': '#1976d2',
            '--border-color': '#a0a0a0',
            '--border-accent': '#b8860b',
            '--shadow': 'rgba(0, 0, 0, 0.1)',
            '--msg-user-bg': '#e3f2fd',
            '--msg-assistant-bg': '#f5f5f5',
            '--msg-private-bg': '#fff9e6',
            '--sidebar-bg': '#f5f5f5',
            '--sidebar-hover': '#e8e8e8',
            '--sidebar-active': '#d4af37',
            '--header-bg': '#ffffff',
            '--header-border': '#d0d0d0',
            '--profile-gradient-end': '#fafafa'
        }
    },
    
    'noir-absolute': {
        name: 'noir-absolute',
        displayName: 'Noir Absolute',
        description: 'Maximum contrast dark theme with stark white accents',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#0a0a0a',
            '--bg-secondary': '#1a1a1a',
            '--bg-tertiary': '#252525',
            '--bg-elevated': '#303030',
            '--accent-primary': '#ffffff',
            '--accent-secondary': '#e0e0e0',
            '--accent-tertiary': '#ffffff',
            '--text-primary': '#f5f5f5',
            '--text-secondary': '#d0d0d0',
            '--text-tertiary': '#a0a0a0',
            '--text-muted': '#707070',
            '--success': '#4caf50',
            '--warning': '#ff9800',
            '--error': '#f44336',
            '--info': '#2196f3',
            '--border-color': '#404040',
            '--border-accent': '#ffffff',
            '--shadow': 'rgba(0, 0, 0, 0.8)',
            '--msg-user-bg': '#1a1a1a',
            '--msg-assistant-bg': '#252525',
            '--msg-private-bg': '#2a2a2a',
            '--sidebar-bg': '#0a0a0a',
            '--sidebar-hover': '#252525',
            '--sidebar-active': '#303030',
            '--header-bg': '#0a0a0a',
            '--header-border': '#404040',
            '--profile-gradient-end': '#0a0a0a'
        }
    },
    
    'cobalt-depths': {
        name: 'cobalt-depths',
        displayName: 'Cobalt Depths',
        description: 'Cool professional blue theme',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#0f1729',
            '--bg-secondary': '#1a2942',
            '--bg-tertiary': '#253a5a',
            '--bg-elevated': '#2f4a6f',
            '--accent-primary': '#4a9eff',
            '--accent-secondary': '#3a7fd5',
            '--accent-tertiary': '#5aaeff',
            '--text-primary': '#e8f2ff',
            '--text-secondary': '#b8d4f1',
            '--text-tertiary': '#88b3d8',
            '--text-muted': '#5a7ea0',
            '--success': '#4caf50',
            '--warning': '#ff9800',
            '--error': '#f44336',
            '--info': '#64b5f6',
            '--border-color': '#2a3f5f',
            '--border-accent': '#4a9eff',
            '--shadow': 'rgba(0, 0, 0, 0.5)',
            '--msg-user-bg': '#1a2942',
            '--msg-assistant-bg': '#253a5a',
            '--msg-private-bg': '#2a3550',
            '--sidebar-bg': '#0f1729',
            '--sidebar-hover': '#253a5a',
            '--sidebar-active': '#2f4a6f',
            '--header-bg': '#0f1729',
            '--header-border': '#2a3f5f',
            '--profile-gradient-end': '#0f1729'
        }
    },
    
    'forest-canopy': {
        name: 'forest-canopy',
        displayName: 'Forest Canopy',
        description: 'Natural green theme for calming focus',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#0f1a12',
            '--bg-secondary': '#1a2b1f',
            '--bg-tertiary': '#243a2c',
            '--bg-elevated': '#2e4a38',
            '--accent-primary': '#4caf50',
            '--accent-secondary': '#388e3c',
            '--accent-tertiary': '#66bb6a',
            '--text-primary': '#e8f5e9',
            '--text-secondary': '#c8e6c9',
            '--text-tertiary': '#a5d6a7',
            '--text-muted': '#689f6a',
            '--success': '#66bb6a',
            '--warning': '#ff9800',
            '--error': '#f44336',
            '--info': '#4caf50',
            '--border-color': '#2a3f2e',
            '--border-accent': '#4caf50',
            '--shadow': 'rgba(0, 0, 0, 0.5)',
            '--msg-user-bg': '#1a2b1f',
            '--msg-assistant-bg': '#243a2c',
            '--msg-private-bg': '#2a3528',
            '--sidebar-bg': '#0f1a12',
            '--sidebar-hover': '#243a2c',
            '--sidebar-active': '#2e4a38',
            '--header-bg': '#0f1a12',
            '--header-border': '#2a3f2e',
            '--profile-gradient-end': '#0f1a12'
        }
    },
    
    'ember-glow': {
        name: 'ember-glow',
        displayName: 'Ember Glow',
        description: 'Warm passionate theme with soft coral accents',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#1a0f0f',
            '--bg-secondary': '#2b1818',
            '--bg-tertiary': '#3a2424',
            '--bg-elevated': '#4a2f2f',
            '--accent-primary': '#ff6b6b',
            '--accent-secondary': '#e85555',
            '--accent-tertiary': '#ff8787',
            '--text-primary': '#fff5f5',
            '--text-secondary': '#ffd4d4',
            '--text-tertiary': '#ffb3b3',
            '--text-muted': '#c07878',
            '--success': '#4caf50',
            '--warning': '#ffa726',
            '--error': '#ef5350',
            '--info': '#ff6b6b',
            '--border-color': '#3f2a2a',
            '--border-accent': '#ff6b6b',
            '--shadow': 'rgba(0, 0, 0, 0.5)',
            '--msg-user-bg': '#2b1818',
            '--msg-assistant-bg': '#3a2424',
            '--msg-private-bg': '#352020',
            '--sidebar-bg': '#1a0f0f',
            '--sidebar-hover': '#3a2424',
            '--sidebar-active': '#4a2f2f',
            '--header-bg': '#1a0f0f',
            '--header-border': '#3f2a2a',
            '--profile-gradient-end': '#1a0f0f'
        }
    },
    
    'monochrome-studio': {
        name: 'monochrome-studio',
        displayName: 'Monochrome Studio',
        description: 'Minimalist grayscale for distraction-free work',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#151515',
            '--bg-secondary': '#252525',
            '--bg-tertiary': '#353535',
            '--bg-elevated': '#454545',
            '--accent-primary': '#888888',
            '--accent-secondary': '#707070',
            '--accent-tertiary': '#a0a0a0',
            '--text-primary': '#e5e5e5',
            '--text-secondary': '#c0c0c0',
            '--text-tertiary': '#909090',
            '--text-muted': '#606060',
            '--success': '#808080',
            '--warning': '#a0a0a0',
            '--error': '#707070',
            '--info': '#888888',
            '--border-color': '#404040',
            '--border-accent': '#888888',
            '--shadow': 'rgba(0, 0, 0, 0.6)',
            '--msg-user-bg': '#252525',
            '--msg-assistant-bg': '#353535',
            '--msg-private-bg': '#2a2a2a',
            '--sidebar-bg': '#151515',
            '--sidebar-hover': '#353535',
            '--sidebar-active': '#454545',
            '--header-bg': '#151515',
            '--header-border': '#404040',
            '--profile-gradient-end': '#151515'
        }
    },
    
    'sunset-boulevard': {
        name: 'sunset-boulevard',
        displayName: 'Sunset Boulevard',
        description: 'Warm theatrical theme with nostalgic orange glow',
        category: 'dark',
        cssVariables: {
            '--bg-primary': '#1a1410',
            '--bg-secondary': '#2b2218',
            '--bg-tertiary': '#3a2f24',
            '--bg-elevated': '#4a3d2f',
            '--accent-primary': '#ff8c42',
            '--accent-secondary': '#e67830',
            '--accent-tertiary': '#ffa05a',
            '--text-primary': '#fff8f0',
            '--text-secondary': '#f5d4b8',
            '--text-tertiary': '#d9b899',
            '--text-muted': '#a08060',
            '--success': '#4caf50',
            '--warning': '#ffa726',
            '--error': '#f44336',
            '--info': '#ff8c42',
            '--border-color': '#3f3025',
            '--border-accent': '#ff8c42',
            '--shadow': 'rgba(0, 0, 0, 0.5)',
            '--msg-user-bg': '#2b2218',
            '--msg-assistant-bg': '#3a2f24',
            '--msg-private-bg': '#352820',
            '--sidebar-bg': '#1a1410',
            '--sidebar-hover': '#3a2f24',
            '--sidebar-active': '#4a3d2f',
            '--header-bg': '#1a1410',
            '--header-border': '#3f3025',
            '--profile-gradient-end': '#1a1410'
        }
    },
    
    'arctic-twilight': {
        name: 'arctic-twilight',
        displayName: 'Arctic Twilight',
        description: 'Cool clean light theme with airy blue tones',
        category: 'light',
        cssVariables: {
            '--bg-primary': '#f5f8fa',
            '--bg-secondary': '#e8f1f5',
            '--bg-tertiary': '#d8e8f0',
            '--bg-elevated': '#ffffff',
            '--accent-primary': '#4a9eff',
            '--accent-secondary': '#3a7fd5',
            '--accent-tertiary': '#5aaeff',
            '--text-primary': '#1a2942',
            '--text-secondary': '#2a4a6a',
            '--text-tertiary': '#4a6a8a',
            '--text-muted': '#5a7a9a',
            '--success': '#2e7d32',
            '--warning': '#f57c00',
            '--error': '#c62828',
            '--info': '#1976d2',
            '--border-color': '#90b8d8',
            '--border-accent': '#4a9eff',
            '--shadow': 'rgba(74, 158, 255, 0.1)',
            '--msg-user-bg': '#e3f2fd',
            '--msg-assistant-bg': '#f5f8fa',
            '--msg-private-bg': '#e8f5e9',
            '--sidebar-bg': '#e8f1f5',
            '--sidebar-hover': '#d8e8f0',
            '--sidebar-active': '#c0d8e8',
            '--header-bg': '#ffffff',
            '--header-border': '#c0d8e8',
            '--profile-gradient-end': '#f5f8fa'
        }
    }
};

/**
 * Theme Manager
 * Handles theme switching and persistence
 */
const ThemeManager = {
    /**
     * Get system default theme from config
     */
    async getSystemTheme() {
        try {
            const response = await fetch('/config');
            const config = await response.json();
            return config?.ui?.color_scheme || 'stage-night';
        } catch (error) {
            console.error('Failed to fetch system config:', error);
            return 'stage-night';
        }
    },
    
    /**
     * Get character's theme preference
     */
    async getCharacterTheme(characterId) {
        try {
            const response = await fetch(`/characters/${characterId}`);
            const character = await response.json();
            return character?.ui_preferences?.color_scheme || null;
        } catch (error) {
            console.error('Failed to fetch character theme:', error);
            return null;
        }
    },
    
    /**
     * Get currently active theme name
     */
    getCurrentTheme() {
        return localStorage.getItem('activeTheme') || 'stage-night';
    },
    
    /**
     * Apply theme by updating CSS variables
     */
    applyTheme(themeName) {
        const theme = THEMES[themeName];
        
        if (!theme) {
            console.warn(`Theme '${themeName}' not found, using stage-night`);
            themeName = 'stage-night';
        }
        
        const themeData = THEMES[themeName];
        const root = document.documentElement;
        
        // Update CSS variables
        Object.entries(themeData.cssVariables).forEach(([key, value]) => {
            root.style.setProperty(key, value);
        });
        
        // Store active theme
        localStorage.setItem('activeTheme', themeName);
        
        console.log(`Applied theme: ${themeData.displayName}`);
    },
    
    /**
     * Initialize theme system on page load
     */
    async initialize() {
        // Always use system default on initial load
        // Character-specific themes will be applied when character is selected
        const systemTheme = await this.getSystemTheme();
        this.applyTheme(systemTheme);
    },
    
    /**
     * Apply theme for a specific character (with fallback to system default)
     */
    async applyCharacterTheme(characterId) {
        if (!characterId) {
            const systemTheme = await this.getSystemTheme();
            this.applyTheme(systemTheme);
            return;
        }
        
        const characterTheme = await this.getCharacterTheme(characterId);
        
        if (characterTheme && THEMES[characterTheme]) {
            this.applyTheme(characterTheme);
        } else {
            const systemTheme = await this.getSystemTheme();
            this.applyTheme(systemTheme);
        }
    },
    
    /**
     * Get all available themes
     */
    getAllThemes() {
        return Object.values(THEMES);
    },
    
    /**
     * Get themes by category
     */
    getThemesByCategory(category) {
        return Object.values(THEMES).filter(theme => theme.category === category);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { THEMES, ThemeManager };
}
