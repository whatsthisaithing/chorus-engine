/**
 * Document Autocomplete Module
 * Provides #doc: and file: reference autocomplete in message input
 */

window.DocumentAutocomplete = (function() {
    const API_BASE = window.location.origin;
    let autocompleteContainer = null;
    let currentSuggestions = [];
    let selectedIndex = -1;
    let currentPrefix = '';
    let currentQuery = '';
    
    /**
     * Initialize autocomplete on the message input
     */
    function init(inputElement) {
        if (!inputElement) {
            console.error('DocumentAutocomplete: Input element not found');
            return;
        }
        
        // Create autocomplete container
        createAutocompleteContainer(inputElement);
        
        // Add event listeners
        inputElement.addEventListener('input', handleInput);
        inputElement.addEventListener('keydown', handleKeydown);
        
        // Close autocomplete when clicking outside
        document.addEventListener('click', (e) => {
            if (e.target !== inputElement && !autocompleteContainer.contains(e.target)) {
                hideAutocomplete();
            }
        });
        
        console.log('Document autocomplete initialized');
    }
    
    /**
     * Create the autocomplete dropdown container
     */
    function createAutocompleteContainer(inputElement) {
        autocompleteContainer = document.createElement('div');
        autocompleteContainer.className = 'document-autocomplete';
        autocompleteContainer.style.cssText = `
            position: absolute;
            background: #2d2d2d;
            border: 1px solid #444;
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
            z-index: 1060;
            display: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            min-width: 300px;
        `;
        
        // Position it relative to the input
        const inputContainer = inputElement.parentElement;
        inputContainer.style.position = 'relative';
        inputContainer.appendChild(autocompleteContainer);
    }
    
    /**
     * Handle input changes to detect #doc: or file: prefixes
     */
    function handleInput(e) {
        const input = e.target;
        const cursorPos = input.selectionStart;
        const text = input.value.substring(0, cursorPos);
        
        // Check for #doc: or file: prefix
        const docMatch = text.match(/#doc:([^\s]*)$/);
        const fileMatch = text.match(/file:([^\s]*)$/);
        
        if (docMatch) {
            currentPrefix = '#doc:';
            currentQuery = docMatch[1];
            fetchSuggestions(currentQuery);
        } else if (fileMatch) {
            currentPrefix = 'file:';
            currentQuery = fileMatch[1];
            fetchSuggestions(currentQuery);
        } else {
            hideAutocomplete();
        }
    }
    
    /**
     * Handle keyboard navigation in autocomplete
     */
    function handleKeydown(e) {
        if (!autocompleteContainer || autocompleteContainer.style.display === 'none') {
            return;
        }
        
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, currentSuggestions.length - 1);
                updateSelection();
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, -1);
                updateSelection();
                break;
                
            case 'Enter':
                if (selectedIndex >= 0) {
                    e.preventDefault();
                    selectSuggestion(currentSuggestions[selectedIndex]);
                }
                break;
                
            case 'Escape':
                e.preventDefault();
                hideAutocomplete();
                break;
        }
    }
    
    /**
     * Fetch autocomplete suggestions from API
     */
    async function fetchSuggestions(query) {
        try {
            // Get current character ID if available
            const characterId = window.App?.state?.selectedCharacterId || null;
            
            let url = `${API_BASE}/documents/autocomplete?query=${encodeURIComponent(query)}&limit=10`;
            if (characterId) {
                url += `&character_id=${characterId}`;
            }
            
            const response = await fetch(url);
            if (!response.ok) {
                console.error('Failed to fetch autocomplete suggestions:', response.status);
                hideAutocomplete();
                return;
            }
            
            const suggestions = await response.json();
            showSuggestions(suggestions);
            
        } catch (error) {
            console.error('Error fetching autocomplete suggestions:', error);
            hideAutocomplete();
        }
    }
    
    /**
     * Display suggestions in the dropdown
     */
    function showSuggestions(suggestions) {
        if (!suggestions || suggestions.length === 0) {
            hideAutocomplete();
            return;
        }
        
        currentSuggestions = suggestions;
        selectedIndex = -1;
        
        autocompleteContainer.innerHTML = '';
        
        suggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            item.style.cssText = `
                padding: 8px 12px;
                cursor: pointer;
                border-bottom: 1px solid #3a3a3a;
                transition: background-color 0.15s;
            `;
            
            // Create suggestion content
            const titleDiv = document.createElement('div');
            titleDiv.style.cssText = 'color: #e0e0e0; font-weight: 500; margin-bottom: 2px;';
            titleDiv.textContent = suggestion.title || suggestion.filename;
            
            const filenameDiv = document.createElement('div');
            filenameDiv.style.cssText = 'color: #999; font-size: 0.85em;';
            filenameDiv.textContent = suggestion.filename;
            
            item.appendChild(titleDiv);
            if (suggestion.title && suggestion.title !== suggestion.filename) {
                item.appendChild(filenameDiv);
            }
            
            // Mouse events
            item.addEventListener('mouseenter', () => {
                selectedIndex = index;
                updateSelection();
            });
            
            item.addEventListener('click', () => {
                selectSuggestion(suggestion);
            });
            
            autocompleteContainer.appendChild(item);
        });
        
        // Position dropdown above input
        const inputElement = autocompleteContainer.parentElement.querySelector('input');
        const inputRect = inputElement.getBoundingClientRect();
        const containerRect = autocompleteContainer.parentElement.getBoundingClientRect();
        
        autocompleteContainer.style.bottom = (containerRect.height + 5) + 'px';
        autocompleteContainer.style.left = '0';
        autocompleteContainer.style.right = '0';
        autocompleteContainer.style.display = 'block';
    }
    
    /**
     * Update visual selection in dropdown
     */
    function updateSelection() {
        const items = autocompleteContainer.querySelectorAll('.autocomplete-item');
        items.forEach((item, index) => {
            if (index === selectedIndex) {
                item.style.backgroundColor = '#404040';
            } else {
                item.style.backgroundColor = 'transparent';
            }
        });
        
        // Scroll selected item into view
        if (selectedIndex >= 0 && items[selectedIndex]) {
            items[selectedIndex].scrollIntoView({ block: 'nearest' });
        }
    }
    
    /**
     * Insert selected suggestion into input
     */
    function selectSuggestion(suggestion) {
        const inputElement = autocompleteContainer.parentElement.querySelector('input');
        const cursorPos = inputElement.selectionStart;
        const text = inputElement.value;
        
        // Find where the current prefix starts
        const beforeCursor = text.substring(0, cursorPos);
        const prefixStart = beforeCursor.lastIndexOf(currentPrefix);
        
        if (prefixStart === -1) {
            hideAutocomplete();
            return;
        }
        
        // Replace the partial reference with the full filename
        const before = text.substring(0, prefixStart);
        const after = text.substring(cursorPos);
        const reference = currentPrefix + suggestion.filename;
        
        inputElement.value = before + reference + ' ' + after;
        
        // Position cursor after the inserted reference
        const newCursorPos = prefixStart + reference.length + 1;
        inputElement.setSelectionRange(newCursorPos, newCursorPos);
        
        hideAutocomplete();
        inputElement.focus();
    }
    
    /**
     * Hide autocomplete dropdown
     */
    function hideAutocomplete() {
        if (autocompleteContainer) {
            autocompleteContainer.style.display = 'none';
        }
        currentSuggestions = [];
        selectedIndex = -1;
        currentPrefix = '';
        currentQuery = '';
    }
    
    // Public API
    return {
        init: init
    };
})();
