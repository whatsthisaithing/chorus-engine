"""
Macro processor for SillyTavern character card imports.

This module handles the replacement of SillyTavern macros ({{char}}, {{user}}, etc.)
with plain text that works naturally with Chorus Engine's system prompts.

Processing occurs ONLY at import time - no runtime overhead or prompt assembly integration.
"""

import re
from typing import Optional


class MacroProcessor:
    """
    Simple macro processor for character card imports.
    
    Replaces SillyTavern macro syntax with LLM-friendly plain text:
    - {{char}} → character name
    - {{user}} → "the user"
    - Utility macros ({{newline}}, etc.)
    - Strips unsupported macros (time, variables, random, etc.)
    """
    
    def __init__(self, character_name: str):
        """
        Initialize the macro processor.
        
        Args:
            character_name: The name of the character (used for {{char}} replacement)
        """
        self.character_name = character_name
    
    def process(self, text: Optional[str]) -> str:
        """
        Process all macros in the given text.
        
        Args:
            text: Text containing SillyTavern macros
            
        Returns:
            Text with all macros processed/removed
        """
        if not text:
            return ""
        
        # Process in order: character → user → utility → strip unsupported
        text = self._replace_character_macros(text)
        text = self._replace_user_macros(text)
        text = self._replace_utility_macros(text)
        text = self._strip_unsupported_macros(text)
        
        # Only strip if there's more than just whitespace
        # (preserve intentional newlines from {{newline}} macros)
        if text and not text.isspace():
            text = text.strip()
        
        return text
    
    def _replace_character_macros(self, text: str) -> str:
        """
        Replace {{char}} and related macros with the character name.
        
        Handles:
        - {{char}}
        - <CHAR>
        - <BOT>
        """
        # {{char}} in all case variations
        text = re.sub(r'\{\{char\}\}', self.character_name, text, flags=re.IGNORECASE)
        
        # Legacy angle bracket formats
        text = re.sub(r'<CHAR>', self.character_name, text, flags=re.IGNORECASE)
        text = re.sub(r'<BOT>', self.character_name, text, flags=re.IGNORECASE)
        
        return text
    
    def _replace_user_macros(self, text: str) -> str:
        """
        Replace {{user}} and related macros with "the user".
        
        The LLM naturally converts "the user" to "you/your" in conversational context.
        
        Handles:
        - {{user}}
        - <USER>
        """
        # {{user}} in all case variations
        text = re.sub(r'\{\{user\}\}', 'the user', text, flags=re.IGNORECASE)
        
        # Legacy angle bracket format
        text = re.sub(r'<USER>', 'the user', text, flags=re.IGNORECASE)
        
        return text
    
    def _replace_utility_macros(self, text: str) -> str:
        """
        Replace utility macros with their functional equivalents.
        
        Handles:
        - {{newline}} → \n
        - {{newline::N}} → N newlines
        - {{trim}} → empty string
        - {{noop}} → empty string
        """
        # {{newline}} → single newline
        text = re.sub(r'\{\{newline\}\}', '\n', text, flags=re.IGNORECASE)
        
        # {{newline::N}} → N newlines
        text = re.sub(
            r'\{\{newline::(\d+)\}\}',
            lambda m: '\n' * int(m.group(1)),
            text,
            flags=re.IGNORECASE
        )
        
        # {{trim}} and {{noop}} → empty string
        text = re.sub(r'\{\{trim\}\}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\{\{noop\}\}', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _strip_unsupported_macros(self, text: str) -> str:
        """
        Remove macros that Chorus Engine doesn't support.
        
        These macros are either:
        - Dynamic (time, random) - would be static after import
        - SillyTavern-specific (variables, instruct mode)
        - Not applicable to Chorus Engine's architecture
        
        Stripped macros:
        - Time/date macros
        - Variable operations
        - Random/pick/roll macros
        - Comments
        - Instruct mode macros
        - Message reference macros
        - System state macros
        """
        # Time and date macros (would be static after import)
        text = re.sub(
            r'\{\{(time|date|weekday|idle_duration|isotime|isodate)\}\}',
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r'\{\{datetimeformat[^}]+\}\}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\{\{time_UTC[^}]+\}\}', '', text, flags=re.IGNORECASE)
        
        # Variable operations (not supported in Chorus)
        text = re.sub(r'\{\{.*?var::[^}]+\}\}', '', text, flags=re.IGNORECASE)
        
        # Random/pick/roll (would become static after import)
        text = re.sub(r'\{\{random::[^}]+\}\}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\{\{pick::[^}]+\}\}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\{\{roll[^}]*\}\}', '', text, flags=re.IGNORECASE)
        
        # Comments
        text = re.sub(r'\{\{//[^}]*\}\}', '', text, flags=re.IGNORECASE)
        
        # Message reference macros (not applicable at import time)
        text = re.sub(
            r'\{\{(lastMessage|lastUserMessage|lastCharMessage|lastMessageId|'
            r'lastSwipeId|currentSwipeId|firstIncludedMessageId|firstDisplayedMessageId)\}\}',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Instruct mode macros (not using instruct mode in Chorus)
        text = re.sub(r'\{\{instruct[^}]*\}\}', '', text, flags=re.IGNORECASE)
        
        # System state macros (not applicable)
        text = re.sub(
            r'\{\{(model|isMobile|maxPrompt|original)\}\}',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Author's note macros (not supported)
        text = re.sub(
            r'\{\{(authorsNote|charAuthorsNote|defaultAuthorsNote)\}\}',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Character card field reference macros (would be circular)
        text = re.sub(
            r'\{\{(description|personality|scenario|persona|mesExamples|mesExamplesRaw|'
            r'charDescription|charPersonality|charScenario|charPrompt|charInstruction|'
            r'version|charVersion|char_version|creatorNotes|charCreatorNotes|charDepthPrompt)\}\}',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Group/multi-character macros (for future multi-character support)
        text = re.sub(
            r'\{\{(group|groupNotMuted|notChar)\}\}',
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r'<GROUP>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<CHARIFNOTGROUP>', '', text, flags=re.IGNORECASE)
        
        # Reverse macro (not needed)
        text = re.sub(r'\{\{reverse:[^}]+\}\}', '', text, flags=re.IGNORECASE)
        
        # Input macro (not applicable)
        text = re.sub(r'\{\{input\}\}', '', text, flags=re.IGNORECASE)
        
        return text


def process_character_card_macros(character_data: dict, character_name: str) -> dict:
    """
    Process all macros in a character card's text fields.
    
    This is a convenience function that applies macro processing to all
    relevant fields in a character card dictionary.
    
    Args:
        character_data: Dictionary containing character card fields
        character_name: Name of the character (for {{char}} replacement)
        
    Returns:
        Dictionary with all text fields processed
    """
    processor = MacroProcessor(character_name)
    
    # Fields that should have macros processed
    text_fields = [
        'description',
        'personality',
        'scenario',
        'first_mes',
        'mes_example',
        'system_prompt',
        'post_history_instructions',
        'alternate_greetings',
        'tags',
        'creator_notes',
    ]
    
    # Process each field if it exists
    processed_data = character_data.copy()
    for field in text_fields:
        if field in processed_data and processed_data[field]:
            if isinstance(processed_data[field], str):
                processed_data[field] = processor.process(processed_data[field])
            elif isinstance(processed_data[field], list):
                # Handle list fields (like alternate_greetings, tags)
                processed_data[field] = [
                    processor.process(item) if isinstance(item, str) else item
                    for item in processed_data[field]
                ]
    
    return processed_data
