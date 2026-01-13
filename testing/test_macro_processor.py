"""
Tests for the macro processor.

Tests cover:
- Basic {{char}} and {{user}} replacement
- Legacy angle bracket formats (<USER>, <CHAR>, <BOT>)
- Utility macros ({{newline}}, {{trim}}, {{noop}})
- Unsupported macro removal (time, variables, random, etc.)
- Edge cases (case insensitivity, multiple occurrences, etc.)
"""

import pytest
from chorus_engine.services.character_cards.macro_processor import (
    MacroProcessor,
    process_character_card_macros
)


class TestMacroProcessor:
    """Test suite for MacroProcessor class."""
    
    def test_basic_char_replacement(self):
        """Test {{char}} is replaced with character name."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{char}} is helpful") == "Nova is helpful"
        assert processor.process("{{CHAR}} is helpful") == "Nova is helpful"  # Case insensitive
        assert processor.process("{{Char}} is helpful") == "Nova is helpful"
    
    def test_basic_user_replacement(self):
        """Test {{user}} is replaced with 'the user'."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{user}} asks a question") == "the user asks a question"
        assert processor.process("{{USER}} asks a question") == "the user asks a question"
        assert processor.process("{{User}} asks a question") == "the user asks a question"
    
    def test_legacy_angle_brackets(self):
        """Test legacy <USER>, <CHAR>, <BOT> formats."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("<USER> walks in") == "the user walks in"
        assert processor.process("<CHAR> responds") == "Nova responds"
        assert processor.process("<BOT> responds") == "Nova responds"
    
    def test_multiple_replacements(self):
        """Test multiple macro occurrences in same text."""
        processor = MacroProcessor("Nova")
        
        text = "{{char}} likes helping {{user}}. {{char}} is friendly to {{user}}."
        expected = "Nova likes helping the user. Nova is friendly to the user."
        assert processor.process(text) == expected
    
    def test_possessive_forms(self):
        """Test possessive forms work correctly."""
        processor = MacroProcessor("Nova")
        
        # Note: We replace the macro itself, LLM handles grammar
        assert processor.process("{{user}}'s question") == "the user's question"
        assert processor.process("{{char}}'s answer") == "Nova's answer"
    
    def test_newline_macro(self):
        """Test {{newline}} macro replacement."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("Line 1{{newline}}Line 2") == "Line 1\nLine 2"
        assert processor.process("Line 1{{NEWLINE}}Line 2") == "Line 1\nLine 2"
    
    def test_newline_with_count(self):
        """Test {{newline::N}} macro with count."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("Line 1{{newline::2}}Line 2") == "Line 1\n\nLine 2"
        assert processor.process("Line 1{{newline::3}}Line 2") == "Line 1\n\n\nLine 2"
    
    def test_utility_macros(self):
        """Test {{trim}} and {{noop}} macros."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("Text{{trim}}") == "Text"
        assert processor.process("Text{{noop}}") == "Text"
        assert processor.process("{{trim}}Text{{noop}}") == "Text"
    
    def test_time_macros_stripped(self):
        """Test time/date macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("Today is {{date}}") == "Today is"
        assert processor.process("The time is {{time}}") == "The time is"
        assert processor.process("It's {{weekday}}") == "It's"
        assert processor.process("{{isotime}} exactly") == "exactly"
        assert processor.process("{{isodate}} today") == "today"
    
    def test_variable_macros_stripped(self):
        """Test variable operation macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{setvar::count::5}}Text") == "Text"
        assert processor.process("{{getvar::count}}") == ""
        assert processor.process("{{incvar::counter}}") == ""
        assert processor.process("{{setglobalvar::name::value}}") == ""
    
    def test_random_macros_stripped(self):
        """Test random/pick/roll macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{random::a::b::c}}") == ""
        assert processor.process("{{pick::x::y::z}}") == ""
        assert processor.process("{{roll 2d6}}") == ""
        assert processor.process("Result: {{roll d20}}") == "Result:"
    
    def test_comment_macros_stripped(self):
        """Test comment macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("Text{{//This is a comment}}More") == "TextMore"
        assert processor.process("{{//Comment at start}}Text") == "Text"
    
    def test_message_reference_macros_stripped(self):
        """Test message reference macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{lastMessage}}") == ""
        assert processor.process("{{lastUserMessage}}") == ""
        assert processor.process("{{lastCharMessage}}") == ""
        assert processor.process("{{lastMessageId}}") == ""
    
    def test_instruct_mode_macros_stripped(self):
        """Test instruct mode macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{instructInput}}") == ""
        assert processor.process("{{instructUserPrefix}}") == ""
        assert processor.process("{{instructSystemPrompt}}") == ""
    
    def test_system_state_macros_stripped(self):
        """Test system state macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{model}}") == ""
        assert processor.process("{{isMobile}}") == ""
        assert processor.process("{{maxPrompt}}") == ""
    
    def test_character_field_macros_stripped(self):
        """Test character field reference macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{description}}") == ""
        assert processor.process("{{personality}}") == ""
        assert processor.process("{{scenario}}") == ""
        assert processor.process("{{charDescription}}") == ""
    
    def test_group_macros_stripped(self):
        """Test group/multi-character macros are removed."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("{{group}}") == ""
        assert processor.process("{{groupNotMuted}}") == ""
        assert processor.process("{{notChar}}") == ""
        assert processor.process("<GROUP>") == ""
    
    def test_empty_and_none_input(self):
        """Test handling of empty and None input."""
        processor = MacroProcessor("Nova")
        
        assert processor.process("") == ""
        assert processor.process(None) == ""
        assert processor.process("   ") == ""  # Whitespace only
    
    def test_no_macros(self):
        """Test text without macros passes through unchanged."""
        processor = MacroProcessor("Nova")
        
        text = "This is normal text without any macros."
        assert processor.process(text) == text
    
    def test_complex_real_world_example(self):
        """Test a realistic character card description."""
        processor = MacroProcessor("Nova")
        
        description = """{{char}} is a curious and empathetic AI assistant.
{{char}} enjoys helping {{user}} explore new ideas.{{newline}}
{{char}}'s personality is warm and engaging.
{{user}} will find {{char}} to be a great conversational partner."""
        
        expected = """Nova is a curious and empathetic AI assistant.
Nova enjoys helping the user explore new ideas.

Nova's personality is warm and engaging.
the user will find Nova to be a great conversational partner."""
        
        assert processor.process(description) == expected
    
    def test_dialogue_example_processing(self):
        """Test processing example dialogue with macros."""
        processor = MacroProcessor("Nova")
        
        dialogue = """{{user}}: Hello!
{{char}}: Hi there! How can I help you today?
{{user}}: I need advice.
{{char}}: I'd be happy to help {{user}} with advice!"""
        
        expected = """the user: Hello!
Nova: Hi there! How can I help you today?
the user: I need advice.
Nova: I'd be happy to help the user with advice!"""
        
        assert processor.process(dialogue) == expected


class TestProcessCharacterCardMacros:
    """Test suite for process_character_card_macros convenience function."""
    
    def test_process_all_text_fields(self):
        """Test that all text fields in character card are processed."""
        character_data = {
            'name': 'Nova',
            'description': '{{char}} is helpful to {{user}}',
            'personality': '{{char}} is curious',
            'scenario': '{{user}} meets {{char}}',
            'first_mes': 'Hello {{user}}!',
            'mes_example': '{{user}}: Hi\n{{char}}: Hello!',
        }
        
        processed = process_character_card_macros(character_data, 'Nova')
        
        assert processed['description'] == 'Nova is helpful to the user'
        assert processed['personality'] == 'Nova is curious'
        assert processed['scenario'] == 'the user meets Nova'
        assert processed['first_mes'] == 'Hello the user!'
        assert processed['mes_example'] == 'the user: Hi\nNova: Hello!'
    
    def test_process_list_fields(self):
        """Test that list fields (alternate_greetings, tags) are processed."""
        character_data = {
            'name': 'Nova',
            'alternate_greetings': [
                'Hello {{user}}!',
                'Welcome, {{user}}!',
            ],
            'tags': [
                '{{char}} is friendly',
                'helpful to {{user}}',
            ],
        }
        
        processed = process_character_card_macros(character_data, 'Nova')
        
        assert processed['alternate_greetings'] == [
            'Hello the user!',
            'Welcome, the user!',
        ]
        assert processed['tags'] == [
            'Nova is friendly',
            'helpful to the user',
        ]
    
    def test_preserves_non_text_fields(self):
        """Test that non-text fields are preserved unchanged."""
        character_data = {
            'name': 'Nova',
            'description': '{{char}} is helpful',
            'age': 25,
            'metadata': {'version': '1.0'},
            'enabled': True,
        }
        
        processed = process_character_card_macros(character_data, 'Nova')
        
        assert processed['age'] == 25
        assert processed['metadata'] == {'version': '1.0'}
        assert processed['enabled'] is True
    
    def test_handles_missing_fields(self):
        """Test that missing fields don't cause errors."""
        character_data = {
            'name': 'Nova',
            'description': '{{char}} is helpful',
        }
        
        # Should not raise error even though most fields are missing
        processed = process_character_card_macros(character_data, 'Nova')
        
        assert processed['description'] == 'Nova is helpful'
        assert 'name' in processed
    
    def test_handles_empty_fields(self):
        """Test that empty fields are handled gracefully."""
        character_data = {
            'name': 'Nova',
            'description': '',
            'personality': None,
        }
        
        processed = process_character_card_macros(character_data, 'Nova')
        
        assert processed['description'] == ''
        assert processed['personality'] == ''  # None becomes empty string


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
