"""
Manual test script for macro processor.
Run this to verify macro processing works correctly.
"""

import sys
sys.path.insert(0, 'j:/Dev/chorus-engine')

from chorus_engine.services.character_cards.macro_processor import MacroProcessor

def test_basic_replacements():
    """Test basic {{char}} and {{user}} replacements."""
    processor = MacroProcessor("Nova")
    
    tests = [
        ("{{char}} is helpful", "Nova is helpful"),
        ("{{user}} asks a question", "the user asks a question"),
        ("<CHAR> responds", "Nova responds"),
        ("<USER> walks in", "the user walks in"),
        ("{{char}} helps {{user}}", "Nova helps the user"),
    ]
    
    print("Testing basic replacements:")
    for input_text, expected in tests:
        result = processor.process(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_text}' → '{result}'")
        if result != expected:
            print(f"     Expected: '{expected}'")
    print()

def test_utility_macros():
    """Test utility macros like {{newline}}."""
    processor = MacroProcessor("Nova")
    
    tests = [
        ("Line1{{newline}}Line2", "Line1\nLine2"),
        ("{{newline::2}}", "\n\n"),
        ("Text{{trim}}", "Text"),
        ("{{noop}}Text", "Text"),
    ]
    
    print("Testing utility macros:")
    for input_text, expected in tests:
        result = processor.process(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{repr(input_text)}' → '{repr(result)}'")
        if result != expected:
            print(f"     Expected: '{repr(expected)}'")
    print()

def test_unsupported_macros():
    """Test that unsupported macros are stripped."""
    processor = MacroProcessor("Nova")
    
    tests = [
        ("Today is {{date}}", "Today is"),
        ("{{time}} exactly", "exactly"),
        ("{{setvar::count::5}}Text", "Text"),
        ("{{random::a::b::c}}", ""),
        ("Text{{//comment}}More", "TextMore"),
    ]
    
    print("Testing unsupported macro removal:")
    for input_text, expected in tests:
        result = processor.process(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_text}' → '{result}'")
        if result != expected:
            print(f"     Expected: '{expected}'")
    print()

def test_real_world_example():
    """Test a realistic character card description."""
    processor = MacroProcessor("Nova")
    
    description = """{{char}} is a curious and empathetic AI assistant.
{{char}} enjoys helping {{user}} explore new ideas.{{newline}}
{{char}}'s personality is warm and engaging.
{{user}} will find {{char}} to be a great conversational partner."""
    
    print("Testing real-world example:")
    print("Input:")
    print(description)
    print("\nOutput:")
    result = processor.process(description)
    print(result)
    print()
    
    # Check key replacements
    if "Nova" in result and "the user" in result and "{{" not in result:
        print("✓ Real-world example processed correctly!")
    else:
        print("✗ Real-world example processing failed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Macro Processor Manual Tests")
    print("=" * 60)
    print()
    
    test_basic_replacements()
    test_utility_macros()
    test_unsupported_macros()
    test_real_world_example()
    
    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)
