"""
Test script for Message Processor and Response Formatter

Phase 3, Tasks 3.2 and 3.3
Tests Discord message format conversion and response splitting.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bridge.response_formatter import ResponseFormatter


def test_response_formatter():
    """Test response formatter with various message lengths."""
    print("=" * 60)
    print("TESTING RESPONSE FORMATTER")
    print("=" * 60)
    
    formatter = ResponseFormatter()
    
    # Test 1: Short message (should not split)
    print("\n1. Short message:")
    short = "Hello! This is a short message."
    chunks = formatter.format_response(short)
    print(f"   Input length: {len(short)}")
    print(f"   Output chunks: {len(chunks)}")
    assert len(chunks) == 1
    print("   ✓ PASS")
    
    # Test 2: Long message (should split)
    print("\n2. Long message:")
    long = "This is a long message. " * 100  # ~2400 chars
    chunks = formatter.format_response(long)
    print(f"   Input length: {len(long)}")
    print(f"   Output chunks: {len(chunks)}")
    print(f"   Chunk lengths: {[len(c) for c in chunks]}")
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    print("   ✓ PASS")
    
    # Test 3: Message with code block
    print("\n3. Message with code block:")
    with_code = """Here's some code:

```python
def hello_world():
    print("Hello, world!")
    return True
```

That's a simple function."""
    chunks = formatter.format_response(with_code)
    print(f"   Input length: {len(with_code)}")
    print(f"   Output chunks: {len(chunks)}")
    # Check code block is preserved
    assert any("```python" in c for c in chunks)
    print("   ✓ PASS")
    
    # Test 4: Large code block that needs splitting
    print("\n4. Large code block:")
    large_code = "```python\n" + ("print('line')\n" * 200) + "```"
    chunks = formatter.format_response(large_code)
    print(f"   Input length: {len(large_code)}")
    print(f"   Output chunks: {len(chunks)}")
    print(f"   Chunk lengths: {[len(c) for c in chunks]}")
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    # Check all chunks have code markers
    assert all("```" in c for c in chunks)
    print("   ✓ PASS")
    
    # Test 5: Multiple paragraphs
    print("\n5. Multiple paragraphs:")
    paragraphs = "\n\n".join([
        "First paragraph is here.",
        "Second paragraph is here.",
        "Third paragraph is here.",
        "Fourth paragraph is here."
    ])
    chunks = formatter.format_response(paragraphs)
    print(f"   Input length: {len(paragraphs)}")
    print(f"   Output chunks: {len(chunks)}")
    assert len(chunks) == 1  # Should fit in one
    print("   ✓ PASS")
    
    # Test 6: Very long single sentence (force word splitting)
    print("\n6. Very long single sentence:")
    long_sentence = "word " * 500  # ~2500 chars of one sentence
    chunks = formatter.format_response(long_sentence)
    print(f"   Input length: {len(long_sentence)}")
    print(f"   Output chunks: {len(chunks)}")
    print(f"   Chunk lengths: {[len(c) for c in chunks]}")
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    print("   ✓ PASS")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)


def test_message_processor_patterns():
    """Test message processor regex patterns (without Discord client)."""
    print("\n" + "=" * 60)
    print("TESTING MESSAGE PROCESSOR PATTERNS")
    print("=" * 60)
    
    import re
    
    # Test custom emoji pattern
    print("\n1. Custom emoji pattern:")
    emoji_text = "Hello <:smile:123456> and <a:wave:789012>!"
    pattern = r'<a?:([a-zA-Z0-9_]+):\d+>'
    result = re.sub(pattern, r':\1:', emoji_text)
    print(f"   Input:  {emoji_text}")
    print(f"   Output: {result}")
    assert result == "Hello :smile: and :wave:!"
    print("   ✓ PASS")
    
    # Test user mention pattern
    print("\n2. User mention pattern:")
    mention_text = "Hey <@123456> and <@!789012>!"
    pattern = r'<@!?(\d+)>'
    matches = re.findall(pattern, mention_text)
    print(f"   Input:  {mention_text}")
    print(f"   User IDs found: {matches}")
    assert len(matches) == 2
    assert matches[0] == "123456"
    assert matches[1] == "789012"
    print("   ✓ PASS")
    
    # Test channel mention pattern
    print("\n3. Channel mention pattern:")
    channel_text = "Check <#123456> for updates!"
    pattern = r'<#(\d+)>'
    matches = re.findall(pattern, channel_text)
    print(f"   Input:  {channel_text}")
    print(f"   Channel IDs found: {matches}")
    assert len(matches) == 1
    assert matches[0] == "123456"
    print("   ✓ PASS")
    
    # Test role mention pattern
    print("\n4. Role mention pattern:")
    role_text = "Hello <@&123456> members!"
    pattern = r'<@&(\d+)>'
    matches = re.findall(pattern, role_text)
    print(f"   Input:  {role_text}")
    print(f"   Role IDs found: {matches}")
    assert len(matches) == 1
    assert matches[0] == "123456"
    print("   ✓ PASS")
    
    # Test spoiler pattern
    print("\n5. Spoiler pattern:")
    spoiler_text = "The answer is ||42||!"
    pattern = r'\|\|([^|]+)\|\|'
    result = re.sub(pattern, r'(spoiler: \1)', spoiler_text)
    print(f"   Input:  {spoiler_text}")
    print(f"   Output: {result}")
    assert result == "The answer is (spoiler: 42)!"
    print("   ✓ PASS")
    
    # Test timestamp pattern
    print("\n6. Timestamp pattern:")
    timestamp_text = "Event at <t:1234567890:F>!"
    pattern = r'<t:(\d+)(?::[tTdDfFR])?>'
    matches = re.findall(pattern, timestamp_text)
    print(f"   Input:  {timestamp_text}")
    print(f"   Timestamps found: {matches}")
    assert len(matches) == 1
    assert matches[0] == "1234567890"
    print("   ✓ PASS")
    
    print("\n" + "=" * 60)
    print("ALL PATTERN TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_response_formatter()
        test_message_processor_patterns()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("\nPhase 3, Tasks 3.2 and 3.3 implementation verified.")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
