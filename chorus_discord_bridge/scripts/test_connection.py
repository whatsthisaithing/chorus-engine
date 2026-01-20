"""Test script to verify connection to Chorus Engine."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge.config import BridgeConfig, ConfigError
from bridge.chorus_client import ChorusClient, ChorusAPIError


def main():
    """Test connection to Chorus Engine."""
    print("=" * 60)
    print("Chorus Engine Connection Test")
    print("=" * 60)
    print()
    
    try:
        # Load configuration
        print("1. Loading configuration...")
        config = BridgeConfig()
        print(f"   ✓ Config loaded")
        print(f"   - API URL: {config.chorus_api_url}")
        print(f"   - Character: {config.chorus_character_id}")
        print()
        
        # Create Chorus client
        print("2. Creating Chorus API client...")
        client = ChorusClient(
            api_url=config.chorus_api_url,
            api_key=config.chorus_api_key,
            timeout=config.chorus_timeout
        )
        print("   ✓ Client created")
        print()
        
        # Test health check
        print("3. Testing API health...")
        if client.health_check():
            print("   ✓ API is healthy")
        else:
            print("   ✗ API health check failed")
            print("   Make sure Chorus Engine is running!")
            return False
        print()
        
        # Test character loading
        print("4. Testing character loading...")
        try:
            character_info = client.get_character_info(config.chorus_character_id)
            print(f"   ✓ Character loaded: {character_info.get('name', 'Unknown')}")
            print(f"   - Description: {character_info.get('description', 'N/A')[:60]}...")
        except ChorusAPIError as e:
            print(f"   ✗ Failed to load character: {e}")
            return False
        print()
        
        # Test conversation creation
        print("5. Testing conversation creation...")
        try:
            conv_data = client.create_conversation(
                character_id=config.chorus_character_id,
                title="Discord Bridge Test Conversation",
                is_private=False,
                source='test'  # Tag as test conversation
            )
            conversation_id = conv_data['conversation_id']
            thread_id = conv_data['thread_id']
            print(f"   ✓ Conversation created")
            print(f"   - Conversation ID: {conversation_id}")
            print(f"   - Thread ID: {thread_id}")
        except ChorusAPIError as e:
            print(f"   ✗ Failed to create conversation: {e}")
            return False
        print()
        
        # Test message sending
        print("6. Testing message sending...")
        try:
            response = client.send_message(
                conversation_id=conversation_id,
                thread_id=thread_id,
                message="Hello! This is a test message from the Discord bridge.",
                user_name="TestUser",
                metadata={'test': True, 'source': 'connection_test'}
            )
            
            assistant_message = response.get('assistant_message', {})
            reply = assistant_message.get('content', '')
            
            print(f"   ✓ Message sent and response received")
            print(f"   - Response preview: {reply[:100]}...")
            
            # Check if metadata was accepted
            if assistant_message.get('metadata'):
                print(f"   ✓ Metadata supported")
            else:
                print(f"   ⚠ Metadata not returned (may need Task 1.6)")
                
        except ChorusAPIError as e:
            print(f"   ✗ Failed to send message: {e}")
            return False
        print()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Your Chorus Engine connection is working correctly.")
        print("You can now run the Discord bridge with:")
        print("  python -m bridge.main")
        print()
        
        return True
        
    except ConfigError as e:
        print(f"\n❌ Configuration Error:\n{e}")
        print("\nPlease check your configuration files:")
        print("  - config.yaml (copy from config.yaml.template)")
        print("  - .env (copy from .env.template)")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
