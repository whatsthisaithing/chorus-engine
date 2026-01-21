"""
Test script for Phase 2: State Management & Persistence

Tests conversation mapper and user tracker functionality.
"""
import sys
from pathlib import Path

# Add bridge directory to path
bridge_dir = Path(__file__).parent
sys.path.insert(0, str(bridge_dir))

from bridge.database import init_database, get_database
from bridge.conversation_mapper import ConversationMapper
from bridge.user_tracker import UserTracker

print("=== Phase 2 State Management Test ===\n")

# Initialize database
print("1. Initializing database...")
db_path = "storage/test_state.db"
if init_database(db_path):
    print("   ✓ Database initialized\n")
else:
    print("   ✗ Database initialization failed\n")
    sys.exit(1)

# Test ConversationMapper
print("2. Testing ConversationMapper...")
mapper = ConversationMapper(db_path)

# Create first conversation mapping
print("   Creating conversation mapping for #general...")
mapping1 = mapper.get_or_create_conversation(
    discord_channel_id="123456789",
    discord_guild_id="999888777",
    chorus_conversation_id="conv_abc123",
    chorus_thread_id=1,
    is_dm=False
)
print(f"   ✓ Created: Discord {mapping1['discord_channel_id']} -> Chorus {mapping1['chorus_conversation_id']}")

# Try to get existing mapping (should not create new)
print("   Fetching existing conversation mapping...")
mapping2 = mapper.get_or_create_conversation(
    discord_channel_id="123456789",
    discord_guild_id="999888777",
    chorus_conversation_id="conv_different",  # Different, but should use existing
    chorus_thread_id=999,  # Different, but should use existing
    is_dm=False
)
assert mapping1['chorus_conversation_id'] == mapping2['chorus_conversation_id'], "Should return existing mapping!"
print(f"   ✓ Retrieved existing mapping (same conversation ID)")

# Create DM conversation
print("   Creating DM conversation...")
dm_mapping = mapper.get_or_create_conversation(
    discord_channel_id="555444333",  # User ID for DMs
    discord_guild_id=None,
    chorus_conversation_id="conv_dm_xyz",
    chorus_thread_id=2,
    is_dm=True
)
print(f"   ✓ Created DM: Discord {dm_mapping['discord_channel_id']} -> Chorus {dm_mapping['chorus_conversation_id']}")

# Update last message time
print("   Updating last message time...")
mapper.update_last_message_time("123456789")
print("   ✓ Updated")

# List active conversations
print("   Listing active conversations...")
conversations = mapper.list_active_conversations()
print(f"   ✓ Found {len(conversations)} conversation(s)")
for conv in conversations:
    conv_type = "DM" if conv['is_dm'] else "Channel"
    print(f"      - {conv_type}: {conv['discord_channel_id']} ({conv['message_count']} messages)")

# Get stats
stats = mapper.get_conversation_stats()
print(f"   ✓ Stats: {stats['total_conversations']} total, {stats['dm_conversations']} DMs, {stats['channel_conversations']} channels\n")

# Test UserTracker
print("3. Testing UserTracker...")
tracker = UserTracker(db_path)

# Track first user
print("   Tracking user Alex...")
user1 = tracker.track_user(
    discord_user_id="111222333",
    username="alex_codes",
    display_name="Alex"
)
print(f"   ✓ Tracked: {user1['username']} (aliases: {user1['known_aliases']})")

# Track same user with different username (simulating username change)
print("   Tracking Alex with new username...")
user2 = tracker.track_user(
    discord_user_id="111222333",
    username="alex_the_dev",
    display_name="Alex"
)
print(f"   ✓ Updated: {user2['username']} (aliases: {user2['known_aliases']})")
assert "alex_codes" in user2['known_aliases'], "Old username should be in aliases!"
assert "alex_the_dev" in user2['known_aliases'], "New username should be in aliases!"

# Track second user
print("   Tracking user Sarah...")
user3 = tracker.track_user(
    discord_user_id="444555666",
    username="sarah_dev",
    display_name="Sarah"
)
print(f"   ✓ Tracked: {user3['username']}")

# Add custom alias
print("   Adding custom alias for Alex...")
tracker.update_aliases("111222333", "Fitzy")
user_updated = tracker.get_user_info("111222333")
print(f"   ✓ Added alias: {user_updated['known_aliases']}")
assert "Fitzy" in user_updated['known_aliases'], "Custom alias should be present!"

# List active users
print("   Listing active users...")
users = tracker.list_active_users()
print(f"   ✓ Found {len(users)} user(s)")
for user in users:
    print(f"      - {user['username']} ({user['message_count']} messages)")

# Get user stats
user_stats = tracker.get_user_stats()
print(f"   ✓ Stats: {user_stats['total_users']} total users, {user_stats['total_messages']} total messages\n")

# Test persistence (close and reopen)
print("4. Testing persistence...")
db = get_database(db_path)
db.close()

print("   Database closed, reopening...")
mapper2 = ConversationMapper(db_path)
tracker2 = UserTracker(db_path)

# Verify data persists
conv_check = mapper2.get_conversation_mapping("123456789")
user_check = tracker2.get_user_info("111222333")

if conv_check and user_check:
    print("   ✓ Data persisted across database close/reopen")
    print(f"      - Conversation: {conv_check['chorus_conversation_id']}")
    print(f"      - User: {user_check['username']} with {len(user_check['known_aliases'])} aliases")
else:
    print("   ✗ Data did not persist!")
    sys.exit(1)

print("\n=== All Phase 2 Tests Passed! ===")
print("\nPhase 2 State Management is working correctly:")
print("  ✓ Database initialization")
print("  ✓ Conversation mapping (create & retrieve)")
print("  ✓ User tracking (create & update)")
print("  ✓ Username change detection & alias tracking")
print("  ✓ Data persistence across restarts")
print("\nReady to test with Discord bot!")
