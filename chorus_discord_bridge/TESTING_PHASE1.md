# Phase 1 Testing Guide

## What We've Built

Phase 1 (Tasks 1.1-1.5) is complete! We now have:

✅ Complete project structure
✅ Configuration system with validation
✅ Chorus API client with retry logic
✅ Discord bot with @mention detection
✅ Basic message flow through Chorus Engine
✅ Connection test script

## Testing Steps

### Step 1: Verify Project Structure

Check that the following structure exists:

```
chorus_discord_bridge/
├── bridge/
│   ├── __init__.py
│   ├── main.py          # Entry point
│   ├── bot.py           # Discord bot
│   ├── config.py        # Configuration loader
│   └── chorus_client.py # Chorus API client
├── scripts/
│   ├── create_bot.md    # Discord bot creation guide
│   └── test_connection.py # Connection test
├── storage/             # Will contain logs and state.db
├── requirements.txt
├── setup.py
├── config.yaml.template
├── .env.template
└── README.md
```

### Step 2: Install Dependencies

**IMPORTANT**: The Discord Bridge uses the same embedded Python / venv as Chorus Engine!

**Windows:**
```bash
cd j:\Dev\chorus-engine\chorus_discord_bridge
install_bridge.bat
```

**Linux/Mac:**
```bash
cd chorus_discord_bridge
./install_bridge.sh
```

This will:
- Use the existing `python_embeded/` (Windows) or `venv/` (Linux/Mac) from Chorus Engine
- Install Discord Bridge dependencies into the same environment
- Create `.env` and `config.yaml` from templates

Expected output: Installation of discord.py, requests, pyyaml, python-dotenv, aiosqlite into the shared Python environment

### Step 3: Configure Environment

1. **Create .env file:**
   ```bash
   copy .env.template .env
   ```

2. **Edit .env** (use your Discord bot token):
   ```
   DISCORD_BOT_TOKEN=your_actual_bot_token_here
   CHORUS_API_URL=http://localhost:5000
   ```

   **Note**: You need to create a Discord bot first (see `scripts/create_bot.md`)

3. **Edit config.yaml** (optional - created by install script):
   - File location: `chorus_discord_bridge\config.yaml`
   - Verify `character_id: "nova"` or change to your character
   - Defaults should work fine for testing

### Step 4: Test Chorus Engine Connection

**Make sure Chorus Engine is running first!** (Run `start.bat` from the main directory)

**Windows:**
```bash
cd j:\Dev\chorus-engine
python_embeded\python.exe chorus_discord_bridge\scripts\test_connection.py
```

**Linux/Mac:**
```bash
cd chorus-engine
source venv/bin/activate
python chorus_discord_bridge/scripts/test_connection.py
```

**Expected Output:**
```
============================================================
Chorus Engine Connection Test
============================================================

1. Loading configuration...
   ✓ Config loaded
   - API URL: http://localhost:5000
   - Character: nova

2. Creating Chorus API client...
   ✓ Client created

3. Testing API health...
   ✓ API is healthy

4. Testing character loading...
   ✓ Character loaded: Nova
   - Description: ...

5. Testing conversation creation...
   ✓ Conversation created
   - Conversation ID: ...
   - Thread ID: ...

6. Testing message sending...
   ✓ Message sent and response received
   - Response preview: ...
   ⚠ Metadata not returned (may need Task 1.6)

============================================================
✓ All tests passed!
============================================================
```

**Troubleshooting:**
- ✗ API health check failed → Start Chorus Engine (`start.bat` in main directory)
- ✗ Failed to load character → Check `character_id` in `config.yaml` matches existing character
- ✗ Configuration Error → Check `.env` and `config.yaml` exist and are valid

### Step 5: Create Discord Bot (if not done yet)

Follow the guide in `scripts/create_bot.md`:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
**Windows:**
```bash
cd j:\Dev\chorus-engine\chorus_discord_bridge
start_bridge.bat
```

**Linux/Mac:**
```bash
cd chorus_discord_bridge
./start_bridge.sh
```

The startup script will:
- Check that Chorus Engine is running
- Use the same embedded Python / venv as Chorus Engine  
- Start the Discord Bridge with proper loggingEnable "MESSAGE CONTENT INTENT" (required!)
6. Generate invite URL and add bot to your test server

### Step 6: Run the Discord Bridge

```bash
cd j:\Dev\chorus-engine\chorus_discord_bridge
python -m bridge.main
```

**Expected Output:**
```
============================================================
Chorus Discord Bridge v0.1.0
============================================================
Loading configuration...
✓ Configuration loaded: BridgeConfig(character='nova', api_url='http://localhost:5000', log_level='INFO')

Starting Discord bot...
Connecting to Discord...
(Press Ctrl+C to stop)

2026-01-20 15:30:45 - bridge.bot - INFO - ChorusBot initialized
2026-01-20 15:30:45 - bridge.bot - INFO - Setting up bot...
2026-01-20 15:30:45 - bridge.bot - INFO - ✓ Connected to Chorus Engine
2026-01-20 15:30:45 - bridge.bot - INFO - ✓ Active character: Nova
2026-01-20 15:30:46 - bridge.bot - INFO - Bot connected as YourBotName (ID: ...)
2026-01-20 15:30:46 - bridge.bot - INFO - Connected to 1 server(s)
2026-01-20 15:30:46 - bridge.bot - INFO - Bot is ready!
```

### Step 7: Test in Discord

**Test 1: @mention in channel**

In your Discord server:
```
@YourBotName hello!
```

Expected behavior:
- Bot shows "typing..."
- Bot responds with character's message
- Check logs for message flow

**Test 2: Direct Message**

Send DM to bot:
```
Hi there! Can you help me?
```

Expected behavior:
- Bot responds in DM
- Separate conversation from channel (different conversation_id)

**Test 3: Multiple messages**

Send several messages:
```
@YourBotName what's your favorite color?
@YourBotName do you remember what I just asked?
@YourBotName tell me about yourself
```

Expected behavior:
- Bot maintains context within the channel
- Conversation continues (uses same conversation_id)

### Step 8: Verify Logs

Check the log file:
```
type storage\bridge.log
```

Look for:
- "Message from [user] in #[channel]"
- "Sending to Chorus: conversation=[id], thread=[id]"
- "Sent reply ([N] chars)"
- No errors or exceptions

## Known Limitations (Phase 1)

1. **No message history** - Only current message sent (Phase 3 will add history)
2. **No deduplication** - Each message creates new Chorus message (Phase 3)
3. **No persistence** - Conversation mappings lost on restart (Phase 2 will add DB)
4. **No rate limiting** - Can be spammed (Phase 4)
5. **No typing refresh** - Long responses may show typing timeout (Phase 4)
6. **Metadata not stored** - Task 1.6 needed for metadata support

## What to Test

### ✅ Should Work
- [x] Bot connects to Discord
- [x] Bot connects to Chorus Engine
- [x] Bot responds to @mentions in channels
- [x] Bot responds to DMs
- [x] Bot maintains separate conversations per channel
- [x] Bot shows typing indicator
- [x] Bot handles long responses (splits at 2000 chars)
- [x] Bot logs all activity

### ⚠️ Not Yet Implemented
- [ ] Message history context (Phase 3)
- [ ] Deduplication (Phase 3)
- [ ] Persistent state across restarts (Phase 2)
- [ ] Rate limiting (Phase 4)
- [ ] Image generation support (Phase 4)
- [ ] Character switching (Phase 5)

## Next Steps After Testing

Once Phase 1 is working:

1. **Confirm Task 1.6**: Should we add metadata support to Chorus API?
   - Currently metadata is sent but not stored
   - Required for Phase 3 (message deduplication)
   - Small change to `chorus_engine/api/app.py`

2. **Proceed to Phase 2**: State management with SQLite
   - Persistent conversation mappings
   - User tracking
   - Survives restarts

## Troubleshooting

### Bot doesn't connect to Discord
- Check `DISCORD_BOT_TOKEN` in `.env`
- Verify token is valid (regenerate if needed)
- Check internet connection

### Bot doesn't respond
- Ensure "MESSAGE CONTENT INTENT" is enabled in Discord Developer Portal
- Check bot has permissions in channel (Read/Send Messages)
- Verify bot is @mentioned or sent a DM
- Check logs for errors

### "Cannot connect to Chorus Engine"
- Start Chorus Engine first (`start.bat`)
- Check `CHORUS_API_URL` in `.env`
- Test directly: `curl http://localhost:5000/api/characters`

### "Failed to load character"
- Check `character_id` in `config.yaml` matches existing character
- List characters: `curl http://localhost:5000/api/characters`

### Bot crashes or errors
- Check `storage/bridge.log` for stack traces
- Common issues:
  - Missing dependencies → `pip install -r requirements.txt`
  - Invalid config → Check YAML syntax
  - Network issues → Check firewalls

## Success Criteria

✅ **Phase 1 Complete When:**
- [x] Bot connects to Discord successfully
- [x] Bot connects to Chorus Engine successfully
- [x] Bot responds to @mentions with character's message
- [x] Bot responds to DMs
- [x] Bot maintains separate conversations per channel
- [x] Logs show clean message flow
- [x] No crashes or errors during normal operation

**Ready to proceed to Phase 2!**
