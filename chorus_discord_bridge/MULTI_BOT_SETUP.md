# Multi-Bot Setup Guide

Run multiple AI characters on Discord from a single bridge instance!

---

## ⚠️ IMPORTANT: LLM Model Configuration

**When running multiple bots, it is highly recommended that all characters use the same LLM model in Chorus Engine.**

### Why This Matters:

The system **will work** with different models per character, but on most consumer hardware you'll experience:
- ⚠️ **Constant model loading/unloading** - Severe performance degradation
- ⚠️ **VRAM exhaustion** - Multiple large models may not fit in GPU memory
- ⚠️ **Slow response times** - Model swapping adds 10-30 seconds per response
- ⚠️ **Potential crashes** - System instability under memory pressure

**Best Practice:**
```yaml
# In character YAML files - use SAME model for all Discord bots
nova.yaml:
  llm:
    provider: "koboldcpp"
    model: "dolphin-2.9.2-qwen2-7b"  # ✅ Same model

marcus.yaml:
  llm:
    provider: "koboldcpp"
    model: "dolphin-2.9.2-qwen2-7b"  # ✅ Same model

alex.yaml:
  llm:
    provider: "koboldcpp"
    model: "dolphin-2.9.2-qwen2-7b"  # ✅ Same model
```

**If you need different models**, consider:
- Running separate Chorus Engine instances (different ports)
- Using different bridge instances pointing to different Chorus engines
- Accepting slower response times when switching between characters

---

## Overview

The Discord Bridge now supports running **multiple characters from a single instance**:

✅ **Single Process**: One bridge instance runs multiple Discord bots  
✅ **Shared Resources**: Efficient use of memory and API connections  
✅ **Independent Conversations**: Each bot maintains separate conversation history  
✅ **Easy Management**: Update code once, all bots benefit  
✅ **Simple Configuration**: One config file, multiple bots  

---

## Architecture

```
chorus-engine/                    # Main engine (runs once)
├── start.bat                    # Start Chorus Engine API
├── characters/                  # All character definitions
│   ├── nova.yaml               # ⚠️ Use same LLM model
│   ├── marcus.yaml             # ⚠️ Use same LLM model
│   └── alex.yaml               # ⚠️ Use same LLM model
└── chorus_discord_bridge/      # Single bridge instance
    ├── bridge/                 # Bridge code
    ├── config.yaml             # Multi-bot configuration
    ├── .env                    # Multiple bot tokens
    ├── install_bridge.bat      # Setup script
    └── start_bridge.bat        # Start all bots
```

**Single instance runs all bots concurrently!**

---

## Setup Process

### Step 1: Start Chorus Engine

```batch
cd J:\Dev\chorus-engine
start.bat
```

Keep this running! All bots connect to this API.

---

### Step 2: Create Discord Bot Applications

For each character, create a Discord application:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application**
3. Name it (e.g., "Nova Bot", "Marcus Bot")
4. Go to **Bot** tab
5. Click **Reset Token** and copy the token
6. Enable these under **Privileged Gateway Intents**:
   - ✅ Message Content Intent
   - ✅ Server Members Intent (optional)

Repeat for each character you want to deploy.

---

### Step 3: Configure Environment Variables

Edit `.env` in the bridge folder:

```env
# Chorus Engine connection
CHORUS_API_URL=http://localhost:8000

# Bot tokens - use character ID in uppercase
DISCORD_BOT_TOKEN_NOVA=your_nova_bot_token_here
DISCORD_BOT_TOKEN_MARCUS=your_marcus_bot_token_here
DISCORD_BOT_TOKEN_ALEX=your_alex_bot_token_here

# Pattern: DISCORD_BOT_TOKEN_<CHARACTER_ID_UPPERCASE>
```

**Important**: Each bot needs a **different** Discord token!

---

### Step 4: Configure Bots Array

Edit `config.yaml`:

```yaml
# Bot Configurations
bots:
  - character_id: "nova"
    bot_token_env: "DISCORD_BOT_TOKEN_NOVA"
    enabled: true
  
  - character_id: "marcus"
    bot_token_env: "DISCORD_BOT_TOKEN_MARCUS"
    enabled: true
  
  - character_id: "alex"
    bot_token_env: "DISCORD_BOT_TOKEN_ALEX"
    enabled: false  # Disabled - won't start

# Discord Settings (shared across all bots)
discord:
  command_prefix: "!"
  rate_limit:
    per_user_cooldown: 2
    global_limit: 10
  max_history_fetch: 10
  enable_dm_support: true

# Chorus Engine Settings (shared)
chorus:
  timeout: 30
  retry_attempts: 3
  retry_delay: 2

# Bridge Settings (shared)
bridge:
  state_db_path: "storage/state.db"
  log_level: "INFO"
  enable_typing_indicator: true
```

**Notes:**
- Set `enabled: false` to temporarily disable a bot without removing it
- All bots share Discord and Chorus settings (rate limits, timeouts, etc.)
- Each bot maintains **separate conversation history** via character_id

---

### Step 5: Configure Character LLM Models

**⚠️ Highly Recommended: Configure all characters to use the same LLM model for optimal performance.**

Edit character YAML files in `chorus-engine/characters/`:

```yaml
# nova.yaml
id: nova
name: Nova
llm:
  provider: "koboldcpp"
  model: "dolphin-2.9.2-qwen2-7b"  # ✅ Recommended: Same for all bots
  # ... other settings

# marcus.yaml
id: marcus
name: Marcus
llm:
  provider: "koboldcpp"
  model: "dolphin-2.9.2-qwen2-7b"  # ✅ Same model for better performance
  # ... other settings

# alex.yaml
id: alex
name: Alex
llm:
  provider: "koboldcpp"
  model: "dolphin-2.9.2-qwen2-7b"  # ✅ Same model avoids swapping delays
  # ... other settings
```

---

### Step 6: Install Bridge Dependencies

```batch
cd chorus_discord_bridge
install_bridge.bat    # Windows
./install_bridge.sh   # Linux/Mac
```

---

### Step 7: Invite Bots to Discord

For each bot:

1. Go to Discord Developer Portal → Your Application → OAuth2 → URL Generator
2. Select scopes:
   - ✅ `bot`
3. Select bot permissions:
   - ✅ Send Messages
   - ✅ Read Message History
   - ✅ Add Reactions (optional)
   - ✅ Use Slash Commands (optional)
4. Copy the generated URL
5. Open in browser and invite to your server

**All bots can be in the same Discord server!**

---

### Step 8: Start All Bots

```batch
cd chorus_discord_bridge
start_bridge.bat      # Windows
./start_bridge.sh     # Linux/Mac
```

Expected output:
```
============================================================
Chorus Discord Bridge v0.2.0 - Multi-Bot Support
============================================================
Loading configuration...
✓ Connected to Chorus Engine: http://localhost:8000
Found 2 enabled bot(s):
  - nova
  - marcus

Initializing 2 bot(s)...
  - nova
  - marcus
✓ All bots initialized

Connecting to Discord...
(Press Ctrl+C to stop)

[2026-01-22 15:30:45] INFO: Bot 'nova' connected as NovaBot#1234
[2026-01-22 15:30:46] INFO: Bot 'marcus' connected as MarcusBot#5678
```

**All bots are now running from a single process!**

---

## Usage in Discord

### In Different Channels

```
#nova-chat:
User: @Nova what do you think about this painting?
Nova: Oh, I love the use of color! The way the artist...

#marcus-chat:
User: @Marcus can you review this code?
Marcus: Let me take a look. First thing I notice is...
```

### In the Same Channel

```
#general:
User: @Nova what's your take on time travel stories?
Nova: I'm fascinated by Ted Chiang's approach...

User: @Marcus what about you?
Marcus: From a logical perspective, I find the paradoxes...

User: @Nova and @Marcus what do you both think?
Nova: I'd say the creative possibilities are endless!
Marcus: While Nova focuses on creativity, I prefer analyzing consistency...
```

**Each bot maintains separate conversation history!**

---

## Management

### Enabling/Disabling Bots

Edit `config.yaml`:

```yaml
bots:
  - character_id: "nova"
    enabled: true   # Active
  
  - character_id: "marcus"
    enabled: false  # Disabled - won't start
```

Restart the bridge to apply changes.

### Adding a New Bot

1. Create Discord application and get token
2. Add to `.env`:
   ```env
   DISCORD_BOT_TOKEN_NEWBOT=your_token_here
   ```
3. Add to `config.yaml`:
   ```yaml
   bots:
     - character_id: "newbot"
       bot_token_env: "DISCORD_BOT_TOKEN_NEWBOT"
       enabled: true
   ```
4. **⚠️ Recommended: Configure character to use same LLM model as other bots**
5. Restart bridge

### Updating Code

When you update bridge code:
1. Stop the bridge (Ctrl+C)
2. Pull/edit code changes
3. Restart with `start_bridge.bat`

**All bots automatically get the update!** No copying to multiple folders.

---

## Resource Usage

### Single Bridge Instance:
- **CPU**: Minimal (only active when bots respond)
- **RAM**: ~100-150MB total (regardless of bot count)
- **Network**: Outbound only (to Discord and Chorus Engine)

### Chorus Engine (the heavy one):
- **GPU/VRAM**: Depends on LLM model size
- **RAM**: Depends on model (typically 4-16GB)
- **CPU**: Model inference

**The bridge is lightweight - Chorus Engine does the heavy lifting!**

---

## Troubleshooting

### Bots Don't Start

**Check logs**: `storage/bridge.log`

**Common issues**:
- ❌ Wrong token in `.env` → Verify token matches character_id
- ❌ Character doesn't exist → Check `characters/` folder
- ❌ Chorus Engine not running → Start with `start.bat`
- ❌ Token environment variable not found → Check spelling (uppercase!)

### Slow Responses

**Symptoms**: Bots take 10+ seconds to respond

**Likely cause**: Different LLM models causing constant loading/unloading

**Solution**:
1. Check character YAML files
2. Ensure all use **same model**
3. Restart Chorus Engine
4. Restart bridge

### Wrong Character Responding

**Check**:
1. Verify `character_id` in `config.yaml` matches character YAML filename
2. Check bot token corresponds to correct Discord application
3. Restart bridge after config changes

### Memory Issues

**If conversation history grows too large**:

Edit `config.yaml`:
```yaml
discord:
  max_history_fetch: 5  # Reduce from 10
```

**If Chorus Engine crashes**:
- Likely model loading/unloading issue (different models per character)
- **⚠️ Recommended: Use same model for all characters**
- Or reduce number of enabled bots
- Or upgrade hardware (more VRAM)

---

## Best Practices

### 1. ⚠️ Use Same LLM Model for All Bots (Highly Recommended)

**For best performance on consumer hardware:**

```yaml
# ✅ RECOMMENDED - All use same model (optimal performance)
nova.yaml:    model: "dolphin-2.9.2-qwen2-7b"
marcus.yaml:  model: "dolphin-2.9.2-qwen2-7b"
alex.yaml:    model: "dolphin-2.9.2-qwen2-7b"

# ⚠️ WORKS BUT SLOWER - Different models (expect model swapping delays)
nova.yaml:    model: "dolphin-2.9.2-qwen2-7b"
marcus.yaml:  model: "llama-3-70b-instruct"     # Will cause loading delays
alex.yaml:    model: "mistral-7b-instruct"      # Will cause loading delays
```

### 2. Organize Bot Tokens Clearly

```env
# Group by character
DISCORD_BOT_TOKEN_NOVA=...
DISCORD_BOT_TOKEN_MARCUS=...
DISCORD_BOT_TOKEN_ALEX=...
```

### 3. Use Descriptive Bot Names

In Discord Developer Portal, name applications clearly:
- ✅ "Nova - Creative AI"
- ✅ "Marcus - Technical Analyst"
- ❌ "Bot 1", "Bot 2"

### 4. Monitor Logs

Logs are in `storage/bridge.log`. Check for:
- Connection errors
- Rate limit warnings
- Character loading issues

### 5. Start Small

Start with 2 bots, verify everything works, then add more.

---

## Advanced Configuration

### Per-Character Behavior

While rate limits are shared, you can control bot behavior via character personality:

```yaml
# nova.yaml - More talkative
personality:
  verbosity: "high"

# marcus.yaml - More concise
personality:
  verbosity: "low"
```

### Different Models (Advanced)

If you **must** use different models:

1. Run separate Chorus Engine instances on different ports:
   ```batch
   # Engine 1 - Port 8000
   set PORT=8000
   set MODEL=dolphin-2.9.2-qwen2-7b
   start.bat
   
   # Engine 2 - Port 8001
   set PORT=8001
   set MODEL=llama-3-70b-instruct
   start.bat
   ```

2. Run separate bridge instances pointing to different engines
3. **Not recommended** - high resource usage!

---

## FAQ

**Q: How many bots can I run?**  
A: Technically unlimited, but practical limit is your Discord API rate limits and Chorus Engine capacity. 3-5 bots is typical.

**Q: Do bots share conversation history?**  
A: No! Each bot maintains separate conversations via `character_id`.

**Q: Can bots respond to each other?**  
A: Yes! In shared channels, bots can see and respond to each other's messages.

**Q: What if I want different LLM models?**  
A: **Not recommended** due to performance issues. If necessary, run separate Chorus Engine instances on different ports, and point different bridges at each.

**Q: Can I run this in Docker?**  
A: Yes, but you'd containerize: 1) Chorus Engine, 2) Bridge instance. Not multiple bridges - one bridge runs all bots.

**Q: Do I need multiple Chorus Engine instances?**  
A: No! One Chorus Engine serves all bots. This is the key efficiency.

**Q: What about model context limits?**  
A: Each character's conversations are separate, so context doesn't mix. Chorus Engine manages per-character context.

---

## Summary

✅ **Single Instance**: One bridge runs all bots  
✅ **Easy Updates**: Change code once, all bots updated  
✅ **Resource Efficient**: Shared API connections  
✅ **Independent Conversations**: Each bot tracks separately  
⚠️ **Same LLM Model**: Highly recommended for optimal performance!  

**Core Principle**: One Chorus Engine + One Bridge = Many Discord Bots
