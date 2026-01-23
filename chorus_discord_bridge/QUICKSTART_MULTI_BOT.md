# Quick Start: Multiple Discord Bots

Run multiple AI characters on Discord from a single bridge instance!

---

## ⚠️ Performance Tip: Same LLM Model Recommended

**For optimal performance, it's highly recommended that all characters use the same LLM model in Chorus Engine.**

Check `characters/*.yaml`:
```yaml
# Recommended: All bots use same model
nova.yaml:    model: "dolphin-2.9.2-qwen2-7b"
marcus.yaml:  model: "dolphin-2.9.2-qwen2-7b"  # Same = faster
alex.yaml:    model: "dolphin-2.9.2-qwen2-7b"  # Same = faster
```

Different models will work but expect slower responses due to model loading/unloading on consumer hardware.

---

## One-Time Setup: Chorus Engine

```batch
cd J:\Dev\chorus-engine
start.bat
```

**Keep this running!** All bots connect to it.

---

## Bridge Setup (5 minutes)

### 1. Install Dependencies

```batch
cd chorus_discord_bridge
install_bridge.bat   # Windows
./install_bridge.sh  # Linux/Mac
```

### 2. Configure Environment Variables

**Edit `.env`** - different Discord token for each bot:
```env
CHORUS_API_URL=http://localhost:8000

# Bot tokens - use character ID in uppercase
DISCORD_BOT_TOKEN_NOVA=your_nova_token_here
DISCORD_BOT_TOKEN_MARCUS=your_marcus_token_here
```

### 3. Configure Bots Array

**Edit `config.yaml`**:
```yaml
bots:
  - character_id: "nova"
    bot_token_env: "DISCORD_BOT_TOKEN_NOVA"
    enabled: true
  
  - character_id: "marcus"
    bot_token_env: "DISCORD_BOT_TOKEN_MARCUS"
    enabled: true

# Discord Settings (shared)
discord:
  command_prefix: "!"
  rate_limit:
    per_user_cooldown: 2
    global_limit: 10
  enable_dm_support: true

# Chorus Settings (shared)
chorus:
  timeout: 30
  retry_attempts: 3

# Bridge Settings (shared)
bridge:
  log_level: "INFO"
  enable_typing_indicator: true
```

### 4. Create Discord Bot Applications

For each bot (Nova, Marcus, etc.):
1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Name it (e.g., "Nova Bot")
4. Go to Bot → Reset Token → Copy token → Paste in `.env`
5. Enable "Message Content Intent"
6. Go to OAuth2 → URL Generator:
   - Scopes: `bot`
   - Permissions: Send Messages, Read Message History
7. Copy URL and invite to your Discord server

### 5. Start All Bots

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
✓ Connected to Chorus Engine: http://localhost:8000
Found 2 enabled bot(s):
  - nova
  - marcus

Initializing 2 bot(s)...
✓ All bots initialized

Connecting to Discord...
Bot 'nova' connected as NovaBot#1234
Bot 'marcus' connected as MarcusBot#5678
```

**Done!** All bots are running from a single process.

---

## Usage

In Discord:
```
User: @Nova what do you think about this painting?
Nova: Oh, I love the use of color! The way...

User: @Marcus can you review this code?
Marcus: Let me take a look. First thing I notice...
```

**Both bots in same channel:**
```
User: @Nova and @Marcus what do you both think?
Nova: I'd say the creative possibilities are endless!
Marcus: While Nova focuses on creativity, I prefer logic...
```

---

## Key Points

✅ **Single Process**: One bridge runs all bots  
✅ **Independent Conversations**: Each bot tracks separately  
✅ **Easy Updates**: Change code once, all bots updated  
⚠️ **Same LLM Model**: Highly recommended for performance!  

---

## Troubleshooting

**Bots won't start?**
- Check logs: `storage/bridge.log`
- Verify tokens in `.env` match character IDs
- Ensure Chorus Engine is running (`start.bat`)
- Check characters exist in `characters/` folder

**Slow responses?**
- **Most likely**: Different models in character configs causing loading delays
- **Recommended**: Configure all characters to use same model
- Or accept slower performance with diverse models
- Restart Chorus Engine and bridge after changes

**Wrong character responding?**
- Verify `character_id` in `config.yaml` matches character YAML filename
- Restart bridge after config changes

---

## Managing Bots

**Disable a bot temporarily:**
```yaml
bots:
  - character_id: "nova"
    enabled: false  # Won't start
```

**Add a new bot:**
1. Create Discord application and get token
2. Add to `.env`: `DISCORD_BOT_TOKEN_NEWBOT=token_here`
3. Add to `config.yaml` bots array
4. **Recommended: Use same LLM model as other bots**
5. Restart bridge

---

## Want More Details?

See [MULTI_BOT_SETUP.md](MULTI_BOT_SETUP.md) for comprehensive guide including:
- Resource usage details
- Advanced configuration (including multi-model setups)
- Migration from folder duplication
- FAQ and best practices
