# Quick Start: Multiple Discord Bots

Run multiple AI characters on Discord simultaneously!

## One-Time Setup: Chorus Engine

```batch
cd J:\Dev\chorus-engine
start.bat
```

**Keep this running!** All bots connect to it.

---

## Per-Bot Setup (5 minutes each)

### 1. Copy Bridge Folder

```batch
# Copy to anywhere you want
xcopy /E /I J:\Dev\chorus-engine\chorus_discord_bridge C:\MyBots\nova-bot
xcopy /E /I J:\Dev\chorus-engine\chorus_discord_bridge C:\MyBots\marcus-bot
```

### 2. Install Each Bot

```batch
cd C:\MyBots\nova-bot
install_bridge.bat
```

### 3. Configure Each Bot

**Edit `.env`** (different Discord token for each):
```env
DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
CHORUS_API_URL=http://localhost:8000
```

**Edit `config.yaml`** (different character for each):
```yaml
chorus:
  character_id: "nova"  # or "marcus", "alex", etc.
```

### 4. Create Discord Bot Applications

For each bot:
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

Open separate terminals:

**Terminal 1:**
```batch
cd C:\MyBots\nova-bot
start_bridge.bat
```

**Terminal 2:**
```batch
cd C:\MyBots\marcus-bot
start_bridge.bat
```

**Done!** All bots are running.

---

## Usage

In Discord:
```
User: @Nova what do you think about this painting?
Nova: Oh, I love the use of color! The way...

User: @Marcus can you review this code?
Marcus: Let me take a look. First thing I notice...
```

---

## Key Points

✅ **No Port Conflicts**: Bots are clients, not servers  
✅ **Lightweight**: ~15MB per bot  
✅ **Independent**: Each has own environment  
✅ **Portable**: Copy folder anywhere  
✅ **Scalable**: Run as many as you want  

❌ **Don't duplicate Chorus Engine** - run once, many bridges!

---

## Troubleshooting

**Bot won't start?**
- Run `install_bridge.bat` in bot folder
- Check Chorus Engine is running (`start.bat`)
- Verify Discord token in `.env`

**Wrong character responding?**
- Check `character_id` in `config.yaml`
- Restart bot after changing config

**Want more details?**  
See [MULTI_BOT_SETUP.md](MULTI_BOT_SETUP.md)
