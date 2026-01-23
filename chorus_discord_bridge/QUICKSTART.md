# Discord Bridge - Quick Reference

## Overview

The Discord Bridge connects Chorus Engine characters to Discord. 

**New**: Multi-bot support! Run multiple characters from one instance. See [QUICKSTART_MULTI_BOT.md](QUICKSTART_MULTI_BOT.md)

**⚠️ Performance Tip**: When running multiple bots, it's highly recommended to use the same LLM model for all characters to avoid model loading/unloading delays on consumer hardware.

## Installation Commands

### Windows
```bash
cd chorus_discord_bridge
install_bridge.bat
```

### Linux/Mac
```bash
cd chorus_discord_bridge
chmod +x install_bridge.sh start_bridge.sh
./install_bridge.sh
```

## Running the Bridge

### Windows
```bash
cd chorus_discord_bridge
start_bridge.bat
```

### Linux/Mac
```bash
cd chorus_discord_bridge
./start_bridge.sh
```

## Configuration

### Single Bot Setup

├── chorus_engine/               # Main Chorus Engine
├── start.bat / start.sh         # Start Chorus Engine
│
└── chorus_discord_bridge/       # Discord Bridge
    ├── bridge/                  # Bridge code
    ├── storage/                 # Bridge state and logs
    ├── config.yaml              # Configuration (bots array)
    ├── .env                     # Bot token
    bot_token_env: "DISCORD_BOT_TOKEN"
    enabled: true
```

### Multiple Bots Setup

See [QUICKSTART_MULTI_BOT.md](QUICKSTART_MULTI_BOT.md) for detailed multi-bot setup.

**⚠️ Performance Tip**: It's highly recommended that all characters use the same LLM model for optimal performance on consumer hardware.

## File Structure

```
chorus-engine/                    # Root (same as before)
├── python_embeded/              # Windows: Shared embedded Python
├── venv/                        # Linux/Mac: Shared virtual env
├── chorus_engine/               # Main Chorus Engine (unchanged)
├── start.bat / start.sh         # Start Chorus Engine
├── install.bat / install.sh     # Install Chorus Engine
│
└── chorus_discord_bridge/       # NEW: Discord Bridge
    ├── bridge/                  # Bridge code
    ├── storage/                 # Bridge state and logs
    ├── scripts/                 # Helper scripts
    ├── install_bridge.bat/sh    # Install bridge deps
    └── start_bridge.bat/sh      # Start bridge
```

## Dependencies

The bridge adds these packages to the **shared** Python environment:
- `discord.py` - Discord bot framework
- `requestsrequires these packages:
- `discord.py` - Discord bot framework
- `requests` - HTTP client
- `pyyaml` - YAML parsing
- `python-dotenv` - Environment variables
- `aiosqlite` - Async SQLite (for state management)

All automatically installed by `install_bridge.bat/sh`.

## Configuration Files

- `.env` - Bot tokens and Chorus Engine URL (secret)
- `config.yaml` - Bridge settings and bots arrayogs

## Common Tasks

### Check if Chorus Engine is running
```bash8000/api/characters
```

### View bridge logs
```bash
type chorus_discord_bridge\storage\bridge.log     # Windows
tail -f chorus_discord_bridge/storage/bridge.log  # Linux/Mac
```

### Add a new bot
1. Create Discord application and get token
2. Add to `.env`: `DISCORD_BOT_TOKEN_NEWBOT=token_here`
3. Add to `config.yaml` bots array
4. **Recommended: Use same LLM model as other bots for best performance**
5. Restart bridge

### Enable/disable a bot
Edit `config.yaml`:
```yaml
bots:
  - character_id: "nova"
    enabled: true   # Active
  - character_id: "marcus"
    enabled: false  # Disabled
```
Restart bridge to apply. install -r chorus_discord_bridge/requirements.txt
```

## Typical Workflow

1. **First time setup:**
   ```bash
   # If you haven't installed Chorus Engine yet:
   install.bat          # Windows
   ./install.sh         # Linux/Mac
   
   # Then install Discord Bridge:
   cd chorus_discord_bridge
   install_bridge.bat   # Windows
   ./install_bridge.sh  # Linux/Mac
   ```
nstall Chorus Engine first (if not done)
   cd chorus-engine
   install.bat          # Windows
   ./install.sh         # Linux/Mac
   
   # Then install Discord Bridge
   cd chorus_discord_bridge
   install_bridge.bat   # Windows
   ./install_bridge.sh  # Linux/Mac
   ```

2. **Edit Discord configuration:**
   ```bash
   # Copy templates
   copy .env.template .env                     # Windows
   copy config.yaml.template config.yaml
   
   cp .env.template .env                       # Linux/Mac
   cp config.yaml.template config.yaml
   
   # Edit files
   notepad .env          # Windows
   notepad config.yaml
   
   nano .env             # Linux/Mac
   nano config.yaml
   ```

3. **Start both services:**
   ```bash
   # Terminal 1: Start Chorus Engine
   cd chorus-ennel
   - Cannot connect to Chorus Engine"
→ Start Chorus Engine first with `start.bat`
→ Check `CHORUS_API_URL` in `.env` (default: http://localhost:8000)

### "discord.py not found"
→ Run `install_bridge.bat` to install bridge dependencies

### Bot responds slowly
→ **Most common**: Different LLM models causing loading/unloading delays
→ **Recommended**: Configure all characters to use same model in `characters/*.yaml`
→ Or accept slower performance with diverse models
→ Restart Chorus Engine and bridge after changes

### Wrong character responding
→ Verify `character_id` in `config.yaml` matches character YAML filename
→ Restart bridge after config changes

### Changes to requirements.txt not taking effect
→ Re-run `install_bridge.bat` to update dependencies

---

## Multi-Bot Setup

See [QUICKSTART_MULTI_BOT.md](QUICKSTART_MULTI_BOT.md) for running multiple characters.

**⚠️ Performance Tip**: It's highly recommended that all characters use the same LLM model for optimal performance!
→ Run `install_bridge.bat` to install bridge dependencies

### Changes to requirements.txt not taking effect
→ Re-run `install_bridge.bat` to update dependencies
