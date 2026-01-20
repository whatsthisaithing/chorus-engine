# Discord Bridge - Quick Reference

## ✅ Uses Shared Python Environment

The Discord Bridge uses the **same** Python environment as Chorus Engine:
- **Windows**: `python_embeded/python.exe` (embedded Python)
- **Linux/Mac**: `venv/` (virtual environment)

**Benefits:**
- ✅ No dependency hell
- ✅ Consistent Python version
- ✅ Shared packages reduce disk space
- ✅ Easier maintenance

## Installation Commands

### Windows
```bash
cd j:\Dev\chorus-engine\chorus_discord_bridge
install_bridge.bat
```

### Linux/Mac
```bash
cd chorus-engine/chorus_discord_bridge
./install_bridge.sh
```

## Running the Bridge

### Windows
```bash
cd j:\Dev\chorus-engine\chorus_discord_bridge
start_bridge.bat
```

### Linux/Mac
```bash
cd chorus-engine/chorus_discord_bridge
./start_bridge.sh
```

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
- `requests` - Already in Chorus Engine
- `pyyaml` - Already in Chorus Engine
- `python-dotenv` - Already in Chorus Engine
- `aiosqlite` - New (for bridge state management)

## Configuration Files

Bridge-specific config (in `chorus_discord_bridge/`):
- `.env` - Discord bot token (secret)
- `config.yaml` - Bridge settings
- `storage/state.db` - Bridge state (created on first run)
- `storage/bridge.log` - Bridge logs

## Common Tasks

### Check if Chorus Engine is running
```bash
curl http://localhost:5000/api/characters
```

### Test bridge connection to Chorus
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

### View bridge logs
```bash
type chorus_discord_bridge\storage\bridge.log     # Windows
tail -f chorus_discord_bridge/storage/bridge.log  # Linux/Mac
```

### Update bridge dependencies
**Windows:**
```bash
cd j:\Dev\chorus-engine
python_embeded\python.exe -m pip install -r chorus_discord_bridge\requirements.txt
```

**Linux/Mac:**
```bash
source venv/bin/activate
pip install -r chorus_discord_bridge/requirements.txt
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

2. **Edit Discord configuration:**
   ```bash
   notepad chorus_discord_bridge\.env          # Windows
   nano chorus_discord_bridge/.env             # Linux/Mac
   ```

3. **Start both services:**
   ```bash
   # Terminal 1: Start Chorus Engine
   start.bat / ./start.sh
   
   # Terminal 2: Start Discord Bridge
   cd chorus_discord_bridge
   start_bridge.bat / ./start_bridge.sh
   ```

4. **Test in Discord:**
   - @mention your bot in a channel
   - Or send it a direct message

## Troubleshooting

### "Embedded Python not found"
→ Run main `install.bat` first to set up Python

### "Cannot connect to Chorus Engine"
→ Start Chorus Engine first with `start.bat`

### "discord.py not found"
→ Run `install_bridge.bat` to install bridge dependencies

### Changes to requirements.txt not taking effect
→ Re-run `install_bridge.bat` to update dependencies
