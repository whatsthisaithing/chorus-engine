#!/bin/bash
# Discord Bridge - Linux/Mac Installation Script
# Uses the same venv as Chorus Engine

echo "============================================"
echo " Discord Bridge - Installation (Linux/Mac)"
echo "============================================"
echo ""

# Change to Chorus Engine root directory
cd "$(dirname "$0")/.."

# Check for venv (should exist from main install)
if [ -d "venv" ]; then
    echo "[OK] Found virtual environment from Chorus Engine"
else
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run the main Chorus Engine installation first:"
    echo "  ./install.sh"
    echo ""
    echo "This will set up the venv that both"
    echo "Chorus Engine and Discord Bridge will use."
    exit 1
fi

# Activate virtual environment
echo ""
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo ""
echo "[1/2] Installing Discord Bridge dependencies..."
echo ""

# Install discord bridge requirements
pip install -r chorus_discord_bridge/requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo "[OK] Dependencies installed"
echo ""

# Check if .env exists
if [ ! -f "chorus_discord_bridge/.env" ]; then
    echo "[INFO] Creating .env from template..."
    cp chorus_discord_bridge/.env.template chorus_discord_bridge/.env
    echo ""
    echo "[ACTION REQUIRED] Please edit chorus_discord_bridge/.env"
    echo "and add your Discord bot token:"
    echo "  DISCORD_BOT_TOKEN=your_bot_token_here"
    echo ""
fi

# Check if config.yaml exists
if [ ! -f "chorus_discord_bridge/config.yaml" ]; then
    echo "[INFO] Creating config.yaml from template..."
    cp chorus_discord_bridge/config.yaml.template chorus_discord_bridge/config.yaml
    echo "[OK] Config file created with defaults"
    echo ""
fi

echo "============================================"
echo " Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Create a Discord bot (see chorus_discord_bridge/scripts/create_bot.md)"
echo "  2. Edit chorus_discord_bridge/.env with your bot token"
echo "  3. Run: ./start_bridge.sh"
echo ""

# Deactivate venv
deactivate
