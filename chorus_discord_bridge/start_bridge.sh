#!/bin/bash
# Discord Bridge - Linux/Mac Startup Script
# Uses venv from Chorus Engine

echo "============================================"
echo "   Discord Bridge - Starting Up"
echo "============================================"
echo ""

# Change to Chorus Engine root directory
cd "$(dirname "$0")/.."

# Check for venv
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run the installation script first:"
    echo "  chorus_discord_bridge/install_bridge.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo "[OK] Using virtual environment from Chorus Engine"

# Check if config files exist
if [ ! -f "chorus_discord_bridge/.env" ]; then
    echo "[ERROR] Configuration not found!"
    echo ""
    echo "Please run the installation script first:"
    echo "  chorus_discord_bridge/install_bridge.sh"
    exit 1
fi

if [ ! -f "chorus_discord_bridge/config.yaml" ]; then
    echo "[ERROR] Config file not found!"
    echo ""
    echo "Please run the installation script first:"
    echo "  chorus_discord_bridge/install_bridge.sh"
    exit 1
fi

echo "[OK] Configuration found"
echo ""

# Check if Chorus Engine is running
echo "[INFO] Checking Chorus Engine connection..."
if python chorus_discord_bridge/scripts/check_chorus.py > /dev/null 2>&1; then
    echo "[OK] Chorus Engine is online"
else
    echo ""
    echo "[WARNING] Cannot connect to Chorus Engine!"
    echo ""
    echo "Make sure Chorus Engine is running first:"
    echo "  ./start.sh"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

echo ""
echo "[INFO] Starting Discord Bridge..."
echo "[INFO] Press Ctrl+C to stop"
echo ""
echo "============================================"
echo ""

# Run the bridge directly
python chorus_discord_bridge/bridge/main.py

# If we get here, the bridge stopped
echo ""
echo "============================================"
echo "   Discord Bridge Stopped"
echo "============================================"

# Deactivate venv
deactivate
