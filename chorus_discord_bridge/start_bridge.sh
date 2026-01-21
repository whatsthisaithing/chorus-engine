#!/bin/bash
# Discord Bridge - Linux/Mac Startup Script
# Uses self-contained virtual environment

echo "============================================"
echo "   Discord Bridge - Starting Up"
echo "============================================"
echo ""

# Stay in bridge directory
cd "$(dirname "$0")"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run the installation script first:"
    echo "  ./install_bridge.sh"
    echo ""
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo "[OK] Using virtual environment"

# Check if config files exist
if [ ! -f ".env" ]; then
    echo "[ERROR] Configuration not found!"
    echo ""
    echo "Please run the installation script first:"
    echo "  ./install_bridge.sh"
    echo ""
    exit 1
fi

if [ ! -f "config.yaml" ]; then
    echo "[ERROR] Config file not found!"
    echo ""
    echo "Please run the installation script first:"
    echo "  ./install_bridge.sh"
    echo ""
    exit 1
fi

echo "[OK] Configuration found"
echo ""

# Check if Chorus Engine is running
echo "[INFO] Checking Chorus Engine connection..."
if python scripts/check_chorus.py > /dev/null 2>&1; then
    echo "[OK] Chorus Engine is online"
else
    echo ""
    echo "[WARNING] Cannot connect to Chorus Engine!"
    echo ""
    echo "Make sure Chorus Engine is running first."
    echo "  (In the main chorus-engine folder, run: ./start.sh)"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Show character info from config
CHARACTER=$(grep "character_id:" config.yaml | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'" | xargs)
if [ ! -z "$CHARACTER" ]; then
    echo "[INFO] Character: $CHARACTER"
fi

echo ""
echo "[INFO] Starting Discord Bridge..."
echo "[INFO] Press Ctrl+C to stop"
echo ""
echo "============================================"
echo ""

# Run the bridge
python bridge/main.py

# If we get here, the bridge stopped
echo ""
echo "============================================"
echo "   Discord Bridge Stopped"
echo "============================================"

# Deactivate venv
deactivate
