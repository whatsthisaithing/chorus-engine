#!/bin/bash
# Discord Bridge - Linux/Mac Installation Script
# Creates a self-contained virtual environment for portability

echo "============================================"
echo " Discord Bridge - Installation (Linux/Mac)"
echo "============================================"
echo ""
echo "This bridge is PORTABLE - you can copy this folder"
echo "anywhere and it will work independently."
echo ""

# Stay in bridge directory
cd "$(dirname "$0")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found!"
    echo ""
    echo "Please install Python 3.10 or newer:"
    echo "  https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[OK] Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment in .venv folder
if [ -d ".venv" ]; then
    echo "[INFO] Virtual environment already exists"
else
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[OK] Virtual environment created"
fi
echo ""

# Activate virtual environment and install dependencies
echo "[2/3] Installing dependencies..."
echo ""
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo "[OK] Dependencies installed"
echo ""

# Check if .env exists
echo "[3/3] Setting up configuration..."
echo ""

if [ ! -f ".env" ]; then
    echo "[INFO] Creating .env from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env
    else
        echo "# Discord Bridge Environment Variables" > .env
        echo "DISCORD_BOT_TOKEN=your_bot_token_here" >> .env
        echo "CHORUS_API_URL=http://localhost:8000" >> .env
    fi
    echo ""
    echo "[ACTION REQUIRED] Please edit .env and add your Discord bot token:"
    echo "  DISCORD_BOT_TOKEN=your_bot_token_here"
    echo ""
fi

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "[INFO] Creating config.yaml from template..."
    if [ -f "config.yaml.template" ]; then
        cp config.yaml.template config.yaml
    else
        echo "[ERROR] config.yaml.template not found!"
        exit 1
    fi
    echo "[OK] Config file created with defaults"
    echo ""
    echo "[ACTION REQUIRED] Edit config.yaml to set your character:"
    echo "  chorus.character_id: \"nova\"  # Change to your character"
    echo ""
fi

# Create storage directory
mkdir -p storage

echo "============================================"
echo " Installation Complete!"
echo "============================================"
echo ""
echo "This bridge folder is now PORTABLE!"
echo "You can copy it anywhere and it will work independently."
echo ""
echo "Next steps:"
echo "  1. Create a Discord bot (see scripts/create_bot.md)"
echo "  2. Edit .env with your bot token"
echo "  3. Edit config.yaml to set your character"
echo "  4. Run: ./start_bridge.sh"
echo ""
echo "To run multiple bots:"
echo "  - Copy this entire folder"
echo "  - Use different bot tokens in each .env"
echo "  - Use different characters in each config.yaml"
echo "  - Run each start_bridge.sh separately"
echo ""

# Deactivate venv
deactivate
