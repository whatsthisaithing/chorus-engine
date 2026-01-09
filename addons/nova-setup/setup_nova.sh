#!/bin/bash
# Nova Character Setup for Chorus Engine
# Configures Nova with profile picture, voice sample, and workflow

echo "========================================"
echo "Nova Character Setup"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to project root (2 levels up from script)
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd "$PROJECT_ROOT"

# Check if venv exists (portable installation)
if [ -d "venv" ]; then
    echo "[*] Using virtual environment Python"
    PYTHON_CMD="venv/bin/python"
else
    # Check if system Python is available
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "ERROR: Python not found"
        echo "Please run install.sh first to set up Python"
        exit 1
    fi
    echo "[*] Using system Python"
fi

# Run the setup script
$PYTHON_CMD "$SCRIPT_DIR/setup_nova.py"

if [ $? -ne 0 ]; then
    echo ""
    echo "Setup encountered errors. Please check the output above."
    exit 1
else
    echo ""
    echo "Setup completed successfully!"
fi
