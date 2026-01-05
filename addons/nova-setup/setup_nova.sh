#!/bin/bash
# Nova Character Setup for Chorus Engine
# Configures Nova with profile picture, voice sample, and workflow

echo "========================================"
echo "Nova Character Setup"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    echo "Please run install.sh first"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
