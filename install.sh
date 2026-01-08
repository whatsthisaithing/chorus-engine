#!/bin/bash

echo "============================================================"
echo "Chorus Engine - Installation Script (Linux/Mac)"
echo "============================================================"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "[*] Virtual environment already exists"
else
    echo "[1/2] Creating Python virtual environment..."
    echo ""
    
    # Try to find Python 3.11 or newer
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "[ERROR] Python 3.11+ not found. Please install Python 3.11 or newer."
        exit 1
    fi
    
    echo "[*] Using: $PYTHON_CMD ($(${PYTHON_CMD} --version))"
    
    # Create virtual environment
    $PYTHON_CMD -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    
    echo "[*] Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

# Create config/system.yaml from template if it doesn't exist
if [ ! -f "config/system.yaml" ]; then
    echo "[*] Creating config/system.yaml from template..."
    cp config/system.yaml.template config/system.yaml
    echo "[*] Config file created - you can customize config/system.yaml"
    echo ""
fi

echo "[2/2] Installing Chorus Engine dependencies..."
echo "This may take several minutes..."
echo ""

# Upgrade pip
pip install --upgrade pip

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upgrade pip"
    exit 1
fi

# Install requirements
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo ""
echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "You can now run ./start.sh to launch Chorus Engine"
echo ""
echo "Virtual environment: venv/"
echo "Python: $(python --version)"
echo "Dependencies installed to: venv/lib/python*/site-packages"
echo ""
