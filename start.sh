#!/bin/bash
# Chorus Engine - Unix Startup Script (Linux/macOS)
# Launches the backend server and opens web UI in browser

echo "============================================"
echo "   Chorus Engine - Starting Up"
echo "============================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check for venv first (portable installation)
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo -e "${CYAN}[INFO] Using virtual environment (portable mode)${NC}"
    source venv/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
elif command -v python3.11 &> /dev/null; then
    echo -e "${CYAN}[INFO] Using system Python 3.11 (developer mode)${NC}"
    PYTHON_CMD="python3.11"
    PIP_CMD="pip3"
elif command -v python3.12 &> /dev/null; then
    echo -e "${CYAN}[INFO] Using system Python 3.12 (developer mode)${NC}"
    PYTHON_CMD="python3.12"
    PIP_CMD="pip3"
elif command -v python3 &> /dev/null; then
    echo -e "${CYAN}[INFO] Using system Python 3 (developer mode)${NC}"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    echo -e "${RED}[ERROR] Python not found!${NC}"
    echo ""
    echo "Please either:"
    echo "  1. Run ./install.sh to set up virtual environment, OR"
    echo "  2. Install Python 3.11+ and ensure it's in PATH"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}[OK] Python found: $PYTHON_VERSION${NC}"

# Check if required packages are installed
if ! $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}[INFO] Installing dependencies...${NC}"
    $PIP_CMD install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to install dependencies${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}[OK] Dependencies installed${NC}"

# Read API host and port from config
API_HOST=$($PYTHON_CMD utilities/get_config.py api_host 2>/dev/null)
API_PORT=$($PYTHON_CMD utilities/get_config.py api_port 2>/dev/null)

# Fallback to defaults if config read fails
if [ -z "$API_HOST" ]; then
    API_HOST="localhost"
fi
if [ -z "$API_PORT" ]; then
    API_PORT="8080"
fi

echo -e "${CYAN}[INFO] Server configured for ${API_HOST}:${API_PORT}${NC}"

# Check if configured port is available
if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}[WARNING] Port $API_PORT is already in use!${NC}"
    echo "Another instance might be running, or another application is using the port."
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "   Starting Chorus Engine Server"
echo "============================================"
echo ""
echo -e "${GREEN}Backend API:    http://${API_HOST}:${API_PORT}/docs${NC}"
echo -e "${GREEN}Web Interface:  http://${API_HOST}:${API_PORT}${NC}"
echo ""
echo -e "${YELLOW}Opening browser...${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo "============================================"
echo ""

# Function to open browser
open_browser() {
    sleep 3
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://${API_HOST}:${API_PORT}" 2>/dev/null
    elif command -v open &> /dev/null; then
        open "http://${API_HOST}:${API_PORT}" 2>/dev/null
    fi
}

# Open browser in background
open_browser &

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Shutting down Chorus Engine...${NC}"; exit 0' INT TERM

# Restart loop
while true; do
    # Start the server
    $PYTHON_CMD -m chorus_engine.main
    EXIT_CODE=$?
    
    # Check if exit code is 42 (restart request)
    if [ $EXIT_CODE -eq 42 ]; then
        echo ""
        echo -e "${YELLOW}[INFO] Restarting server...${NC}"
        echo ""
        sleep 2
        continue
    fi
    
    # If server exits with error or normal exit, stop loop
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo -e "${RED}[ERROR] Server exited with an error${NC}"
        exit 1
    fi
    
    # Normal exit (0), stop
    break
done
