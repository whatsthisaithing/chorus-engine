#!/bin/bash

echo "============================================================"
echo "Chorus Engine - Update Script (Linux/macOS)"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Pull latest code from Git (if available)"
echo "  2. Update Python dependencies"
echo "  3. Run database migrations"
echo ""
echo "============================================================"
echo "                   IMPORTANT WARNING"
echo "============================================================"
echo ""
echo "Before updating, we recommend backing up your Chorus Engine"
echo "folder, especially:"
echo "  - Your custom characters (characters/*.yaml)"
echo "  - Your data folder (conversations, documents, etc.)"
echo "  - Any configuration changes"
echo ""
echo "You can also create a Git stash to save local changes:"
echo "  git stash save 'backup before update'"
echo ""
echo "============================================================"
echo ""
read -p "Continue with update? (y/n): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Update cancelled"
    exit 0
fi

echo ""
echo "============================================================"
echo "Starting Update Process"
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run pre-update diagnostics
echo "[1/5] Running pre-update diagnostics..."
if ! python3 check_before_update.py --fix; then
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 2 ]; then
        echo ""
        echo "============================================================"
        echo "CRITICAL ERROR DETECTED"
        echo "============================================================"
        echo "Pre-update diagnostics found critical issues that could not"
        echo "be fixed automatically. Please review the errors above and"
        echo "consider backing up your data before proceeding."
        echo ""
        read -p "Continue with update anyway? (y/n): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Update cancelled"
            exit 1
        fi
    else
        echo ""
        echo "[INFO] Issues were detected and fixed automatically."
        echo ""
    fi
fi
echo ""

# Check for Git and pull latest code
echo "[2/5] Checking for code updates..."
if command -v git &> /dev/null; then
    echo "[INFO] Git found - pulling latest code..."
    if git pull; then
        echo "[OK] Code updated successfully"
    else
        echo "[WARNING] Git pull failed (may have local changes)"
        echo "You can manually resolve conflicts or use:"
        echo "  git stash     - Save local changes"
        echo "  git pull      - Get updates"
        echo "  git stash pop - Restore local changes"
        echo ""
        read -p "Continue anyway? (y/n): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Update cancelled"
            exit 1
        fi
    fi
else
    echo "[SKIP] Git not found - skipping code update"
fi
echo ""

# Determine Python command
if [ -f "python_embeded/bin/python3" ]; then
    echo "[INFO] Using embedded Python (portable mode)"
    PYTHON_CMD="python_embeded/bin/python3"
    PIP_CMD="python_embeded/bin/pip3"
elif command -v python3 &> /dev/null; then
    echo "[INFO] Using system Python (developer mode)"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    echo "[ERROR] Python not found!"
    echo "Please run install.sh first"
    exit 1
fi

echo "[OK] Python found"
echo ""

echo "[3/5] Upgrading pip..."
$PIP_CMD install --upgrade pip
echo ""

echo "[4/5] Updating dependencies from requirements.txt..."
$PIP_CMD install --upgrade -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to update dependencies"
    exit 1
fi

echo ""
echo "[5/5] Running database migrations..."
$PYTHON_CMD -c "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); command.upgrade(cfg, 'head')"

if [ $? -ne 0 ]; then
    echo "[WARNING] Database migration failed (may be okay if no migrations needed)"
fi

echo ""
echo "============================================================"
echo "Update Complete!"
echo "============================================================"
echo ""
echo "Summary:"
echo "- Dependencies updated from requirements.txt"
echo "- Database migrations applied"
echo ""
echo "You can now run ./start.sh to launch Chorus Engine"
echo ""
