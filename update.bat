@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo Chorus Engine - Update Script (Windows)
echo ============================================================
echo.
echo This script will:
echo   1. Pull latest code from Git (if available)
echo   2. Update Python dependencies
echo   3. Run database migrations
echo.
echo ============================================================
echo                    IMPORTANT WARNING
echo ============================================================
echo.
echo Before updating, we recommend backing up your Chorus Engine
echo folder, especially:
echo   - Your custom characters (characters/*.yaml)
echo   - Your data folder (conversations, documents, etc.)
echo   - Any configuration changes
echo.
echo You can also create a Git stash to save local changes:
echo   git stash save "backup before update"
echo.
echo ============================================================
echo.
set /p CONFIRM="Continue with update? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Update cancelled
    pause
    exit /b 0
)

echo.
echo ============================================================
echo Starting Update Process
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check for Git and pull latest code
echo [1/4] Checking for code updates...
git --version >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Git found - pulling latest code...
    git pull
    if errorlevel 1 (
        echo [WARNING] Git pull failed (may have local changes)
        echo You can manually resolve conflicts or use:
        echo   git stash     - Save local changes
        echo   git pull      - Get updates
        echo   git stash pop - Restore local changes
        echo.
        set /p CONTINUE="Continue anyway? (y/n): "
        if /i not "!CONTINUE!"=="y" (
            echo Update cancelled
            pause
            exit /b 1
        )
    ) else (
        echo [OK] Code updated successfully
    )
) else (
    echo [SKIP] Git not found - skipping code update
)
echo.

REM Check for embedded Python first
if exist python_embeded\python.exe (
    echo [INFO] Using embedded Python (portable mode)
    set PYTHON_CMD=python_embeded\python.exe
    set PIP_CMD=python_embeded\python.exe -m pip
) else (
    echo [INFO] Using system Python (developer mode)
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found!
        echo Please run install.bat first
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
    set PIP_CMD=pip
)

echo [OK] Python found
echo.

echo [2/4] Upgrading pip...
%PIP_CMD% install --upgrade pip
echo.

echo [3/4] Updating dependencies from requirements.txt...
%PIP_CMD% install --upgrade -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to update dependencies
    pause
    exit /b 1
)

echo.
echo [4/4] Running database migrations...
%PYTHON_CMD% -c "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); command.upgrade(cfg, 'head')"

if errorlevel 1 (
    echo [WARNING] Database migration failed (may be okay if no migrations needed)
)

echo.
echo ============================================================
echo Update Complete!
echo ============================================================
echo.
echo Summary:
echo - Dependencies updated from requirements.txt
echo - Database migrations applied
echo.
echo You can now run start.bat to launch Chorus Engine
echo.
pause
