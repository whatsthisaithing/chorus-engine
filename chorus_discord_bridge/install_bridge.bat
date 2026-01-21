@echo off
REM Discord Bridge - Windows Installation Script
REM Creates a self-contained virtual environment for portability

echo ============================================
echo  Discord Bridge - Installation (Windows)
echo ============================================
echo.
echo This bridge is PORTABLE - you can copy this folder
echo anywhere and it will work independently.
echo.

REM Stay in bridge directory
cd /d "%~dp0"

REM Check for Python on PATH
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH!
    echo.
    echo Please install Python 3.10 or newer:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment in .venv folder
if exist .venv (
    echo [INFO] Virtual environment already exists
) else (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment and install dependencies
echo [2/3] Installing dependencies...
echo.
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

:check_config
echo [3/3] Setting up configuration...
echo.

REM Check if .env exists
if not exist .env (
    echo [INFO] Creating .env from template...
    if exist .env.template (
        copy .env.template .env
    ) else (
        echo # Discord Bridge Environment Variables > .env
        echo DISCORD_BOT_TOKEN=your_bot_token_here >> .env
        echo CHORUS_API_URL=http://localhost:8000 >> .env
    )
    echo.
    echo [ACTION REQUIRED] Please edit .env and add your Discord bot token:
    echo   DISCORD_BOT_TOKEN=your_bot_token_here
    echo.
)

REM Check if config.yaml exists
if not exist config.yaml (
    echo [INFO] Creating config.yaml from template...
    if exist config.yaml.template (
        copy config.yaml.template config.yaml
    ) else (
        echo [ERROR] config.yaml.template not found!
        pause
        exit /b 1
    )
    echo [OK] Config file created with defaults
    echo.
    echo [ACTION REQUIRED] Edit config.yaml to set your character:
    echo   chorus.character_id: "nova"  # Change to your character
    echo.
)

REM Create storage directory
if not exist storage mkdir storage

echo ============================================
echo  Installation Complete!
echo ============================================
echo.
echo This bridge folder is now PORTABLE!
echo You can copy it anywhere and it will work independently.
echo.
echo Next steps:
echo   1. Create a Discord bot (see scripts\create_bot.md)
echo   2. Edit .env with your bot token
echo   3. Edit config.yaml to set your character
echo   4. Run: start_bridge.bat
echo.
echo To run multiple bots:
echo   - Copy this entire folder
 echo   - Use different bot tokens in each .env
   echo   - Use different characters in each config.yaml
echo   - Run each start_bridge.bat separately
echo.
pause
