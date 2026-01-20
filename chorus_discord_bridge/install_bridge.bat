@echo off
REM Discord Bridge - Windows Installation Script
REM Uses the same embedded Python as Chorus Engine

echo ============================================
echo  Discord Bridge - Installation (Windows)
echo ============================================
echo.

REM Change to Chorus Engine root directory
cd /d "%~dp0\.."

REM Check for embedded Python (should exist from main install)
if exist python_embeded\python.exe (
    echo [OK] Found embedded Python from Chorus Engine
    set PYTHON_CMD=python_embeded\python.exe
    set PIP_CMD=python_embeded\python.exe -m pip
    goto :install_deps
)

REM If not found, direct user to run main install
echo [ERROR] Embedded Python not found!
echo.
echo Please run the main Chorus Engine installation first:
echo   install.bat
echo.
echo This will set up the embedded Python that both
echo Chorus Engine and Discord Bridge will use.
pause
exit /b 1

:install_deps
echo.
echo [1/2] Installing Discord Bridge dependencies...
echo.

REM Install discord bridge requirements
%PIP_CMD% install -r chorus_discord_bridge\requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

:check_config
echo [2/2] Setting up configuration...
echo.

REM Check if .env exists
if not exist chorus_discord_bridge\.env (
    echo [INFO] Creating .env from template...
    copy chorus_discord_bridge\.env.template chorus_discord_bridge\.env
    echo.
    echo [ACTION REQUIRED] Please edit chorus_discord_bridge\.env
    echo and add your Discord bot token:
    echo   DISCORD_BOT_TOKEN=your_bot_token_here
    echo.
)

REM Check if config.yaml exists
if not exist chorus_discord_bridge\config.yaml (
    echo [INFO] Creating config.yaml from template...
    copy chorus_discord_bridge\config.yaml.template chorus_discord_bridge\config.yaml
    echo [OK] Config file created with defaults
    echo.
)

echo ============================================
echo  Installation Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Create a Discord bot (see chorus_discord_bridge\scripts\create_bot.md)
echo   2. Edit chorus_discord_bridge\.env with your bot token
echo   3. Run: start_bridge.bat
echo.
pause
