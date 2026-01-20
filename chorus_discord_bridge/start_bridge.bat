@echo off
REM Discord Bridge - Windows Startup Script
REM Uses embedded Python from Chorus Engine

echo ============================================
echo    Discord Bridge - Starting Up
echo ============================================
echo.

REM Change to Chorus Engine root directory
cd /d "%~dp0\.."

REM Check for embedded Python
if exist python_embeded\python.exe (
    echo [OK] Using embedded Python from Chorus Engine
    set PYTHON_CMD=python_embeded\python.exe
    set PIP_CMD=python_embeded\python.exe -m pip
    REM Set PYTHONPATH to include current directory
    set PYTHONPATH=%~dp0\..
    goto :check_config
)

echo [ERROR] Embedded Python not found!
echo.
echo Please run the installation script first:
echo   chorus_discord_bridge\install_bridge.bat
pause
exit /b 1

:check_config
REM Check if config files exist
if not exist chorus_discord_bridge\.env (
    echo [ERROR] Configuration not found!
    echo.
    echo Please run the installation script first:
    echo   chorus_discord_bridge\install_bridge.bat
    pause
    exit /b 1
)

if not exist chorus_discord_bridge\config.yaml (
    echo [ERROR] Config file not found!
    echo.
    echo Please run the installation script first:
    echo   chorus_discord_bridge\install_bridge.bat
    pause
    exit /b 1
)

echo [OK] Configuration found
echo.

REM Check if Chorus Engine is running
echo [INFO] Checking Chorus Engine connection...
%PYTHON_CMD% chorus_discord_bridge\scripts\check_chorus.py >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Cannot connect to Chorus Engine!
    echo.
    echo Make sure Chorus Engine is running first:
    echo   start.bat
    echo.
    echo Press any key to continue anyway, or Ctrl+C to exit...
    pause >nul
) else (
    echo [OK] Chorus Engine is online
)

echo.
echo [INFO] Starting Discord Bridge...
echo [INFO] Press Ctrl+C to stop
echo.
echo ============================================
echo.

REM Run the bridge directly
%PYTHON_CMD% chorus_discord_bridge\bridge\main.py

REM If we get here, the bridge stopped
echo.
echo ============================================
echo    Discord Bridge Stopped
echo ============================================
pause
