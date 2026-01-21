@echo off
setlocal enabledelayedexpansion
REM Discord Bridge - Windows Startup Script
REM Uses self-contained virtual environment

echo ============================================
echo    Discord Bridge - Starting Up
echo ============================================
echo.

REM Stay in bridge directory
cd /d "%~dp0"

REM Check for virtual environment
if exist .venv\Scripts\python.exe (
    echo [OK] Using virtual environment
    goto :check_config
)

echo [ERROR] Virtual environment not found!
echo.
echo Please run the installation script first:
echo   install_bridge.bat
echo.
pause
exit /b 1

:check_config
REM Check if config files exist
if not exist .env (
    echo [ERROR] Configuration not found!
    echo.
    echo Please run the installation script first:
    echo   install_bridge.bat
    echo.
    pause
    exit /b 1
)

if not exist config.yaml (
    echo [ERROR] Config file not found!
    echo.
    echo Please run the installation script first:
    echo   install_bridge.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Configuration found
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if Chorus Engine is running
echo [INFO] Checking Chorus Engine connection...
.venv\Scripts\python.exe scripts\check_chorus.py >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Chorus Engine is online
    goto :start_bridge
)

echo.
echo [WARNING] Cannot connect to Chorus Engine!
echo.
echo Make sure Chorus Engine is running first.
echo   (In the main chorus-engine folder, run: start.bat)
echo.
echo Press any key to continue anyway, or Ctrl+C to exit...
pause >nul

:start_bridge
echo.

REM Show character info from config
for /f "tokens=2 delims=:" %%i in ('findstr /C:"character_id" config.yaml 2^>nul') do (
    set CHARACTER=%%i
)
if defined CHARACTER (
    echo [INFO] Character: !CHARACTER!
)

echo.
echo [INFO] Starting Discord Bridge...
echo [INFO] Press Ctrl+C to stop
echo.
echo ============================================
echo.

REM Run the bridge
.venv\Scripts\python.exe bridge\main.py

REM If we get here, the bridge stopped
echo.
echo ============================================
echo    Discord Bridge Stopped
echo ============================================
pause
