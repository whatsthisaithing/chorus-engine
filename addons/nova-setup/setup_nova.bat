@echo off
REM Nova Character Setup for Chorus Engine
REM Configures Nova with profile picture, voice sample, and workflow

echo ========================================
echo Nova Character Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Run the setup script
python "%~dp0setup_nova.py"

if errorlevel 1 (
    echo.
    echo Setup encountered errors. Please check the output above.
) else (
    echo.
    echo Setup completed successfully!
)

echo.
pause
