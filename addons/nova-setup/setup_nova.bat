@echo off
REM Nova Character Setup for Chorus Engine
REM Configures Nova with profile picture, voice sample, and workflow

echo ========================================
echo Nova Character Setup
echo ========================================
echo.

REM Navigate to project root (2 levels up from this script)
cd /d "%~dp0..\.."

REM Check if embedded Python exists (portable installation)
if exist python_embeded\python.exe (
    echo [*] Using embedded Python
    set PYTHON_CMD=python_embeded\python.exe
) else (
    REM Check if system Python is available
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found
        echo Please run install.bat first to set up Python
        pause
        exit /b 1
    )
    echo [*] Using system Python
    set PYTHON_CMD=python
)

REM Run the setup script
%PYTHON_CMD% "%~dp0setup_nova.py"

if errorlevel 1 (
    echo.
    echo Setup encountered errors. Please check the output above.
) else (
    echo.
    echo Setup completed successfully!
)

echo.
pause
