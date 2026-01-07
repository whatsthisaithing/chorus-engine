@echo off
REM Chorus Engine - Windows Startup Script
REM Launches the backend server and opens web UI in browser

echo ============================================
echo    Chorus Engine - Starting Up
echo ============================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check for embedded Python first (portable installation)
if exist python_embeded\python.exe (
    echo [INFO] Using embedded Python ^(portable mode^)
    set PYTHON_CMD=python_embeded\python.exe
    set PIP_CMD=python_embeded\python.exe -m pip
    REM Set PYTHONPATH to include current directory for module imports
    set PYTHONPATH=%~dp0
    goto :check_deps
)

REM Fall back to system Python (developer mode)
echo [INFO] Using system Python ^(developer mode^)
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please either:
    echo   1. Run install.bat to set up portable Python, OR
    echo   2. Install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

set PYTHON_CMD=python
set PIP_CMD=pip

:check_deps
echo [OK] Python found

REM Check if required packages are installed
%PIP_CMD% show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    %PIP_CMD% install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [OK] Dependencies installed

REM Check if port 8080 is available
netstat -ano | findstr ":8080" >nul
if not errorlevel 1 (
    echo [WARNING] Port 8080 is already in use!
    echo Another instance might be running, or another application is using the port.
    choice /C YN /M "Do you want to continue anyway?"
    if errorlevel 2 exit /b 1
)

echo.
echo ============================================
echo    Starting Chorus Engine Server
echo ============================================
echo.
echo Backend API: http://localhost:8080/docs
echo Web Interface: http://localhost:8080
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

:restart_loop
REM Start the server
if exist python_embeded\python.exe (
    REM Embedded Python - run directly from current directory
    %PYTHON_CMD% chorus_engine\main.py
) else (
    REM System Python - use -m module syntax
    %PYTHON_CMD% -m chorus_engine.main
)

REM Check exit code
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==42 (
    echo.
    echo [INFO] Restarting server...
    echo.
    timeout /t 2 /nobreak >nul
    goto :restart_loop
)

REM If server exits with error, pause to show any error messages
if %EXIT_CODE% NEQ 0 (
    echo.
    echo [ERROR] Server exited with an error
    pause
)
