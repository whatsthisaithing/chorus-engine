@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo Chorus Engine - Update Script (Windows)
echo ============================================================
echo.
echo This script will:
echo   1. Pull latest code from Git (if available)
echo   2. Update Python dependencies
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
if errorlevel 1 (
    echo [SKIP] Git not found - skipping code update
    goto :after_git_update
)

echo [INFO] Git found - pulling latest code...
git pull
if errorlevel 1 (
    echo [WARNING] Git pull failed ^(may have local changes^)
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

:after_git_update
echo.

REM Determine Python environment
set PYTHON_CMD=
set PIP_CMD=

if exist python_embeded\python.exe (
    echo [INFO] Using embedded Python ^(portable mode^)
    set "PYTHON_CMD=python_embeded\python.exe"
    set "PIP_CMD=python_embeded\python.exe -m pip"
    goto :python_found
)

echo [INFO] Using system Python ^(developer mode^)
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please run install.bat first
    pause
    exit /b 1
)
set "PYTHON_CMD=python"
set "PIP_CMD=pip"

:python_found
echo [OK] Python found
echo.

echo [2/4] Upgrading pip...
%PIP_CMD% install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed, continuing anyway...
)
echo.

echo [3/4] Updating PyTorch with CUDA 13.0 support...
%PIP_CMD% install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo [WARNING] PyTorch CUDA update failed, keeping existing version
)

REM Verify PyTorch still has CUDA after update
%PYTHON_CMD% -c "import torch; exit(0 if torch.cuda.is_available() or 'cpu' in torch.__version__ else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] PyTorch CUDA not available after update
)
echo.

echo [4/5] Updating core dependencies...
%PIP_CMD% install --upgrade fastapi uvicorn[standard] pydantic pydantic-settings httpx pyyaml python-multipart sqlalchemy alembic chromadb sentence-transformers transformers huggingface_hub nvidia-ml-py
if errorlevel 1 (
    echo [ERROR] Failed to update dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Update Complete!
============================================================
echo.
echo Summary:
echo - Dependencies updated from requirements.txt
echo - Database migrations will run automatically on next start
echo - Model Manager requires Ollama (download from https://ollama.com/download)
echo.
echo You can now run start.bat to launch Chorus Engine
echo.
pause
