@echo off
echo ============================================================
echo Chorus Engine - Environment Verification
echo ============================================================
echo.

REM Determine Python command
if exist python_embeded\python.exe (
    set "PYTHON_CMD=python_embeded\python.exe"
    echo [*] Using embedded Python
) else (
    set "PYTHON_CMD=python"
    echo [*] Using system Python
)

echo.
echo [1/5] Checking Python version...
%PYTHON_CMD% --version
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo.
echo [2/5] Checking PyTorch installation...
%PYTHON_CMD% -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if errorlevel 1 (
    echo [ERROR] PyTorch check failed!
    pause
    exit /b 1
)

echo.
echo [3/5] Checking core dependencies...
%PYTHON_CMD% -c "import fastapi, uvicorn, chromadb, sentence_transformers, transformers; print('Core dependencies OK')"

if errorlevel 1 (
    echo [ERROR] Core dependencies check failed!
    pause
    exit /b 1
)

echo.
echo [4/5] Checking chatterbox-tts...
%PYTHON_CMD% -c "import chatterbox; print('Chatterbox TTS:', chatterbox.__version__)"

if errorlevel 1 (
    echo [WARNING] Chatterbox TTS not available (optional)
) else (
    echo [OK] Chatterbox TTS available
)

echo.
echo [5/5] Checking Ollama availability...
%PYTHON_CMD% -c "import httpx; r = httpx.get('http://localhost:11434/api/tags', timeout=2); print('Ollama API: Available'); print('Models:', len(r.json().get('models', [])))" 2>nul

if errorlevel 1 (
    echo [WARNING] Ollama not running or not installed
    echo [INFO] Download from: https://ollama.com/download
    echo [INFO] Model Manager requires Ollama for LLM inference
    echo.
) else (
    echo [OK] Ollama is running
)

echo.
echo ============================================================
echo Verification Summary
echo ============================================================
%PYTHON_CMD% -c "import torch; cuda = torch.cuda.is_available(); print('PyTorch CUDA:', '✓ Available' if cuda else '✗ CPU-only'); print('GPU:', torch.cuda.get_device_name(0) if cuda else 'None'); print('VRAM:', str(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)) + ' GB' if cuda else 'N/A')"

echo.
echo Ollama Status:
%PYTHON_CMD% -c "import httpx; r = httpx.get('http://localhost:11434/api/tags', timeout=2); print('  ✓ Running (' + str(len(r.json().get('models', []))) + ' models)')" 2>nul || echo   ✗ Not running

echo.
echo [INFO] If everything shows ✓, your setup is complete!
echo [INFO] For Model Manager, ensure Ollama is installed and running
echo.
pause
