@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo Chorus Engine - Installation Script (Windows)
echo ============================================================
echo.

REM Check if Python embedded already exists
if exist python_embeded\python.exe (
    echo [*] Embedded Python already installed
    goto :install_deps
)

echo [1/3] Downloading Python 3.11.7 embedded...
echo.

REM Create temp directory
if not exist temp mkdir temp

REM Download Python embedded
echo Downloading from python.org...
curl -L https://www.python.org/ftp/python/3.11.7/python-3.11.7-embed-amd64.zip -o temp\python_embeded.zip

if errorlevel 1 (
    echo [ERROR] Failed to download Python. Please check your internet connection.
    pause
    exit /b 1
)

echo [*] Download complete
echo.

echo [2/3] Extracting Python...
powershell -command "Expand-Archive -Path temp\python_embeded.zip -DestinationPath python_embeded -Force"

if errorlevel 1 (
    echo [ERROR] Failed to extract Python
    pause
    exit /b 1
)

REM Clean up
del temp\python_embeded.zip
rmdir temp

echo [*] Python extracted successfully
echo.

REM Configure embedded Python to use pip and site-packages
echo [*] Configuring embedded Python...

REM Modify python311._pth to enable site-packages and parent directory access
(
    echo python311.zip
    echo .
    echo ..
    echo # Uncomment to run site.main^(^) automatically
    echo import site
) > python_embeded\python311._pth

REM Download get-pip.py
echo [*] Installing pip...
curl -L https://bootstrap.pypa.io/get-pip.py -o python_embeded\get-pip.py

if errorlevel 1 (
    echo [ERROR] Failed to download get-pip.py
    pause
    exit /b 1
)

REM Install pip
python_embeded\python.exe python_embeded\get-pip.py

if errorlevel 1 (
    echo [ERROR] Failed to install pip
    pause
    exit /b 1
)

REM Clean up get-pip.py
del python_embeded\get-pip.py

echo [*] Pip installed successfully
echo.

:install_deps

REM Create config/system.yaml from template if it doesn't exist
if not exist config\system.yaml (
    echo [*] Creating config\system.yaml from template...
    copy config\system.yaml.template config\system.yaml
    echo [*] Config file created - you can customize config\system.yaml
    echo.
)

echo [3/3] Installing Chorus Engine dependencies...
echo This may take several minutes...
echo.

REM Upgrade pip first
python_embeded\python.exe -m pip install --upgrade pip

REM Install PyTorch with CUDA support FIRST (before requirements.txt)
echo.
echo Installing PyTorch with CUDA support...
echo (This is required for GPU-accelerated TTS and may take a few minutes)
echo.

REM Use CUDA 13.0 (supports RTX 30/40/50 series including sm_120)
echo Installing PyTorch 2.9 with CUDA 13.0 (supports all RTX 30/40/50 series)...
python_embeded\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130

if errorlevel 1 (
    echo [WARNING] CUDA PyTorch installation failed, falling back to CPU version...
    python_embeded\python.exe -m pip install torch torchaudio
)

REM Verify PyTorch CUDA installation
echo.
echo Verifying PyTorch installation...
python_embeded\python.exe -c "import torch; print('PyTorch version:', torch.__version__); cuda_available = torch.cuda.is_available(); print('CUDA available:', cuda_available); exit(0 if cuda_available or 'cpu' in torch.__version__ else 1)"

if errorlevel 1 (
    echo [WARNING] PyTorch CUDA verification failed!
    echo [INFO] PyTorch may have been downgraded or installed incorrectly
    echo [INFO] TTS will be significantly slower without GPU acceleration
    echo.
) else (
    python_embeded\python.exe -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only mode')"
)

REM Install remaining dependencies
echo.
echo Installing remaining dependencies...
echo (PyTorch already installed, ensuring no downgrade...)
echo.
REM Use --no-deps for chatterbox-tts to prevent torch downgrade, install others normally
python_embeded\python.exe -m pip install fastapi==0.115.5 uvicorn[standard]==0.32.1 pydantic==2.10.3 pydantic-settings==2.6.1 httpx==0.28.1 pyyaml==6.0.2 python-multipart==0.0.21 sqlalchemy==2.0.36 alembic==1.13.1 chromadb==0.5.23 sentence-transformers==3.3.1 transformers==4.46.3 huggingface_hub nvidia-ml-py

if errorlevel 1 (
    echo [ERROR] Failed to install core dependencies
    pause
    exit /b 1
)

echo.
echo Installing chatterbox-tts (without deps to preserve PyTorch CUDA)...
python_embeded\python.exe -m pip install chatterbox-tts --no-deps

echo.
echo Installing chatterbox-tts dependencies (except torch/torchaudio)...
python_embeded\python.exe -m pip install conformer diffusers gradio librosa numpy omegaconf pykakasi pyloudnorm resemble-perth s3tokenizer safetensors spacy-pkuseg transformers

if errorlevel 1 (
    echo [WARNING] Some chatterbox-tts dependencies may be missing
    echo TTS functionality might be limited
)

REM Note: Model Manager uses Ollama for LLM inference
REM No additional LLM libraries needed - download Ollama separately:
REM https://ollama.com/download

REM Install chatterbox-tts without dependency checks
REM (Works with PyTorch 2.9 despite requiring 2.6 in metadata)
echo.
echo Installing chatterbox-tts (AI TTS with voice cloning)...
python_embeded\python.exe -m pip install chatterbox-tts --no-deps

if errorlevel 1 (
    echo [WARNING] Chatterbox TTS installation failed (optional component)
)

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo You can now run start.bat to launch Chorus Engine
echo.
echo Python location: python_embeded\python.exe
echo Dependencies installed to: python_embeded\Lib\site-packages
echo.
pause
