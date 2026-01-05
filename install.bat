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

REM Install remaining dependencies
echo.
echo Installing remaining dependencies...
echo.
python_embeded\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

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
