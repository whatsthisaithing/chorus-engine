# Chorus Engine - Installation Guide

This guide covers different installation methods for Chorus Engine, from simple portable setup to developer installation.

## Quick Start (Portable - Recommended for Most Users)

### Windows

1. **Download** or clone Chorus Engine
2. **Run the installer:**
   ```batch
   install.bat
   ```
   This will:
   - Download Python 3.11.7 embedded (~100MB)
   - Install all required dependencies
   - Set up everything in the `python_embeded/` folder

3. **Start Chorus Engine:**
   ```batch
   start.bat
   ```

That's it! The browser will open automatically to http://localhost:8080

### Linux/Mac

1. **Download** or clone Chorus Engine
2. **Make scripts executable:**
   ```bash
   chmod +x install.sh start.sh
   ```

3. **Run the installer:**
   ```bash
   ./install.sh
   ```
   This will:
   - Create a Python virtual environment in `venv/`
   - Install all required dependencies
   - Set up everything isolated from your system Python

4. **Start Chorus Engine:**
   ```bash
   ./start.sh
   ```

The browser will open automatically to http://localhost:8080

## Developer Installation (Advanced)

If you're a developer who wants to use your own Python environment:

### Prerequisites
- Python 3.11 or newer
- pip (usually comes with Python)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chorus-engine.git
   cd chorus-engine
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run directly:**
   ```bash
   python -m chorus_engine.main
   ```

The startup scripts (`start.bat` / `start.sh`) will automatically detect and use your system Python if no portable installation exists.

## Installation Modes Explained

Chorus Engine supports two installation modes that work seamlessly together:

### Portable Mode (End Users)
- **Windows**: Uses `python_embeded/` with bundled Python 3.11
- **Linux/Mac**: Uses `venv/` with isolated environment
- **Pros**: 
  - No Python installation required
  - Consistent versions across all installations
  - No dependency conflicts with other Python software
  - Easy to move/backup (just copy the folder)
- **Cons**: 
  - Larger download size (~500MB with dependencies)
  - Separate Python just for Chorus Engine

### Developer Mode (Contributors)
- Uses your system Python installation
- **Pros**:
  - Full control over Python version
  - Can use your preferred IDE/debugger
  - Easy to modify dependencies
  - Shares Python installation with other projects
- **Cons**:
  - Requires Python 3.11+ installed
  - Potential for dependency conflicts
  - Need to manage your own virtual environment

**The scripts automatically detect which mode to use:**
- If `python_embeded/` (Windows) or `venv/` (Linux/Mac) exists → Portable mode
- Otherwise → Developer mode with system Python

## Requirements

### System Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for application + models
- **GPU**: Optional but recommended for faster processing

### External Services (Choose One or More)

**LLM Backend** (Required - choose one):
- [Ollama](https://ollama.com/) - Local models (recommended)
- [LM Studio](https://lmstudio.ai/) - Local models with GUI
- OpenAI API - Cloud-based

**Image Generation** (Optional):
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Local Stable Diffusion

## Troubleshooting

### Windows: "Python not found" error
Even after running `install.bat`:
- Make sure `python_embeded\python.exe` exists
- Try deleting `python_embeded/` folder and run `install.bat` again
- Check your antivirus isn't blocking the download

### Linux/Mac: "Permission denied"
- Make sure scripts are executable: `chmod +x install.sh start.sh`
- Try running with `bash install.sh` instead of `./install.sh`

### "Failed to install dependencies"
- **Portable mode**: Delete `python_embeded/` or `venv/` and run installer again
- **Developer mode**: Make sure you have pip installed: `python -m ensurepip`
- Check your internet connection
- Try upgrading pip: `pip install --upgrade pip`

### Port 8080 already in use
- Another Chorus Engine instance might be running
- Check for other applications using port 8080
- You can change the port in `config/system.yaml`

### Import errors when starting
- Make sure dependencies are installed: see relevant error message
- **Portable mode**: Re-run `install.bat` or `install.sh`
- **Developer mode**: Run `pip install -r requirements.txt` again

## Updating Chorus Engine

### Portable Installation
```bash
# Windows
git pull
install.bat

# Linux/Mac
git pull
./install.sh
```

### Developer Installation
```bash
git pull
pip install -r requirements.txt --upgrade
```

## Uninstalling

### Portable Installation
Simply delete the Chorus Engine folder. Everything is self-contained:
- Windows: `python_embeded/` contains the Python installation
- Linux/Mac: `venv/` contains the virtual environment
- `data/` contains your conversations and memories

### Developer Installation
```bash
pip uninstall -r requirements.txt -y
```

## File Structure

After installation, your directory structure will look like:

```
chorus-engine/
├── python_embeded/          # Windows: Embedded Python (portable mode)
├── venv/                    # Linux/Mac: Virtual environment (portable mode)
├── chorus_engine/           # Application code
├── data/                    # Your data (conversations, images, etc.)
├── config/                  # Configuration files
├── web/                     # Web interface
├── install.bat/sh           # Installation scripts
├── start.bat/sh             # Startup scripts
└── requirements.txt         # Python dependencies
```

**Note**: `python_embeded/` and `venv/` are in `.gitignore` and won't be committed to git.

## Next Steps

After installation:
1. Configure your LLM backend (Ollama, LM Studio, or OpenAI)
2. (Optional) Set up ComfyUI for image generation
3. Read the [User Guide](Documentation/USER_GUIDE.md)
4. Explore the character system
5. Start chatting!

## Getting Help

- **Documentation**: See `Documentation/` folder
- **Issues**: Open an issue on GitHub
- **Community**: [Link to Discord/Forum if available]
