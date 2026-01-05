# Installation Modes - Quick Reference

## Overview

Chorus Engine supports two installation modes that coexist seamlessly:

| Feature | 🚀 Portable Mode | 👨‍💻 Developer Mode |
|---------|------------------|---------------------|
| **Python Required** | ❌ No | ✅ Yes (3.11+) |
| **Setup Command** | `install.bat/.sh` | `pip install -r requirements.txt` |
| **Run Command** | `start.bat/.sh` | `python -m chorus_engine.main` |
| **Python Location** | `python_embeded/` (Win) or `venv/` (Unix) | System Python |
| **Isolation** | ✅ Fully isolated | Depends on venv usage |
| **Disk Space** | ~500MB (with deps) | ~200MB (deps only) |
| **Best For** | End users, non-technical | Developers, contributors |
| **Update Method** | Re-run installer | `pip install -r requirements.txt --upgrade` |

## Directory Structure After Installation

### Portable Mode (Windows)
```
chorus-engine/
├── python_embeded/           ← Downloaded Python 3.11 (~100MB)
│   ├── python.exe
│   ├── Lib/
│   │   └── site-packages/    ← All dependencies here
│   └── Scripts/
├── chorus_engine/            ← Your code
├── data/                     ← Your data
├── install.bat               ← Run once to setup
└── start.bat                 ← Run to start
```

### Portable Mode (Linux/Mac)
```
chorus-engine/
├── venv/                     ← Python virtual environment
│   ├── bin/
│   │   └── python            ← Python 3.11+
│   └── lib/
│       └── python3.11/
│           └── site-packages/  ← All dependencies here
├── chorus_engine/            ← Your code
├── data/                     ← Your data
├── install.sh                ← Run once to setup
└── start.sh                  ← Run to start
```

### Developer Mode
```
chorus-engine/
├── chorus_engine/            ← Your code
├── data/                     ← Your data
├── requirements.txt          ← Dependencies list
└── start.bat/.sh             ← Auto-detects system Python
```

## How Scripts Detect Mode

Both `start.bat` and `start.sh` automatically detect which mode to use:

```
IF python_embeded/ or venv/ exists:
    → Use portable mode (isolated Python)
ELSE:
    → Use developer mode (system Python)
```

This means:
- **Users** run `install.bat` → Get portable mode automatically
- **Developers** skip installer → Get developer mode automatically
- **Both** use the same startup scripts

## Switching Between Modes

### Portable → Developer
```bash
# Just delete the portable Python
rm -rf python_embeded/  # Windows
rm -rf venv/            # Linux/Mac

# Next run of start.bat/.sh uses system Python
```

### Developer → Portable
```bash
# Just run the installer
install.bat     # Windows
./install.sh    # Linux/Mac

# Next run of start.bat/.sh uses portable Python
```

## When to Use Each Mode

### Use Portable Mode If You:
- ✅ Don't want to install Python separately
- ✅ Want consistent behavior across machines
- ✅ Plan to distribute to non-technical users
- ✅ Want zero dependency conflicts
- ✅ Need to bundle specific library versions (like TTS)

### Use Developer Mode If You:
- ✅ Already have Python 3.11+ installed
- ✅ Want to use your IDE's debugger
- ✅ Are contributing to the project
- ✅ Want to customize dependencies
- ✅ Need to test with different Python versions

## Technical Details

### Windows Portable (python_embeded/)
- Uses official Python "embeddable package" from python.org
- Modified `python311._pth` to enable site-packages
- Pip installed via `get-pip.py`
- ~95MB base + dependencies
- Completely self-contained

### Unix Portable (venv/)
- Standard Python venv module
- Uses system Python to create, then isolated
- Slightly smaller than Windows (no duplicate binaries)
- Can use any Python 3.11+ as base

### Developer Mode
- No special setup
- Uses whatever `python` or `python3` is in PATH
- Respects existing virtual environments
- Standard Python development workflow

## FAQ

**Q: Can I use both modes on the same machine?**  
A: Yes! Just have multiple Chorus Engine folders. One with `python_embeded/`, one without.

**Q: Which mode is faster?**  
A: Both are identical speed. Python runs the same code.

**Q: Can I switch modes without losing data?**  
A: Yes! Your `data/` folder is independent of Python installation.

**Q: Do I need to re-run installer after git pull?**  
A: Only if `requirements.txt` changed. Check release notes.

**Q: Can I customize the portable Python version?**  
A: Yes! Edit `install.bat` and change the download URL. Use any Python 3.11+.

**Q: Why Python 3.11 and not 3.12?**  
A: 3.11 is stable and has excellent library support. 3.12 works too if you modify the scripts.

## Recommendations by Use Case

| Use Case | Recommended Mode |
|----------|-----------------|
| First-time user | 🚀 Portable |
| Windows end user | 🚀 Portable |
| Linux/Mac end user | 🚀 Portable |
| Contributing code | 👨‍💻 Developer |
| Running in IDE | 👨‍💻 Developer |
| CI/CD testing | 👨‍💻 Developer |
| Distributing to others | 🚀 Portable |
| Adding TTS support | 🚀 Portable |
| Multi-machine deployment | 🚀 Portable |
