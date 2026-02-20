@echo off
REM Helper script to run Python scripts with the correct environment
REM
REM Usage: run_script.bat testing\test_chatterbox.py
REM    or: run_script.bat utilities\vector_lookup_probe\vector_lookup_probe.py --character nova_custom

setlocal

REM Keep proxy changes local to this script execution.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "GIT_HTTP_PROXY="
set "GIT_HTTPS_PROXY="

REM Check if embedded Python exists
if exist python_embeded\python.exe (
    echo [Using Embedded Python]
    set PYTHON_CMD=python_embeded\python.exe
) else (
    echo [Using System Python]
    set PYTHON_CMD=python
)

REM Check if arguments provided
if "%~1"=="" (
    echo.
    echo Usage: run_script.bat ^<script_path^> [args...]
    echo.
    echo Examples:
    echo   run_script.bat testing\test_chatterbox.py
    echo   run_script.bat testing\check_memories.py
    echo   run_script.bat utilities\vector_lookup_probe\vector_lookup_probe.py --character nova_custom
    echo   run_script.bat view_debug_log.py
    echo.
    exit /b 1
)

REM Run the script with all arguments
echo Running: %PYTHON_CMD% %*
echo.
%PYTHON_CMD% %*

endlocal
