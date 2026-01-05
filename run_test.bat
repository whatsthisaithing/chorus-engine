@echo off
REM Helper script to run test scripts with correct Python environment
REM
REM Usage: run_test.bat testing\test_chatterbox.py
REM    or: run_test.bat testing\test_chatterbox.py --verbose

setlocal

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
    echo Usage: run_test.bat ^<script_path^> [args...]
    echo.
    echo Examples:
    echo   run_test.bat testing\test_chatterbox.py
    echo   run_test.bat testing\check_memories.py
    echo   run_test.bat view_debug_log.py
    echo.
    exit /b 1
)

REM Run the script with all arguments
echo Running: %PYTHON_CMD% %*
echo.
%PYTHON_CMD% %*

endlocal
