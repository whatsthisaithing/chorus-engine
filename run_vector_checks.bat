@echo off
setlocal

REM Verify summary/memory/moment-pin vector presence for one conversation.
REM Usage:
REM   run_vector_checks.bat <conversation_id>
REM   run_vector_checks.bat <conversation_id> --strict

if "%~1"=="" (
    echo Usage: run_vector_checks.bat ^<conversation_id^> [--strict]
    exit /b 1
)

REM Keep proxy changes local to this script execution.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "GIT_HTTP_PROXY="
set "GIT_HTTPS_PROXY="

set "PYTHON_CMD="
if exist "python_embeded\python.exe" (
    echo [Using Embedded Python: python_embeded]
    set "PYTHON_CMD=python_embeded\python.exe"
) else if exist "python_embedded\python.exe" (
    echo [Using Embedded Python: python_embedded]
    set "PYTHON_CMD=python_embedded\python.exe"
) else (
    echo [Using System Python]
    set "PYTHON_CMD=python"
)

echo Running: %PYTHON_CMD% utilities\check_conversation_vectors.py %*
%PYTHON_CMD% utilities\check_conversation_vectors.py %*

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Vector check exited with code %EXIT_CODE%.
)

endlocal & exit /b %EXIT_CODE%
