@echo off
setlocal

REM ENS v3 harness launcher using embedded Python when available.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "GIT_HTTP_PROXY="
set "GIT_HTTPS_PROXY="

set "PYTHON_CMD="
if exist "python_embeded\python.exe" (
    set "PYTHON_CMD=python_embeded\python.exe"
) else if exist "python_embedded\python.exe" (
    set "PYTHON_CMD=python_embedded\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Running: %PYTHON_CMD% -m chorus_engine.devtools.ens_v3_harness %*
%PYTHON_CMD% -m chorus_engine.devtools.ens_v3_harness %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

