@echo off
setlocal

REM Run pytest with embedded Python when available.
REM Usage:
REM   run_test.bat
REM   run_test.bat all
REM   run_test.bat all -k vector_store
REM   run_test.bat testing\test_ens_slice01_integration.py -q

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

REM Clean pytest temp dir used by tests (best-effort).
if exist ".pytest_tmp" (
    rmdir /s /q ".pytest_tmp" >nul 2>&1
)

echo.
if "%~1"=="" goto :smoke
if /i "%~1"=="all" goto :full
goto :custom

:smoke
echo Running: %PYTHON_CMD% -m pytest -q testing\test_ens_slice01_integration.py testing\test_ens_slice2_integration.py testing\test_ens_slice2_media_refinements_integration.py testing\test_ens_slice25_media_gating_integration.py testing\test_ens_slice3_integration.py testing\test_ens_slice4_config_integration.py testing\test_ens_slice5_integration.py testing\test_ens_slice6_integration.py testing\test_ens_slice65_egress_integration.py testing\test_ens_slice7_unified_llm_invocation_integration.py
%PYTHON_CMD% -m pytest -q testing\test_ens_slice01_integration.py testing\test_ens_slice2_integration.py testing\test_ens_slice2_media_refinements_integration.py testing\test_ens_slice25_media_gating_integration.py testing\test_ens_slice3_integration.py testing\test_ens_slice4_config_integration.py testing\test_ens_slice5_integration.py testing\test_ens_slice6_integration.py testing\test_ens_slice65_egress_integration.py testing\test_ens_slice7_unified_llm_invocation_integration.py
goto :done

:full
shift
if "%~1"=="" (
    echo Running: %PYTHON_CMD% -m pytest -q
    %PYTHON_CMD% -m pytest -q
) else (
    echo Running: %PYTHON_CMD% -m pytest %*
    %PYTHON_CMD% -m pytest %*
)
goto :done

:custom
echo Running: %PYTHON_CMD% -m pytest %*
%PYTHON_CMD% -m pytest %*

:done

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Pytest exited with code %EXIT_CODE%.
)

REM Clean pytest temp dir after run to avoid stale tracked artifacts.
if exist ".pytest_tmp" (
    rmdir /s /q ".pytest_tmp" >nul 2>&1
)

endlocal & exit /b %EXIT_CODE%
