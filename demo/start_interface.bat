@echo off
chcp 65001 >nul
REM ======================================================================
REM  start_interface.bat
REM  Loads venv and launches the Gradio Web UI in one step.
REM  Usage: double-click in Explorer, or run in cmd.
REM  Note: kept ASCII-only on purpose (Chinese GBK cmd would mangle UTF-8).
REM ======================================================================

setlocal

REM ---- Config (edit these 3 lines if paths change) ---------------------
set "VENV_PY=D:\Document\Course\Team_intership\LLM\.venv-gpu\Scripts\python.exe"
set "MODEL_DIR=D:\Document\Course\Team_intership\LLM\models\qwen25-coder-7b"
set "ENTRY=demo_interface.py"
REM ----------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"

echo ======================================================================
echo  LLM Topology Generation - Web Interface
echo ======================================================================
echo  Python : %VENV_PY%
echo  Model  : %MODEL_DIR%
echo  Script : %SCRIPT_DIR%%ENTRY%
echo ======================================================================

if not exist "%VENV_PY%" (
    echo [ERROR] venv Python not found: %VENV_PY%
    echo         Create the venv first, or fix the VENV_PY path above.
    pause
    exit /b 1
)
if not exist "%MODEL_DIR%" (
    echo [WARN ] Model directory not found: %MODEL_DIR%
    echo         Continuing anyway - the script may fail on first request.
)
if not exist "%SCRIPT_DIR%%ENTRY%" (
    echo [ERROR] Entry script not found: %SCRIPT_DIR%%ENTRY%
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
echo.
echo Launching Gradio... press Ctrl+C in this window to stop.
echo Browser will open automatically at http://127.0.0.1:7860
echo.

"%VENV_PY%" "%ENTRY%"

echo.
echo ======================================================================
echo  Server stopped.
echo ======================================================================
pause
endlocal
