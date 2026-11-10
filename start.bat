@echo off
setlocal
cd /d "%~dp0"
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto error
)
".venv\Scripts\python.exe" -c "import gradio, faster_whisper" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
    if errorlevel 1 goto error
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto error
exit /b 0
:error
echo Application failed. See the error above.
pause
exit /b 1
