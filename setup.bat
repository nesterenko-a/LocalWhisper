@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto error
)
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto error
echo Setup complete. Run start.bat.
exit /b 0
:error
echo Setup failed. See the error above.
pause
exit /b 1
