@echo off
REM CameraLM launcher.
REM
REM The camera recognition loop and the admin web server share a single Python
REM process - start_admin_server() runs on a daemon thread inside main.py - so
REM one launch covers both. This script:
REM   1. starts that process in its own titled console window
REM   2. waits a few seconds for the admin server to come up
REM   3. opens http://localhost:8765 (the admin UI) in the default browser

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [start.bat] .venv\Scripts\python.exe not found.
    echo Run setup.ps1 first to create the virtual environment.
    pause
    exit /b 1
)

start "CameraLM" ".venv\Scripts\python.exe" -m cameralm.main
echo Waiting for the admin server to come up...
timeout /t 8 /nobreak >nul
start "" "http://localhost:8765"
