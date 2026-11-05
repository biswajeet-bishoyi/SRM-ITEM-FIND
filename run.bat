@echo off
REM Starts the SRM CampusFind server and opens it in the browser.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment and installing requirements. This takes several minutes the first time...
    py -3.11 -m venv venv || python -m venv venv
    venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
)

echo Starting SRM CampusFind at http://localhost:8000 (models take ~30 seconds to load)...
start "" cmd /c "timeout /t 30 >nul && start http://localhost:8000"
venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
pause
