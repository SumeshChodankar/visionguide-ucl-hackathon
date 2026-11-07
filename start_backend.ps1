# VisionGuide Backend Startup Script (PowerShell)
Write-Host "Starting VisionGuide Multi-Agent FastAPI Backend..." -ForegroundColor Cyan

if (-Not (Test-Path "backend\.venv")) {
    Write-Host "Creating Python virtual environment in backend\.venv..." -ForegroundColor Yellow
    python -m venv backend\.venv
}

& "backend\.venv\Scripts\Activate.ps1"
pip install -r backend\requirements.txt

Set-Location backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

