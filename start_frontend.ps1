# VisionGuide Mobile Frontend Startup Script (PowerShell)
Write-Host "Starting VisionGuide React Native (Expo SDK 51+) Mobile App..." -ForegroundColor Yellow

Set-Location frontend
if (-Not (Test-Path "node_modules")) {
    Write-Host "Installing npm dependencies..." -ForegroundColor Cyan
    npm install
}

npx expo start

