# VisionGuide: Multi-Agent Outdoor-to-Indoor Assistive Navigation

**VisionGuide** is a full-stack, real-time multi-agent navigation platform built for blind and visually impaired users to navigate seamlessly from outdoor GPS coordinates to exact indoor rooms, gates, elevators, and reception desks.

---

## 1. System Architecture & Multi-Agent Pipeline

```
+-------------------------------------------------------------------------+
|                 React Native (Expo SDK 51+) Mobile App                  |
|  - expo-camera (3.5s auto frame loop)   - expo-speech (Low-latency TTS) |
|  - expo-location (Live GPS stream)      - expo-haptics (Urgent alerts)  |
|  - expo-av (Giant bottom Push-to-Talk)  - WCAG AAA High-Contrast HUD    |
+------------------------------------+------------------------------------+
                                     |
                   REST / Multipart (FastAPI Async)
                                     |
+------------------------------------v------------------------------------+
|              FastAPI Backend + SQLite (visionguide.db)                  |
|                                                                         |
|  1. VoiceAgent (OpenAI Whisper-1 + Intent & Floor Regex/LLM Parser)     |
|                                                                         |
|  Parallel Frame & Route Execution via `asyncio.gather`:                 |
|  +------------------+ +------------------+ +------------------+         |
|  | 2. Navigation    | | 3. HazardAgent   | | 4. OCRAgent      |         |
|  | Google Maps Walk | | Ultralytics      | | EasyOCR Plaques, |         |
|  | Directions &     | | YOLOv8n (<40ms)  | | Floor Signs &    |         |
|  | Haversine Step   | | Clock-Face & Dist| | Left/Center/Right|         |
|  +--------+---------+ +--------+---------+ +--------+---------+         |
|           |                    |                    |                   |
|           |           +--------+---------+          |                   |
|           |           | 5. VisionAgent   |          |                   |
|           |           | Gemini 2.5 Flash |          |                   |
|           |           | Wet Floor, Glass |          |                   |
|           |           | Doors, Elevators |          |                   |
|           |           +--------+---------+          |                   |
|           +--------------------+--------------------+                   |
|                                |                                        |
|                                v                                        |
|  +-------------------------------------------------------------------+  |
|  | 6. ReasoningAgent (Orchestrator)                                  |  |
|  | - Safety Priority Interruption (`hazard_priority == "HIGH"`)      |  |
|  | - Intelligent Floor Reasoning (`Room 443B` -> Floor 4 vs Floor 1) |  |
|  | - Arrival Verification (OCR plaque match / Vision reception)      |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

---

## 2. Project Structure

```text
.
├── .env.example                 # Environment variables template (API keys & ports)
├── start_backend.ps1            # One-command PowerShell launcher for FastAPI backend
├── start_frontend.ps1           # One-command PowerShell launcher for Expo mobile app
├── backend/
│   ├── requirements.txt         # Python dependencies (FastAPI, YOLOv8, EasyOCR, OpenAI, Gemini)
│   ├── database.py              # SQLite session persistence & telemetry logging (visionguide.db)
│   ├── agents.py                # All 6 agents + parallel asyncio.gather orchestrator
│   ├── main.py                  # FastAPI server (/api/v1/voice-command & /api/v1/analyze-frame)
│   └── test_pipeline.py         # End-to-end automated verification suite
└── frontend/
    ├── package.json             # Expo SDK 51+ dependencies
    ├── app.json                 # iOS/Android camera, microphone & location permissions
    └── App.js                   # Accessible WCAG AAA HUD, 3.5s loop, haptics & giant PTT button
```

---

## 3. Quick Start

### Step 1: Configure Environment Variables
Copy `.env.example` to `.env` (in root or `backend/.env`) and add your API keys:
```powershell
Copy-Item .env.example .env
```

### Step 2: Start the FastAPI Multi-Agent Backend
```powershell
.\start_backend.ps1
```
Or manually:
```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Start the React Native (Expo) Mobile Frontend
Update `EXPO_PUBLIC_API_BASE_URL` in `.env` to your machine's LAN IP (e.g., `http://192.168.1.100:8000`), then run:
```powershell
.\start_frontend.ps1
```
Scan the QR code in the **Expo Go** app on iOS or Android.

