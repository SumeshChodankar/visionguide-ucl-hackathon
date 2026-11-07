"""
VisionGuide FastAPI Backend Server (backend/main.py)
Exposes low-latency endpoints for push-to-talk voice commands and 3.5-second
continuous camera frame analysis powered by parallel multi-agent execution.
"""

import base64
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
from pydantic import BaseModel

load_dotenv()

try:
    from backend.agents import orchestrator
    from backend.database import db
except ImportError:
    from agents import orchestrator
    from database import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite schema and warm up orchestrator on server startup."""
    db._init_db()
    yield


app = FastAPI(
    title="VisionGuide Multi-Agent API",
    description=(
        "Outdoor-to-Indoor Assistive Navigation API orchestrating Whisper-1, "
        "Google Maps Directions, YOLOv8n, EasyOCR, and Gemini 2.5 Flash."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetSessionRequest(BaseModel):
    session_id: str = "default-user"


@app.get("/", response_class=HTMLResponse)
async def serve_web_hud():
    """Serve the interactive VisionGuide Web HUD & Multi-Agent Simulator."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return HTMLResponse("<h1>VisionGuide API is running. Visit <a href='/docs'>/docs</a></h1>")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint reporting multi-agent readiness."""
    return {
        "status": "ok",
        "service": "VisionGuide Multi-Agent Backend",
        "agents": [
            "VoiceAgent (Whisper-1 + Intent/Floor Parser)",
            "NavigationAgent (Google Maps Walking Directions)",
            "HazardAgent (Ultralytics YOLOv8n <40ms)",
            "OCRAgent (EasyOCR Spatial Sign & Room Reader)",
            "VisionAgent (Gemini 2.5 Flash Multimodal)",
            "ReasoningAgent (Parallel asyncio.gather Orchestrator)",
        ],
    }


@app.get("/api/v1/session/{session_id}")
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """Retrieve current navigation state and recent telemetry for a session."""
    session = db.get_or_create_session(session_id)
    telemetry = db.get_recent_telemetry(session_id, limit=10)
    return {
        "session": session,
        "recent_telemetry": telemetry,
    }


@app.post("/api/v1/voice-command")
async def handle_voice_command(
    session_id: str = Form("default-user"),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    transcript_override: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Process a Push-to-Talk voice command:
    1. Transcribe audio via OpenAI Whisper-1 (`whisper-1`) or use `transcript_override`.
    2. Extract destination building, target room, and inferred target floor.
    3. Compute outdoor walking route via Google Maps Directions API (`mode=walking`).
    4. Persist active goal and route in SQLite (`visionguide.db`).
    """
    t0 = time.perf_counter()
    session = db.get_or_create_session(session_id)

    transcript = (transcript_override or "").strip()
    if not transcript and audio is not None:
        audio_bytes = await audio.read()
        transcript = await orchestrator.voice_agent.transcribe_audio(
            audio_bytes, filename=audio.filename or "command.m4a"
        )

    intent = await orchestrator.voice_agent.extract_intent(transcript)
    intent_type = intent["intent_type"]

    if intent_type == "STOP":
        updated_session = db.reset_session(session_id)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        db.log_telemetry(
            session_id=session_id,
            event_type="VOICE_COMMAND_STOP",
            hazard_priority="NONE",
            spoken_guidance=intent["spoken_confirmation"],
            payload={"transcript": transcript, "intent": intent},
            latency_ms=latency_ms,
        )
        return {
            "session_id": session_id,
            "transcript": transcript,
            "intent": intent,
            "spoken_guidance": intent["spoken_confirmation"],
            "hazard_priority": "NONE",
            "interrupt_speech": False,
            "haptic_pattern": "NONE",
            "session": updated_session,
            "latency_ms": latency_ms,
        }

    if intent_type == "NAVIGATE":
        dest_query = (
            intent.get("destination_building")
            or intent.get("active_goal")
            or "Target Building"
        )
        route_info = await orchestrator.navigation_agent.compute_route(
            origin_lat=lat if lat is not None else session.get("last_lat"),
            origin_lng=lng if lng is not None else session.get("last_lng"),
            destination_query=dest_query,
        )
        steps = route_info.get("steps", [])
        first_instruction = (
            steps[0]["instruction"] if steps else "Proceed forward toward your destination."
        )
        spoken = f"{intent['spoken_confirmation']} First step: {first_instruction}."

        updated_session = db.update_session(
            session_id,
            active_goal=intent.get("active_goal"),
            destination_building=intent.get("destination_building"),
            target_room=intent.get("target_room"),
            target_floor=intent.get("target_floor"),
            navigation_mode="OUTDOOR",
            last_lat=lat if lat is not None else session.get("last_lat"),
            last_lng=lng if lng is not None else session.get("last_lng"),
            route_steps=steps,
            current_step_index=0,
            arrived=False,
        )
    else:
        spoken = intent["spoken_confirmation"]
        updated_session = session

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    db.log_telemetry(
        session_id=session_id,
        event_type="VOICE_COMMAND",
        hazard_priority="NONE",
        spoken_guidance=spoken,
        payload={"transcript": transcript, "intent": intent},
        latency_ms=latency_ms,
    )

    return {
        "session_id": session_id,
        "transcript": transcript,
        "intent": intent,
        "spoken_guidance": spoken,
        "hazard_priority": "NONE",
        "interrupt_speech": False,
        "haptic_pattern": "COMMAND_ACCEPTED",
        "session": updated_session,
        "latency_ms": latency_ms,
    }


@app.post("/api/v1/analyze-frame")
async def analyze_frame(
    session_id: str = Form("default-user"),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    frame_base64: Optional[str] = Form(None),
    simulated_overrides_json: Optional[str] = Form(None),
    frame: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Analyze a live mobile camera frame + GPS coordinate every 3.5 seconds:
    - Runs HazardAgent (YOLOv8n), OCRAgent (EasyOCR), VisionAgent (Gemini 2.5 Flash),
      and NavigationAgent concurrently via `asyncio.gather`.
    - Performs intelligent floor reasoning (e.g., Room 443B -> Floor 4 vs current Floor 1).
    - Preempts speech immediately with `interrupt_speech=True` when `hazard_priority == "HIGH"`.
    - Verifies arrival when OCR reads the target room plaque or Vision spots the target entrance.
    """
    session = db.get_or_create_session(session_id)

    image_bytes = b""
    if frame is not None:
        image_bytes = await frame.read()
    elif frame_base64:
        clean_b64 = frame_base64.split(",")[-1].strip()
        try:
            image_bytes = base64.b64decode(clean_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid frame_base64 payload: {exc}"
            )

    simulated_overrides: Optional[Dict[str, Any]] = None
    if simulated_overrides_json:
        try:
            simulated_overrides = json.loads(simulated_overrides_json)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid simulated_overrides_json: {exc}"
            )

    result = await orchestrator.analyze_frame_parallel(
        session=session,
        image_bytes=image_bytes,
        lat=lat if lat is not None else session.get("last_lat"),
        lng=lng if lng is not None else session.get("last_lng"),
        simulated_overrides=simulated_overrides,
    )

    # Persist updated floor, step index, GPS, and arrival state in SQLite
    nav_agent_res = result.get("agents", {}).get("navigation", {})
    updated_session = db.update_session(
        session_id,
        current_floor=result["current_floor"],
        target_floor=result["target_floor"],
        navigation_mode=result["navigation_mode"],
        last_lat=lat if lat is not None else session.get("last_lat"),
        last_lng=lng if lng is not None else session.get("last_lng"),
        route_steps=nav_agent_res.get("steps", session.get("route_steps", [])),
        current_step_index=nav_agent_res.get(
            "step_index", session.get("current_step_index", 0)
        ),
        arrived=result["arrived"],
    )
    result["session"] = updated_session

    db.log_telemetry(
        session_id=session_id,
        event_type="ANALYZE_FRAME",
        hazard_priority=result["hazard_priority"],
        spoken_guidance=result["spoken_guidance"],
        payload={
            "navigation_mode": result["navigation_mode"],
            "current_floor": result["current_floor"],
            "target_floor": result["target_floor"],
            "floor_mismatch": result["floor_mismatch"],
            "arrived": result["arrived"],
            "hazards_count": len(
                result.get("agents", {}).get("hazard", {}).get("hazards", [])
            ),
            "ocr_texts_count": len(
                result.get("agents", {}).get("ocr", {}).get("texts", [])
            ),
        },
        latency_ms=result["total_latency_ms"],
    )

    return result


@app.post("/api/v1/reset-session")
async def reset_session_endpoint(req: ResetSessionRequest) -> Dict[str, Any]:
    """Reset the navigation session state back to IDLE."""
    updated = db.reset_session(req.session_id)
    return {
        "status": "reset",
        "session": updated,
        "spoken_guidance": "Session reset. Hold Push-to-Talk to set a new destination.",
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)

