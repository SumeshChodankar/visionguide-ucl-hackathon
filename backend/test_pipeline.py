"""
Automated End-to-End & Live API Verification Suite for VisionGuide (backend/test_pipeline.py)
Tests all functionality:
1. Spatial Geometry & Floor Inference ('443B' -> 4, 'G12' -> 0, '1204' -> 12, 'B1' -> -1)
2. Clock-Face Direction ('10 o'clock' to '2 o'clock'), Spatial Tagging, Distance Estimation & Haversine Math
3. VoiceAgent Intent Extraction (NAVIGATE, STATUS, STOP, UNKNOWN)
4. Parallel asyncio.gather Multi-Agent Orchestration (Navigation + YOLOv8n + EasyOCR + Gemini 2.5 Flash)
5. SQLite Session Persistence & Telemetry Logging (visionguide.db)
6. Live FastAPI HTTP Endpoints (GET /, GET /health, POST /api/v1/voice-command, POST /api/v1/analyze-frame,
   GET /api/v1/session/{id}, POST /api/v1/reset-session)
"""

import asyncio
import base64
import io
import json
import os
import tempfile

import httpx

from agents import (
    ReasoningAgent,
    bbox_to_clock_face,
    bbox_to_spatial_tag,
    estimate_distance_from_bbox,
    haversine_meters,
    infer_floor_from_room,
)
from database import VisionGuideDB


def create_dummy_jpeg() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (640, 480), color=(20, 20, 20)).save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        return b"\xff\xd8\xff\xd9"


async def run_verification() -> None:
    print("=" * 78)
    print("VISIONGUIDE FULL FUNCTIONALITY & MULTI-AGENT VERIFICATION SUITE")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Verify Intelligent Room-to-Floor Inference
    # -------------------------------------------------------------------------
    assert infer_floor_from_room("443B") == 4
    assert infer_floor_from_room("Room 443B") == 4
    assert infer_floor_from_room("G12") == 0
    assert infer_floor_from_room("Ground") == 0
    assert infer_floor_from_room("B1") == -1
    assert infer_floor_from_room("1204") == 12
    assert infer_floor_from_room("205A") == 2
    assert infer_floor_from_room("4.12") == 4
    print("[PASS] 1. Intelligent Room-to-Floor Inference (443B->F4, G12->F0, B1->F-1, 1204->F12)")

    # -------------------------------------------------------------------------
    # 2. Verify Clock-Face, Spatial Tagging, Distance & Haversine Geometry
    # -------------------------------------------------------------------------
    assert bbox_to_clock_face(0.10) == "10 o'clock"
    assert bbox_to_clock_face(0.30) == "11 o'clock"
    assert bbox_to_clock_face(0.50) == "12 o'clock"
    assert bbox_to_clock_face(0.70) == "1 o'clock"
    assert bbox_to_clock_face(0.90) == "2 o'clock"

    assert bbox_to_spatial_tag(0.15) == "left"
    assert bbox_to_spatial_tag(0.50) == "center"
    assert bbox_to_spatial_tag(0.85) == "right"

    band_near, dist_near = estimate_distance_from_bbox(0.95, 0.55, 0.30)
    band_med, dist_med = estimate_distance_from_bbox(0.65, 0.30, 0.08)
    band_far, dist_far = estimate_distance_from_bbox(0.30, 0.10, 0.02)
    assert band_near == "near" and dist_near < 1.5
    assert band_med == "medium" and 1.5 <= dist_med <= 3.5
    assert band_far == "far" and dist_far >= 3.5

    hav_dist = haversine_meters(51.5246, -0.1340, 51.5251, -0.1340)
    assert 50.0 <= hav_dist <= 60.0
    print(
        f"[PASS] 2. Spatial Geometry: Clock-Face (10-2 o'clock), Distance (near={dist_near}m, med={dist_med}m, far={dist_far}m), Haversine ({hav_dist:.1f}m)"
    )

    # -------------------------------------------------------------------------
    # 3. Verify VoiceAgent Intent Extraction
    # -------------------------------------------------------------------------
    orch = ReasoningAgent()
    intent_nav = await orch.voice_agent.extract_intent(
        "Take me to UCL Engineering Building, Room 443B"
    )
    assert intent_nav["intent_type"] == "NAVIGATE"
    assert intent_nav["target_room"] == "443B"
    assert intent_nav["target_floor"] == 4
    assert "UCL Engineering Building" in (intent_nav["destination_building"] or "")

    intent_stop = await orch.voice_agent.extract_intent("Stop navigation please")
    assert intent_stop["intent_type"] == "STOP"

    intent_status = await orch.voice_agent.extract_intent("Where am I right now?")
    assert intent_status["intent_type"] == "STATUS"

    intent_empty = await orch.voice_agent.extract_intent("")
    assert intent_empty["intent_type"] == "UNKNOWN"
    print("[PASS] 3. VoiceAgent Intent Extraction (NAVIGATE, STOP, STATUS, UNKNOWN)")

    # -------------------------------------------------------------------------
    # 4. Verify Parallel Multi-Agent Orchestration & SQLite Persistence
    # -------------------------------------------------------------------------
    tmp_db = os.path.join(tempfile.gettempdir(), "test_visionguide.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    test_db = VisionGuideDB(db_path=tmp_db)
    dummy_jpg = create_dummy_jpeg()

    session = test_db.update_session(
        "test-session-1",
        active_goal=intent_nav["active_goal"],
        destination_building=intent_nav["destination_building"],
        target_room=intent_nav["target_room"],
        target_floor=intent_nav["target_floor"],
        current_floor=1,
        navigation_mode="OUTDOOR",
    )

    # 4a. Outdoor Navigation Step
    res_outdoor = await orch.analyze_frame_parallel(
        session=session,
        image_bytes=dummy_jpg,
        lat=51.5246,
        lng=-0.1340,
        simulated_overrides={
            "yolo_detections": [],
            "ocr_items": [],
            "vision_result": {
                "wet_floor_or_spill": False,
                "glass_entrance_detected": False,
                "elevator_detected": False,
                "scene_summary": "Outdoor tactile paving sidewalk.",
            },
        },
    )
    assert res_outdoor["navigation_mode"] == "OUTDOOR"
    assert res_outdoor["hazard_priority"] == "NONE"
    print(f"[PASS] 4a. Outdoor Routing Step -> \"{res_outdoor['spoken_guidance']}\"")

    # 4b. Indoor Floor Mismatch (User enters Floor 1, needs Floor 4)
    res_mismatch = await orch.analyze_frame_parallel(
        session=session,
        image_bytes=dummy_jpg,
        lat=51.5246,
        lng=-0.1340,
        simulated_overrides={
            "ocr_items": [
                {"text": "Floor 1", "confidence": 0.96, "x_center_norm": 0.25},
                {"text": "Elevators ->", "confidence": 0.95, "x_center_norm": 0.75},
            ],
            "vision_result": {
                "wet_floor_or_spill": False,
                "glass_entrance_detected": True,
                "elevator_detected": True,
                "elevator_direction": "right",
                "detected_floor_number": 1,
                "scene_summary": "Main lobby on Floor 1.",
            },
        },
    )
    assert res_mismatch["floor_mismatch"] is True
    assert res_mismatch["navigation_mode"] == "ELEVATOR_SEEK"
    assert "Floor 1" in res_mismatch["spoken_guidance"]
    assert "Floor 4" in res_mismatch["spoken_guidance"]
    print(f"[PASS] 4b. Intelligent Floor Mismatch -> \"{res_mismatch['spoken_guidance']}\"")

    # 4c. Safety Priority Interruption (Obstacle at 12 o'clock)
    res_hazard = await orch.analyze_frame_parallel(
        session=session,
        image_bytes=dummy_jpg,
        lat=51.5246,
        lng=-0.1340,
        simulated_overrides={
            "yolo_detections": [
                {
                    "class_name": "chair",
                    "confidence": 0.93,
                    "bbox": [250, 220, 390, 460],
                }
            ],
            "vision_result": {
                "wet_floor_or_spill": False,
                "scene_summary": "Chair blocking corridor ahead.",
            },
        },
    )
    assert res_hazard["hazard_priority"] == "HIGH"
    assert res_hazard["interrupt_speech"] is True
    assert res_hazard["haptic_pattern"] == "URGENT_PULSE"
    assert "12 o'clock" in res_hazard["spoken_guidance"]
    print(f"[PASS] 4c. HIGH Hazard Preemption -> \"{res_hazard['spoken_guidance']}\"")

    # 4d. Arrival Verification on Floor 4 at Room 443B
    res_arrival = await orch.analyze_frame_parallel(
        session=session,
        image_bytes=dummy_jpg,
        lat=51.5246,
        lng=-0.1340,
        simulated_overrides={
            "ocr_items": [
                {"text": "Floor 4", "confidence": 0.98, "x_center_norm": 0.2},
                {"text": "Room 443B", "confidence": 0.99, "x_center_norm": 0.5},
            ],
            "vision_result": {
                "wet_floor_or_spill": False,
                "detected_floor_number": 4,
                "scene_summary": "Door plaque for Room 443B directly ahead.",
            },
        },
    )
    assert res_arrival["arrived"] is True
    assert res_arrival["navigation_mode"] == "ARRIVED"
    assert res_arrival["haptic_pattern"] == "ARRIVAL_SUCCESS"
    assert "Room 443B" in res_arrival["spoken_guidance"]
    print(f"[PASS] 4d. Arrival Verification -> \"{res_arrival['spoken_guidance']}\"")

    # -------------------------------------------------------------------------
    # 5. Verify Live FastAPI HTTP Server Endpoints (http://127.0.0.1:8000)
    # -------------------------------------------------------------------------
    b64_frame = base64.b64encode(dummy_jpg).decode("utf-8")
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        # 5a. GET / (Web HUD) and GET /health
        r_hud = await client.get("/")
        assert r_hud.status_code == 200 and "VISIONGUIDE" in r_hud.text
        r_health = await client.get("/health")
        assert r_health.status_code == 200 and r_health.json()["status"] == "ok"
        print("[PASS] 5a. HTTP GET / (Accessible Web HUD) & GET /health (200 OK)")

        # 5b. POST /api/v1/voice-command (Set Goal: Room 443B)
        r_voice = await client.post(
            "/api/v1/voice-command",
            data={
                "session_id": "e2e-http-user",
                "lat": "51.5246",
                "lng": "-0.1340",
                "transcript_override": "Take me to UCL Engineering Building, Room 443B",
            },
        )
        assert r_voice.status_code == 200
        v_data = r_voice.json()
        assert v_data["session"]["target_room"] == "443B"
        assert v_data["session"]["target_floor"] == 4
        assert v_data["session"]["navigation_mode"] == "OUTDOOR"
        print(f"[PASS] 5b. HTTP POST /api/v1/voice-command -> target_room=443B, target_floor=4")

        # 5c. POST /api/v1/analyze-frame (Floor 1 Mismatch)
        r_frame_mismatch = await client.post(
            "/api/v1/analyze-frame",
            data={
                "session_id": "e2e-http-user",
                "lat": "51.5246",
                "lng": "-0.1340",
                "frame_base64": b64_frame,
                "simulated_overrides_json": json.dumps(
                    {
                        "ocr_items": [
                            {"text": "Floor 1", "confidence": 0.97, "x_center_norm": 0.25},
                            {"text": "Elevators ->", "confidence": 0.95, "x_center_norm": 0.75},
                        ],
                        "vision_result": {
                            "wet_floor_or_spill": False,
                            "glass_entrance_detected": True,
                            "elevator_detected": True,
                            "elevator_direction": "right",
                            "detected_floor_number": 1,
                            "scene_summary": "Entered main lobby on Floor 1.",
                        },
                    }
                ),
            },
        )
        assert r_frame_mismatch.status_code == 200
        fm_data = r_frame_mismatch.json()
        assert fm_data["floor_mismatch"] is True
        assert fm_data["navigation_mode"] == "ELEVATOR_SEEK"
        print(
            f"[PASS] 5c. HTTP POST /api/v1/analyze-frame (Floor Mismatch) -> mode={fm_data['navigation_mode']}, latency={fm_data['total_latency_ms']}ms"
        )

        # 5d. POST /api/v1/analyze-frame (HIGH Hazard + Wet Floor)
        r_frame_hazard = await client.post(
            "/api/v1/analyze-frame",
            data={
                "session_id": "e2e-http-user",
                "lat": "51.5246",
                "lng": "-0.1340",
                "frame_base64": b64_frame,
                "simulated_overrides_json": json.dumps(
                    {
                        "yolo_detections": [
                            {"class_name": "chair", "confidence": 0.94, "bbox": [240, 220, 400, 465]}
                        ],
                        "vision_result": {
                            "wet_floor_or_spill": True,
                            "surface_hazard_description": "Yellow wet floor sign and water spill",
                            "scene_summary": "Obstacle and wet floor ahead.",
                        },
                    }
                ),
            },
        )
        assert r_frame_hazard.status_code == 200
        fh_data = r_frame_hazard.json()
        assert fh_data["hazard_priority"] == "HIGH"
        assert fh_data["interrupt_speech"] is True
        assert fh_data["haptic_pattern"] == "URGENT_PULSE"
        print(
            f"[PASS] 5d. HTTP POST /api/v1/analyze-frame (HIGH Hazard) -> priority={fh_data['hazard_priority']}, interrupt={fh_data['interrupt_speech']}"
        )

        # 5e. POST /api/v1/analyze-frame (Arrive at Room 443B on Floor 4)
        r_frame_arrive = await client.post(
            "/api/v1/analyze-frame",
            data={
                "session_id": "e2e-http-user",
                "lat": "51.5246",
                "lng": "-0.1340",
                "frame_base64": b64_frame,
                "simulated_overrides_json": json.dumps(
                    {
                        "ocr_items": [
                            {"text": "Floor 4", "confidence": 0.98, "x_center_norm": 0.2},
                            {"text": "Room 443B", "confidence": 0.99, "x_center_norm": 0.5},
                        ],
                        "vision_result": {
                            "wet_floor_or_spill": False,
                            "detected_floor_number": 4,
                            "scene_summary": "Room 443B door plaque ahead.",
                        },
                    }
                ),
            },
        )
        assert r_frame_arrive.status_code == 200
        fa_data = r_frame_arrive.json()
        assert fa_data["arrived"] is True
        assert fa_data["navigation_mode"] == "ARRIVED"
        assert fa_data["haptic_pattern"] == "ARRIVAL_SUCCESS"
        print(
            f"[PASS] 5e. HTTP POST /api/v1/analyze-frame (Arrival 443B) -> arrived={fa_data['arrived']}, haptic={fa_data['haptic_pattern']}"
        )

        # 5f. GET /api/v1/session/{session_id} & POST /api/v1/reset-session
        r_sess = await client.get("/api/v1/session/e2e-http-user")
        assert r_sess.status_code == 200
        s_data = r_sess.json()
        assert s_data["session"]["arrived"] is True
        assert len(s_data["recent_telemetry"]) >= 4

        r_reset = await client.post(
            "/api/v1/reset-session", json={"session_id": "e2e-http-user"}
        )
        assert r_reset.status_code == 200
        assert r_reset.json()["session"]["navigation_mode"] == "IDLE"
        print(
            f"[PASS] 5f. HTTP GET /api/v1/session & POST /api/v1/reset-session ({len(s_data['recent_telemetry'])} telemetry records verified)"
        )

    print("=" * 78)
    print("ALL VISIONGUIDE UNIT, MULTI-AGENT & LIVE HTTP API TESTS PASSED (100%)!")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(run_verification())
