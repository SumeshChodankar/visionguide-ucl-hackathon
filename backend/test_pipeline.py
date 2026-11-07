"""
Automated End-to-End Verification Suite for VisionGuide (backend/test_pipeline.py)
Tests:
1. Intelligent floor inference from room identifiers ('443B' -> 4, 'G12' -> 0, '1204' -> 12, 'B1' -> -1)
2. Clock-face direction mapping ('10 o'clock' through '2 o'clock') and spatial tagging ('left', 'center', 'right')
3. VoiceAgent intent extraction ('Take me to UCL Engineering Building, Room 443B', 'Stop navigation', 'Where am I')
4. Parallel asyncio.gather multi-agent orchestration:
   - Outdoor navigation step evaluation
   - Floor mismatch detection (Floor 1 vs Floor 4 -> routes to Elevator)
   - Safety Priority Interruption (HIGH hazard at 12 o'clock and wet floor preempts navigation speech)
   - Arrival Verification (OCR reads 'Room 443B' -> arrived=True)
5. SQLite session persistence, telemetry logging, and session reset
"""

import asyncio
import io
import os
import tempfile

from agents import (
    ReasoningAgent,
    bbox_to_clock_face,
    bbox_to_spatial_tag,
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
    # 1. Verify Floor Inference
    assert infer_floor_from_room("443B") == 4
    assert infer_floor_from_room("Room 443B") == 4
    assert infer_floor_from_room("G12") == 0
    assert infer_floor_from_room("Ground") == 0
    assert infer_floor_from_room("B1") == -1
    assert infer_floor_from_room("1204") == 12
    assert infer_floor_from_room("205A") == 2
    assert infer_floor_from_room("4.12") == 4

    # 2. Verify Clock-Face & Spatial Tag Mapping
    assert bbox_to_clock_face(0.10) == "10 o'clock"
    assert bbox_to_clock_face(0.30) == "11 o'clock"
    assert bbox_to_clock_face(0.50) == "12 o'clock"
    assert bbox_to_clock_face(0.70) == "1 o'clock"
    assert bbox_to_clock_face(0.90) == "2 o'clock"

    assert bbox_to_spatial_tag(0.15) == "left"
    assert bbox_to_spatial_tag(0.50) == "center"
    assert bbox_to_spatial_tag(0.85) == "right"

    # 3. Verify SQLite Persistence & Orchestrator Scenarios
    tmp_db = os.path.join(tempfile.gettempdir(), "test_visionguide.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    test_db = VisionGuideDB(db_path=tmp_db)
    orch = ReasoningAgent()
    dummy_jpg = create_dummy_jpeg()

    # Step A: Voice Command Intent Extraction
    intent = await orch.voice_agent.extract_intent(
        "Take me to UCL Engineering Building, Room 443B"
    )
    assert intent["intent_type"] == "NAVIGATE"
    assert intent["target_room"] == "443B"
    assert intent["target_floor"] == 4
    assert "UCL Engineering Building" in (intent["destination_building"] or "")

    stop_intent = await orch.voice_agent.extract_intent("Stop navigation please")
    assert stop_intent["intent_type"] == "STOP"

    status_intent = await orch.voice_agent.extract_intent("Where am I right now?")
    assert status_intent["intent_type"] == "STATUS"

    session = test_db.update_session(
        "test-session-1",
        active_goal=intent["active_goal"],
        destination_building=intent["destination_building"],
        target_room=intent["target_room"],
        target_floor=intent["target_floor"],
        current_floor=1,
        navigation_mode="OUTDOOR",
    )

    # Step B: Outdoor Navigation Mode (No indoor signs yet)
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

    # Step C: Indoor Floor Mismatch (User enters Floor 1, needs Floor 4)
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

    # Step D: Safety Priority Interruption (Obstacle at 12 o'clock)
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

    # Step E: Arrival Verification on Floor 4 at Room 443B
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

    # Step F: Verify Telemetry Logging & Reset
    log_id = test_db.log_telemetry(
        session_id="test-session-1",
        event_type="ANALYZE_FRAME",
        hazard_priority=res_arrival["hazard_priority"],
        spoken_guidance=res_arrival["spoken_guidance"],
        payload={"arrived": True},
        latency_ms=res_arrival["total_latency_ms"],
    )
    assert log_id > 0
    logs = test_db.get_recent_telemetry("test-session-1", limit=5)
    assert len(logs) == 1
    assert logs[0]["payload"]["arrived"] is True

    reset_state = test_db.reset_session("test-session-1")
    assert reset_state["navigation_mode"] == "IDLE"
    assert reset_state["active_goal"] is None

    print("ALL VISIONGUIDE MULTI-AGENT PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_verification())
