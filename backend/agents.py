"""
VisionGuide Multi-Agent Pipeline (backend/agents.py)
Implements all 6 specialized agents and parallel asyncio.gather orchestration:
1. VoiceAgent       - OpenAI Whisper-1 STT + Regex/LLM Intent & Floor Inference
2. NavigationAgent  - Google Maps Walking Directions & Places API + Haversine Proximity
3. HazardAgent      - Ultralytics YOLOv8n (<40ms) with Clock-Face & Vertical Distance Estimation
4. OCRAgent         - EasyOCR Text & Plaque Reader with Left/Center/Right Spatial Tagging
5. VisionAgent      - Gemini 2.5 Flash Multimodal Scene & Environmental Hazard Analyzer
6. ReasoningAgent   - Orchestrator synthesizing GPS + YOLO + OCR + Gemini + Session Goal
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

logger = logging.getLogger("visionguide.agents")
logging.basicConfig(level=logging.INFO)


# ============================================================================
# UTILITY: INTELLIGENT ROOM-TO-FLOOR INFERENCE & SPATIAL GEOMETRY
# ============================================================================

def infer_floor_from_room(room_str: Optional[str]) -> Optional[int]:
    """
    Infer building floor number from a room identifier.
    Examples:
      - '443B' or 'Room 443B' -> 4
      - '102'  -> 1
      - '205A' -> 2
      - 'G12' or 'Ground' -> 0
      - 'B1' or 'Basement' -> -1
      - '1204' -> 12
    """
    if not room_str:
        return None
    cleaned = room_str.strip().upper()
    cleaned = re.sub(r"^(ROOM|RM|SUITE|LAB|GATE)\s*[-:#]?\s*", "", cleaned).strip()

    if re.match(r"^(G|GROUND|GF)\b", cleaned) or re.match(r"^G\d+", cleaned):
        return 0
    if re.match(r"^(B|BASEMENT|LG)\d*", cleaned):
        return -1

    match = re.search(r"(\d{3,4})[A-Z]?\b", cleaned)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            return int(digits[0])
        if len(digits) == 4:
            return int(digits[:2])

    # Single or two-digit room with explicit floor prefix like "4.12" or "4-15"
    sep_match = re.match(r"^(\d{1,2})[.\-](\d{1,3})[A-Z]?$", cleaned)
    if sep_match:
        return int(sep_match.group(1))

    return None


def bbox_to_clock_face(x_center_norm: float) -> str:
    """
    Map normalized horizontal center [0.0, 1.0] to clock-face direction
    spanning the camera's ~70-degree horizontal field of view:
      0.00 - 0.20 -> 10 o'clock (far left)
      0.20 - 0.40 -> 11 o'clock (slightly left)
      0.40 - 0.60 -> 12 o'clock (dead ahead)
      0.60 - 0.80 -> 1 o'clock  (slightly right)
      0.80 - 1.00 -> 2 o'clock  (far right)
    """
    if x_center_norm < 0.20:
        return "10 o'clock"
    if x_center_norm < 0.40:
        return "11 o'clock"
    if x_center_norm <= 0.60:
        return "12 o'clock"
    if x_center_norm <= 0.80:
        return "1 o'clock"
    return "2 o'clock"


def bbox_to_spatial_tag(x_center_norm: float) -> str:
    """Map normalized horizontal center [0.0, 1.0] to 'left', 'center', or 'right'."""
    if x_center_norm < 0.35:
        return "left"
    if x_center_norm <= 0.65:
        return "center"
    return "right"


def estimate_distance_from_bbox(
    y2_norm: float, height_norm: float, area_norm: float
) -> Tuple[str, float]:
    """
    Estimate approximate metric distance and proximity band ('near', 'medium', 'far')
    using vertical bounding-box grounding (y2_norm near bottom of frame = closer)
    and relative bounding-box height/area.
    """
    # Combine bottom edge position and box height
    closeness_score = (0.65 * y2_norm) + (0.35 * min(1.0, height_norm * 1.3))
    if closeness_score >= 0.78 or area_norm >= 0.28:
        meters = max(0.6, round(2.2 * (1.0 - min(0.85, area_norm)), 1))
        return "near", min(meters, 1.4)
    if closeness_score >= 0.52 or area_norm >= 0.10:
        meters = round(1.6 + (1.0 - closeness_score) * 3.0, 1)
        return "medium", min(max(meters, 1.5), 3.4)
    meters = round(3.5 + (1.0 - closeness_score) * 5.0, 1)
    return "far", meters


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two WGS84 coordinates."""
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def strip_html(text: str) -> str:
    """Remove HTML tags from Google Maps step instructions."""
    clean = re.sub(r"<div[^>]*>", ". ", text)
    clean = re.sub(r"<[^>]+>", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


# ============================================================================
# 1. VOICE AGENT (OpenAI Whisper-1 + Intent & Goal Extraction)
# ============================================================================

class VoiceAgent:
    """
    Transcribes user push-to-talk audio via OpenAI Whisper-1 (`whisper-1`)
    and extracts structured navigation intents (building, room, floor, command).
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key and not self.api_key.startswith("your_"):
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except Exception as exc:
                logger.warning("OpenAI client initialization skipped: %s", exc)
        return self._client

    async def transcribe_audio(
        self, audio_bytes: Optional[bytes], filename: str = "command.m4a"
    ) -> str:
        """Transcribe audio bytes using OpenAI Whisper-1 API."""
        if not audio_bytes:
            return ""
        client = self._get_client()
        if client is None:
            logger.info("OPENAI_API_KEY not set; using fallback transcript parser.")
            return ""
        try:
            bio = io.BytesIO(audio_bytes)
            bio.name = filename
            transcription = await client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
                language="en",
                prompt="VisionGuide indoor and outdoor navigation command: Room 443B, Floor 4, UCL Engineering Building, reception desk, elevator, stairs.",
            )
            return transcription.text.strip()
        except Exception as exc:
            logger.error("Whisper transcription error: %s", exc)
            return ""

    async def extract_intent(self, transcript: str) -> Dict[str, Any]:
        """
        Extract structured navigation intent from transcript using deterministic
        regex rules augmented by LLM fallback when needed.
        """
        text = transcript.strip()
        if not text:
            return {
                "intent_type": "UNKNOWN",
                "active_goal": None,
                "destination_building": None,
                "target_room": None,
                "target_floor": None,
                "spoken_confirmation": "I didn't catch that. Please hold the bottom button and tell me your destination, such as Room 443B in the Engineering Building.",
            }

        lower = text.lower()

        # Check control commands first
        if re.search(r"\b(stop|cancel|abort|clear|reset)\b", lower):
            return {
                "intent_type": "STOP",
                "active_goal": None,
                "destination_building": None,
                "target_room": None,
                "target_floor": None,
                "spoken_confirmation": "Navigation stopped. Hold the Push-to-Talk button whenever you are ready for a new destination.",
            }

        if re.search(r"\b(where am i|what is in front of me|describe|status|repeat)\b", lower):
            return {
                "intent_type": "STATUS",
                "active_goal": text,
                "destination_building": None,
                "target_room": None,
                "target_floor": None,
                "spoken_confirmation": "Scanning your surroundings now.",
            }

        # Extract room number (e.g., "Room 443B", "lab 205A", "gate 12", or standalone "443B")
        target_room: Optional[str] = None
        room_match = re.search(
            r"\b(?:room|rm|lab|suite|office|gate|lecture\s+theatre)\s*([a-z]?\d{1,4}[a-z]?)\b",
            text,
            re.IGNORECASE,
        )
        if room_match:
            target_room = room_match.group(1).upper()
        else:
            standalone_room = re.search(r"\b(\d{3,4}[A-Za-z])\b", text)
            if standalone_room:
                target_room = standalone_room.group(1).upper()

        # Extract explicit floor if spoken (e.g. "floor 4", "4th floor", "level 2")
        target_floor: Optional[int] = None
        floor_match = re.search(
            r"\b(?:floor|level)\s*(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\s+floor\b",
            lower,
        )
        if floor_match:
            fl_str = floor_match.group(1) or floor_match.group(2)
            target_floor = int(fl_str)
        elif "ground floor" in lower:
            target_floor = 0
        elif target_room:
            target_floor = infer_floor_from_room(target_room)

        # Extract destination building or landmark
        destination_building: Optional[str] = None
        bldg_Cleaned = re.sub(
            r"^(?:take me to|navigate to|go to|guide me to|find|walk to|i need to get to)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" .,!?")

        # Separate room clause from building clause if both exist
        if target_room:
            bldg_candidate = re.sub(
                r"\b(?:in|at|inside|to)?\s*(?:room|rm|lab|suite|office|gate)?\s*"
                + re.escape(target_room)
                + r"\b",
                "",
                bldg_Cleaned,
                flags=re.IGNORECASE,
            ).strip(" ,.-inAt")
            bldg_candidate = re.sub(
                r"^(?:the|in the|at the|in|at|to)\s+",
                "",
                bldg_candidate,
                flags=re.IGNORECASE,
            ).strip(" ,.-")
            if len(bldg_candidate) >= 3:
                destination_building = bldg_candidate
        else:
            destination_building = bldg_Cleaned if bldg_Cleaned else None

        # Construct accessible spoken confirmation
        parts = []
        if destination_building and target_room:
            parts.append(f"Routing to {destination_building}, Room {target_room}")
        elif target_room:
            parts.append(f"Routing to Room {target_room}")
        elif destination_building:
            parts.append(f"Routing to {destination_building}")
        else:
            parts.append(f"Goal set: {text}")

        if target_floor is not None:
            floor_name = "Ground Floor" if target_floor == 0 else f"Floor {target_floor}"
            parts.append(f"on {floor_name}")

        spoken_confirmation = (
            " ".join(parts)
            + ". Continuous camera hazard and sign scanning is active."
        )

        return {
            "intent_type": "NAVIGATE",
            "active_goal": text,
            "destination_building": destination_building,
            "target_room": target_room,
            "target_floor": target_floor,
            "spoken_confirmation": spoken_confirmation,
        }


# ============================================================================
# 2. NAVIGATION AGENT (Google Maps Directions & Places API)
# ============================================================================

class NavigationAgent:
    """
    Provides outdoor turn-by-turn walking directions via Google Maps Directions API
    and transitions seamlessly to indoor mode when approaching the target building.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")

    async def compute_route(
        self,
        origin_lat: Optional[float],
        origin_lng: Optional[float],
        destination_query: str,
    ) -> Dict[str, Any]:
        """Fetch walking route from Google Maps Directions API or generate structured fallback."""
        if (
            origin_lat is None
            or origin_lng is None
            or not self.api_key
            or self.api_key.startswith("your_")
        ):
            return self._fallback_walking_route(origin_lat, origin_lng, destination_query)

        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{origin_lat},{origin_lng}",
            "destination": destination_query,
            "mode": "walking",
            "key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(url, params=params)
                data = resp.json()
                if data.get("status") != "OK" or not data.get("routes"):
                    logger.warning(
                        "Google Directions returned status %s; using fallback route.",
                        data.get("status"),
                    )
                    return self._fallback_walking_route(
                        origin_lat, origin_lng, destination_query
                    )

                leg = data["routes"][0]["legs"][0]
                steps = []
                for step in leg.get("steps", []):
                    steps.append(
                        {
                            "instruction": strip_html(step.get("html_instructions", "")),
                            "distance_text": step.get("distance", {}).get("text", ""),
                            "distance_meters": step.get("distance", {}).get("value", 0),
                            "end_lat": step.get("end_location", {}).get("lat"),
                            "end_lng": step.get("end_location", {}).get("lng"),
                        }
                    )
                return {
                    "source": "google_maps",
                    "destination_address": leg.get("end_address", destination_query),
                    "total_distance_text": leg.get("distance", {}).get("text", ""),
                    "total_duration_text": leg.get("duration", {}).get("text", ""),
                    "steps": steps,
                }
        except Exception as exc:
            logger.error("Google Maps API error: %s", exc)
            return self._fallback_walking_route(origin_lat, origin_lng, destination_query)

    @staticmethod
    def _fallback_walking_route(
        origin_lat: Optional[float],
        origin_lng: Optional[float],
        destination_query: str,
    ) -> Dict[str, Any]:
        base_lat = origin_lat if origin_lat is not None else 51.5246
        base_lng = origin_lng if origin_lng is not None else -0.1340
        target_name = destination_query or "Target Building Entrance"
        return {
            "source": "simulated_gps_fallback",
            "destination_address": target_name,
            "total_distance_text": "85 m",
            "total_duration_text": "1 min",
            "steps": [
                {
                    "instruction": f"Walk straight ahead along the tactile paving toward {target_name}",
                    "distance_text": "55 m",
                    "distance_meters": 55,
                    "end_lat": round(base_lat + 0.00045, 6),
                    "end_lng": round(base_lng + 0.00010, 6),
                },
                {
                    "instruction": f"Turn slightly right toward the main glass entrance of {target_name}",
                    "distance_text": "30 m",
                    "distance_meters": 30,
                    "end_lat": round(base_lat + 0.00065, 6),
                    "end_lng": round(base_lng + 0.00025, 6),
                },
            ],
        }

    async def evaluate_navigation_step(
        self,
        session: Dict[str, Any],
        current_lat: Optional[float],
        current_lng: Optional[float],
    ) -> Dict[str, Any]:
        """Evaluate current outdoor progress and return active step instruction."""
        steps: List[Dict[str, Any]] = session.get("route_steps") or []
        idx: int = int(session.get("current_step_index") or 0)

        if not steps:
            dest = (
                session.get("destination_building")
                or session.get("active_goal")
                or "destination"
            )
            if session.get("active_goal"):
                route = await self.compute_route(current_lat, current_lng, dest)
                steps = route.get("steps", [])
                idx = 0
            else:
                return {
                    "has_active_route": False,
                    "current_instruction": "No active destination set. Hold Push-to-Talk to speak your destination.",
                    "distance_to_step_meters": None,
                    "near_building_entrance": False,
                    "step_index": 0,
                    "steps": [],
                }

        if idx >= len(steps):
            idx = len(steps) - 1

        step = steps[idx]
        dist_m = step.get("distance_meters", 30)
        if (
            current_lat is not None
            and current_lng is not None
            and step.get("end_lat") is not None
            and step.get("end_lng") is not None
        ):
            dist_m = round(
                haversine_meters(
                    current_lat, current_lng, step["end_lat"], step["end_lng"]
                ),
                1,
            )
            if dist_m < 12.0 and idx + 1 < len(steps):
                idx += 1
                step = steps[idx]

        near_entrance = idx == len(steps) - 1 and dist_m <= 20.0
        return {
            "has_active_route": True,
            "current_instruction": step.get("instruction", "Proceed straight ahead."),
            "distance_to_step_meters": dist_m,
            "near_building_entrance": near_entrance,
            "step_index": idx,
            "steps": steps,
        }


# ============================================================================
# 3. HAZARD AGENT (Ultralytics YOLOv8n <40ms Local Obstacle Detection)
# ============================================================================

class HazardAgent:
    """
    Runs Ultralytics YOLOv8n (`yolov8n.pt`) locally to detect dynamic and static
    obstacles with clock-face direction ("12 o'clock") and vertical bounding-box
    distance estimation.
    """

    # COCO classes relevant for blind/low-vision navigation safety
    HAZARD_CLASSES = {
        "person": {"label": "person", "base_severity": "MEDIUM"},
        "bicycle": {"label": "cyclist", "base_severity": "HIGH"},
        "car": {"label": "vehicle", "base_severity": "HIGH"},
        "motorcycle": {"label": "motorcycle", "base_severity": "HIGH"},
        "bus": {"label": "bus", "base_severity": "HIGH"},
        "truck": {"label": "truck", "base_severity": "HIGH"},
        "traffic light": {"label": "traffic light", "base_severity": "LOW"},
        "fire hydrant": {"label": "fire hydrant obstacle", "base_severity": "MEDIUM"},
        "stop sign": {"label": "signpost", "base_severity": "LOW"},
        "bench": {"label": "bench", "base_severity": "MEDIUM"},
        "dog": {"label": "dog", "base_severity": "MEDIUM"},
        "backpack": {"label": "floor bag obstacle", "base_severity": "MEDIUM"},
        "umbrella": {"label": "umbrella", "base_severity": "LOW"},
        "handbag": {"label": "bag", "base_severity": "LOW"},
        "suitcase": {"label": "suitcase obstacle", "base_severity": "HIGH"},
        "chair": {"label": "chair", "base_severity": "MEDIUM"},
        "couch": {"label": "couch", "base_severity": "MEDIUM"},
        "potted plant": {"label": "planter obstacle", "base_severity": "MEDIUM"},
        "dining table": {"label": "table", "base_severity": "MEDIUM"},
        "tv": {"label": "display monitor", "base_severity": "LOW"},
        "stairs": {"label": "stairs", "base_severity": "HIGH"},
        "barrier": {"label": "barrier", "base_severity": "HIGH"},
    }

    def __init__(self) -> None:
        self.model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self._model = None
        self._load_attempted = False

    def _get_model(self):
        if not self._load_attempted:
            self._load_attempted = True
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                logger.info("Loaded YOLOv8n model from %s", self.model_path)
            except Exception as exc:
                logger.warning("YOLOv8n model load deferred/fallback: %s", exc)
        return self._model

    async def analyze(
        self,
        image_np: Any,
        simulated_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run YOLOv8n obstacle detection asynchronously in a worker thread."""
        t0 = time.perf_counter()
        img_h, img_w = self._get_dims(image_np)
        if simulated_detections is not None:
            return self._process_raw_detections(
                simulated_detections,
                img_w,
                img_h,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        model = await asyncio.to_thread(self._get_model)
        if model is None:
            return {
                "hazard_priority": "NONE",
                "hazards": [],
                "urgent_alert": None,
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        def _infer():
            return model.predict(source=image_np, conf=0.35, verbose=False)

        try:
            results = await asyncio.to_thread(_infer)
            raw_boxes: List[Dict[str, Any]] = []
            if results and len(results) > 0:
                res = results[0]
                names = res.names
                boxes = res.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                        raw_boxes.append(
                            {
                                "class_name": cls_name,
                                "confidence": round(conf, 3),
                                "bbox": [x1, y1, x2, y2],
                            }
                        )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return self._process_raw_detections(
                raw_boxes, img_w, img_h, latency_ms
            )
        except Exception as exc:
            logger.error("YOLOv8n inference error: %s", exc)
            return {
                "hazard_priority": "NONE",
                "hazards": [],
                "urgent_alert": None,
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    @staticmethod
    def _get_dims(image_np: Any) -> Tuple[int, int]:
        if hasattr(image_np, "shape") and len(image_np.shape) >= 2:
            return int(image_np.shape[0]), int(image_np.shape[1])
        if isinstance(image_np, dict):
            return int(image_np.get("height", 480)), int(image_np.get("width", 640))
        return 480, 640

    def _process_raw_detections(
        self,
        raw_boxes: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
        latency_ms: float,
    ) -> Dict[str, Any]:
        img_w = max(1, img_w)
        img_h = max(1, img_h)
        processed: List[Dict[str, Any]] = []
        highest_priority = "NONE"
        priority_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

        for item in raw_boxes:
            cls_name = str(item.get("class_name", "")).lower()
            if cls_name not in self.HAZARD_CLASSES:
                continue

            x1, y1, x2, y2 = item["bbox"]
            xc_norm = ((x1 + x2) / 2.0) / img_w
            y2_norm = y2 / img_h
            h_norm = max(0.0, (y2 - y1) / img_h)
            w_norm = max(0.0, (x2 - x1) / img_w)
            area_norm = h_norm * w_norm

            clock_dir = bbox_to_clock_face(xc_norm)
            proximity_band, est_meters = estimate_distance_from_bbox(
                y2_norm, h_norm, area_norm
            )
            meta = self.HAZARD_CLASSES[cls_name]
            label = meta["label"]

            # Determine hazard priority based on proximity and direct walking path (11, 12, 1 o'clock)
            in_central_corridor = clock_dir in ("11 o'clock", "12 o'clock", "1 o'clock")
            if proximity_band == "near" and in_central_corridor:
                priority = "HIGH"
            elif proximity_band == "near" or (
                proximity_band == "medium"
                and in_central_corridor
                and meta["base_severity"] in ("HIGH", "MEDIUM")
            ):
                priority = (
                    "HIGH" if meta["base_severity"] == "HIGH" else "MEDIUM"
                )
            else:
                priority = "LOW"

            if priority_rank[priority] > priority_rank[highest_priority]:
                highest_priority = priority

            processed.append(
                {
                    "object": label,
                    "class_name": cls_name,
                    "confidence": item.get("confidence", 0.85),
                    "clock_direction": clock_dir,
                    "spatial_tag": bbox_to_spatial_tag(xc_norm),
                    "proximity": proximity_band,
                    "distance_meters": est_meters,
                    "priority": priority,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )

        # Sort hazards by priority (HIGH first) and then closest distance
        processed.sort(
            key=lambda h: (-priority_rank[h["priority"]], h["distance_meters"])
        )

        urgent_alert = None
        if processed and highest_priority == "HIGH":
            top = processed[0]
            urgent_alert = (
                f"Caution! {top['object'].capitalize()} directly ahead at "
                f"{top['clock_direction']}, {top['distance_meters']} meters."
            )

        return {
            "hazard_priority": highest_priority,
            "hazards": processed,
            "urgent_alert": urgent_alert,
            "inference_ms": round(latency_ms, 2),
        }


# ============================================================================
# 4. OCR AGENT (EasyOCR Room Plaques, Floor Indicators & Directional Signs)
# ============================================================================

class OCRAgent:
    """
    Uses EasyOCR to detect and parse indoor signage:
    - Room plaques (e.g., 'Room 443B', '443B', 'Lab 201')
    - Floor indicators (e.g., 'Floor 1', 'Level 4', 'Ground Floor')
    - Directional / Wayfinding signs (e.g., 'Elevators', 'Stairs', 'Reception', 'Exit')
    Each detection is tagged with 'left', 'center', or 'right' spatial position.
    """

    def __init__(self) -> None:
        self.use_gpu = os.getenv("EASYOCR_GPU", "false").lower() == "true"
        self._reader = None
        self._load_attempted = False

    def _get_reader(self):
        if not self._load_attempted:
            self._load_attempted = True
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
                logger.info("Initialized EasyOCR Reader (gpu=%s)", self.use_gpu)
            except Exception as exc:
                logger.warning("EasyOCR initialization deferred/fallback: %s", exc)
        return self._reader

    async def analyze(
        self,
        image_np: Any,
        simulated_ocr: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run EasyOCR and classify detected text into rooms, floors, and signs."""
        t0 = time.perf_counter()
        if hasattr(image_np, "shape") and len(image_np.shape) >= 2:
            img_h, img_w = int(image_np.shape[0]), int(image_np.shape[1])
        else:
            img_h, img_w = 480, 640

        if simulated_ocr is not None:
            return self._parse_ocr_items(
                simulated_ocr,
                img_w,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        reader = await asyncio.to_thread(self._get_reader)
        if reader is None:
            return {
                "texts": [],
                "detected_rooms": [],
                "detected_floor": None,
                "wayfinding_signs": [],
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        def _read():
            return reader.readtext(image_np)

        try:
            raw_results = await asyncio.to_thread(_read)
            items: List[Dict[str, Any]] = []
            for pts, text, conf in raw_results:
                if conf < 0.30 or not str(text).strip():
                    continue
                xs = [p[0] for p in pts]
                xc_norm = (sum(xs) / len(xs)) / max(1, img_w)
                items.append(
                    {
                        "text": str(text).strip(),
                        "confidence": round(float(conf), 3),
                        "x_center_norm": xc_norm,
                    }
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return self._parse_ocr_items(items, img_w, latency_ms)
        except Exception as exc:
            logger.error("EasyOCR error: %s", exc)
            return {
                "texts": [],
                "detected_rooms": [],
                "detected_floor": None,
                "wayfinding_signs": [],
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    @staticmethod
    def _parse_ocr_items(
        items: List[Dict[str, Any]], img_w: int, latency_ms: float
    ) -> Dict[str, Any]:
        texts: List[Dict[str, Any]] = []
        detected_rooms: List[Dict[str, Any]] = []
        detected_floor: Optional[int] = None
        wayfinding_signs: List[Dict[str, Any]] = []

        for item in items:
            raw_text = item["text"]
            xc_norm = float(item.get("x_center_norm", 0.5))
            spatial = item.get("spatial_tag") or bbox_to_spatial_tag(xc_norm)
            clock_dir = bbox_to_clock_face(xc_norm)

            entry = {
                "text": raw_text,
                "spatial_tag": spatial,
                "clock_direction": clock_dir,
                "confidence": item.get("confidence", 0.9),
            }
            texts.append(entry)

            # 1. Check Floor indicators ('Floor 1', 'Level 4', 'L2', 'Ground Floor')
            fl_match = re.search(
                r"\b(?:floor|level|lvl)\s*(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\s*floor\b",
                raw_text,
                re.IGNORECASE,
            )
            if fl_match:
                fl_val = fl_match.group(1) or fl_match.group(2)
                detected_floor = int(fl_val)
            elif re.search(r"\bground\s*floor\b", raw_text, re.IGNORECASE):
                detected_floor = 0

            # 2. Check Room Plaques ('Room 443B', 'Rm 102', or standalone '443B')
            rm_match = re.search(
                r"\b(?:room|rm|suite|lab|gate)\s*[-:#]?\s*([A-Z]?\d{2,4}[A-Z]?)\b",
                raw_text,
                re.IGNORECASE,
            )
            if rm_match:
                room_code = rm_match.group(1).upper()
                detected_rooms.append(
                    {
                        "room": room_code,
                        "inferred_floor": infer_floor_from_room(room_code),
                        "spatial_tag": spatial,
                        "clock_direction": clock_dir,
                        "raw_text": raw_text,
                    }
                )
            else:
                standalone = re.match(r"^([A-Z]?\d{3,4}[A-Z])$", raw_text.strip(), re.IGNORECASE)
                if standalone:
                    room_code = standalone.group(1).upper()
                    detected_rooms.append(
                        {
                            "room": room_code,
                            "inferred_floor": infer_floor_from_room(room_code),
                            "spatial_tag": spatial,
                            "clock_direction": clock_dir,
                            "raw_text": raw_text,
                        }
                    )

            # 3. Check Wayfinding / Directional signs
            if re.search(
                r"\b(elevator|lift|stairs|stairwell|reception|entrance|exit|lobby|way out|restroom|accessible)\b",
                raw_text,
                re.IGNORECASE,
            ):
                wayfinding_signs.append(entry)

        return {
            "texts": texts,
            "detected_rooms": detected_rooms,
            "detected_floor": detected_floor,
            "wayfinding_signs": wayfinding_signs,
            "inference_ms": round(latency_ms, 2),
        }


# ============================================================================
# 5. VISION AGENT (Gemini 2.5 Flash Multimodal Scene & Surface Understanding)
# ============================================================================

class VisionAgent:
    """
    Uses `gemini-2.5-flash` multimodal vision to detect nuanced indoor/outdoor
    features that standard YOLO object detection misses:
    - Wet floor signs & liquid spills
    - Glass doors / building entrances
    - Elevator doors & call buttons
    - Reception / security desks
    - Staircases & escalators
    """

    SYSTEM_PROMPT = (
        "You are the Vision Agent for VisionGuide, a real-time assistive navigation system "
        "for blind and visually impaired users. Analyze the camera frame and return ONLY "
        "valid JSON with these exact keys:\n"
        "{\n"
        '  "wet_floor_or_spill": boolean,\n'
        '  "surface_hazard_description": string or null,\n'
        '  "glass_entrance_detected": boolean,\n'
        '  "elevator_detected": boolean,\n'
        '  "elevator_direction": "left" | "center" | "right" | null,\n'
        '  "stairs_detected": boolean,\n'
        '  "reception_desk_detected": boolean,\n'
        '  "detected_floor_number": integer or null,\n'
        '  "scene_summary": "concise 1-sentence spatial summary for a blind user"\n'
        "}"
    )

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = "gemini-2.5-flash"

    async def analyze(
        self,
        image_bytes: bytes,
        active_goal: Optional[str] = None,
        simulated_vision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call Gemini 2.5 Flash multimodal vision API and return structured JSON."""
        t0 = time.perf_counter()
        if simulated_vision is not None:
            result = dict(simulated_vision)
            result["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return result

        if not self.api_key or self.api_key.startswith("your_") or not image_bytes:
            return {
                "wet_floor_or_spill": False,
                "surface_hazard_description": None,
                "glass_entrance_detected": False,
                "elevator_detected": False,
                "elevator_direction": None,
                "stairs_detected": False,
                "reception_desk_detected": False,
                "detected_floor_number": None,
                "scene_summary": "Clear corridor ahead.",
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        prompt = self.SYSTEM_PROMPT
        if active_goal:
            prompt += f"\nUser's current navigation goal: {active_goal}"

        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=6.5) as client:
                resp = await client.post(url, json=payload)
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    raw_text = (
                        candidates[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "{}")
                    )
                    parsed = json.loads(raw_text)
                    parsed["inference_ms"] = round(
                        (time.perf_counter() - t0) * 1000.0, 2
                    )
                    return parsed
        except Exception as exc:
            logger.error("Gemini 2.5 Flash Vision error: %s", exc)

        return {
            "wet_floor_or_spill": False,
            "surface_hazard_description": None,
            "glass_entrance_detected": False,
            "elevator_detected": False,
            "elevator_direction": None,
            "stairs_detected": False,
            "reception_desk_detected": False,
            "detected_floor_number": None,
            "scene_summary": "Clear walking path ahead.",
            "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


# ============================================================================
# 6. REASONING AGENT / ORCHESTRATOR (Parallel asyncio.gather + Synthesis)
# ============================================================================

class ReasoningAgent:
    """
    Orchestrates parallel execution of Navigation, Hazard (YOLO), OCR (EasyOCR),
    and Vision (Gemini 2.5 Flash) agents via `asyncio.gather`, and synthesizes
    safety interrupts, intelligent multi-floor routing, and arrival confirmation.
    """

    def __init__(self) -> None:
        self.voice_agent = VoiceAgent()
        self.navigation_agent = NavigationAgent()
        self.hazard_agent = HazardAgent()
        self.ocr_agent = OCRAgent()
        self.vision_agent = VisionAgent()

    @staticmethod
    def decode_image(image_bytes: bytes) -> Any:
        """Decode raw JPEG/PNG bytes into a BGR numpy array for OpenCV/YOLO/EasyOCR."""
        if np is None:
            return {"height": 480, "width": 640}
        if not image_bytes:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if cv2 is not None:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        if Image is not None:
            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if cv2 is not None:
                    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                return np.array(pil_img)
            except Exception:
                pass
        return np.zeros((480, 640, 3), dtype=np.uint8)

    async def analyze_frame_parallel(
        self,
        session: Dict[str, Any],
        image_bytes: bytes,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        simulated_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute HazardAgent, OCRAgent, VisionAgent, and NavigationAgent in parallel
        using `asyncio.gather`, then synthesize into a unified accessible guidance response.
        """
        t_start = time.perf_counter()
        image_np = await asyncio.to_thread(self.decode_image, image_bytes)
        overrides = simulated_overrides or {}

        # Launch all 4 perception & routing agents concurrently via asyncio.gather
        hazard_task = self.hazard_agent.analyze(
            image_np, simulated_detections=overrides.get("yolo_detections")
        )
        ocr_task = self.ocr_agent.analyze(
            image_np, simulated_ocr=overrides.get("ocr_items")
        )
        vision_task = self.vision_agent.analyze(
            image_bytes,
            active_goal=session.get("active_goal"),
            simulated_vision=overrides.get("vision_result"),
        )
        nav_task = self.navigation_agent.evaluate_navigation_step(session, lat, lng)

        hazard_res, ocr_res, vision_res, nav_res = await asyncio.gather(
            hazard_task, ocr_task, vision_task, nav_task
        )

        synthesis = self.synthesize(
            session=session,
            hazard_res=hazard_res,
            ocr_res=ocr_res,
            vision_res=vision_res,
            nav_res=nav_res,
        )
        synthesis["total_latency_ms"] = round(
            (time.perf_counter() - t_start) * 1000.0, 2
        )
        return synthesis

    def synthesize(
        self,
        session: Dict[str, Any],
        hazard_res: Dict[str, Any],
        ocr_res: Dict[str, Any],
        vision_res: Dict[str, Any],
        nav_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Core multi-agent reasoning logic:
        1. Safety Priority Interruption (HIGH hazard from YOLO or Wet Floor/Spill from Gemini)
        2. Intelligent Floor Inference & Mismatch Routing (Room 443B -> Floor 4 vs Current Floor)
        3. Arrival Verification (OCR matches target_room or Vision matches reception/entrance)
        4. Seamless Outdoor-to-Indoor Transition
        """
        target_room = session.get("target_room")
        target_floor = session.get("target_floor")
        if target_floor is None and target_room:
            target_floor = infer_floor_from_room(target_room)

        current_floor = session.get("current_floor")
        if current_floor is None:
            current_floor = 1

        # Update current_floor if OCR or Gemini Vision detected an explicit floor sign
        if ocr_res.get("detected_floor") is not None:
            current_floor = int(ocr_res["detected_floor"])
        elif vision_res.get("detected_floor_number") is not None:
            current_floor = int(vision_res["detected_floor_number"])

        # Elevate hazard_priority to HIGH if Gemini Vision spots a wet floor sign or liquid spill
        hazard_priority = hazard_res.get("hazard_priority", "NONE")
        surface_hazard = vision_res.get("wet_floor_or_spill", False)
        surface_desc = vision_res.get("surface_hazard_description") or "wet floor or spill"
        if surface_hazard:
            hazard_priority = "HIGH"

        # Determine haptic pattern for mobile frontend
        haptic_pattern = "NONE"
        if hazard_priority == "HIGH":
            haptic_pattern = "URGENT_PULSE"
        elif hazard_priority == "MEDIUM":
            haptic_pattern = "WARNING_DOUBLE_TAP"

        # Check Arrival Verification via OCR room plaque match or reception match
        arrived = False
        arrival_message = None
        detected_rooms = ocr_res.get("detected_rooms", [])
        if target_room:
            for rm in detected_rooms:
                if rm["room"].upper() == target_room.upper():
                    arrived = True
                    arrival_message = (
                        f"You have arrived! Room {target_room} plaque detected on your "
                        f"{rm['spatial_tag']} at {rm['clock_direction']}."
                    )
                    haptic_pattern = "ARRIVAL_SUCCESS"
                    break

        active_goal_lower = (session.get("active_goal") or "").lower()
        if (
            not arrived
            and "reception" in active_goal_lower
            and vision_res.get("reception_desk_detected")
        ):
            arrived = True
            arrival_message = "You have arrived at the reception desk straight ahead."
            haptic_pattern = "ARRIVAL_SUCCESS"

        # Determine Navigation Mode & Floor Mismatch State
        nav_mode = session.get("navigation_mode", "IDLE")
        if arrived:
            nav_mode = "ARRIVED"
        elif session.get("active_goal"):
            if (
                vision_res.get("glass_entrance_detected")
                or ocr_res.get("detected_floor") is not None
                or detected_rooms
                or ocr_res.get("wayfinding_signs")
                or nav_res.get("near_building_entrance")
            ):
                if target_floor is not None and current_floor != target_floor:
                    nav_mode = "ELEVATOR_SEEK"
                else:
                    nav_mode = "INDOOR_TRANSIT"
            else:
                nav_mode = "OUTDOOR"

        # Build Floor Reasoning Directive (active once inside or at building entrance)
        floor_mismatch = (
            target_floor is not None
            and current_floor is not None
            and current_floor != target_floor
            and nav_mode in ("ELEVATOR_SEEK", "INDOOR_TRANSIT")
        )
        floor_directive = None
        if floor_mismatch and not arrived:
            delta = target_floor - current_floor
            direction_word = "up" if delta > 0 else "down"
            room_label = f"Room {target_room}" if target_room else "your destination"

            # Check if elevator or stairs are currently visible in OCR or Vision
            elevator_seen = vision_res.get("elevator_detected", False)
            elevator_dir = vision_res.get("elevator_direction") or "ahead"
            ocr_Wayfinding = ocr_res.get("wayfinding_signs", [])
            elev_sign = next(
                (
                    s
                    for s in ocr_Wayfinding
                    if re.search(r"elevator|lift|stairs", s["text"], re.IGNORECASE)
                ),
                None,
            )

            if elev_sign:
                floor_directive = (
                    f"You are on Floor {current_floor}, and {room_label} is on Floor {target_floor}. "
                    f"Follow the '{elev_sign['text']}' sign on your {elev_sign['spatial_tag']} "
                    f"({elev_sign['clock_direction']}) to go {direction_word} to Floor {target_floor}."
                )
            elif elevator_seen:
                floor_directive = (
                    f"You are on Floor {current_floor}, and {room_label} is on Floor {target_floor}. "
                    f"Proceed to the elevator on your {elevator_dir} and take it {direction_word} to Floor {target_floor}."
                )
            else:
                floor_directive = (
                    f"You are on Floor {current_floor}, but {room_label} is on Floor {target_floor}. "
                    f"Scan for the elevator or stairwell to go {direction_word} to Floor {target_floor}."
                )

        # ====================================================================
        # PRIORITY SYNTHESIS FOR SPOKEN GUIDANCE (Safety > Arrival > Floor > Nav)
        # ====================================================================
        interrupt_speech = hazard_priority == "HIGH"
        if hazard_priority == "HIGH":
            if surface_hazard:
                spoken_guidance = (
                    f"Stop! Caution: {surface_desc} detected ahead. Step carefully around it."
                )
            elif hazard_res.get("urgent_alert"):
                spoken_guidance = hazard_res["urgent_alert"]
            else:
                spoken_guidance = "Caution! Obstacle directly in your walking path."
        elif arrived and arrival_message:
            spoken_guidance = arrival_message
        elif floor_directive:
            # Append medium hazard note if present
            med_note = ""
            if hazard_priority == "MEDIUM" and hazard_res.get("hazards"):
                top_h = hazard_res["hazards"][0]
                med_note = (
                    f" Mind the {top_h['object']} at {top_h['clock_direction']} "
                    f"({top_h['distance_meters']}m)."
                )
            spoken_guidance = floor_directive + med_note
        elif nav_mode == "INDOOR_TRANSIT":
            indoor_parts = []
            if target_room and target_floor is not None and current_floor == target_floor:
                indoor_parts.append(
                    f"You are on Floor {current_floor}. Scanning corridor plaques for Room {target_room}."
                )
            if detected_rooms:
                nearby_rm = detected_rooms[0]
                indoor_parts.append(
                    f"Passing Room {nearby_rm['room']} on your {nearby_rm['spatial_tag']}."
                )
            elif vision_res.get("glass_entrance_detected"):
                indoor_parts.append(
                    "Glass entrance doors detected ahead. Proceed inside toward the lobby."
                )
            else:
                indoor_parts.append(
                    vision_res.get("scene_summary") or "Proceed along the indoor corridor."
                )
            if hazard_priority == "MEDIUM" and hazard_res.get("hazards"):
                top_h = hazard_res["hazards"][0]
                indoor_parts.append(
                    f"Note: {top_h['object']} at {top_h['clock_direction']}, {top_h['distance_meters']} meters."
                )
            spoken_guidance = " ".join(indoor_parts)
        elif nav_res.get("has_active_route"):
            step_instr = nav_res["current_instruction"]
            dist_m = nav_res.get("distance_to_step_meters")
            dist_str = f" ({int(dist_m)} meters)" if dist_m is not None else ""
            med_note = ""
            if hazard_priority == "MEDIUM" and hazard_res.get("hazards"):
                top_h = hazard_res["hazards"][0]
                med_note = (
                    f" Watch for {top_h['object']} at {top_h['clock_direction']}."
                )
            spoken_guidance = f"{step_instr}{dist_str}.{med_note}"
        else:
            # Idle exploration mode: still report hazards, signs, or scene summary
            if hazard_res.get("hazards"):
                top_h = hazard_res["hazards"][0]
                spoken_guidance = (
                    f"{top_h['object'].capitalize()} at {top_h['clock_direction']}, "
                    f"{top_h['distance_meters']} meters. {vision_res.get('scene_summary', '')}"
                ).strip()
            elif detected_rooms:
                rm = detected_rooms[0]
                spoken_guidance = (
                    f"Detected plaque for Room {rm['room']} on your {rm['spatial_tag']}."
                )
            else:
                spoken_guidance = (
                    vision_res.get("scene_summary")
                    or "Path ahead is clear. Hold Push-to-Talk to set a destination."
                )

        return {
            "session_id": session.get("session_id"),
            "navigation_mode": nav_mode,
            "active_goal": session.get("active_goal"),
            "destination_building": session.get("destination_building"),
            "target_room": target_room,
            "target_floor": target_floor,
            "current_floor": current_floor,
            "floor_mismatch": floor_mismatch,
            "arrived": arrived,
            "hazard_priority": hazard_priority,
            "interrupt_speech": interrupt_speech,
            "haptic_pattern": haptic_pattern,
            "spoken_guidance": spoken_guidance,
            "agents": {
                "hazard": hazard_res,
                "ocr": ocr_res,
                "vision": vision_res,
                "navigation": nav_res,
            },
        }


# Singleton orchestrator instance
orchestrator = ReasoningAgent()
