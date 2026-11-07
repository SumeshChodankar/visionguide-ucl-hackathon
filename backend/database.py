"""
VisionGuide Database Layer (backend/database.py)
Provides SQLite persistence for user navigation sessions, active goals,
detected floor state, route steps, and real-time multi-agent telemetry logs.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.getenv("VISIONGUIDE_DB_PATH", "visionguide.db")


class VisionGuideDB:
    """Thread-safe SQLite manager for VisionGuide sessions and telemetry."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    active_goal TEXT,
                    destination_building TEXT,
                    target_room TEXT,
                    target_floor INTEGER,
                    current_floor INTEGER DEFAULT 1,
                    navigation_mode TEXT DEFAULT 'IDLE',
                    last_lat REAL,
                    last_lng REAL,
                    route_steps_json TEXT DEFAULT '[]',
                    current_step_index INTEGER DEFAULT 0,
                    arrived INTEGER DEFAULT 0,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    hazard_priority TEXT,
                    spoken_guidance TEXT,
                    payload_json TEXT,
                    latency_ms REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telemetry_session
                ON telemetry_logs(session_id, created_at DESC)
                """
            )

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieve an existing session or initialize a default session state."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                return self._deserialize_session(dict(row))

            now = datetime.now(timezone.utc).isoformat()
            default_session = {
                "session_id": session_id,
                "active_goal": None,
                "destination_building": None,
                "target_room": None,
                "target_floor": None,
                "current_floor": 1,
                "navigation_mode": "IDLE",  # IDLE | OUTDOOR | INDOOR_TRANSIT | ELEVATOR_SEEK | FLOOR_SEEK | ARRIVED
                "last_lat": None,
                "last_lng": None,
                "route_steps_json": "[]",
                "current_step_index": 0,
                "arrived": 0,
                "updated_at": now,
            }
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, active_goal, destination_building, target_room,
                    target_floor, current_floor, navigation_mode, last_lat, last_lng,
                    route_steps_json, current_step_index, arrived, updated_at
                ) VALUES (
                    :session_id, :active_goal, :destination_building, :target_room,
                    :target_floor, :current_floor, :navigation_mode, :last_lat, :last_lng,
                    :route_steps_json, :current_step_index, :arrived, :updated_at
                )
                """,
                default_session,
            )
            return self._deserialize_session(default_session)

    def update_session(self, session_id: str, **updates: Any) -> Dict[str, Any]:
        """Update specific fields on a session record and return the updated state."""
        current = self.get_or_create_session(session_id)
        allowed_fields = {
            "active_goal",
            "destination_building",
            "target_room",
            "target_floor",
            "current_floor",
            "navigation_mode",
            "last_lat",
            "last_lng",
            "route_steps",
            "current_step_index",
            "arrived",
        }

        db_updates: Dict[str, Any] = {}
        for key, value in updates.items():
            if key not in allowed_fields:
                continue
            if key == "route_steps":
                db_updates["route_steps_json"] = json.dumps(value)
            elif key == "arrived":
                db_updates["arrived"] = 1 if value else 0
            else:
                db_updates[key] = value

        db_updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{col} = :{col}" for col in db_updates.keys())
        db_updates["session_id"] = session_id

        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE session_id = :session_id",
                db_updates,
            )

        return self.get_or_create_session(session_id)

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        """Reset a user session back to IDLE while preserving the session record."""
        return self.update_session(
            session_id,
            active_goal=None,
            destination_building=None,
            target_room=None,
            target_floor=None,
            current_floor=1,
            navigation_mode="IDLE",
            route_steps=[],
            current_step_index=0,
            arrived=False,
        )

    def log_telemetry(
        self,
        session_id: str,
        event_type: str,
        hazard_priority: str,
        spoken_guidance: str,
        payload: Dict[str, Any],
        latency_ms: float = 0.0,
    ) -> int:
        """Append a structured telemetry entry for auditing and debugging."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO telemetry_logs (
                    session_id, event_type, hazard_priority, spoken_guidance,
                    payload_json, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event_type,
                    hazard_priority,
                    spoken_guidance,
                    json.dumps(payload, default=str),
                    round(latency_ms, 2),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_recent_telemetry(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Fetch recent telemetry logs for a given session."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM telemetry_logs
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["payload"] = json.loads(item.pop("payload_json", "{}"))
                except Exception:
                    item["payload"] = {}
                results.append(item)
            return results

    @staticmethod
    def _deserialize_session(raw: Dict[str, Any]) -> Dict[str, Any]:
        route_steps_raw = raw.pop("route_steps_json", "[]")
        try:
            route_steps = json.loads(route_steps_raw) if route_steps_raw else []
        except Exception:
            route_steps = []
        raw["route_steps"] = route_steps
        raw["arrived"] = bool(raw.get("arrived", 0))
        return raw


# Global singleton instance
db = VisionGuideDB()

