"""
geofence_monitor.py - Geofence violation detection and tracking.

Monitors visitor positions relative to predefined boundaries and triggers
alerts when visitors spend too long outside authorized zones.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from db import events_col

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class BoundaryConfig:
    """Polygon boundary for a camera."""
    camera_id: str
    points: List[Tuple[int, int]]  # [(x1,y1), (x2,y2), ...]

    def is_inside(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the polygon boundary."""
        if len(self.points) < 3:
            return True  # No boundary = all points valid

        # Convert to numpy array for cv2
        contour = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(contour, point, False)
        return result >= 0  # 1=inside, 0=on_edge, -1=outside


@dataclass
class ViolationState:
    """Track when a visitor started violating geofence."""
    visitor_id: str
    name: str
    started_at: datetime
    last_seen_at: datetime
    position: Tuple[int, int]


# ──────────────────────────────────────────────────────────
# Geofence Monitor
# ──────────────────────────────────────────────────────────

class GeofenceMonitor:
    """
    Monitor visitor positions against geofence boundaries.

    Triggers alerts when visitors remain outside boundaries for too long.
    """

    def __init__(
        self,
        violation_threshold_seconds: int = 60,
        duplicate_suppression_seconds: int = 30,
    ):
        self.violation_threshold = timedelta(seconds=violation_threshold_seconds)
        self.duplicate_window = timedelta(seconds=duplicate_suppression_seconds)

        # camera_id → BoundaryConfig
        self._boundaries: Dict[str, BoundaryConfig] = {}

        # (camera_id, visitor_id) → ViolationState
        self._violations: Dict[Tuple[str, str], ViolationState] = {}

        # (camera_id, visitor_id) → last alert time
        self._last_alert: Dict[Tuple[str, str], datetime] = {}

    # ----------------------------------------------------------
    # Boundary management
    # ----------------------------------------------------------

    def load_boundary(self, camera_id: str, filepath: Optional[str] = None) -> bool:
        """
        Load boundary from JSON file.

        If filepath is None, tries to load from:
        boundaries/{camera_id}_boundary.json
        """
        if filepath is None:
            filepath = f"boundaries/{camera_id}_boundary.json"

        path = Path(filepath)
        if not path.exists():
            logger.warning("Boundary file not found: %s", filepath)
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            points = [tuple(p) for p in data["points"]]
            self._boundaries[camera_id] = BoundaryConfig(
                camera_id=camera_id,
                points=points,
            )
            logger.info(
                "Loaded boundary for camera '%s': %d points",
                camera_id,
                len(points),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load boundary from %s: %s", filepath, exc)
            return False

    def set_boundary(self, camera_id: str, points: List[Tuple[int, int]]):
        """Manually set boundary points (alternative to loading from file)."""
        self._boundaries[camera_id] = BoundaryConfig(
            camera_id=camera_id,
            points=points,
        )
        logger.info("Boundary set for camera '%s': %d points", camera_id, len(points))

    def get_boundary(self, camera_id: str) -> Optional[BoundaryConfig]:
        """Get boundary config for a camera."""
        return self._boundaries.get(camera_id)

    # ----------------------------------------------------------
    # Monitoring
    # ----------------------------------------------------------

    def check_position(
        self,
        visitor_id: str,
        name: str,
        camera_id: str,
        position: Tuple[int, int],
        location: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """
        Check if visitor is violating geofence and return alert if needed.

        Parameters
        ----------
        visitor_id : str
            Unique visitor identifier.
        name : str
            Visitor display name.
        camera_id : str
            Camera identifier.
        position : Tuple[int, int]
            (x, y) coordinates of visitor in frame.
        location : str
            Human-readable location name.
        confidence : float
            Recognition confidence.
        timestamp : datetime, optional
            Defaults to now (UTC).

        Returns
        -------
        Optional[Dict]
            Alert document if violation threshold exceeded, else None.
        """
        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Get boundary for this camera
        boundary = self._boundaries.get(camera_id)
        if boundary is None:
            # No boundary defined = no geofence monitoring
            return None

        key = (camera_id, visitor_id)
        is_inside = boundary.is_inside(position)

        if is_inside:
            # Visitor is inside boundary → clear violation state
            if key in self._violations:
                logger.debug(
                    "Visitor %s returned inside boundary (camera=%s)",
                    visitor_id,
                    camera_id,
                )
                del self._violations[key]
            return None

        # Visitor is OUTSIDE boundary
        if key not in self._violations:
            # Start tracking violation
            self._violations[key] = ViolationState(
                visitor_id=visitor_id,
                name=name,
                started_at=now,
                last_seen_at=now,
                position=position,
            )
            logger.debug(
                "Geofence violation started: visitor=%s camera=%s",
                visitor_id,
                camera_id,
            )
        else:
            # Update existing violation
            state = self._violations[key]
            state.last_seen_at = now
            state.position = position

        # Check if violation duration exceeds threshold
        state = self._violations[key]
        duration = now - state.started_at

        if duration >= self.violation_threshold:
            # Check if we should emit alert (avoid duplicates)
            if self._should_emit_alert(camera_id, visitor_id, now):
                return self._create_alert(
                    visitor_id=visitor_id,
                    name=name,
                    camera_id=camera_id,
                    location=location,
                    position=position,
                    confidence=confidence,
                    duration_seconds=int(duration.total_seconds()),
                    timestamp=now,
                )

        return None

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Fetch recent geofence violation alerts from MongoDB."""
        projection = {"_id": 0}
        cursor = (
            events_col()
            .find({"event_type": "geofence_violation"}, projection)
            .sort("timestamp", -1)
            .limit(limit)
        )
        results = []
        for doc in cursor:
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            results.append(doc)
        return results

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _should_emit_alert(
        self,
        camera_id: str,
        visitor_id: str,
        now: datetime,
    ) -> bool:
        """Check if we should emit alert (duplicate suppression)."""
        key = (camera_id, visitor_id)
        last = self._last_alert.get(key)
        if last and (now - last) < self.duplicate_window:
            return False
        self._last_alert[key] = now
        return True

    def _create_alert(
        self,
        visitor_id: str,
        name: str,
        camera_id: str,
        location: str,
        position: Tuple[int, int],
        confidence: float,
        duration_seconds: int,
        timestamp: datetime,
    ) -> Dict:
        """Create and persist a geofence violation alert."""
        doc = {
            "visitor_id": visitor_id,
            "name": name,
            "camera_id": camera_id,
            "location": location,
            "event_type": "geofence_violation",
            "confidence": round(confidence, 4),
            "timestamp": timestamp,
            "position_x": position[0],
            "position_y": position[1],
            "duration_seconds": duration_seconds,
        }

        try:
            events_col().insert_one(doc)
        except Exception as exc:
            logger.warning("Failed to insert geofence alert: %s", exc)

        doc.pop("_id", None)
        doc["timestamp"] = timestamp.isoformat()
        logger.warning(
            "🚨 GEOFENCE VIOLATION: visitor=%s camera=%s duration=%ds",
            visitor_id,
            camera_id,
            duration_seconds,
        )
        return doc

    # ----------------------------------------------------------
    # Visualization helpers
    # ----------------------------------------------------------

    def draw_boundary(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Draw the boundary polygon on a frame."""
        boundary = self._boundaries.get(camera_id)
        if boundary is None or len(boundary.points) < 3:
            return frame

        # Draw polygon outline
        pts = np.array(boundary.points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

        return frame

    def draw_violations(
        self,
        frame: np.ndarray,
        camera_id: str,
    ) -> np.ndarray:
        """Draw violation indicators for visitors outside boundary."""
        for (cam_id, visitor_id), state in self._violations.items():
            if cam_id != camera_id:
                continue

            x, y = state.position
            duration = (datetime.now(timezone.utc) - state.started_at).total_seconds()

            # Draw red circle at visitor position
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            # Draw duration text
            text = f"{visitor_id}: {int(duration)}s outside"
            cv2.putText(
                frame,
                text,
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        return frame
