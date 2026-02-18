"""
tracker.py - Multi-camera visitor movement tracker.

Records every detection sighting and derives entry/exit events.
All state lives in MongoDB (events collection) so it survives restarts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from db import events_col

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────────────────

class MultiCameraTracker:
    """
    Records identity-tracking events and queries movement history.

    This class is intentionally thin – it writes events to MongoDB and
    lets callers query back the history they need. The BehaviorAnalyzer
    (behavior_analyzer.py) handles alert logic on top of these events.
    """

    # ----------------------------------------------------------
    # Write path
    # ----------------------------------------------------------

    def record(
        self,
        visitor_id: str,
        name: str,
        camera_id: str,
        location: str,
        zone_type: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> Dict:
        """
        Persist an identity-tracking event.

        Returns the document that was inserted (without _id).
        """
        ts = timestamp or datetime.now(timezone.utc)

        doc = {
            "visitor_id": visitor_id,
            "name": name,
            "camera_id": camera_id,
            "location": location,
            "zone_type": zone_type,
            "event_type": "identity_tracking",
            "confidence": round(confidence, 4),
            "timestamp": ts,
        }

        try:
            events_col().insert_one(doc)
        except Exception as exc:
            logger.warning("Tracker insert failed: %s", exc)

        doc.pop("_id", None)
        doc["timestamp"] = ts.isoformat()
        return doc

    # ----------------------------------------------------------
    # Read path
    # ----------------------------------------------------------

    def get_visitor_history(
        self,
        visitor_id: str,
        limit: int = 100,
    ) -> List[Dict]:
        """Return recent tracking events for a specific visitor."""
        projection = {"_id": 0}
        cursor = (
            events_col()
            .find({"visitor_id": visitor_id}, projection)
            .sort("timestamp", -1)
            .limit(limit)
        )
        return self._serialize(cursor)

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict]:
        """Return the most recent events, optionally filtered by type."""
        query: Dict = {}
        if event_type:
            query["event_type"] = event_type

        projection = {"_id": 0}
        cursor = (
            events_col()
            .find(query, projection)
            .sort("timestamp", -1)
            .limit(limit)
        )
        return self._serialize(cursor)

    def get_camera_activity(self, camera_id: str, limit: int = 50) -> List[Dict]:
        """Return recent detections for a specific camera."""
        projection = {"_id": 0}
        cursor = (
            events_col()
            .find({"camera_id": camera_id}, projection)
            .sort("timestamp", -1)
            .limit(limit)
        )
        return self._serialize(cursor)

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    @staticmethod
    def _serialize(cursor) -> List[Dict]:
        """Convert MongoDB cursor to a JSON-serialisable list."""
        results = []
        for doc in cursor:
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            results.append(doc)
        return results
