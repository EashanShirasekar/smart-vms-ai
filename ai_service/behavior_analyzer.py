"""
behavior_analyzer.py - Stateful alert engine for visitor anomalies.

Detects:
  - Loitering (same person, same camera, exceeded dwell threshold)
  - Restricted zone entry
  - Unknown person presence
  - Re-entry without exit

All generated alert events are persisted to MongoDB.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

from db import events_col

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Defaults (overridable via constructor)
# ──────────────────────────────────────────────────────────

DEFAULT_LOITERING_SECONDS = 60
DEFAULT_DUPLICATE_SUPPRESSION_SECONDS = 30
DEFAULT_UNKNOWN_ALERT_INTERVAL_SECONDS = 45


# ──────────────────────────────────────────────────────────
# Internal state
# ──────────────────────────────────────────────────────────

@dataclass
class _PresenceState:
    first_seen: datetime
    last_seen: datetime
    camera_id: str
    location: str


# ──────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────

class BehaviorAnalyzer:
    """
    Stateful per-process behavior analyzer.

    Call `analyze()` once per detection frame. It returns a (possibly
    empty) list of alert dicts that have already been persisted to DB.
    """

    def __init__(
        self,
        loitering_threshold_seconds: int = DEFAULT_LOITERING_SECONDS,
        duplicate_suppression_seconds: int = DEFAULT_DUPLICATE_SUPPRESSION_SECONDS,
        unknown_alert_interval_seconds: int = DEFAULT_UNKNOWN_ALERT_INTERVAL_SECONDS,
    ) -> None:
        self._loitering_threshold = timedelta(seconds=loitering_threshold_seconds)
        self._dup_window = timedelta(seconds=duplicate_suppression_seconds)
        self._unknown_interval = timedelta(seconds=unknown_alert_interval_seconds)

        # visitor_id → PresenceState  (camera-scoped: key = (visitor_id, camera_id))
        self._presence: Dict[Tuple[str, str], _PresenceState] = {}

        # (visitor_id, camera_id, event_type) → last emitted time
        self._last_emitted: Dict[Tuple[str, str, str], datetime] = {}

        # Track who is "inside" (entry seen, no exit yet)
        self._inside: Set[str] = set()

        # Last unknown alert per camera
        self._last_unknown: Dict[str, datetime] = defaultdict(
            lambda: datetime.min.replace(tzinfo=timezone.utc)
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def analyze(
        self,
        visitor_id: str,
        name: str,
        camera_id: str,
        location: str,
        zone_type: str,
        category: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Evaluate one detection and return any new alert events.

        Parameters
        ----------
        visitor_id : str
            "unknown" for unrecognised faces.
        name : str
            Display name or "Unknown".
        camera_id : str
            Source camera identifier.
        location : str
            Human-readable location label (e.g. "Entrance", "Lab A").
        zone_type : str
            "general" | "restricted"
        category : str
            Enrollment category or "unknown".
        confidence : float
            Recognition confidence 0..1.
        timestamp : datetime, optional
            Defaults to now (UTC).

        Returns
        -------
        List[Dict]
            Alert event documents (already saved to DB).
        """
        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        alerts: List[Dict] = []
        pk = (visitor_id, camera_id)

        # Update presence state
        if pk not in self._presence:
            self._presence[pk] = _PresenceState(now, now, camera_id, location)
        else:
            self._presence[pk].last_seen = now
            self._presence[pk].location = location

        dwell = self._presence[pk].last_seen - self._presence[pk].first_seen

        # 1. Loitering
        if dwell >= self._loitering_threshold:
            if self._should_emit(visitor_id, camera_id, "loitering", now):
                alerts.append(self._emit(
                    visitor_id, name, camera_id, location,
                    "loitering", confidence, now,
                    {"dwell_seconds": int(dwell.total_seconds())}
                ))

        # 2. Restricted zone entry (fire on every new arrival window)
        if zone_type.lower() == "restricted":
            if self._should_emit(visitor_id, camera_id, "restricted_zone_entry", now):
                alerts.append(self._emit(
                    visitor_id, name, camera_id, location,
                    "restricted_zone_entry", confidence, now
                ))

        # 3. Unknown person
        if visitor_id == "unknown":
            if (now - self._last_unknown[camera_id]) >= self._unknown_interval:
                if self._should_emit(visitor_id, camera_id, "unknown_person", now):
                    self._last_unknown[camera_id] = now
                    alerts.append(self._emit(
                        visitor_id, name, camera_id, location,
                        "unknown_person", confidence, now
                    ))

        # 4. Re-entry without exit (only for known visitors)
        if visitor_id != "unknown":
            self._check_entry_exit(visitor_id, name, camera_id, location,
                                   confidence, now, alerts)

        return alerts

    def get_recent_alerts(self, limit: int = 100) -> List[Dict]:
        """Fetch persisted alerts from MongoDB."""
        alert_types = {
            "loitering", "restricted_zone_entry",
            "unknown_person", "re_entry_without_exit",
        }
        projection = {"_id": 0}
        cursor = (
            events_col()
            .find({"event_type": {"$in": list(alert_types)}}, projection)
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

    def _should_emit(
        self,
        visitor_id: str,
        camera_id: str,
        event_type: str,
        now: datetime,
    ) -> bool:
        key = (visitor_id, camera_id, event_type)
        prev = self._last_emitted.get(key)
        if prev and (now - prev) < self._dup_window:
            return False
        self._last_emitted[key] = now
        return True

    def _emit(
        self,
        visitor_id: str,
        name: str,
        camera_id: str,
        location: str,
        event_type: str,
        confidence: float,
        timestamp: datetime,
        extra: Optional[Dict] = None,
    ) -> Dict:
        doc = {
            "visitor_id": visitor_id,
            "name": name,
            "camera_id": camera_id,
            "location": location,
            "event_type": event_type,
            "confidence": round(confidence, 4),
            "timestamp": timestamp,
        }
        if extra:
            doc.update(extra)

        try:
            events_col().insert_one(doc)
        except Exception as exc:
            logger.warning("Alert insert failed: %s", exc)

        doc.pop("_id", None)
        doc["timestamp"] = timestamp.isoformat()
        logger.info("ALERT [%s] visitor=%s camera=%s", event_type, visitor_id, camera_id)
        return doc

    def _check_entry_exit(
        self,
        visitor_id: str,
        name: str,
        camera_id: str,
        location: str,
        confidence: float,
        now: datetime,
        alerts: List[Dict],
    ) -> None:
        lower = location.lower()
        is_entry = "entry" in lower or "entrance" in lower
        is_exit = "exit" in lower

        if is_entry:
            if visitor_id in self._inside:
                if self._should_emit(visitor_id, camera_id, "re_entry_without_exit", now):
                    alerts.append(self._emit(
                        visitor_id, name, camera_id, location,
                        "re_entry_without_exit", confidence, now
                    ))
            self._inside.add(visitor_id)

        if is_exit and visitor_id in self._inside:
            self._inside.discard(visitor_id)
