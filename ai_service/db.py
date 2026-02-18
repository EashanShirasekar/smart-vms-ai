"""
db.py - MongoDB connection and collection management.

All database access goes through this module.
Run `ensure_indexes()` once at startup to apply indexes.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# ──────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("DB_NAME", "smart_vms")

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db() -> Database:
    return get_client()[DB_NAME]


# ──────────────────────────────────────────────────────────
# Collection accessors
# ──────────────────────────────────────────────────────────

def embeddings_col() -> Collection:
    """Stores face embeddings for enrolled visitors."""
    return get_db()["embeddings"]


def events_col() -> Collection:
    """Stores all detection and alert events."""
    return get_db()["events"]


def cameras_col() -> Collection:
    """Stores camera configuration."""
    return get_db()["cameras"]


# ──────────────────────────────────────────────────────────
# Schema / Indexes
# ──────────────────────────────────────────────────────────

def ensure_indexes() -> None:
    """Create indexes on startup. Safe to call multiple times."""
    embeddings_col().create_index([("visitor_id", ASCENDING)], unique=True)

    events_col().create_index([("timestamp", DESCENDING)])
    events_col().create_index([("visitor_id", ASCENDING), ("timestamp", DESCENDING)])
    events_col().create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
    events_col().create_index([("event_type", ASCENDING), ("timestamp", DESCENDING)])

    cameras_col().create_index([("camera_id", ASCENDING)], unique=True)


# ──────────────────────────────────────────────────────────
# MongoDB document shapes (for documentation only)
# ──────────────────────────────────────────────────────────

"""
embeddings collection:
{
    "visitor_id":  str,          # unique
    "name":        str,
    "embedding":   list[float],  # ArcFace vector (512-dim)
    "category":    str,          # "visitor" | "staff" | "vip"
    "created_at":  datetime (UTC)
}

events collection:
{
    "visitor_id":  str | "unknown",
    "camera_id":   str,
    "location":    str,
    "event_type":  str,          # see EventType in behavior_analyzer.py
    "confidence":  float,
    "timestamp":   datetime (UTC)
}

cameras collection:
{
    "camera_id":    str,          # unique
    "source_type":  str,          # "webcam" | "video" | "rtsp"
    "source_value": str | int,
    "location":     str,
    "zone_type":    str,          # "general" | "restricted"
    "created_at":   datetime (UTC)
}
"""
