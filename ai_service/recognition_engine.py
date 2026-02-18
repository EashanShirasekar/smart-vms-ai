"""
recognition_engine.py - Real-time face detection and identification.

Loads ArcFace embeddings from MongoDB and performs cosine-distance matching
against each detected face in a video frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
from deepface import DeepFace

from db import embeddings_col

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
DEFAULT_DISTANCE_THRESHOLD = 0.40   # L2 on unit vectors; tune as needed


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class FaceMatch:
    """All information returned for a single detected face."""
    visitor_id: str
    name: str
    category: str
    confidence: float                    # 0.0 – 1.0
    bounding_box: Dict[str, int]         # {x, y, w, h}
    embedding: np.ndarray = field(repr=False, compare=False)


# ──────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────

class RecognitionEngine:
    """
    Thread-safe face identification engine.

    Typical usage
    -------------
    engine = RecognitionEngine()
    engine.load_embeddings()          # call once at startup
    matches = engine.identify(frame)  # call per frame
    engine.load_embeddings()          # call again to refresh after new enrolments
    """

    def __init__(
        self,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        model_name: str = MODEL_NAME,
        detector_backend: str = DETECTOR_BACKEND,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.model_name = model_name
        self.detector_backend = detector_backend

        self._cache: List[Dict] = []
        self._lock = Lock()

    # ----------------------------------------------------------
    # Cache management
    # ----------------------------------------------------------

    def load_embeddings(self) -> int:
        """
        Pull all embeddings from MongoDB into memory.
        Returns the number of records loaded.
        """
        records = list(embeddings_col().find({}, {"_id": 0}))
        cache: List[Dict] = []

        for rec in records:
            raw = rec.get("embedding")
            if not raw:
                continue
            vector = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            cache.append({
                "visitor_id": rec.get("visitor_id", "unknown"),
                "name": rec.get("name", "Unknown"),
                "category": rec.get("category", "unknown"),
                "embedding": vector / norm,
            })

        with self._lock:
            self._cache = cache

        logger.info("Loaded %d face embeddings from database.", len(cache))
        return len(cache)

    # ----------------------------------------------------------
    # Identification
    # ----------------------------------------------------------

    def identify(self, frame: np.ndarray) -> List[FaceMatch]:
        """
        Detect all faces in `frame` and return identification results.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (from OpenCV).

        Returns
        -------
        List[FaceMatch]
            One entry per detected face.
        """
        try:
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
            )
        except Exception as exc:
            logger.warning("Face detection failed: %s", exc)
            return []

        matches: List[FaceMatch] = []

        for det in detections:
            confidence_score = det.get("confidence", 0)
            if confidence_score < 0.70:
                # Low-confidence detections are likely false positives
                continue

            area = det.get("facial_area", {})
            x, y = int(area.get("x", 0)), int(area.get("y", 0))
            w, h = int(area.get("w", 0)), int(area.get("h", 0))

            if w <= 0 or h <= 0:
                continue

            crop = frame[y: y + h, x: x + w]
            if crop.size == 0:
                continue

            embedding = self._embed(crop)
            if embedding is None:
                continue

            visitor_id, name, category, conf = self._match(embedding)

            matches.append(FaceMatch(
                visitor_id=visitor_id,
                name=name,
                category=category,
                confidence=conf,
                bounding_box={"x": x, "y": y, "w": w, "h": h},
                embedding=embedding,
            ))

        return matches

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _embed(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Generate a normalised embedding for a face crop."""
        try:
            result = DeepFace.represent(
                img_path=face_crop,
                model_name=self.model_name,
                detector_backend="skip",      # face already cropped
                enforce_detection=False,
                align=False,
            )
            vector = np.array(result[0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vector)
            return vector / norm if norm > 0 else vector
        except Exception as exc:
            logger.debug("Embedding extraction failed: %s", exc)
            return None

    def _match(self, probe: np.ndarray) -> Tuple[str, str, str, float]:
        """
        Find the closest entry in the cache.

        Returns (visitor_id, name, category, confidence).
        """
        with self._lock:
            candidates = list(self._cache)

        if not candidates:
            return "unknown", "Unknown", "unknown", 0.0

        distances = np.array([
            np.linalg.norm(probe - c["embedding"])
            for c in candidates
        ])
        idx = int(np.argmin(distances))
        best_dist = float(distances[idx])

        if best_dist > self.distance_threshold:
            # Normalise distance to a rough confidence score
            conf = float(max(0.0, 1.0 - best_dist))
            return "unknown", "Unknown", "unknown", conf

        conf = float(max(0.0, 1.0 - (best_dist / self.distance_threshold)))
        best = candidates[idx]
        return best["visitor_id"], best["name"], best["category"], conf
