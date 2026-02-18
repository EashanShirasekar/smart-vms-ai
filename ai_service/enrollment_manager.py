"""
enrollment_manager.py - Visitor enrollment into MongoDB.

Accepts a raw image (numpy array), extracts an ArcFace embedding
via DeepFace, then upserts the record into the embeddings collection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from deepface import DeepFace
from pymongo.errors import PyMongoError

from db import embeddings_col


# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

class EnrollmentManager:
    """Enroll visitors and manage the embeddings collection."""

    # ----------------------------------------------------------
    # Enrollment
    # ----------------------------------------------------------

    def enroll(
        self,
        visitor_id: str,
        name: str,
        image: np.ndarray,
        category: str = "visitor",
    ) -> Dict:
        """
        Enroll a visitor from a numpy BGR image.

        Returns a dict with keys: success, message, visitor_id.
        Raises ValueError if no face is detected.
        """
        embedding = self._extract_embedding(image)

        doc = {
            "visitor_id": visitor_id,
            "name": name,
            "embedding": embedding.tolist(),
            "category": category,
            "created_at": datetime.now(timezone.utc),
        }

        try:
            embeddings_col().replace_one(
                {"visitor_id": visitor_id},
                doc,
                upsert=True,
            )
        except PyMongoError as exc:
            return {"success": False, "message": f"Database error: {exc}", "visitor_id": visitor_id}

        return {
            "success": True,
            "message": f"Visitor '{name}' enrolled successfully.",
            "visitor_id": visitor_id,
        }

    # ----------------------------------------------------------
    # Retrieval helpers
    # ----------------------------------------------------------

    def get_all_visitors(self) -> List[Dict]:
        """Return all enrolled visitors (without embeddings for brevity)."""
        projection = {"_id": 0, "embedding": 0}
        return list(embeddings_col().find({}, projection))

    def get_visitor(self, visitor_id: str) -> Optional[Dict]:
        """Return a single visitor record (without embedding vector)."""
        return embeddings_col().find_one(
            {"visitor_id": visitor_id},
            {"_id": 0, "embedding": 0},
        )

    def delete_visitor(self, visitor_id: str) -> Dict:
        result = embeddings_col().delete_one({"visitor_id": visitor_id})
        if result.deleted_count:
            return {"success": True, "message": f"Visitor {visitor_id} deleted."}
        return {"success": False, "message": "Visitor not found."}

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    @staticmethod
    def _extract_embedding(image: np.ndarray) -> np.ndarray:
        """
        Extract a normalised ArcFace embedding from a BGR numpy image.
        Raises ValueError if DeepFace finds no face.
        """
        try:
            results = DeepFace.represent(
                img_path=image,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
        except Exception as exc:
            raise ValueError(f"Face not detected in image: {exc}") from exc

        if not results:
            raise ValueError("DeepFace returned an empty result.")

        vector = np.array(results[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
