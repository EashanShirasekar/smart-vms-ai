"""
app.py - Smart Visitor Management System — AI Service

FastAPI application that wires together:
  - Enrollment  (POST /enroll)
  - Recognition (POST /recognize)
  - Camera management (POST /cameras/register, /cameras/{id}/start, etc.)
  - Alerts (GET /alerts)
  - Visitors (GET /visitors)
  - Stats / health
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from behavior_analyzer import BehaviorAnalyzer
from camera_manager import CameraConfig, CameraManager
from db import ensure_indexes
from enrollment_manager import EnrollmentManager
from event_dispatcher import EventDispatcher
from recognition_engine import RecognitionEngine
from tracker import MultiCameraTracker

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smart_vms")

# ──────────────────────────────────────────────────────────
# Component initialisation
# ──────────────────────────────────────────────────────────

recognition_engine = RecognitionEngine()
enrollment_manager = EnrollmentManager()
tracker = MultiCameraTracker()
behavior_analyzer = BehaviorAnalyzer(
    loitering_threshold_seconds=int(os.getenv("LOITERING_SECONDS", 60)),
    duplicate_suppression_seconds=int(os.getenv("DUP_SUPPRESS_SECONDS", 30)),
    unknown_alert_interval_seconds=int(os.getenv("UNKNOWN_ALERT_SECONDS", 45)),
)
dispatcher = EventDispatcher(
    backend_url=os.getenv("BACKEND_WEBHOOK_URL", ""),
)


# ──────────────────────────────────────────────────────────
# Frame processing pipeline (called by CameraWorker)
# ──────────────────────────────────────────────────────────

async def on_frame(
    camera_id: str,
    location: str,
    zone_type: str,
    timestamp: datetime,
    frame: np.ndarray,
) -> None:
    """
    Process one video frame end-to-end:
      detect → identify → track → analyse → dispatch
    """
    matches = await _run_recognition(frame)

    for match in matches:
        # 1. Persist tracking event
        tracker.record(
            visitor_id=match.visitor_id,
            name=match.name,
            camera_id=camera_id,
            location=location,
            zone_type=zone_type,
            confidence=match.confidence,
            timestamp=timestamp,
        )

        # 2. Behaviour analysis → alerts
        alerts = behavior_analyzer.analyze(
            visitor_id=match.visitor_id,
            name=match.name,
            camera_id=camera_id,
            location=location,
            zone_type=zone_type,
            category=match.category,
            confidence=match.confidence,
            timestamp=timestamp,
        )

        # 3. Dispatch alerts to backend (fire-and-forget)
        for alert in alerts:
            await dispatcher.dispatch(alert)


async def _run_recognition(frame: np.ndarray):
    """Run recognition in a thread to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(recognition_engine.identify, frame)


# ──────────────────────────────────────────────────────────
# Camera manager (needs on_frame callback)
# ──────────────────────────────────────────────────────────

camera_manager = CameraManager(on_frame=on_frame)


# ──────────────────────────────────────────────────────────
# App lifespan
# ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Smart VMS AI Service starting...")
    ensure_indexes()
    recognition_engine.load_embeddings()
    camera_manager.load_from_db()
    logger.info("✅ System ready.")
    yield
    logger.info("🛑 Shutting down cameras...")
    await camera_manager.stop_all()
    await dispatcher.close()
    logger.info("👋 Shutdown complete.")


# ──────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Visitor Management AI Service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# Pydantic request models
# ──────────────────────────────────────────────────────────

class CameraRegisterRequest(BaseModel):
    camera_id: str
    source_type: str            # "webcam" | "video" | "rtsp"
    source_value: Union[int, str]
    location: str
    zone_type: str = "general"
    target_fps: float = 5.0


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _decode_image(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG or PNG.")
    return img


# ──────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Smart VMS AI",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Enrollment ──────────────────────────────────────────

@app.post("/enroll")
async def enroll_visitor(
    visitor_id: str = Form(...),
    name: str = Form(...),
    category: str = Form("visitor"),
    file: UploadFile = File(...),
):
    """
    Enroll a visitor by uploading their photo.

    Form fields:
      - visitor_id  (string, unique)
      - name        (string)
      - category    (optional: visitor / staff / vip)
      - file        (image: jpg/png)
    """
    contents = await file.read()
    try:
        image = _decode_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = enrollment_manager.enroll(
            visitor_id=visitor_id,
            name=name,
            image=image,
            category=category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])

    # Reload embeddings so cameras pick up the new face immediately
    recognition_engine.load_embeddings()

    return JSONResponse(content=result)


# ── Recognition ─────────────────────────────────────────

@app.post("/recognize")
async def recognize_face(
    camera_id: str = Form("manual"),
    file: UploadFile = File(...),
):
    """
    Identify faces in an uploaded image (one-shot, not from live camera).

    Useful for testing / security desk manual check.
    """
    contents = await file.read()
    try:
        image = _decode_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    import asyncio
    matches = await asyncio.to_thread(recognition_engine.identify, image)

    results = [
        {
            "visitor_id": m.visitor_id,
            "name": m.name,
            "category": m.category,
            "confidence": round(m.confidence, 4),
            "bounding_box": m.bounding_box,
        }
        for m in matches
    ]

    return JSONResponse(content={
        "success": True,
        "camera_id": camera_id,
        "faces_detected": len(results),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Camera management ────────────────────────────────────

@app.post("/cameras/register")
async def register_camera(req: CameraRegisterRequest):
    """Register a camera (webcam, video file, or RTSP)."""
    config = CameraConfig(
        camera_id=req.camera_id,
        source_type=req.source_type,
        source_value=req.source_value,
        location=req.location,
        zone_type=req.zone_type,
        target_fps=req.target_fps,
    )
    camera_manager.register(config)
    return {"success": True, "message": f"Camera '{req.camera_id}' registered."}


@app.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start live processing for a registered camera."""
    ok = await camera_manager.start(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"success": True, "message": f"Camera '{camera_id}' started."}


@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stop a running camera worker."""
    ok = await camera_manager.stop(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not running.")
    return {"success": True, "message": f"Camera '{camera_id}' stopped."}


@app.get("/cameras")
async def list_cameras():
    """List all registered cameras and their status."""
    cameras = camera_manager.list()
    return {"success": True, "count": len(cameras), "cameras": list(cameras.values())}


# ── Alerts ───────────────────────────────────────────────

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Return recent behavior alerts."""
    alerts = behavior_analyzer.get_recent_alerts(limit=limit)
    return {"success": True, "count": len(alerts), "alerts": alerts}


# ── Visitors ─────────────────────────────────────────────

@app.get("/visitors")
async def list_visitors():
    """Return all enrolled visitors."""
    visitors = enrollment_manager.get_all_visitors()
    return {"success": True, "count": len(visitors), "visitors": visitors}


@app.get("/visitors/{visitor_id}")
async def get_visitor(visitor_id: str):
    visitor = enrollment_manager.get_visitor(visitor_id)
    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    return {"success": True, "visitor": visitor}


@app.delete("/visitors/{visitor_id}")
async def delete_visitor(visitor_id: str):
    result = enrollment_manager.delete_visitor(visitor_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    recognition_engine.load_embeddings()
    return result


# ── Embeddings ───────────────────────────────────────────

@app.post("/embeddings/reload")
async def reload_embeddings():
    """Force-reload face embeddings from MongoDB into memory."""
    count = recognition_engine.load_embeddings()
    return {"success": True, "embeddings_loaded": count}


# ── Tracking ─────────────────────────────────────────────

@app.get("/tracking/{visitor_id}")
async def get_visitor_tracking(visitor_id: str, limit: int = 50):
    """Get movement history for a specific visitor."""
    history = tracker.get_visitor_history(visitor_id, limit=limit)
    return {"success": True, "visitor_id": visitor_id, "history": history}


# ── Stats ────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    visitors = enrollment_manager.get_all_visitors()
    cameras = camera_manager.list()
    alerts = behavior_analyzer.get_recent_alerts(limit=1000)
    return {
        "success": True,
        "stats": {
            "enrolled_visitors": len(visitors),
            "registered_cameras": len(cameras),
            "active_cameras": sum(1 for c in cameras.values() if c["active"]),
            "total_alerts": len(alerts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,   # reload=True breaks asyncio tasks — keep False
        log_level="info",
    )
