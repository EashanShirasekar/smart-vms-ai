"""
camera_manager.py - Camera ingestion orchestrator.

Supports:
  - Webcam (source_type="webcam", source_value=0 or 1)
  - Video file (source_type="video", source_value="path/to/file.mp4")
  - RTSP stream (source_type="rtsp", source_value="rtsp://...")

Each camera runs in its own asyncio Task. Frames are forwarded to the
`on_frame` callback supplied at construction.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional, Union

import cv2
import numpy as np

from db import cameras_col

logger = logging.getLogger(__name__)

# Signature: (camera_id, location, zone_type, timestamp, frame) -> None
FrameCallback = Callable[
    [str, str, str, datetime, np.ndarray],
    Awaitable[None],
]


# ──────────────────────────────────────────────────────────
# Config dataclass
# ──────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    camera_id: str
    source_type: str          # "webcam" | "video" | "rtsp"
    source_value: Union[int, str]   # camera index OR file path OR rtsp url
    location: str
    zone_type: str = "general"
    target_fps: float = 5.0   # keep low for demo (reduce CPU load)

    def to_cv2_source(self) -> Union[int, str]:
        """Return what to pass to cv2.VideoCapture."""
        if self.source_type == "webcam":
            return int(self.source_value)
        # "video" and "rtsp" both use the string/url as-is
        return str(self.source_value)


# ──────────────────────────────────────────────────────────
# Camera Worker (one per camera)
# ──────────────────────────────────────────────────────────

class CameraWorker:
    """Reads frames from one source and fires the on_frame callback."""

    def __init__(self, config: CameraConfig, on_frame: FrameCallback) -> None:
        self.config = config
        self._on_frame = on_frame
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run(),
            name=f"camera-{self.config.camera_id}",
        )
        logger.info("Camera '%s' started (%s → %s).",
                    self.config.camera_id,
                    self.config.source_type,
                    self.config.source_value)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Camera '%s' stopped.", self.config.camera_id)

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    async def _run(self) -> None:
        source = self.config.to_cv2_source()
        capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            logger.error("Cannot open camera '%s' (source=%s).",
                         self.config.camera_id, source)
            self._running = False
            return

        frame_interval = 1.0 / max(1.0, self.config.target_fps)

        try:
            while self._running:
                t0 = asyncio.get_running_loop().time()

                ret, frame = await asyncio.to_thread(capture.read)

                if not ret:
                    if self.config.source_type == "video":
                        # Loop video file for demo purposes
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        await asyncio.sleep(0.1)
                        continue
                    # Webcam/RTSP read failure – retry briefly
                    await asyncio.sleep(0.5)
                    continue

                timestamp = datetime.now(timezone.utc)

                try:
                    await self._on_frame(
                        self.config.camera_id,
                        self.config.location,
                        self.config.zone_type,
                        timestamp,
                        frame,
                    )
                except Exception as exc:
                    logger.warning("on_frame error (camera=%s): %s",
                                   self.config.camera_id, exc)

                elapsed = asyncio.get_running_loop().time() - t0
                sleep_for = max(0.0, frame_interval - elapsed)
                if sleep_for:
                    await asyncio.sleep(sleep_for)
        finally:
            capture.release()
            self._running = False


# ──────────────────────────────────────────────────────────
# Manager
# ──────────────────────────────────────────────────────────

class CameraManager:
    """Registers, starts, and stops camera workers."""

    def __init__(self, on_frame: FrameCallback) -> None:
        self._on_frame = on_frame
        self._configs: Dict[str, CameraConfig] = {}
        self._workers: Dict[str, CameraWorker] = {}

    # ----------------------------------------------------------
    # Registration
    # ----------------------------------------------------------

    def register(self, config: CameraConfig) -> None:
        """Persist config to DB and store in memory."""
        self._configs[config.camera_id] = config

        doc = {
            "camera_id": config.camera_id,
            "source_type": config.source_type,
            "source_value": str(config.source_value),
            "location": config.location,
            "zone_type": config.zone_type,
            "target_fps": config.target_fps,
        }
        cameras_col().replace_one(
            {"camera_id": config.camera_id}, doc, upsert=True
        )
        logger.info("Camera '%s' registered.", config.camera_id)

    def remove(self, camera_id: str) -> None:
        self._configs.pop(camera_id, None)
        cameras_col().delete_one({"camera_id": camera_id})

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    async def start(self, camera_id: str) -> bool:
        """Start streaming from a registered camera. Returns False if not found."""
        if camera_id in self._workers and self._workers[camera_id].is_running:
            return True

        config = self._configs.get(camera_id)
        if not config:
            logger.warning("Camera '%s' not registered.", camera_id)
            return False

        worker = CameraWorker(config=config, on_frame=self._on_frame)
        self._workers[camera_id] = worker
        worker.start()
        return True

    async def stop(self, camera_id: str) -> bool:
        worker = self._workers.pop(camera_id, None)
        if not worker:
            return False
        await worker.stop()
        return True

    async def stop_all(self) -> None:
        for cid in list(self._workers.keys()):
            await self.stop(cid)

    # ----------------------------------------------------------
    # Query
    # ----------------------------------------------------------

    def list(self) -> Dict[str, Dict]:
        return {
            cid: {
                "camera_id": cfg.camera_id,
                "source_type": cfg.source_type,
                "source_value": cfg.source_value,
                "location": cfg.location,
                "zone_type": cfg.zone_type,
                "target_fps": cfg.target_fps,
                "active": cid in self._workers and self._workers[cid].is_running,
            }
            for cid, cfg in self._configs.items()
        }

    def load_from_db(self) -> int:
        """Re-populate in-memory configs from MongoDB (useful after restart)."""
        records = list(cameras_col().find({}, {"_id": 0}))
        for rec in records:
            source_value: Union[int, str] = rec["source_value"]
            if rec.get("source_type") == "webcam":
                try:
                    source_value = int(source_value)
                except (ValueError, TypeError):
                    pass
            config = CameraConfig(
                camera_id=rec["camera_id"],
                source_type=rec["source_type"],
                source_value=source_value,
                location=rec["location"],
                zone_type=rec.get("zone_type", "general"),
                target_fps=float(rec.get("target_fps", 5.0)),
            )
            self._configs[config.camera_id] = config
        return len(records)
