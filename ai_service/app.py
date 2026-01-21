"""
Smart Visitor Management AI Engine - Main Application
Production-grade facial recognition and tracking system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import cv2
import numpy as np
from datetime import datetime
import json
import base64

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.face_detector import FaceDetector
from recognition.face_recognizer import FaceRecognizer
from enrollment.enroll import EnrollmentManager
from tracking.tracker import MultiCameraTracker
from alerts.behavior_monitor import BehaviorMonitor
from camera.camera_manager import CameraManager
from utils.helpers import decode_image, encode_image

app = FastAPI(title="Smart Visitor Management AI", version="1.0.0")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
enrollment_manager = EnrollmentManager(face_detector, face_recognizer)
tracker = MultiCameraTracker()
behavior_monitor = BehaviorMonitor()
camera_manager = CameraManager()

# Pydantic models for API
class EnrollRequest(BaseModel):
    visitor_id: str
    name: str
    metadata: Optional[dict] = {}

class RecognizeRequest(BaseModel):
    camera_id: str
    image_base64: Optional[str] = None

class CameraConfig(BaseModel):
    camera_id: str
    stream_url: str
    location: str
    zone_type: str = "general"

class AlertResponse(BaseModel):
    alert_id: str
    alert_type: str
    visitor_id: Optional[str]
    camera_id: str
    timestamp: str
    details: dict

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🚀 Smart Visitor Management AI Engine Starting...")
    print("✅ Face Detector Ready")
    print("✅ Face Recognizer Ready")
    print("✅ Tracker Initialized")
    print("✅ Behavior Monitor Active")
    print("🎯 System Ready for Operations")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Smart Visitor Management AI",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/enroll")
async def enroll_visitor(
    visitor_id: str,
    name: str,
    file: UploadFile = File(...)
):
    """
    Enroll new visitor with face image
    
    Args:
        visitor_id: Unique visitor identifier
        name: Visitor full name
        file: Face image file (jpg, png)
    
    Returns:
        Enrollment confirmation with embedding status
    """
    try:
        # Read and decode image
        contents = await file.read()
        image = decode_image(contents)
        
        # Enroll visitor
        result = enrollment_manager.enroll_visitor(
            visitor_id=visitor_id,
            name=name,
            image=image
        )
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "message": f"Visitor {name} enrolled successfully",
                "visitor_id": visitor_id,
                "embedding_generated": True,
                "face_detected": True,
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Enrollment failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment error: {str(e)}")

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...), camera_id: str = "camera_01"):
    """
    Recognize face from uploaded image
    
    Args:
        file: Image file containing face
        camera_id: Camera identifier for tracking
    
    Returns:
        Recognition result with visitor details
    """
    try:
        # Read and decode image
        contents = await file.read()
        image = decode_image(contents)
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return JSONResponse(content={
                "success": False,
                "message": "No faces detected",
                "recognized": False
            })
        
        results = []
        for face in faces:
            x, y, w, h = face["box"]
            face_img = image[y:y+h, x:x+w]
            
            # Recognize face
            match = face_recognizer.recognize(face_img)
            
            if match:
                # Update tracker
                tracker.update_tracking(
                    camera_id=camera_id,
                    visitor_id=match["visitor_id"],
                    location=face["box"],
                    timestamp=datetime.now()
                )
                
                results.append({
                    "recognized": True,
                    "visitor_id": match["visitor_id"],
                    "name": match["name"],
                    "confidence": match["confidence"],
                    "bounding_box": face["box"]
                })
            else:
                results.append({
                    "recognized": False,
                    "message": "Unknown face",
                    "bounding_box": face["box"]
                })
        
        return JSONResponse(content={
            "success": True,
            "faces_detected": len(faces),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

@app.post("/camera/register")
async def register_camera(config: CameraConfig):
    """
    Register new camera for monitoring
    
    Args:
        config: Camera configuration (ID, URL, location, zone)
    
    Returns:
        Registration confirmation
    """
    try:
        result = camera_manager.register_camera(
            camera_id=config.camera_id,
            stream_url=config.stream_url,
            location=config.location,
            zone_type=config.zone_type
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Camera registered successfully",
            "camera_id": config.camera_id,
            "location": config.location,
            "zone_type": config.zone_type
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera registration error: {str(e)}")

@app.get("/camera/list")
async def list_cameras():
    """Get list of all registered cameras"""
    cameras = camera_manager.get_all_cameras()
    return JSONResponse(content={
        "success": True,
        "count": len(cameras),
        "cameras": cameras
    })

@app.post("/track")
async def track_visitor(visitor_id: str, camera_id: str):
    """
    Get tracking history for specific visitor
    
    Args:
        visitor_id: Visitor identifier
        camera_id: Optional camera filter
    
    Returns:
        Movement history across cameras
    """
    try:
        history = tracker.get_visitor_history(visitor_id, camera_id)
        
        return JSONResponse(content={
            "success": True,
            "visitor_id": visitor_id,
            "tracking_history": history,
            "total_detections": len(history)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tracking error: {str(e)}")

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """
    Get recent security alerts
    
    Args:
        limit: Maximum number of alerts to return
    
    Returns:
        List of recent alerts
    """
    try:
        alerts = behavior_monitor.get_alerts(limit=limit)
        
        return JSONResponse(content={
            "success": True,
            "alert_count": len(alerts),
            "alerts": alerts
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert retrieval error: {str(e)}")

@app.get("/visitors")
async def get_enrolled_visitors():
    """Get list of all enrolled visitors"""
    visitors = enrollment_manager.get_all_visitors()
    
    return JSONResponse(content={
        "success": True,
        "count": len(visitors),
        "visitors": visitors
    })

@app.delete("/visitor/{visitor_id}")
async def delete_visitor(visitor_id: str):
    """
    Remove visitor from system
    
    Args:
        visitor_id: Visitor identifier to remove
    
    Returns:
        Deletion confirmation
    """
    try:
        result = enrollment_manager.delete_visitor(visitor_id)
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "message": f"Visitor {visitor_id} removed successfully"
            })
        else:
            raise HTTPException(status_code=404, detail="Visitor not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion error: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and metrics"""
    return JSONResponse(content={
        "success": True,
        "statistics": {
            "enrolled_visitors": len(enrollment_manager.get_all_visitors()),
            "active_cameras": len(camera_manager.get_all_cameras()),
            "total_alerts": len(behavior_monitor.get_alerts(limit=1000)),
            "system_uptime": "operational",
            "timestamp": datetime.now().isoformat()
        }
    })

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost
        port=8000,
        reload=True,
        log_level="info"
    )