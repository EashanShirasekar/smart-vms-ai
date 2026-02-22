# Smart Visitor Management System — AI Service

> **Final Year Project / IEEE Demo Edition**
> Python · FastAPI · DeepFace (ArcFace) · MongoDB · OpenCV

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Folder Structure](#2-folder-structure)
3. [MongoDB Schema](#3-mongodb-schema)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Running the Service](#6-running-the-service)
7. [API Reference with Examples](#7-api-reference-with-examples)
8. [Demo Workflow (Step-by-Step)](#8-demo-workflow-step-by-step)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI app.py                         │
│                                                              │
│  POST /enroll          POST /recognize      GET /alerts      │
│  POST /cameras/...     GET /visitors        GET /stats       │
└────────────┬─────────────────┬──────────────────────────────┘
             │                 │
    ┌─────────▼──────┐  ┌──────▼───────────┐
    │EnrollmentMgr   │  │ RecognitionEngine │
    │(DeepFace embed)│  │(ArcFace + cosine) │
    └────────┬───────┘  └──────┬────────────┘
             │                 │
    ┌─────────▼─────────────────▼────────────┐
    │           MongoDB  (local)              │
    │  embeddings │ events │ cameras         │
    └─────────────────────────────────────────┘
                          ▲
             ┌────────────┴──────────────┐
             │       CameraManager        │
             │  (webcam / video / rtsp)   │
             │  asyncio task per camera   │
             └────────────┬──────────────┘
                          │ frame
             ┌────────────▼──────────────┐
             │   BehaviorAnalyzer         │
             │  loitering │ restricted    │
             │  unknown   │ re-entry      │
             └────────────┬──────────────┘
                          │ alert
             ┌────────────▼──────────────┐
             │   EventDispatcher (opt.)   │
             │  HTTP POST to backend API  │
             └───────────────────────────┘
```

---

## 2. Folder Structure

```
ai_service/
├── app.py                 # FastAPI entrypoint, endpoint definitions
├── recognition_engine.py  # DeepFace ArcFace face detection & matching
├── enrollment_manager.py  # Visitor enrolment → MongoDB
├── behavior_analyzer.py   # Loitering, restricted zone, unknown alerts
├── camera_manager.py      # Webcam/video/RTSP ingestion (asyncio)
├── tracker.py             # Multi-camera movement recording & queries
├── event_dispatcher.py    # HTTP event forwarding to backend (optional)
├── db.py                  # MongoDB connection & collection helpers
├── requirements.txt
└── README.md
```

---

## 3. MongoDB Schema

### Collection: `embeddings`
Stores one document per enrolled visitor.

```json
{
  "visitor_id":  "V001",
  "name":        "Alice Johnson",
  "embedding":   [0.023, -0.117, ...],   // 512-dim ArcFace vector
  "category":    "visitor",               // visitor | staff | vip
  "created_at":  "2024-01-15T09:00:00Z"
}
```

### Collection: `events`
Every detection (tracking + alerts) is a document here.

```json
{
  "visitor_id":  "V001",
  "name":        "Alice Johnson",
  "camera_id":   "cam1",
  "location":    "Entrance",
  "zone_type":   "general",
  "event_type":  "identity_tracking",     // see below
  "confidence":  0.872,
  "timestamp":   "2024-01-15T09:05:32Z",

  // alert-only extra field (loitering):
  "dwell_seconds": 75
}
```

**event_type values:**
| Value | Trigger |
|---|---|
| `identity_tracking` | Every recognised face per frame |
| `loitering` | Same person, same camera > threshold (default 60s) |
| `restricted_zone_entry` | Camera with zone_type = "restricted" |
| `unknown_person` | Unrecognised face persists (default 45s interval) |
| `re_entry_without_exit` | Seen at "entry/entrance" location before exiting |

### Collection: `cameras`
Registered camera configurations.

```json
{
  "camera_id":    "cam1",
  "source_type":  "webcam",
  "source_value": "0",
  "location":     "Entrance",
  "zone_type":    "general",
  "target_fps":   5.0
}
```

---

## 4. Installation

### Prerequisites
- Python 3.10 or 3.11
- MongoDB 6+ running locally (`mongod` on port 27017)
- Webcam or video file

### Step 1 — Clone / extract the project

```bash
cd ai_service
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** DeepFace will automatically download the ArcFace
> and RetinaFace model weights (~500 MB) on the first call.
> Ensure you have internet access the first time.

### Step 4 — Start MongoDB

```bash
# Linux / macOS
mongod --dbpath /data/db

# Windows (MongoDB installed as service)
net start MongoDB

# Or with Docker
docker run -d -p 27017:27017 --name mongo mongo:7
```

---

## 5. Configuration

All settings can be changed via environment variables (no code changes needed):

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DB_NAME` | `smart_vms` | Database name |
| `LOITERING_SECONDS` | `60` | Seconds before loitering alert fires |
| `DUP_SUPPRESS_SECONDS` | `30` | Minimum gap between repeated alerts |
| `UNKNOWN_ALERT_SECONDS` | `45` | Interval for unknown person alerts per camera |
| `BACKEND_WEBHOOK_URL` | *(empty)* | Optional: POST alerts to this URL |

Example:
```bash
export LOITERING_SECONDS=30
export MONGO_URI="mongodb://localhost:27017"
python app.py
```

---

## 6. Running the Service

```bash
cd ai_service
source venv/bin/activate   # or venv\Scripts\activate on Windows
python app.py
```

The service starts on **http://127.0.0.1:8000**

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 7. API Reference with Examples

All examples use `curl`. You can also use Postman or the `/docs` Swagger UI.

---

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{ "status": "ok", "service": "Smart VMS AI", "version": "2.0.0" }
```

---

### POST /enroll
Enroll a visitor with their photo.

```bash
curl -X POST http://localhost:8000/enroll \
  -F "visitor_id=V001" \
  -F "name=Alice Johnson" \
  -F "category=visitor" \
  -F "file=@/path/to/alice.jpg"
```
```json
{
  "success": true,
  "message": "Visitor 'Alice Johnson' enrolled successfully.",
  "visitor_id": "V001"
}
```

---

### POST /recognize
Manually identify faces in an image (useful for desk-side checks).

```bash
curl -X POST http://localhost:8000/recognize \
  -F "camera_id=manual" \
  -F "file=@/path/to/test_face.jpg"
```
```json
{
  "success": true,
  "faces_detected": 1,
  "results": [
    {
      "visitor_id": "V001",
      "name": "Alice Johnson",
      "category": "visitor",
      "confidence": 0.874,
      "bounding_box": { "x": 102, "y": 45, "w": 180, "h": 210 }
    }
  ]
}
```

---

### POST /cameras/register
Register a camera source.

**Webcam (index 0 = built-in, 1 = USB):**
```bash
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam1",
    "source_type": "webcam",
    "source_value": 0,
    "location": "Entrance",
    "zone_type": "general",
    "target_fps": 5
  }'
```

**Restricted zone webcam:**
```bash
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam2",
    "source_type": "webcam",
    "source_value": 1,
    "location": "Server Room",
    "zone_type": "restricted",
    "target_fps": 5
  }'
```

**Video file:**
```bash
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam3",
    "source_type": "video",
    "source_value": "test_video.mp4",
    "location": "Corridor",
    "zone_type": "general",
    "target_fps": 10
  }'
```

---

### POST /cameras/{id}/start
```bash
curl -X POST http://localhost:8000/cameras/cam1/start
```
```json
{ "success": true, "message": "Camera 'cam1' started." }
```

### POST /cameras/{id}/stop
```bash
curl -X POST http://localhost:8000/cameras/cam1/stop
```

### GET /cameras
```bash
curl http://localhost:8000/cameras
```
```json
{
  "cameras": [
    {
      "camera_id": "cam1",
      "source_type": "webcam",
      "source_value": 0,
      "location": "Entrance",
      "zone_type": "general",
      "active": true
    }
  ]
}
```

---

### GET /alerts
```bash
curl http://localhost:8000/alerts?limit=20
```
```json
{
  "alerts": [
    {
      "visitor_id": "unknown",
      "camera_id": "cam1",
      "location": "Entrance",
      "event_type": "unknown_person",
      "confidence": 0.0,
      "timestamp": "2024-01-15T09:10:45Z"
    },
    {
      "visitor_id": "V001",
      "camera_id": "cam2",
      "location": "Server Room",
      "event_type": "restricted_zone_entry",
      "confidence": 0.88,
      "timestamp": "2024-01-15T09:12:03Z"
    }
  ]
}
```

---

### GET /visitors
```bash
curl http://localhost:8000/visitors
```

### DELETE /visitors/{id}
```bash
curl -X DELETE http://localhost:8000/visitors/V001
```

### POST /embeddings/reload
Force-reload embeddings after bulk DB changes.
```bash
curl -X POST http://localhost:8000/embeddings/reload
```

### GET /tracking/{visitor_id}
```bash
curl http://localhost:8000/tracking/V001?limit=20
```

### GET /stats
```bash
curl http://localhost:8000/stats
```

---

## 8. Demo Workflow (Step-by-Step)

This is the exact sequence for a live demo or IEEE presentation.

### Phase 1 — Start the system

```bash
# Terminal 1: Start MongoDB
mongod

# Terminal 2: Start AI service
cd ai_service
source venv/bin/activate
python app.py
```

Open browser: http://127.0.0.1:8000/docs

---

### Phase 2 — Enroll visitors (simulate security desk registration)

Prepare 3 photos: `alice.jpg`, `bob.jpg`, `charlie.jpg`

```bash
curl -X POST http://localhost:8000/enroll \
  -F "visitor_id=V001" -F "name=Alice" -F "file=@alice.jpg"

curl -X POST http://localhost:8000/enroll \
  -F "visitor_id=V002" -F "name=Bob" -F "file=@bob.jpg"

curl -X POST http://localhost:8000/enroll \
  -F "visitor_id=V003" -F "name=Charlie" -F "category=staff" -F "file=@charlie.jpg"
```

Confirm: `curl http://localhost:8000/visitors`

---

### Phase 3 — Register cameras

```bash
# Camera 1: Built-in webcam — Entrance (general zone)
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","source_type":"webcam","source_value":0,"location":"Entrance","zone_type":"general","target_fps":5}'

# Camera 2: USB webcam — Restricted lab (triggers alerts)
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam2","source_type":"webcam","source_value":1,"location":"Lab Entry","zone_type":"restricted","target_fps":5}'

# Camera 3: Video file — simulate corridor footage
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam3","source_type":"video","source_value":"corridor.mp4","location":"Main Corridor","zone_type":"general","target_fps":10}'
```

---

### Phase 4 — Start live monitoring

```bash
curl -X POST http://localhost:8000/cameras/cam1/start
curl -X POST http://localhost:8000/cameras/cam2/start
curl -X POST http://localhost:8000/cameras/cam3/start
```

The system is now:
- Capturing frames every 200ms (5 fps)
- Detecting and identifying faces
- Writing `identity_tracking` events to MongoDB

---

### Phase 5 — Trigger and observe alerts

| Scenario | How to trigger | Alert type |
|---|---|---|
| Unknown person | Walk in front of cam1 (not enrolled) | `unknown_person` |
| Restricted zone | Enrolled visitor in front of cam2 | `restricted_zone_entry` |
| Loitering | Stay in frame for >60 seconds | `loitering` |

Check alerts:
```bash
curl http://localhost:8000/alerts
```

---

### Phase 6 — Manual recognition test

Use the `/recognize` endpoint to test identification without a live camera:

```bash
curl -X POST http://localhost:8000/recognize \
  -F "camera_id=desk" \
  -F "file=@test_face.jpg"
```

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `DeepFace: face not detected` | Image too dark, or face not looking at camera. Try a clearer photo. |
| Camera won't open | Check `source_value` index. Use `0` for built-in, `1` for USB. Run `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` |
| MongoDB connection error | Ensure `mongod` is running. Try `mongosh` to confirm. |
| Slow recognition | Normal on CPU. RetinaFace + ArcFace takes 1–3s/frame. Reduce `target_fps` to `2` for demo. |
| TensorFlow errors | Ensure `tensorflow==2.16.1` and `tf-keras==2.16.0` are both installed. |
| Embedding not loading | Call `POST /embeddings/reload` after enrolling new visitors. |
| `re_entry_without_exit` never fires | Location string must contain "entry" or "entrance" (case-insensitive). |















# NEW


# 🚀 Geofencing System - Complete Setup & Execution Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Step-by-Step Execution](#step-by-step-execution)
4. [API Reference](#api-reference)
5. [Troubleshooting](#troubleshooting)

---

## System Overview

### What's New?
The system now includes **spatial geofencing** for precise loitering detection:

- ✅ **Draw custom boundaries** on each camera view
- ✅ **Track visitor positions** using face detection centroids
- ✅ **Alert when visitors stay outside boundaries** for >60 seconds
- ✅ **Visual feedback** showing boundaries and violations in real-time

### Alert Types
| Alert Type | Trigger | Enabled |
|------------|---------|---------|
| `geofence_violation` | Person outside boundary >60s | ✅ NEW |
| `restricted_zone_entry` | Camera with zone_type="restricted" | ✅ |
| `unknown_person` | Unrecognized face persists | ✅ |
| `re_entry_without_exit` | Entry without prior exit | ✅ |
| ~~`loitering`~~ | ~~Time in same camera~~ | ❌ REMOVED |

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- MongoDB running on localhost:27017
- Webcam or video file

### Install Dependencies

No new dependencies needed! The geofencing system uses only OpenCV (already installed).

If starting fresh:
```bash
cd ai_service
python -m venv venv

# Activate
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Step-by-Step Execution

### Phase 1: Draw Geofence Boundaries (One-time Setup)

**For each camera, define the "allowed zone" boundary:**

```bash
# Camera 1 - Entrance (webcam 0)
python boundary_setup.py --camera_id cam1 --source 0

# Camera 2 - Restricted Lab (webcam 1)
python boundary_setup.py --camera_id cam2 --source 1

# Camera 3 - Corridor (video file)
python boundary_setup.py --camera_id cam3 --source corridor.mp4
```

**Instructions while running:**
1. **Click** on the video to mark boundary points (minimum 3 points)
2. The polygon will appear in **green** as you add points
3. Press **'c'** to clear and start over
4. Press **'s'** to **save** the boundary
5. Press **'q'** to quit without saving

**Output:**
```
boundaries/
├── cam1_boundary.json
├── cam2_boundary.json
└── cam3_boundary.json
```

---

### Phase 2: Enroll Visitors

```bash
# Start the FastAPI service first
python app.py
```

Open another terminal and enroll visitors:

```bash
# Using curl
curl -X POST http://localhost:8000/enroll \
  -F "visitor_id=V001" \
  -F "name=Alice Johnson" \
  -F "category=visitor" \
  -F "file=@alice.jpg"

curl -X POST http://localhost:8000/enroll \
  -F "visitor_id=V002" \
  -F "name=Bob Smith" \
  -F "file=@bob.jpg"
```

Or use the Swagger UI: http://localhost:8000/docs

---

### Phase 3: Register Cameras

```bash
# Camera 1 - Entrance
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam1",
    "source_type": "webcam",
    "source_value": 0,
    "location": "Main Entrance",
    "zone_type": "general",
    "target_fps": 3
  }'

# Camera 2 - Restricted Area
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam2",
    "source_type": "webcam",
    "source_value": 1,
    "location": "Research Lab Entry",
    "zone_type": "restricted",
    "target_fps": 3
  }'
```

---

### Phase 4: Start Monitoring

```bash
# Start camera 1
curl -X POST http://localhost:8000/cameras/cam1/start

# Start camera 2
curl -X POST http://localhost:8000/cameras/cam2/start
```

**What's happening now:**
- Cameras are capturing frames at 3 FPS
- Faces are being detected and recognized
- Positions are checked against boundaries
- Timers start when visitors go outside boundaries
- Alerts fire after 60 seconds

---

### Phase 5: Test & Monitor

#### Test Scenario 1: Geofence Violation
1. Stand in front of webcam
2. **Move outside the green boundary** you drew
3. **Wait 60 seconds**
4. Check for alert:
   ```bash
   curl http://localhost:8000/alerts/geofence
   ```

Expected response:
```json
{
  "alerts": [
    {
      "visitor_id": "V001",
      "name": "Alice Johnson",
      "camera_id": "cam1",
      "event_type": "geofence_violation",
      "duration_seconds": 75,
      "position_x": 450,
      "position_y": 320,
      "timestamp": "2024-02-17T14:35:20Z"
    }
  ]
}
```

#### Test Scenario 2: Restricted Zone Entry
```bash
# Check if someone entered the restricted lab camera
curl http://localhost:8000/alerts?limit=10
```

#### View All Stats
```bash
curl http://localhost:8000/stats
```

---

## API Reference

### Geofencing Endpoints (NEW)

#### POST /cameras/{camera_id}/set-boundary
Manually set boundary via API (alternative to boundary_setup.py)

```bash
curl -X POST http://localhost:8000/cameras/cam1/set-boundary \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      [100, 100],
      [500, 100],
      [500, 400],
      [100, 400]
    ]
  }'
```

#### GET /cameras/{camera_id}/boundary
Get current boundary configuration

```bash
curl http://localhost:8000/cameras/cam1/boundary
```

Response:
```json
{
  "success": true,
  "camera_id": "cam1",
  "points": [[100,100], [500,100], [500,400], [100,400]],
  "num_points": 4
}
```

#### GET /alerts/geofence
Get only geofence violation alerts

```bash
curl http://localhost:8000/alerts/geofence?limit=20
```

---

## Configuration

### Environment Variables

```bash
# Geofence violation threshold (default: 60 seconds)
export GEOFENCE_VIOLATION_SECONDS=60

# How often to allow repeat alerts (default: 30 seconds)
export DUP_SUPPRESS_SECONDS=30

# Unknown person alert interval
export UNKNOWN_ALERT_SECONDS=45
```

---

## Troubleshooting

### Issue: Boundary not loading

**Check if file exists:**
```bash
ls boundaries/
```

**Manually load boundary at runtime:**
```python
# In Python console or script
from geofence_monitor import GeofenceMonitor
monitor = GeofenceMonitor()
monitor.load_boundary("cam1", "boundaries/cam1_boundary.json")
```

---

### Issue: No geofence alerts appearing

**Checklist:**
1. ✅ Boundary file exists in `boundaries/` folder
2. ✅ Camera is started: `POST /cameras/{id}/start`
3. ✅ Person is **outside** the boundary (not inside)
4. ✅ Waited full 60 seconds outside boundary
5. ✅ Face is being detected (check with `POST /recognize`)

**Debug: Check if face is being detected:**
```bash
# Upload a test image
curl -X POST http://localhost:8000/recognize \
  -F "camera_id=test" \
  -F "file=@test_face.jpg"
```

---

### Issue: Camera not opening

**Error:** `Cannot open camera/video source: 0`

**Solution:**
```bash
# Test camera in Python
python -c "import cv2; cap = cv2.VideoCapture(0); print('Opened:', cap.isOpened())"

# If False, try index 1
python -c "import cv2; cap = cv2.VideoCapture(1); print('Opened:', cap.isOpened())"
```

Use the working index in your camera registration.

---

### Issue: Boundary drawing window not showing

**Linux users:** Install these packages:
```bash
sudo apt-get install python3-opencv libopencv-dev
```

**Windows users:** Make sure you're not running in a remote/SSH session (GUI needed).

---

## Demo Workflow Summary

```
1. Draw Boundaries
   └─ python boundary_setup.py --camera_id cam1 --source 0
   
2. Start Service
   └─ python app.py
   
3. Enroll Visitors
   └─ POST /enroll (upload photos)
   
4. Register Cameras
   └─ POST /cameras/register
   
5. Start Cameras
   └─ POST /cameras/{id}/start
   
6. Test Geofencing
   └─ Move outside boundary → wait 60s → check alerts
   
7. View Results
   └─ GET /alerts/geofence
   └─ GET /stats
```

---

## File Structure After Setup

```
ai_service/
├── app.py
├── recognition_engine.py
├── enrollment_manager.py
├── behavior_analyzer.py        (loitering removed)
├── camera_manager.py
├── tracker.py
├── geofence_monitor.py         ← NEW
├── boundary_setup.py           ← NEW
├── event_dispatcher.py
├── db.py
├── requirements.txt
├── README.md
├── GEOFENCING_GUIDE.md         ← This file
│
└── boundaries/                 ← Created by boundary_setup.py
    ├── cam1_boundary.json
    ├── cam2_boundary.json
    └── cam3_boundary.json
```

---

## Next Steps

1. **Test with your webcam** following Phase 1-5 above
2. **Adjust thresholds** if 60 seconds is too long/short
3. **Draw different boundaries** for different camera zones
4. **Integrate with frontend dashboard** (if building UI)
5. **Export alerts** to CSV for analysis

---

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify MongoDB is running: `mongosh` → `db.runCommand({ping:1})`
3. Check camera connection: `ls /dev/video*` (Linux) or Device Manager (Windows)
4. Review the troubleshooting section above

---





# 🚀 Simple Integration Guide - Smart VMS AI Module

**Crystal Clear Workflow: One-Time Setup + Web App Control**

---

## 📋 Overview

```
┌─────────────────────────────────────────────────────────┐
│  ONE-TIME SETUP (Command Line)                          │
│  Run ONCE at the beginning                              │
└─────────────────────────────────────────────────────────┘
    python boundary_setup.py --camera_id cam1 --source 1
    Creates: boundaries/cam1_boundary.json

┌─────────────────────────────────────────────────────────┐
│  EVERYTHING ELSE (React Web App)                        │
│  NO Swagger UI, NO command line                         │
└─────────────────────────────────────────────────────────┘
    ├─ Register Visitors (auto ID: V001, V002, V003...)
    ├─ View Alerts Dashboard
    ├─ Manage Cameras (start/stop)
    ├─ Track Visitors
    └─ View System Stats
```

---

## ⚙️ System Components

### **1. Python AI Service (Port 8000)**
- **NO Swagger UI** ✅ (disabled in app.py)
- Only REST API endpoints
- Called by C# backend

### **2. C# .NET Backend (Port 5260)**
- Bridge between React and Python AI
- Auto-generates visitor IDs (V001, V002, V003...)
- Handles file uploads

### **3. React Frontend (Port 3000)**
- All user interactions
- Camera capture (source 1)
- Dashboards and management

### **4. MongoDB (Port 27017)**
- Stores face embeddings
- Stores events and alerts

---

## 🎯 Step-by-Step Setup

### **Phase 1: Start All Services**

#### 1.1 Start MongoDB
```bash
# Windows
mongod --dbpath C:\data\db

# Linux/macOS
mongod --dbpath /data/db
```

#### 1.2 Start Python AI Service
```bash
cd ai_service
python app.py
```

**Expected output:**
```
🚀 Smart VMS AI Service starting...
✅ System ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Test it:**
```bash
curl http://localhost:8000/health
```

**Important:** NO Swagger UI will open. This is correct! ✅

#### 1.3 Start C# Backend
```bash
cd backend
dotnet run
```

#### 1.4 Start React App
```bash
cd new_svms
npm start
```

---

### **Phase 2: ONE-TIME Boundary Setup (Command Line)**

**Run this ONCE for each camera:**

```bash
cd ai_service
python boundary_setup.py --camera_id entrance_cam --source 1
```

**Instructions:**
1. Your **USB webcam (source 1)** will open
2. Click points on video to draw polygon boundary
3. Press **'s'** to save
4. File saved: `boundaries/entrance_cam_boundary.json`

**That's it! You never need to run this again unless you want to change the boundary.**

---

### **Phase 3: Everything Else via Web App**

#### 3.1 Register Camera
1. Open browser: `http://localhost:3000`
2. Navigate to: **Security → Camera Management**
3. Click **"Register New Camera"**
4. Fill in:
   - Camera ID: `entrance_cam`
   - Location: `Main Entrance`
   - Webcam Index: `1` (USB camera)
   - Zone Type: `General` or `Restricted`
5. Click **"Register Camera"**
6. Click **"Start"** button

**Done!** Camera is now monitoring.

#### 3.2 Register Visitor
1. Navigate to: **Security → Register Visitor**
2. Fill in visitor details (name, email, phone, etc.)
3. Step 3: Click **"Open Camera"**
   - USB webcam (source 1) opens automatically
4. Click **"Capture Photo"**
5. Review and submit

**What happens:**
- Visitor ID auto-generated: V001, V002, V003...
- Photo sent to AI service
- Face enrolled automatically
- Notification sent to host

#### 3.3 View Alerts
1. Navigate to: **Security → Alerts Dashboard**
2. See real-time alerts:
   - Geofence violations
   - Restricted zone entries
   - Unknown persons
   - Re-entry without exit
3. Auto-refreshes every 10 seconds

#### 3.4 Manage Cameras
1. Navigate to: **Security → Camera Management**
2. See all registered cameras
3. Start/Stop cameras with buttons
4. View system stats

---

## 📁 Files to Add to Your Project

### **React Components** (add to `new_svms/src/pages/`)
- `SecurityRegisterVisitor.jsx` (UPDATED - with camera capture)
- `SecurityAlertsDashboard.jsx` (NEW - view alerts)
- `SecurityCameraManagement.jsx` (NEW - manage cameras)

### **C# Backend** (add to `backend/Controllers/`)
- `VisitorsController.cs` (NEW - visitor + alerts endpoints)
- `CamerasController.cs` (INCLUDED in VisitorsController.cs)

### **Python AI Service** (already in `ai_service/`)
- `app.py` (UPDATED - Swagger UI disabled)
- `boundary_setup.py` (KEEP as separate tool)
- All other files unchanged

---

## 🔄 Complete Workflow Example

### **Scenario: Register and Monitor a Visitor**

```
Day 1 - Setup (ONE TIME):
├─ 1. Start all services (MongoDB, Python, C#, React)
├─ 2. Run: python boundary_setup.py --camera_id entrance_cam --source 1
├─ 3. Draw boundary, press 's' to save
└─ 4. Register camera via web app, click "Start"

Day 2 - Normal Operations (ALWAYS):
├─ 1. Open web app
├─ 2. Security registers visitor (auto ID: V001)
├─ 3. Webcam opens, captures photo
├─ 4. Visitor enrolled automatically
├─ 5. Camera detects visitor face
├─ 6. Tracks position relative to boundary
├─ 7. Alert if outside boundary >60s
└─ 8. Security sees alert in dashboard

Day 3+ - Just use the web app!
```

---

## 🎨 React Router Setup

Add these routes to your `App.js`:

```javascript
import SecurityRegisterVisitor from './pages/SecurityRegisterVisitor';
import SecurityAlertsDashboard from './pages/SecurityAlertsDashboard';
import SecurityCameraManagement from './pages/SecurityCameraManagement';

<Routes>
  <Route path="/security/register-visitor" element={<SecurityRegisterVisitor />} />
  <Route path="/security/alerts" element={<SecurityAlertsDashboard />} />
  <Route path="/security/cameras" element={<SecurityCameraManagement />} />
  {/* ... other routes ... */}
</Routes>
```

---

## 🔧 Configuration Summary

| Setting | Value | Where |
|---------|-------|-------|
| **Camera Source** | `1` (USB webcam) | Everywhere |
| **AI Service URL** | `http://localhost:8000` | C# controllers |
| **Backend URL** | `http://localhost:5260` | React components |
| **MongoDB URI** | `mongodb://localhost:27017` | Python app.py |
| **Visitor ID Format** | V001, V002, V003... | Auto-generated |
| **Swagger UI** | Disabled ✅ | app.py |
| **Boundary Tool** | Separate ✅ | Run once |

---

## ✅ Checklist Before Demo

- [ ] MongoDB running
- [ ] Python AI service running (port 8000)
- [ ] C# backend running (port 5260)
- [ ] React app running (port 3000)
- [ ] Boundary drawn (`boundary_setup.py` executed once)
- [ ] Camera registered via web app
- [ ] Camera started via web app
- [ ] Test visitor enrolled via web form
- [ ] Alerts appearing in dashboard

---

## 🐛 Troubleshooting

### **Issue: "Cannot connect to AI service"**
**Solution:** Check if Python service is running on port 8000
```bash
curl http://localhost:8000/health
```

### **Issue: "Webcam not opening in browser"**
**Solution:** Check browser permissions, use `https://` or `localhost`

### **Issue: "Visitor ID not auto-incrementing"**
**Solution:** Check if AI service `/visitors` endpoint is accessible

### **Issue: "Boundary not loading"**
**Solution:** Check if `boundaries/entrance_cam_boundary.json` exists

---

## 📊 API Endpoints Summary

### **From React → C# Backend**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/visitors/register` | POST | Register visitor + enroll face |
| `/api/visitors/alerts` | GET | Get all alerts |
| `/api/visitors/alerts/geofence` | GET | Get geofence alerts only |
| `/api/visitors/stats` | GET | Get system statistics |
| `/api/cameras/register` | POST | Register new camera |
| `/api/cameras/list` | GET | List all cameras |
| `/api/cameras/{id}/start` | POST | Start camera monitoring |
| `/api/cameras/{id}/stop` | POST | Stop camera monitoring |

### **From C# Backend → Python AI**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/enroll` | POST | Enroll visitor face |
| `/alerts` | GET | Get alerts |
| `/alerts/geofence` | GET | Get geofence alerts |
| `/visitors` | GET | Get enrolled visitors |
| `/stats` | GET | Get system stats |
| `/cameras/register` | POST | Register camera |
| `/cameras` | GET | List cameras |
| `/cameras/{id}/start` | POST | Start camera |
| `/cameras/{id}/stop` | POST | Stop camera |

---

## 🎯 Key Points to Remember

1. **boundary_setup.py is separate** — Run it ONCE, not via web app ✅
2. **NO Swagger UI** — Python service has NO UI ✅
3. **Everything via React** — Registration, alerts, camera management ✅
4. **Auto visitor IDs** — V001, V002, V003... automatically ✅
5. **Camera source = 1** — USB webcam everywhere ✅

---

## 🚀 You're Ready!

Your system is now fully integrated with ZERO command-line tools (except the one-time boundary setup). Everything is controlled through your React web app!

**Demo Flow:**
1. Open web app
2. Register visitor → camera opens → photo captured → enrolled automatically
3. Go to Camera Management → see camera status
4. Go to Alerts Dashboard → see real-time alerts
5. Impress your professors! 🎓
