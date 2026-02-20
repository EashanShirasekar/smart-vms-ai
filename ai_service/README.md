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