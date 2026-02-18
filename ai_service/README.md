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
