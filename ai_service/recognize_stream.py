import cv2
import numpy as np
from deepface import DeepFace
from db import get_all_faces, log_event
from datetime import datetime

def is_live_face(frame, prev_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_std = np.std(gray)
    motion_level = 0
    if prev_gray is not None:
        motion_level = np.linalg.norm(gray.astype("float") - prev_gray.astype("float"))

    live = lap_var > 25 and brightness_std > 30 and motion_level > 1.5
    return live, gray, lap_var, brightness_std, motion_level

def recognize_face(frame, known_encodings, known_names):
    try:
        faces = DeepFace.represent(img_path=frame, model_name="ArcFace", enforce_detection=False)
    except Exception:
        return []
    results = []
    for face in faces:
        embedding = np.array(face["embedding"])
        distances = [np.linalg.norm(embedding - known) for known in known_encodings]
        min_dist = min(distances) if distances else 999
        name = "Unknown"
        if min_dist < 3.2:
            idx = distances.index(min_dist)
            name = known_names[idx]
        results.append((name, face["facial_area"], min_dist))
    return results

def run():
    all_faces = get_all_faces()
    if not all_faces:
        print("[ERROR] No faces in DB. Please register first.")
        return

    known_encodings = [np.array(f["embedding"]) for f in all_faces]
    known_names = [f["name"] for f in all_faces]

    print("[INFO] Starting webcam... Press Q to quit.")
    cap = cv2.VideoCapture(0)
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        live, gray, lap_var, brightness_std, motion_level = is_live_face(frame, prev_gray)
        prev_gray = gray

        if not live:
            cv2.putText(frame, "FAKE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        cv2.putText(frame, "LIVE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


        results = recognize_face(frame, known_encodings, known_names)

        for name, fa, dist in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (fa["x"], fa["y"]), (fa["x"] + fa["w"], fa["y"] + fa["h"]), color, 2)
            cv2.putText(frame, f"{name}", (fa["x"], fa["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            log_event({
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "camera": "webcam-0",
                "liveness": True
            })

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
