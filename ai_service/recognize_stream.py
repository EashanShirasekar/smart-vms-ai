import cv2
import numpy as np
from deepface import DeepFace
from db import get_all_faces, log_event
from datetime import datetime

def run():
    # Load embeddings from DB
    all_faces = get_all_faces()
    if not all_faces:
        print("[ERROR] No faces in DB. Please register first.")
        return

    known_encodings = [np.array(f["embedding"]) for f in all_faces]
    known_names = [f["name"] for f in all_faces]

    print("[INFO] Starting webcam... Press Q to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # DeepFace returns a list of detected faces with embeddings
        try:
            detected_faces = DeepFace.represent(
                img_path = frame,
                model_name = "Facenet",
                enforce_detection = False
            )
        except Exception as e:
            print("Detection error:", e)
            detected_faces = []

        for face in detected_faces:
            embedding = np.array(face["embedding"])
            distances = [np.linalg.norm(embedding - known) for known in known_encodings]
            min_dist = min(distances) if distances else 999
            name = "Unknown"

            if distances and min_dist < 0.9:  # tweak threshold if needed
                idx = distances.index(min_dist)
                name = known_names[idx]

            # Draw box if DeepFace provided region info
            if "facial_area" in face:
                fa = face["facial_area"]
                cv2.rectangle(frame,
                              (fa["x"], fa["y"]),
                              (fa["x"] + fa["w"], fa["y"] + fa["h"]),
                              (0, 255, 0), 2)
                cv2.putText(frame, name,
                            (fa["x"], fa["y"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

            log_event({
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "camera": "webcam-0"
            })

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
