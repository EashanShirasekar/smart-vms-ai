"""
Smart Visitor Recognition - Desktop Application
Real-time face recognition using webcam

CONTROLS:
E = Enroll Visitor
Q = Quit
"""

import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
import os
import json
import pickle
from datetime import datetime

class VisitorRecognitionApp:

    def __init__(self):
        self.data_dir = "visitor_data"
        self.embeddings_file = f"{self.data_dir}/embeddings.pkl"
        self.visitors_file = f"{self.data_dir}/visitors.json"
        self.photos_dir = f"{self.data_dir}/photos"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.photos_dir, exist_ok=True)

        self.embeddings = {}
        self.visitors = {}
        self.load_data()

        # MODEL SETTINGS
        self.model_name = "ArcFace"
        self.threshold = 0.52

        self.camera = None
        self.frame_skip = 4
        self.frame_count = 0

        self.last_faces = []
        self.last_results = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.window = "Smart Visitor Recognition"

    # ---------------- DATA ---------------- #

    def load_data(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as f:
                self.embeddings = pickle.load(f)

        if os.path.exists(self.visitors_file):
            with open(self.visitors_file, "r") as f:
                self.visitors = json.load(f)

    def save_data(self):
        with open(self.embeddings_file, "wb") as f:
            pickle.dump(self.embeddings, f)

        with open(self.visitors_file, "w") as f:
            json.dump(self.visitors, f, indent=2)

    # ---------------- FACE PIPELINE ---------------- #

    def detect_faces(self, frame):
        faces = RetinaFace.detect_faces(frame)
        detected = []

        if isinstance(faces, dict):
            for face in faces.values():
                if face["score"] > 0.9:
                    x1, y1, x2, y2 = face["facial_area"]
                    detected.append([x1, y1, x2-x1, y2-y1])
        return detected

    def generate_embedding(self, face_img):
        if face_img.size == 0:
            return None

        embeds = []

        for img in [face_img, cv2.convertScaleAbs(face_img, alpha=1.15, beta=10)]:
            obj = DeepFace.represent(
                img_path=img,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip"
            )
            embeds.append(np.array(obj[0]["embedding"]))

        return np.mean(embeds, axis=0)

    def cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize(self, face_img):
        query = self.generate_embedding(face_img)
        if query is None:
            return None

        best_name, best_dist = None, 1.0

        for name, emb in self.embeddings.items():
            dist = self.cosine_distance(query, emb)
            if dist < best_dist:
                best_name, best_dist = name, dist

        if best_dist <= self.threshold:
            return best_name, 1 - best_dist
        return None

    # ---------------- ENROLLMENT ---------------- #

    def enroll(self, frame, name):
        faces = self.detect_faces(frame)
        if len(faces) != 1:
            return False

        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]

        emb = self.generate_embedding(face_img)
        if emb is None:
            return False

        self.embeddings[name] = emb
        self.visitors[name] = {
            "name": name,
            "enrolled_at": datetime.now().isoformat()
        }

        cv2.imwrite(f"{self.photos_dir}/{name}.jpg", face_img)
        self.save_data()
        return True

    # ---------------- UI ---------------- #

    def draw_ui(self, frame, faces, results):
        h, w = frame.shape[:2]

        # Bottom bar
        cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "E = Enroll   Q = Quit", (10, h-12),
                    self.font, 0.6, (255, 255, 255), 2)

        for i, face in enumerate(faces):
            x, y, fw, fh = face
            result = results[i] if i < len(results) else None

            if result:
                name, conf = result
                color = (0, 255, 0)
                label = f"{name} ({conf:.0%})"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        self.font, 0.7, color, 2)

        return frame

    # ---------------- RUN ---------------- #

    def run(self):
        self.camera = cv2.VideoCapture(0)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            self.frame_count += 1

            if self.frame_count % self.frame_skip == 0:
                faces = self.detect_faces(frame)
                results = []

                for x, y, w, h in faces:
                    face_img = frame[y:y+h, x:x+w]
                    res = self.recognize(face_img)
                    results.append(res)

                self.last_faces = faces
                self.last_results = results

            frame = self.draw_ui(frame, self.last_faces, self.last_results)
            cv2.imshow(self.window, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q")]:
                break

            elif key in [ord("e"), ord("E")]:
                name = input("Enter visitor name: ").strip()
                if name:
                    print("Press SPACE to capture, ESC to cancel")
                    while True:
                        ret, f = self.camera.read()
                        cv2.imshow(self.window, f)
                        k = cv2.waitKey(1) & 0xFF
                        if k == 32:
                            self.enroll(f, name)
                            break
                        elif k == 27:
                            break

        self.camera.release()
        cv2.destroyAllWindows()

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    VisitorRecognitionApp().run()
