import sys, face_recognition
import numpy as np
from db import save_face

if len(sys.argv) < 4:
    print("Usage: python register_face.py \"Name\" EMP_ID image.jpg")
    sys.exit(1)

name, emp_id, img_path = sys.argv[1], sys.argv[2], sys.argv[3]

img = face_recognition.load_image_file(img_path)
encs = face_recognition.face_encodings(img)

if not encs:
    print("No face found in image.")
    sys.exit(1)

embedding = encs[0]
save_face(name, emp_id, embedding)
print(f"[OK] Registered {name} with {img_path}")
