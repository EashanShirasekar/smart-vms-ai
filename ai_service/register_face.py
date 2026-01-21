import sys
import numpy as np
from deepface import DeepFace
from db import save_face

if len(sys.argv) < 4:
    print("Usage: python register_face.py \"Name\" EMP_ID image.jpg")
    sys.exit(1)

name, emp_id, img_path = sys.argv[1], sys.argv[2], sys.argv[3]

embedding_objs = DeepFace.represent(
    img_path=img_path,
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True
)

embedding = np.array(embedding_objs[0]["embedding"])
embedding = embedding / np.linalg.norm(embedding)

save_face(name, emp_id, embedding)
print(f"[OK] Registered {name} with ArcFace + side-profile support")
