from pymongo import MongoClient

# Connect to MongoDB (adjust URL if needed)
client = MongoClient("mongodb://localhost:27017/")
db = client["smart_vms"]

faces_col = db["faces"]
logs_col = db["ai_logs"]

def save_face(name, emp_id, embedding):
    faces_col.insert_one({
        "name": name,
        "employee_id": emp_id,
        "embedding": embedding.tolist()
    })

def get_all_faces():
    return list(faces_col.find({}))

def log_event(event):
    logs_col.insert_one(event)
