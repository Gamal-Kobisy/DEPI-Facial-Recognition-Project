import cv2
import os
import uuid
import requests
import numpy as np
import logging
import threading
from flask import Flask, Response, jsonify
from flask_cors import CORS
from core_logic import FaceRecognitionCore
import time
import datetime

LAST_ALERT_TIME = {} 
COOLDOWN_SECONDS = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

AI_LOCK = threading.Lock()

CASCADE_PATH    = "../models/haarcascade_frontalface_default.xml"
BLACKLIST_PATH  = "../data/blacklist_db/"
VISITORS_PATH   = "../data/visitors_db/"

BLACKLIST_THRESHOLD = 0.40  
VISITOR_THRESHOLD   = 0.52  

BACKEND_ALERT_URL = "http://localhost:5000/api/alerts"

logger.info("Starting AI Engine...")

with AI_LOCK:
    ai_engine    = FaceRecognitionCore(model_name="Facenet")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise FileNotFoundError(f"Haar cascade not found at '{CASCADE_PATH}'.")

os.makedirs(BLACKLIST_PATH, exist_ok=True)
os.makedirs(VISITORS_PATH,  exist_ok=True)

def process_saved_images(folder_path):
    db = {}
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(path)
            if img is None: continue
            
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0: continue
            
            x, y, w, h   = faces[0]
            face_crop     = img[y:y+h, x:x+w]
            face_rgb      = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            name          = os.path.splitext(img_name)[0]
            
            with AI_LOCK:
                embedding = ai_engine.generate_embedding(face_rgb)
            
            if embedding is not None:
                db[name] = embedding
        except Exception as e:
            logger.error(f"Error processing {img_name}: {e}")
    return db

def save_new_visitor(face_bgr, face_rgb, visitors_db):
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")
    
    today_folder = os.path.join(VISITORS_PATH, today_date)
    os.makedirs(today_folder, exist_ok=True)
    
    unique_id = f"Visitor_{current_time}"
    save_path = os.path.join(today_folder, f"{unique_id}.jpg")
    cv2.imwrite(save_path, face_bgr)
    
    with AI_LOCK:
        embedding = ai_engine.generate_embedding(face_rgb)
        
    if embedding is not None:
        visitors_db[unique_id] = embedding
        
    logger.info(f"✅ New Visitor Registered: {unique_id}")
    
    payload = {
        "identity": unique_id,
        "threat_level": "Safe",
        "distance_score": None,
        "type": "visitor"
    }
    try:
        requests.post(BACKEND_ALERT_URL, json=payload, timeout=2)
    except Exception:
        pass
        
    return unique_id

def send_alert_to_web(name, distance):
    logger.warning(f"🚨 ALERT: Blacklisted individual detected — {name}")
    payload = {
        "identity": name,
        "threat_level": "High",
        "distance_score": float(distance),
        "type": "threat"
    }
    try:
        requests.post(BACKEND_ALERT_URL, json=payload, timeout=2)
    except Exception as e:
        logger.error(f"Alert failed: {e}")

logger.info("Loading databases into memory...")
GLOBAL_BLACKLIST_DB = process_saved_images(BLACKLIST_PATH)
GLOBAL_VISITORS_DB  = process_saved_images(VISITORS_PATH)
logger.info(f"DB Loaded → Blacklist: {len(GLOBAL_BLACKLIST_DB)} | Known Visitors: {len(GLOBAL_VISITORS_DB)}")

# ==========================================
# 4. Video Streaming Generator
# ==========================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera (index 0).")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))

            for (x, y, w, h) in faces:
                face_crop_bgr = frame[y:y+h, x:x+w]
                face_rgb      = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                
                with AI_LOCK:
                    current_emb = ai_engine.generate_embedding(face_rgb)

                if current_emb is None: continue

                identity = None
                color    = (0, 255, 0)

                # 1. Check blacklist
                for name, saved_emb in GLOBAL_BLACKLIST_DB.items():
                    match, dist = ai_engine.is_match(current_emb, saved_emb, BLACKLIST_THRESHOLD)
                    if match:
                        identity = f"BLACKLIST: {name}"
                        color    = (0, 0, 255)
                        
                        current_time = time.time()
                        if name not in LAST_ALERT_TIME or (current_time - LAST_ALERT_TIME[name]) > COOLDOWN_SECONDS:
                            send_alert_to_web(name, dist)
                            LAST_ALERT_TIME[name] = current_time
                            
                        break

                # 2. Check known visitors
                if identity is None:
                    for name, saved_emb in GLOBAL_VISITORS_DB.items():
                        match, dist = ai_engine.is_match(current_emb, saved_emb, VISITOR_THRESHOLD)
                        if match:
                            identity = name
                            break

                # 3. Register as new visitor
                if identity is None:
                    identity = save_new_visitor(face_crop_bgr, face_rgb, GLOBAL_VISITORS_DB)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret: continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()
        logger.info("Camera released.")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/reload_blacklist", methods=["POST"])
def reload_blacklist():
    global GLOBAL_BLACKLIST_DB
    logger.info("🔄 Reloading Blacklist Database from folder...")
    GLOBAL_BLACKLIST_DB = process_saved_images(BLACKLIST_PATH)
    logger.info("✅ Blacklist Reloaded Successfully!")
    return jsonify({"status": "success", "count": len(GLOBAL_BLACKLIST_DB)})

if __name__ == "__main__":
    logger.info("Video Stream Server running on http://localhost:5001/video_feed")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)