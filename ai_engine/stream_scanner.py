import cv2
import os
import requests
import numpy as np
import logging
import threading
import winsound
from flask import Flask, Response, jsonify
from flask_cors import CORS
from core_logic import FaceRecognitionCore
import time
import datetime
from collections import deque

# ==========================================
# CONFIG
# ==========================================
LAST_ALERT_TIME  = {}
COOLDOWN_SECONDS = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

AI_LOCK = threading.Lock()

CASCADE_PATH   = "../models/haarcascade_frontalface_default.xml"
BLACKLIST_PATH = "../data/blacklist_db/"
VISITORS_PATH  = "../data/visitors_db/"

# ┌─────────────────────────────────────────────────────┐
# │  الـ Blacklist threshold رُفع لـ 0.55 عشان الـ     │
# │  robust embedding أكثر مرونة → يتحمل صور مختلفة   │
# └─────────────────────────────────────────────────────┘
BLACKLIST_THRESHOLD = 0.55
VISITOR_THRESHOLD   = 0.55

BACKEND_ALERT_URL = "http://localhost:5000/api/alerts"
TELEGRAM_BOT_TOKEN = "8571098636:AAHyB4tjNeFbm3Wj9GduFV23voIXqwS5AJA"
TELEGRAM_CHAT_ID = "7321127646"

def trigger_hardware_and_social_alarms(suspect_name):
    """تشغيل صوت الإنذار وإرسال رسالة تليجرام في مسارات منفصلة لمنع تهنيج الكاميرا"""
    
    # 1. تشغيل صوت الويندوز
    def play_sound():
        try:
            winsound.Beep(2500, 1000) # تردد عالي لمدة ثانية
        except: pass
    
    # 2. إرسال رسالة تليجرام
    def send_telegram():
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        # جلب الوقت والتاريخ الحالي وتنسيقهم
        now = datetime.datetime.now()
        current_time = now.strftime("%I:%M:%S %p") # هيطلع بالشكل ده: 04:20:15 PM
        current_date = now.strftime("%Y-%m-%d")    # هيطلع بالشكل ده: 2026-04-10
        
        # تصميم الرسالة الجديد
        text = (
            f"🚨 *إنذار أمني عاجل* 🚨\n\n"
            f"⚠️ *تم تحديد مشتبه به (Suspect Identified)!*\n"
            f"👤 *الاسم:* {suspect_name}\n"
            f"🕒 *الساعة:* {current_time}\n"
            f"📅 *التاريخ:* {current_date}\n"
            f"📍 *الموقع:* الكاميرا الرئيسية (Smart Gate)\n\n"
            f"يرجى مراجعة الكاميرات واتخاذ اللازم فوراً!"
        )
        
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=5)
            logger.info(f"📱 Telegram Alert Sent for: {suspect_name}")
        except Exception as e:
            logger.error(f"Telegram failed: {e}")

    # تشغيلهم في الخلفية
    threading.Thread(target=play_sound, daemon=True).start()
    threading.Thread(target=send_telegram, daemon=True).start()

# Temporal tracker settings
CONFIRMATION_FRAMES  = 8
MAX_CENTROID_DISTANCE = 90
EMBEDDING_BUFFER_SIZE = 6

# ==========================================
# FACE TRACKER
# ==========================================
class FaceTracker:
    def __init__(self):
        self.tracks  = {}
        self.next_id = 0
        self.lock    = threading.Lock()

    def _centroid(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _dist(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def update(self, detections):
        with self.lock:
            current_centroids = [self._centroid(b) for b, _ in detections]
            matched_ids       = set()
            results           = []

            for i, (bbox, emb) in enumerate(detections):
                cx, cy   = current_centroids[i]
                best_id, best_dist = None, float('inf')

                for tid, track in self.tracks.items():
                    if tid in matched_ids:
                        continue
                    d = self._dist(track['centroid'], (cx, cy))
                    if d < best_dist and d < MAX_CENTROID_DISTANCE:
                        best_dist, best_id = d, tid

                if best_id is not None:
                    track = self.tracks[best_id]
                    track['centroid'] = (cx, cy)
                    track['hits']    += 1
                    track['missed']   = 0
                    if emb is not None:
                        track['embeddings'].append(emb)
                    matched_ids.add(best_id)
                    results.append((best_id, bbox, track))
                else:
                    new_id    = self.next_id
                    self.next_id += 1
                    emb_deque = deque(maxlen=EMBEDDING_BUFFER_SIZE)
                    if emb is not None:
                        emb_deque.append(emb)
                    self.tracks[new_id] = {
                        'centroid':   (cx, cy),
                        'hits':       1,
                        'missed':     0,
                        'embeddings': emb_deque,
                        'identity':   None,
                        'registered': False
                    }
                    results.append((new_id, bbox, self.tracks[new_id]))

            # امسح الـ tracks القديمة
            dead = [tid for tid, t in self.tracks.items() if t['missed'] > 10]
            for tid in dead:
                del self.tracks[tid]

            # زوّد missed للـ tracks اللي متشافتش
            for tid in self.tracks:
                if tid not in matched_ids:
                    self.tracks[tid]['missed'] += 1

            return results

    def get_avg_embedding(self, track):
        embeddings = list(track['embeddings'])
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)

# ==========================================
# INIT
# ==========================================
logger.info("Starting AI Engine...")
with AI_LOCK:
    ai_engine = FaceRecognitionCore(model_name="Facenet")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise FileNotFoundError(f"Haar cascade not found at '{CASCADE_PATH}'.")

os.makedirs(BLACKLIST_PATH, exist_ok=True)
os.makedirs(VISITORS_PATH,  exist_ok=True)

face_tracker = FaceTracker()

# ==========================================
# DATABASE LOADING
# ==========================================
def _extract_face_from_image(img_path):
    """
    يستخرج crop الوجه من الصورة.
    يجرب الـ Haar cascade الأول، لو مش لاقي وجه يرجع الصورة كلها.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    if len(faces) > 0:
        # اختار أكبر وجه (الأوضح في الصورة)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop_bgr   = img[y:y+h, x:x+w]
    else:
        # مش لاقي وجه بالـ cascade → استخدم الصورة كلها (ممكن تكون crop جاهز)
        logger.warning(f"No face detected in {os.path.basename(img_path)}, using full image.")
        crop_bgr = img

    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _base_name(filename):
    """
    يرجع الاسم الأساسي للشخص من اسم الملف.
    مثال: 'Ahmed_1.jpg' → 'Ahmed'
            'Ahmed_2.jpg' → 'Ahmed'
            'Ahmed.jpg'   → 'Ahmed'
    """
    name = os.path.splitext(filename)[0]
    # لو في underscore + رقم في الآخر → اشطبه
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def process_blacklist_images(folder_path):
    """
    يبني قاعدة بيانات الـ Blacklist مع دعم:
      - أكثر من صورة لنفس الشخص  (Ahmed_1, Ahmed_2, ...)
      - Robust embedding لكل صورة (multi-augmentation average)
      - تجميع كل embeddings الشخص في قائمة
    """
    # person_name → list of embeddings
    person_embeddings = {}

    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, img_name)
        name     = _base_name(img_name)

        face_rgb = _extract_face_from_image(img_path)
        if face_rgb is None:
            continue

        try:
            # استخدم الـ ROBUST embedding للـ Blacklist
            with AI_LOCK:
                embedding = ai_engine.generate_robust_embedding(face_rgb)

            if embedding is not None:
                if name not in person_embeddings:
                    person_embeddings[name] = []
                person_embeddings[name].append(embedding)
                logger.info(f"✅ Blacklist: {img_name} → '{name}' processed")
        except Exception as e:
            logger.error(f"Error processing blacklist {img_name}: {e}")

    # اعمل average لكل embeddings الشخص لو عنده أكثر من صورة
    final_db = {}
    for name, emb_list in person_embeddings.items():
        avg = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        final_db[name] = avg
        logger.info(f"📌 Blacklist '{name}': averaged {len(emb_list)} photo(s)")

    return final_db


def process_visitor_images(folder_path):
    """
    يبني قاعدة بيانات الـ Visitors (بدون robust — سرعة أهم هنا).
    """
    db = {}
    for root, dirs, files in os.walk(folder_path):
        for img_name in files:
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, img_name)
            face_rgb = _extract_face_from_image(img_path)
            if face_rgb is None:
                continue
            try:
                with AI_LOCK:
                    embedding = ai_engine.generate_embedding(face_rgb)
                if embedding is not None:
                    name    = os.path.splitext(img_name)[0]
                    db[name] = embedding
            except Exception as e:
                logger.error(f"Error processing visitor {img_name}: {e}")
    return db

# ==========================================
# SAVE & ALERT HELPERS
# ==========================================
def save_new_visitor(face_bgr, face_rgb, visitors_db):
    now          = datetime.datetime.now()
    today_folder = os.path.join(VISITORS_PATH, now.strftime("%Y-%m-%d"))
    os.makedirs(today_folder, exist_ok=True)

    unique_id = f"Visitor_{now.strftime('%H-%M-%S')}"
    cv2.imwrite(os.path.join(today_folder, f"{unique_id}.jpg"), face_bgr)

    with AI_LOCK:
        embedding = ai_engine.generate_embedding(face_rgb)
    if embedding is not None:
        visitors_db[unique_id] = embedding

    logger.info(f"✅ New Visitor Registered: {unique_id}")
    payload = {
        "identity":      unique_id,
        "threat_level":  "Safe",
        "distance_score": None,
        "type":          "new_visitor"
    }
    try:
        requests.post(BACKEND_ALERT_URL, json=payload, timeout=2)
    except Exception:
        pass
    return unique_id


def send_alert_to_web(name, distance):
    logger.warning(f"🚨 ALERT: Blacklisted individual detected — {name}")
    
    # <--- السطر الجديد: تشغيل الإنذار والتليجرام
    trigger_hardware_and_social_alarms(name) 

    payload = {
        "identity":      name,
        "threat_level":  "High",
        "distance_score": float(distance),
        "type":          "suspect"
    }
    try:
        requests.post(BACKEND_ALERT_URL, json=payload, timeout=2)
    except Exception as e:
        logger.error(f"Alert failed: {e}")

def send_visitor_alert_to_web(name, distance):
    logger.info(f"✅ ALERT: Known visitor returned — {name}")
    payload = {
        "identity":      name,
        "threat_level":  "Safe",
        "distance_score": float(distance) if distance is not None else None,
        "type":          "old_visitor"
    }
    try:
        requests.post(BACKEND_ALERT_URL, json=payload, timeout=2)
    except Exception as e:
        logger.error(f"Alert failed: {e}")

# ==========================================
# LOAD DBS
# ==========================================
logger.info("Loading databases into memory...")
GLOBAL_BLACKLIST_DB = process_blacklist_images(BLACKLIST_PATH)
GLOBAL_VISITORS_DB  = process_visitor_images(VISITORS_PATH)
logger.info(f"DB Loaded → Blacklist: {len(GLOBAL_BLACKLIST_DB)} | Known Visitors: {len(GLOBAL_VISITORS_DB)}")

# ==========================================
# VIDEO STREAMING
# ==========================================
# ==========================================
# VIDEO STREAMING
# ==========================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera (index 0).")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))

            detections = []
            for (x, y, w, h) in faces:
                face_bgr = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                with AI_LOCK:
                    emb = ai_engine.generate_embedding(face_rgb)
                detections.append(((x, y, w, h), emb))

            tracked = face_tracker.update(detections)

            for track_id, (x, y, w, h), track in tracked:
                face_bgr = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                # 🌟 [التعديل الجديد]: حساب دقة الصورة والاحتفاظ بأفضل وأوضح فريم
                gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()

                if track.get('best_score', -1) < sharpness:
                    track['best_score']    = sharpness
                    track['best_face_bgr'] = face_bgr.copy()
                    track['best_face_rgb'] = face_rgb.copy()

                # لو عنده identity → ارسمه مباشرة
                if track['identity'] is not None:
                    identity = track['identity']
                    color    = (0, 0, 255) if identity.startswith("BLACKLIST") else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, identity, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    continue

                avg_emb = face_tracker.get_avg_embedding(track)

                if avg_emb is None or len(track['embeddings']) < 2:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                    cv2.putText(frame, f"Scanning... ({track['hits']})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                    continue

                identity = None
                color    = (0, 255, 0)

                # 1. Blacklist
                for name, saved_emb in GLOBAL_BLACKLIST_DB.items():
                    match, dist = ai_engine.is_match(avg_emb, saved_emb, BLACKLIST_THRESHOLD)
                    if match:
                        identity         = f"BLACKLIST: {name}"
                        color            = (0, 0, 255)
                        track['identity'] = identity

                        current_time = time.time()
                        if name not in LAST_ALERT_TIME or \
                           (current_time - LAST_ALERT_TIME[name]) > COOLDOWN_SECONDS:
                            send_alert_to_web(name, dist)
                            LAST_ALERT_TIME[name] = current_time
                        break

                # 2. Known visitors
                if identity is None:
                    for name, saved_emb in GLOBAL_VISITORS_DB.items():
                        match, dist = ai_engine.is_match(avg_emb, saved_emb, VISITOR_THRESHOLD)
                        if match:
                            identity          = name
                            track['identity'] = identity
                            
                            current_time = time.time()
                            if name not in LAST_ALERT_TIME or \
                               (current_time - LAST_ALERT_TIME[name]) > COOLDOWN_SECONDS:
                                send_visitor_alert_to_web(name, dist)
                                LAST_ALERT_TIME[name] = current_time
                            break

                # 3. New visitor (بعد confirmation frames)
                if identity is None:
                    if track['hits'] >= CONFIRMATION_FRAMES and not track['registered']:
                        track['registered'] = True
                        
                        # 🌟 [التعديل الجديد]: تسجيل الزائر بأفضل صورة تم التقاطها مش بالصورة الحالية
                        best_bgr = track.get('best_face_bgr', face_bgr)
                        best_rgb = track.get('best_face_rgb', face_rgb)
                        
                        identity            = save_new_visitor(best_bgr, best_rgb, GLOBAL_VISITORS_DB)
                        track['identity']   = identity
                    else:
                        remaining = CONFIRMATION_FRAMES - track['hits']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                        cv2.putText(frame, f"Verifying ({remaining})", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                        continue

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, identity, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ret2, buffer = cv2.imencode(".jpg", frame)
            if not ret2:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()
        logger.info("Camera released.")

# ==========================================
# ROUTES
# ==========================================
@app.route("/reload_visitors", methods=["POST"])
def reload_visitors():
    global GLOBAL_VISITORS_DB
    logger.info("🔄 Reloading Visitors database from UI command...")
    GLOBAL_VISITORS_DB = process_visitor_images(VISITORS_PATH)

    # نفرمت الذاكرة المؤقتة للكاميرا عشان تقرأ الأسماء الجديدة
    with face_tracker.lock:
        for track in face_tracker.tracks.values():
            track['identity']   = None
            track['registered'] = False

    logger.info(f"✅ Visitors Reloaded: {len(GLOBAL_VISITORS_DB)} person(s)")
    return jsonify({"status": "success", "count": len(GLOBAL_VISITORS_DB)})

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/reload_blacklist", methods=["POST"])
def reload_blacklist():
    global GLOBAL_BLACKLIST_DB
    logger.info("🔄 Reloading Blacklist with robust embeddings...")
    GLOBAL_BLACKLIST_DB = process_blacklist_images(BLACKLIST_PATH)

    with face_tracker.lock:
        for track in face_tracker.tracks.values():
            track['identity']   = None
            track['registered'] = False

    logger.info(f"✅ Blacklist Reloaded: {len(GLOBAL_BLACKLIST_DB)} person(s)")
    return jsonify({"status": "success", "count": len(GLOBAL_BLACKLIST_DB)})


if __name__ == "__main__":
    logger.info("Video Stream Server running on http://localhost:5001/video_feed")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)