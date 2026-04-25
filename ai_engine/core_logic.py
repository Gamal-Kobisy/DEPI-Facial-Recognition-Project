import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FaceRecognitionCore:
    def __init__(self, model_name="Facenet"):
        logger.info(f"Initializing {model_name} Core...")
        try:
            built = DeepFace.build_model(model_name)
            self.model = built.model if hasattr(built, "model") else built
        except Exception as e:
            logger.error(f"Failed to build model '{model_name}': {e}")
            raise

        self.target_size = (160, 160)
        logger.info(f"{model_name} Core is ready.")

    # ──────────────────────────────────────────────
    # PREPROCESSING
    # ──────────────────────────────────────────────
    def preprocess_face(self, face_img):
        """Resize + normalize لـ FaceNet."""
        if face_img is None or face_img.size == 0:
            raise ValueError("preprocess_face received an empty or None image.")
        face_resized = cv2.resize(face_img, self.target_size)
        return np.expand_dims(face_resized, axis=0).astype(np.float32) / 255.0

    def _align_face(self, face_img):
        """
        محاذاة الوجه باستخدام كاشف العيون.
        لو العيون اتشافت → يدور الصورة عشان تبقى أفقية.
        لو لأ → يرجع الصورة زي ما هي.
        """
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        gray  = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        eyes  = eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))

        if len(eyes) < 2:
            return face_img  # مش قادر يشوف عيون → رجع الصورة كما هي

        # رتب العيون من اليسار لليمين
        eyes = sorted(eyes, key=lambda e: e[0])
        (x1, y1, w1, h1) = eyes[0]
        (x2, y2, w2, h2) = eyes[1]

        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

        # حساب زاوية الميل
        dY = cy2 - cy1
        dX = cx2 - cx1
        angle = np.degrees(np.arctan2(dY, dX))

        # تدوير الصورة
        h, w = face_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            face_img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    # ──────────────────────────────────────────────
    # SINGLE EMBEDDING (للكاميرا المباشرة - سريع)
    # ──────────────────────────────────────────────
    def generate_embedding(self, face_img):
        """
        Embedding سريع لصور الكاميرا الحية.
        """
        try:
            preprocessed = self.preprocess_face(face_img)
            return self.model.predict(preprocessed, verbose=0)[0]
        except Exception as e:
            logger.warning(f"generate_embedding failed: {e}")
            return None

    # ──────────────────────────────────────────────
    # ROBUST EMBEDDING (للـ Blacklist - دقيق)
    # ──────────────────────────────────────────────
    def generate_robust_embedding(self, face_img):
        """
        يولّد embedding قوي بأخذ متوسط عدة augmentations.
        مناسب لصور الـ Blacklist عشان يغطي أكبر نطاق ممكن.

        الـ augmentations:
          1. الصورة الأصلية بعد المحاذاة
          2. Mirror أفقي
          3. زيادة الإضاءة
          4. تقليل الإضاءة
          5. اقتصاص مركزي خفيف (central crop)
          6. تدوير طفيف +10°
          7. تدوير طفيف -10°
        """
        if face_img is None or face_img.size == 0:
            return None

        try:
            aligned   = self._align_face(face_img)
            variants  = self._build_augmentations(aligned)
            embeddings = []

            for variant in variants:
                try:
                    preprocessed = self.preprocess_face(variant)
                    emb = self.model.predict(preprocessed, verbose=0)[0]
                    embeddings.append(emb)
                except Exception:
                    continue

            if not embeddings:
                return None

            avg_embedding = np.mean(embeddings, axis=0)
            # normalize عشان cosine distance يشتغل صح
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            logger.info(f"Robust embedding built from {len(embeddings)} augmentations.")
            return avg_embedding

        except Exception as e:
            logger.warning(f"generate_robust_embedding failed: {e}")
            return None

    def _build_augmentations(self, face_img):
        """يرجع list من الـ augmentations للصورة الواحدة."""
        h, w = face_img.shape[:2]
        variants = []

        # 1. الصورة الأصلية
        variants.append(face_img.copy())

        # 2. Mirror أفقي
        variants.append(cv2.flip(face_img, 1))

        # 3. إضاءة أعلى
        bright = cv2.convertScaleAbs(face_img, alpha=1.3, beta=20)
        variants.append(bright)

        # 4. إضاءة أقل
        dark = cv2.convertScaleAbs(face_img, alpha=0.7, beta=-20)
        variants.append(dark)

        # 5. Central crop (5% من كل ناحية)
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        if margin_x > 0 and margin_y > 0:
            crop = face_img[margin_y:h-margin_y, margin_x:w-margin_x]
            if crop.size > 0:
                variants.append(crop)

        # 6. تدوير +10°
        M_pos = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
        rot_pos = cv2.warpAffine(face_img, M_pos, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        variants.append(rot_pos)

        # 7. تدوير -10°
        M_neg = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
        rot_neg = cv2.warpAffine(face_img, M_neg, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        variants.append(rot_neg)

        return variants

    # ──────────────────────────────────────────────
    # SIMILARITY & MATCHING
    # ──────────────────────────────────────────────
    def compute_similarity(self, embedding1, embedding2):
        """Cosine distance — أقل = أكثر تشابهاً."""
        if embedding1 is None or embedding2 is None:
            return 1.0
        return cosine(embedding1, embedding2)

    def is_match(self, embedding1, embedding2, threshold=0.25):
        """
        يقرر هل الـ embeddings لنفس الشخص.
        Returns: (bool match, float distance)
        """
        distance = self.compute_similarity(embedding1, embedding2)
        return distance < threshold, distance