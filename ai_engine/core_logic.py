import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FaceRecognitionCore:
    def __init__(self, model_name="Facenet"):
        """
        Initialize the core engine and load the selected model.
        FaceNet produces facial embeddings of 128 dimensions.
        """
        logger.info(f"Initializing {model_name} Core...")

        try:
            built = DeepFace.build_model(model_name)
            self.model = built.model if hasattr(built, "model") else built
        except Exception as e:
            logger.error(f"Failed to build model '{model_name}': {e}")
            raise

        self.target_size = (160, 160)  # FaceNet required input size
        logger.info(f"{model_name} Core is ready.")

    def preprocess_face(self, face_img):
        """
        Preprocess face image before inference:
        - Validates input is not None/empty
        - Resizes to 160x160
        - Normalizes pixel values to [0, 1]
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("preprocess_face received an empty or None image.")

        face_resized = cv2.resize(face_img, self.target_size)
        face_array = np.expand_dims(face_resized, axis=0).astype(np.float32) / 255.0
        return face_array

    def generate_embedding(self, face_img):
        """
        Convert a face image into a 128-D embedding (digital fingerprint).
        Returns None if the image is invalid instead of crashing.
        """
        try:
            preprocessed = self.preprocess_face(face_img)
            embedding = self.model.predict(preprocessed, verbose=0)[0]
            return embedding
        except Exception as e:
            logger.warning(f"generate_embedding failed: {e}")
            return None

    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine distance between two embeddings.
        Lower value = more similar faces.
        """
        if embedding1 is None or embedding2 is None:
            return 1.0
        return cosine(embedding1, embedding2)

    def is_match(self, embedding1, embedding2, threshold=0.25):
        """
        Decide whether two embeddings belong to the same person.
        Returns (bool match, float distance).
        """
        distance = self.compute_similarity(embedding1, embedding2)
        return distance < threshold, distance