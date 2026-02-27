"""
Flask web application for the Facial Recognition System.

Exposes two endpoints:
- POST /identify  – Identify a person from an uploaded facial image.
- POST /verify    – Verify whether two uploaded images show the same person.
"""

import io
import os
from pathlib import Path

import numpy as np

try:
    from flask import Flask, request, jsonify
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from src.data.preprocessing import preprocess_image

app = Flask(__name__) if _FLASK_AVAILABLE else None

# Lazy-loaded model; populated on first request
_model = None
_identity_labels: list = []


def _load_model():
    """Load the trained model from the models/ directory."""
    global _model
    if _model is not None:
        return _model

    try:
        import torch
        from src.models.model import FaceEmbeddingCNN

        models_dir = Path(__file__).resolve().parents[2] / "models"
        checkpoint = models_dir / "best_model.pt"
        model = FaceEmbeddingCNN()
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        _model = model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    return _model


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes into an OpenCV BGR image array."""
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required.")
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")
    return image


if _FLASK_AVAILABLE:

    @app.route("/health", methods=["GET"])
    def health():
        """Health-check endpoint."""
        return jsonify({"status": "ok"})

    @app.route("/identify", methods=["POST"])
    def identify():
        """
        Identify a person from an uploaded facial image.

        Form field: ``image`` (multipart/form-data)

        Returns JSON with the predicted identity label and confidence score.
        """
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file_bytes = request.files["image"].read()
        try:
            raw = _decode_image(file_bytes)
            preprocessed = preprocess_image(raw)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422

        if preprocessed is None:
            return jsonify({"error": "No face detected in the image."}), 422

        try:
            import torch
            model = _load_model()
            tensor = torch.from_numpy(preprocessed.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                embedding = model(tensor)
            # Placeholder: return embedding norm as a proxy confidence
            confidence = float(embedding.norm().item())
            return jsonify({"identity": "unknown", "confidence": confidence})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/verify", methods=["POST"])
    def verify():
        """
        Verify whether two uploaded images show the same person.

        Form fields: ``image1``, ``image2`` (multipart/form-data)

        Returns JSON with a boolean ``same_person`` flag and a similarity score.
        """
        for field in ("image1", "image2"):
            if field not in request.files:
                return jsonify({"error": f"Missing field: {field}"}), 400

        try:
            img1 = _decode_image(request.files["image1"].read())
            img2 = _decode_image(request.files["image2"].read())
            pre1 = preprocess_image(img1)
            pre2 = preprocess_image(img2)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422

        if pre1 is None or pre2 is None:
            return jsonify({"error": "Face not detected in one or both images."}), 422

        try:
            import torch
            model = _load_model()
            t1 = torch.from_numpy(pre1.transpose(2, 0, 1)).unsqueeze(0)
            t2 = torch.from_numpy(pre2.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                e1 = model(t1)
                e2 = model(t2)
            # Cosine similarity
            similarity = float(torch.nn.functional.cosine_similarity(e1, e2).item())
            threshold = float(os.environ.get("VERIFICATION_THRESHOLD", "0.6"))
            return jsonify({"same_person": similarity >= threshold, "similarity": similarity})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
