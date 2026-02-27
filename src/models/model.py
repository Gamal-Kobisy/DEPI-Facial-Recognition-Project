"""
Model architectures for the Facial Recognition System.

Provides a factory function to load or build facial recognition models:
- FaceNet (via facenet-pytorch)
- VGG-Face (via DeepFace)
- Custom lightweight CNN for experimentation
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom lightweight CNN
# ---------------------------------------------------------------------------

class FaceEmbeddingCNN(nn.Module if _TORCH_AVAILABLE else object):
    """
    A simple CNN that maps a 224×224 RGB image to a 128-dimensional
    embedding vector.

    This custom architecture is intended for experimentation and baseline
    comparisons.  For production use, prefer fine-tuning FaceNet or
    VGG-Face.
    """

    def __init__(self, embedding_dim: int = 128):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use FaceEmbeddingCNN.")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x)
        # L2-normalise embeddings
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("facenet", "vgg-face", "custom-cnn")


def build_model(model_name: str = "custom-cnn", embedding_dim: int = 128, pretrained: bool = True):
    """
    Build and return a facial recognition model.

    Parameters
    ----------
    model_name : str
        One of ``"facenet"``, ``"vgg-face"``, or ``"custom-cnn"``.
    embedding_dim : int
        Size of the output embedding vector (used for ``"custom-cnn"``).
    pretrained : bool
        Whether to load pre-trained weights where applicable.

    Returns
    -------
    model
        An instantiated model object.

    Raises
    ------
    ValueError
        If *model_name* is not one of the supported values.
    ImportError
        If the required library for the requested model is not installed.
    """
    name = model_name.lower()

    if name == "custom-cnn":
        return FaceEmbeddingCNN(embedding_dim=embedding_dim)

    if name == "facenet":
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise ImportError("facenet-pytorch is required for the FaceNet model.") from exc
        return InceptionResnetV1(pretrained="vggface2" if pretrained else None).eval()

    if name == "vgg-face":
        try:
            from deepface import DeepFace  # noqa: F401 – validate availability
        except ImportError as exc:
            raise ImportError("deepface is required for the VGG-Face model.") from exc
        # DeepFace wraps model building; return the model name as a handle for
        # the training / inference code to pass back to DeepFace.
        return "VGG-Face"

    raise ValueError(f"Unknown model '{model_name}'. Choose from: {SUPPORTED_MODELS}")
