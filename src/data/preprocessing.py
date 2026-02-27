"""
Data preprocessing utilities for the Facial Recognition System.

Provides functions for:
- Resizing images to a target dimension (default 224×224 for FaceNet)
- Normalizing pixel values to [0, 1]
- Face detection and cropping using OpenCV Haar cascades
- Basic data augmentation (horizontal flip, rotation)
"""

import os
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


TARGET_SIZE = (224, 224)

# Path to OpenCV's pre-trained frontal face Haar cascade
_CASCADE_PATH = (
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if _CV2_AVAILABLE
    else ""
)


def resize_image(image: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """
    Resize *image* to *target_size* using bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input image array (H×W×C or H×W).
    target_size : tuple of int
        (width, height) of the output image.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required for resize_image.")
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to the range [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image array with dtype uint8 or float.

    Returns
    -------
    np.ndarray
        Float32 array with pixel values in [0, 1].
    """
    return image.astype(np.float32) / 255.0


def detect_and_crop_face(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> np.ndarray | None:
    """
    Detect the largest face in *image* and return the cropped region.

    Uses OpenCV's Haar cascade classifier. Returns ``None`` if no face
    is detected.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale input image.
    scale_factor : float
        Parameter specifying how much the image size is reduced at each
        image scale in the cascade detector.
    min_neighbors : int
        Minimum number of neighbor rectangles each candidate face must
        retain to be considered a valid detection.

    Returns
    -------
    np.ndarray or None
        Cropped face region (BGR), or ``None`` if no face was found.
    """
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required for detect_and_crop_face.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.CascadeClassifier(_CASCADE_PATH)
    faces = detector.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return image[y : y + h, x : x + w]


def augment_image(image: np.ndarray) -> list:
    """
    Apply basic augmentation transforms to *image*.

    Augmentations applied:
    - Horizontal flip
    - 15° clockwise rotation
    - 15° counter-clockwise rotation

    Parameters
    ----------
    image : np.ndarray
        Input image array (H×W×C).

    Returns
    -------
    list of np.ndarray
        List of augmented images (not including the original).
    """
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required for augment_image.")

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    augmented = []

    # Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # Rotations
    for angle in (15, -15):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented.append(rotated)

    return augmented


def preprocess_image(image: np.ndarray, target_size: tuple = TARGET_SIZE, detect_face: bool = True) -> np.ndarray | None:
    """
    Full preprocessing pipeline: face detection → resize → normalize.

    Parameters
    ----------
    image : np.ndarray
        Raw input image (BGR).
    target_size : tuple of int
        (width, height) for resizing.
    detect_face : bool
        Whether to run face detection and crop before resizing.

    Returns
    -------
    np.ndarray or None
        Preprocessed float32 image of shape (*target_size*, 3), or ``None``
        if face detection was requested but no face was found.
    """
    if detect_face:
        image = detect_and_crop_face(image)
        if image is None:
            return None

    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image
