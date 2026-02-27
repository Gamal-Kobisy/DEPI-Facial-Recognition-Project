"""
Tests for preprocessing utilities.
"""

import numpy as np
import pytest

from src.data.preprocessing import normalize_image, augment_image


def make_dummy_image(h: int = 224, w: int = 224, c: int = 3, dtype=np.uint8) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, c), dtype=dtype)


class TestNormalizeImage:
    def test_output_range(self):
        img = make_dummy_image()
        result = normalize_image(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_is_float32(self):
        img = make_dummy_image()
        result = normalize_image(img)
        assert result.dtype == np.float32

    def test_shape_preserved(self):
        img = make_dummy_image(100, 150, 3)
        result = normalize_image(img)
        assert result.shape == (100, 150, 3)

    def test_all_zeros_input(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = normalize_image(img)
        assert np.all(result == 0.0)

    def test_all_255_input(self):
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = normalize_image(img)
        assert np.allclose(result, 1.0)


class TestAugmentImage:
    def test_returns_three_augmentations(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        img = make_dummy_image()
        result = augment_image(img)
        assert len(result) == 3

    def test_augmented_shape_matches_input(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        img = make_dummy_image(128, 128, 3)
        for aug in augment_image(img):
            assert aug.shape == img.shape

    def test_flip_is_different_from_original(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        # Use an asymmetric image so flip is guaranteed to differ
        img = make_dummy_image(64, 64, 3)
        img[:, :32] = 0
        img[:, 32:] = 255
        flipped = augment_image(img)[0]
        assert not np.array_equal(img, flipped)
