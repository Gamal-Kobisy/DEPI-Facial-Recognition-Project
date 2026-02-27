"""
Tests for model utilities.
"""

import pytest
import numpy as np


class TestComputeFar:
    def test_zero_far_all_correct(self):
        from src.models.train import compute_far
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_far(y_true, y_pred) == 0.0

    def test_full_far_all_wrong(self):
        from src.models.train import compute_far
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert compute_far(y_true, y_pred) == 1.0

    def test_partial_far(self):
        from src.models.train import compute_far
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_far(y_true, y_pred) == pytest.approx(0.5)

    def test_no_impostors_returns_zero(self):
        from src.models.train import compute_far
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert compute_far(y_true, y_pred) == 0.0


class TestBuildModel:
    def test_build_custom_cnn(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from src.models.model import build_model, FaceEmbeddingCNN
        model = build_model("custom-cnn")
        assert isinstance(model, FaceEmbeddingCNN)

    def test_custom_cnn_forward_pass(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from src.models.model import build_model
        model = build_model("custom-cnn", embedding_dim=64)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (2, 64)

    def test_custom_cnn_embeddings_l2_normalised(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        from src.models.model import build_model
        model = build_model("custom-cnn", embedding_dim=128)
        dummy = torch.randn(4, 3, 224, 224)
        out = model(dummy)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            from src.models.model import build_model
            build_model("unknown-model")
