"""
Training utilities for the Facial Recognition System.

Handles:
- Dataset preparation (train/validation split)
- Training loop with early stopping
- Metric logging (accuracy, loss, FAR)
- Model checkpoint saving
"""

import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def compute_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the False Acceptance Rate (FAR).

    FAR = number of impostors incorrectly accepted /
          total number of impostor attempts.

    Parameters
    ----------
    y_true : np.ndarray of int
        Ground-truth labels (0 = genuine, 1 = impostor).
    y_pred : np.ndarray of int
        Predicted labels.

    Returns
    -------
    float
        FAR value in [0, 1].
    """
    impostor_mask = y_true == 1
    total_impostors = impostor_mask.sum()
    if total_impostors == 0:
        return 0.0
    false_acceptances = ((y_pred == 0) & impostor_mask).sum()
    return float(false_acceptances) / float(total_impostors)


def train(
    model,
    dataset,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    checkpoint_name: str = "best_model.pt",
) -> dict:
    """
    Train *model* on *dataset* and return training history.

    Parameters
    ----------
    model : torch.nn.Module
        The facial recognition model to train.
    dataset : torch.utils.data.Dataset
        Labelled dataset of (image_tensor, label) pairs.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Initial learning rate for the Adam optimiser.
    val_split : float
        Fraction of *dataset* to use for validation.
    checkpoint_name : str
        File name for the best-model checkpoint saved under ``models/``.

    Returns
    -------
    dict
        Training history with keys ``"train_loss"``, ``"val_loss"``,
        ``"val_accuracy"``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / train_size

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_loss /= val_size
        val_accuracy = correct / val_size

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / checkpoint_name)

    return history
