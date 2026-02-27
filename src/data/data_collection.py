"""
Data collection utilities for the Facial Recognition System.

Supports loading standard datasets such as LFW (Labeled Faces in the Wild)
and VGGFace for training and evaluation.
"""

import os
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"


def get_raw_dir() -> Path:
    """Return the path to the raw data directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR


def list_identities(raw_dir: Path = RAW_DIR) -> list:
    """
    List all identity (person) folders found in the raw data directory.

    Each sub-folder is expected to correspond to one individual and contain
    one or more facial images.

    Parameters
    ----------
    raw_dir : Path
        Root directory that contains per-identity sub-folders.

    Returns
    -------
    list of str
        Sorted list of identity names found in *raw_dir*.
    """
    if not raw_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in raw_dir.iterdir()
        if entry.is_dir()
    )


def collect_image_paths(raw_dir: Path = RAW_DIR, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """
    Recursively collect all image file paths from *raw_dir*.

    Parameters
    ----------
    raw_dir : Path
        Root directory to search.
    extensions : tuple of str
        Allowed file extensions (lower-case).

    Returns
    -------
    list of Path
        Sorted list of image paths found under *raw_dir*.
    """
    if not raw_dir.exists():
        return []
    return sorted(
        p for p in raw_dir.rglob("*") if p.suffix.lower() in extensions
    )
