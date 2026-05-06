"""Utility functions for image preprocessing and normalization."""

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale)

    Returns
    -------
    np.ndarray
        Grayscale image (uint8)
    """
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Clamp values to [0, 255] and convert to uint8.
    """
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_3ch(image: np.ndarray) -> np.ndarray:
    """
    Ensure image has 3 channels (BGR format).

    Converts grayscale images to BGR if needed.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def resize_if_needed(image: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize image if its largest dimension exceeds max_dim.

    Keeps aspect ratio unchanged.
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_to_0_255(image: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 255] range using min-max scaling.

    Useful for visualization of heatmaps.
    """
    mn, mx = image.min(), image.max()

    if mx - mn < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - mn) / (mx - mn)
    return to_uint8(normalized * 255)