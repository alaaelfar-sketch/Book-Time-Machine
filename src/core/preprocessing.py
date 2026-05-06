"""Lightweight preprocessing applied before analysis and restoration."""

import cv2
import numpy as np

from src.utils.helpers import to_gray, to_uint8, ensure_3ch


# =========================
# 🔍 Analysis Preprocessing
# =========================
def preprocess_for_analysis(image: np.ndarray) -> np.ndarray:
    """
    Prepare image for analysis tasks such as:
    - Damage detection
    - Segmentation
    - Structure analysis

    Pipeline:
    1. Convert to grayscale
    2. Apply light Gaussian blur to reduce noise while preserving edges

    Returns:
        np.ndarray: Clean grayscale image
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input image for analysis preprocessing")

    gray = to_gray(image)

    # Light smoothing to reduce sensor noise while preserving text edges
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)

    return smoothed


# =========================
# 🎨 Restoration Preprocessing
# =========================
def preprocess_for_restoration(image: np.ndarray) -> np.ndarray:
    """
    Prepare image for restoration pipeline.

    Goal:
    Improve illumination consistency and enhance contrast
    while preserving original color information.

    Pipeline:
    1. Ensure 3-channel format
    2. Convert to LAB color space
    3. Normalize L (lightness) channel using percentile scaling
    4. Convert back to BGR

    Returns:
        np.ndarray: Preprocessed BGR image ready for restoration
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input image for restoration preprocessing")

    img = ensure_3ch(image.copy())

    # Convert to LAB color space (better for luminance correction)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_channel = lab[:, :, 0].astype(np.float64)

    # Percentile-based normalization for robust contrast enhancement
    p_low, p_high = np.percentile(l_channel, (1, 99))

    if p_high - p_low > 1e-3:
        l_channel = (l_channel - p_low) / (p_high - p_low)
        l_channel = l_channel * 245 + 5  # keep safe intensity range [0–255]

    lab[:, :, 0] = to_uint8(l_channel)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)