"""
Visualization module for:
1. Damage heatmap overlays
2. OCR confidence bounding-box visualization

Used in Streamlit UI for interpretability of results.
"""

import cv2
import numpy as np

from config.constants import (
    OCR_COLOR_HIGH,
    OCR_COLOR_MEDIUM,
    OCR_COLOR_LOW,
)

from src.utils.helpers import ensure_3ch
from src.core.ocr_engine import OCRResult


# =========================
# 🌡 Damage Heatmap Overlay
# =========================
def build_damage_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay a colored damage heatmap on the original image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image
    heatmap : np.ndarray
        Grayscale damage intensity map (0–255)
    alpha : float
        Blending factor between image and heatmap

    Returns
    -------
    np.ndarray
        Visual overlay image
    """

    # Convert heatmap to colored representation
    colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    base = ensure_3ch(image)

    # Blend original image with heatmap
    overlay = cv2.addWeighted(base, 1.0 - alpha, colored_map, alpha, 0)

    return overlay


# =========================
# 🧠 OCR Confidence Map
# =========================
def build_ocr_confidence_map(
    image: np.ndarray,
    ocr_result: OCRResult
) -> np.ndarray:
    """
    Draw OCR bounding boxes with confidence-based coloring.

    Color scheme:
    - Green  → High confidence
    - Yellow → Medium confidence
    - Red    → Low confidence
    """

    canvas = ensure_3ch(image.copy())

    for word in ocr_result.words:

        # -------------------------
        # Select color by confidence level
        # -------------------------
        if word.level == "high":
            color = OCR_COLOR_HIGH
        elif word.level == "medium":
            color = OCR_COLOR_MEDIUM
        else:
            color = OCR_COLOR_LOW

        x, y, w, h = word.x, word.y, word.w, word.h

        # -------------------------
        # Draw bounding box
        # -------------------------
        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # -------------------------
        # Confidence label text
        # -------------------------
        label = f"{word.confidence:.0f}"

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            1
        )

        # Background rectangle for text readability
        cv2.rectangle(
            canvas,
            (x, y - th - 4),
            (x + tw, y),
            color,
            -1
        )

        # Text rendering
        cv2.putText(
            canvas,
            label,
            (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return canvas


# =========================
# 🧩 Multi-overlay generator
# =========================
def build_individual_damage_overlays(
    image: np.ndarray,
    damage_maps: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Generate separate overlay for each damage type.

    Returns:
        dict: damage_type → colored overlay image
    """

    overlays = {}

    for dtype, dmap in damage_maps.items():
        overlays[dtype] = build_damage_heatmap_overlay(
            image,
            dmap,
            alpha=0.5
        )

    return overlays