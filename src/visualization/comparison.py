"""
Comparison utilities for Book Time Machine UI.

Provides:
1. Image grid builder for side-by-side comparison
2. Stage slider data preparation for Streamlit

Used to visualize:
- Original vs restored outputs
- Intermediate pipeline stages
"""

import numpy as np
import cv2

from src.utils.helpers import ensure_3ch


# =========================
# 🧱 Comparison Grid Builder
# =========================
def build_comparison_grid(
    images: dict[str, np.ndarray],
    cols: int = 3
) -> list[np.ndarray]:
    """
    Arrange labeled images into a grid format (row-wise concatenation).

    Steps:
    1. Normalize channel count
    2. Resize all images to smallest common size
    3. Group into rows
    4. Horizontally concatenate each row

    Returns:
        list of rows (each row is a single image)
    """

    if not images:
        return []

    # -------------------------
    # Determine minimum size
    # -------------------------
    min_h = min(img.shape[0] for img in images.values())
    min_w = min(img.shape[1] for img in images.values())

    resized = {}

    # -------------------------
    # Normalize images
    # -------------------------
    for label, img in images.items():
        img = ensure_3ch(img)

        if img.shape[:2] != (min_h, min_w):
            img = _resize_with_aspect(img, min_h, min_w)

        resized[label] = img

    labels = list(resized.keys())

    rows = []

    # -------------------------
    # Build grid rows
    # -------------------------
    for i in range(0, len(labels), cols):
        batch = labels[i:i + cols]
        row_imgs = [resized[l] for l in batch]

        # Pad row if needed (optional safety)
        if len(row_imgs) < cols:
            pad = np.zeros_like(row_imgs[0])
            row_imgs += [pad] * (cols - len(row_imgs))

        rows.append(np.hstack(row_imgs))

    return rows


# =========================
# 🎛 Streamlit slider data
# =========================
def build_stage_slider_data(
    stage_images: dict[str, np.ndarray],
    stage_labels: list[str]
) -> dict:
    """
    Prepare structured data for UI slider navigation.

    Returns:
        dict:
        {
            "labels": [...],
            "images": {label: image}
        }
    """

    return {
        "labels": stage_labels,
        "images": stage_images
    }


# =========================
# 📐 Resize helper
# =========================
def _resize_with_aspect(
    image: np.ndarray,
    max_h: int,
    max_w: int
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.

    Ensures all images align properly in comparison grid.
    """

    h, w = image.shape[:2]

    # Compute scale factor
    scale = min(max_h / h, max_w / w)

    # No upscaling
    if scale >= 1.0:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )