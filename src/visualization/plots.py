"""
Image Visualization Utility

This module converts NumPy images (BGR or grayscale)
into byte-encoded format suitable for Streamlit or web display.

Main purpose:
- Prepare images for st.image()
- Ensure consistent encoding across pipeline outputs
"""

import cv2
import numpy as np


# =========================
# 🖼 Image Encoder
# =========================
def render_image(image: np.ndarray, format: str = "png") -> np.ndarray:
    """
    Convert a NumPy image into encoded image bytes.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale uint8)
    format : str
        Output format (png, jpg, etc.)

    Returns
    -------
    bytes
        Encoded image buffer suitable for display or streaming
    """

    # -------------------------
    # Validate input
    # -------------------------
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input: expected numpy.ndarray")

    # -------------------------
    # Ensure 3-channel format
    # -------------------------
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # -------------------------
    # Encode image
    # -------------------------
    success, buffer = cv2.imencode(f".{format}", image)

    if not success:
        raise RuntimeError("cv2.imencode failed to encode image")

    # Return raw bytes (important for Streamlit / web apps)
    return buffer.tobytes()