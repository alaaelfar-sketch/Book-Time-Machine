"""Image loading and validation utilities."""

import os
from pathlib import Path

import cv2
import numpy as np

from config.constants import SUPPORTED_FORMATS
from src.utils.logger import get_logger
from src.utils.helpers import resize_if_needed

logger = get_logger(__name__)


def validate_image(file_path: str) -> bool:
    """
    Validate that the image file exists and has a supported format.

    Parameters
    ----------
    file_path : str
        Path to the image file

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    p = Path(file_path)

    if not p.exists():
        logger.error("File not found: %s", file_path)
        return False

    if p.suffix.lower() not in SUPPORTED_FORMATS:
        logger.error("Unsupported format: %s", p.suffix)
        return False

    return True


def load_image(file_path: str, max_dim: int = 4000) -> np.ndarray | None:
    """
    Load an image from disk with validation and resizing.

    Parameters
    ----------
    file_path : str
        Path to image file
    max_dim : int
        Maximum allowed image dimension

    Returns
    -------
    np.ndarray | None
        Loaded BGR image or None if failed
    """
    if not validate_image(file_path):
        return None

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("OpenCV failed to read image: %s", file_path)
        return None

    image = resize_if_needed(image, max_dim)

    logger.info(
        "Image loaded successfully | name=%s | shape=%s",
        os.path.basename(file_path),
        image.shape,
    )

    return image