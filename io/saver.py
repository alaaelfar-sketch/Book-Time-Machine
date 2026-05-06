"""Image and text saving utilities."""

from pathlib import Path

import cv2

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_image(image, path: str, quality: int = 95) -> bool:
    """
    Save image to disk.

    Parameters
    ----------
    image : np.ndarray
        Image to save
    path : str
        Output path
    quality : int
        JPEG quality (0-100)

    Returns
    -------
    bool
        True if saved successfully
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    if ok:
        logger.info("Image saved → %s", path)

    return ok


def save_text(text: str, path: str) -> bool:
    """
    Save extracted text to a file.

    Parameters
    ----------
    text : str
        Text content
    path : str
        Output file path

    Returns
    -------
    bool
        True if successful
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        Path(path).write_text(text, encoding="utf-8")
        logger.info("Text saved → %s", path)
        return True
    except OSError as e:
        logger.error("Failed to save text: %s", e)
        return False