"""Dataset structure and file management utilities."""

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dataset_dirs(base: str = "data") -> None:
    """
    Create required dataset directory structure if missing.

    Parameters
    ----------
    base : str
        Base dataset directory
    """
    dirs = [
        "raw/historical_docs",
        "raw/damage_dataset/blur",
        "raw/damage_dataset/fade",
        "raw/damage_dataset/noise",
        "raw/damage_dataset/stain",
        "raw/ocr_dataset/images",
        "raw/ocr_dataset/labels",
        "processed/restored",
        "processed/enhanced",
        "processed/heatmaps",
        "processed/text",
    ]

    for d in dirs:
        Path(base, d).mkdir(parents=True, exist_ok=True)

    logger.info("Dataset structure initialized at %s", base)


def list_images(directory: str) -> list[str]:
    """
    List all image files in a directory.

    Parameters
    ----------
    directory : str
        Target folder

    Returns
    -------
    list[str]
        Sorted list of image file paths
    """
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    p = Path(directory)

    if not p.is_dir():
        return []

    return sorted(
        str(f) for f in p.iterdir()
        if f.suffix.lower() in exts
    )