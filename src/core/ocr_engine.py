"""
Classical OCR Engine using Tesseract.

This module extracts:
- Full text
- Word-level bounding boxes
- Confidence scores per word
- Aggregate OCR quality metrics

No deep learning models are used — only classical OCR (Tesseract).
"""

import numpy as np
from dataclasses import dataclass, field

from config.constants import OCR_CONF_HIGH, OCR_CONF_MEDIUM
from config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy import for Tesseract (safe fallback if not installed)
try:
    import pytesseract

    # 🔧 Fix: set Tesseract path manually (Windows)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    _tesseract_available = True
except ImportError:
    pytesseract = None
    _tesseract_available = False
    logger.warning("pytesseract not installed — OCR features disabled")


# =========================
# 📦 Word-level OCR result
# =========================
@dataclass
class WordResult:
    """
    Represents a single detected word with:
    - text
    - confidence score
    - bounding box coordinates
    """

    text: str
    confidence: float
    x: int
    y: int
    w: int
    h: int

    @property
    def level(self) -> str:
        """Convert confidence into categorical quality level."""
        if self.confidence >= OCR_CONF_HIGH:
            return "high"
        if self.confidence >= OCR_CONF_MEDIUM:
            return "medium"
        return "low"


# =========================
# 📦 Full OCR result
# =========================
@dataclass
class OCRResult:
    """
    Complete OCR output containing:
    - full extracted text
    - word-level results
    - confidence statistics
    """

    full_text: str = ""
    words: list[WordResult] = field(default_factory=list)

    mean_confidence: float = 0.0
    high_conf_ratio: float = 0.0

    available: bool = False


# =========================
# 🧠 OCR Engine
# =========================
class OCREngine:
    """
    Thin wrapper around Tesseract OCR.

    Responsibilities:
    - Extract text from image
    - Parse word-level metadata
    - Compute confidence statistics
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()
        self.available = _tesseract_available

    # -------------------------------------------------
    # 🚀 Main OCR API
    # -------------------------------------------------
    def extract(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on input image.

        Steps:
        1. Validate availability
        2. Run Tesseract OCR
        3. Parse word-level output
        4. Compute statistics
        """

        result = OCRResult(available=self.available)

        # -------------------------
        # Check OCR availability
        # -------------------------
        if not self.available:
            result.full_text = "[OCR unavailable — install pytesseract + tesseract]"
            return result

        # -------------------------
        # Run Tesseract OCR
        # -------------------------
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.s.TESSERACT_LANG,
                config=self.s.TESSERACT_CONFIG,
                output_type=pytesseract.Output.DICT,
            )

        except Exception as e:
            logger.error("Tesseract OCR failed: %s", e)
            result.full_text = f"[OCR error: {e}]"
            return result

        # -------------------------
        # Parse OCR output
        # -------------------------
        confidences = []

        for i in range(len(data["text"])):
            word = data["text"][i].strip()

            if not word:
                continue

            conf = float(data["conf"][i])
            confidences.append(conf)

            result.words.append(
                WordResult(
                    text=word,
                    confidence=conf,
                    x=data["left"][i],
                    y=data["top"][i],
                    w=data["width"][i],
                    h=data["height"][i],
                )
            )

        # -------------------------
        # Build full text
        # -------------------------
        result.full_text = " ".join(w.text for w in result.words)

        # -------------------------
        # Confidence statistics
        # -------------------------
        if confidences:
            result.mean_confidence = float(np.mean(confidences))

            result.high_conf_ratio = sum(
                1 for c in confidences if c >= OCR_CONF_HIGH
            ) / len(confidences)

        # -------------------------
        # Logging summary
        # -------------------------
        logger.info(
            "OCR completed | words=%d | mean_conf=%.1f | high_ratio=%.2f",
            len(result.words),
            result.mean_confidence,
            result.high_conf_ratio,
        )

        return result