"""
OCR Quality Metrics Module

This module evaluates OCR performance by comparing:
- Original document OCR output
- Restored document OCR output

It focuses on:
- Confidence improvements
- Word detection changes
- High-confidence ratio improvement

No external ML models used — purely statistical comparison.
"""

import numpy as np
from dataclasses import dataclass

from src.core.ocr_engine import OCRResult


# =========================
# 📦 OCR Metrics Container
# =========================
@dataclass
class OCRMetrics:
    """
    Stores OCR evaluation results comparing original vs restored images.
    """

    conf_improvement: float = 0.0
    high_conf_gain: float = 0.0

    word_count_change: int = 0

    original_word_count: int = 0
    restored_word_count: int = 0

    original_mean_conf: float = 0.0
    restored_mean_conf: float = 0.0

    # -------------------------------------------------
    # 🚀 Main computation
    # -------------------------------------------------
    @staticmethod
    def compute(orig: OCRResult, rest: OCRResult) -> "OCRMetrics":
        """
        Compare OCR outputs from original and restored images.

        Returns:
            OCRMetrics object with improvement statistics.
        """

        m = OCRMetrics()

        # =========================
        # 📊 Confidence metrics
        # =========================
        m.original_mean_conf = float(orig.mean_confidence)
        m.restored_mean_conf = float(rest.mean_confidence)

        m.conf_improvement = (
            m.restored_mean_conf - m.original_mean_conf
        )

        # =========================
        # 🟢 High-confidence ratio gain
        # =========================
        m.high_conf_gain = (
            float(rest.high_conf_ratio) - float(orig.high_conf_ratio)
        )

        # =========================
        # 🧾 Word statistics
        # =========================
        m.original_word_count = len(orig.words)
        m.restored_word_count = len(rest.words)

        m.word_count_change = (
            m.restored_word_count - m.original_word_count
        )

        return m