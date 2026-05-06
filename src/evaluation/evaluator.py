"""
Final Evaluation Module — Book Time Machine

This module combines:
- Image quality metrics
- OCR performance metrics
- Damage analysis signals

into a single unified restoration score (0–100).

It acts as the "grading system" of the entire pipeline.
"""

from dataclasses import dataclass, field
import numpy as np

from config.constants import DAMAGE_TYPES
from config.settings import Settings

from src.evaluation.image_metrics import ImageMetrics
from src.evaluation.ocr_metrics import OCRMetrics
from src.core.damage_analysis import DamageReport

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =========================
# 📦 Final Evaluation Output
# =========================
@dataclass
class EvaluationReport:
    """
    Final evaluation summary for a processed document.
    """

    final_score: float = 0.0

    image_score: float = 0.0
    ocr_score: float = 0.0

    damage_reduction: float = 0.0

    details: dict = field(default_factory=dict)
    text_report: str = ""


# =========================
# 🧠 Evaluator Engine
# =========================
class Evaluator:
    """
    Combines multiple metrics into a single interpretable score.

    Weight distribution:
    - Image quality: 45%
    - OCR quality: 30%
    - Damage reduction: 25%
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()

    # -------------------------------------------------
    # 🚀 Main evaluation function
    # -------------------------------------------------
    def evaluate(
        self,
        img_m: ImageMetrics,
        ocr_m: OCRMetrics,
        damage: DamageReport
    ) -> EvaluationReport:

        r = EvaluationReport()

        # =========================
        # 🖼 Image Quality Score
        # =========================

        # Sharpness improvement ratio
        sharp_ratio = (
            img_m.sharpness_restored /
            (img_m.sharpness_original + 1e-10)
        )

        sharp_gain = min(1.0, max(0.0, sharp_ratio - 0.5))

        # Noise reduction normalized
        noise_ok = np.clip(img_m.noise_reduction, 0.0, 1.0)

        # Contrast closeness to ideal document range
        contrast_ok = max(
            0.0,
            1.0 - abs(img_m.contrast_improvement - 1.4) / 1.4
        )

        # Structural similarity
        ssim_ok = np.clip(img_m.ssim, 0.0, 1.0)

        r.image_score = float(
            np.mean([sharp_gain, noise_ok, contrast_ok, ssim_ok]) * 100
        )

        # =========================
        # 📄 OCR Quality Score
        # =========================

        conf_gain = np.clip(ocr_m.conf_improvement / 30.0, 0.0, 1.0)
        high_gain = np.clip(ocr_m.high_conf_gain, 0.0, 1.0)

        word_stability = 1.0 - (
            abs(ocr_m.word_count_change) /
            max(1, ocr_m.original_word_count)
        )
        word_stability = np.clip(word_stability, 0.0, 1.0)

        r.ocr_score = float(
            np.mean([conf_gain, high_gain, word_stability]) * 100
        )

        # =========================
        # 🌫 Damage Reduction Score
        # =========================

        # Heuristic: inferred from image improvements
        r.damage_reduction = float(
            np.mean([noise_ok, contrast_ok]) * 100
        )

        # =========================
        # 🧮 Final Weighted Score
        # =========================
        weights = {
            "image": 0.45,
            "ocr": 0.30,
            "damage": 0.25
        }

        r.final_score = (
            r.image_score * weights["image"] +
            r.ocr_score * weights["ocr"] +
            r.damage_reduction * weights["damage"]
        )

        r.final_score = float(np.clip(r.final_score, 0.0, 100.0))

        # =========================
        # 📊 Detailed Metrics
        # =========================
        r.details = {
            "PSNR (dB)": round(img_m.psnr, 2),
            "SSIM": round(img_m.ssim, 4),

            "Sharpness ratio": round(sharp_ratio, 2),
            "Noise reduction": round(img_m.noise_reduction, 3),
            "Contrast ratio": round(img_m.contrast_improvement, 2),

            "OCR confidence Δ": round(ocr_m.conf_improvement, 2),
            "OCR high-conf gain": round(ocr_m.high_conf_gain, 3),
        }

        # =========================
        # 🧾 Human-readable report
        # =========================
        r.text_report = self._format_report(r, damage)

        logger.info("Final Score: %.2f / 100", r.final_score)

        return r

    # -------------------------------------------------
    # 🧾 Report Formatter
    # -------------------------------------------------
    @staticmethod
    def _format_report(r: EvaluationReport, damage: DamageReport) -> str:
        """
        Generate human-readable evaluation summary.
        """

        lines = [
            "=" * 55,
            " BOOK TIME MACHINE — FINAL EVALUATION REPORT",
            "=" * 55,
            "",
            f" Final Score           : {r.final_score:.1f} / 100",
            f" Image Quality Score   : {r.image_score:.1f} / 100",
            f" OCR Quality Score     : {r.ocr_score:.1f} / 100",
            f" Damage Reduction      : {r.damage_reduction:.1f} %",
            "",
            " --- Damage Overview ---",
        ]

        for dtype in DAMAGE_TYPES:
            lines.append(
                f" {dtype:20s}: {damage.scores.get(dtype, 0):.1f}"
            )

        lines += [
            "",
            f" Global Severity       : {damage.global_severity:.1f} / 100",
            "",
            " --- Key Metrics ---",
        ]

        for k, v in r.details.items():
            lines.append(f" {k:24s}: {v}")

        lines += ["", "=" * 55]

        return "\n".join(lines)