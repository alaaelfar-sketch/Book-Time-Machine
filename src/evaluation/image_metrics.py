"""
Classical Image Quality Metrics Module

This module evaluates restoration quality using deterministic (non-learning) metrics.

Metrics included:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Contrast improvement
- Sharpness (Laplacian variance)
- Noise reduction estimate

All computations are fully classical CV-based.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from config.settings import Settings
from src.utils.helpers import to_gray

# Optional dependency (graceful fallback if missing)
try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None


# =========================
# 📦 Metrics Container
# =========================
@dataclass
class ImageMetrics:
    """
    Stores quantitative evaluation results of restoration quality.
    """

    psnr: float = 0.0
    ssim: float = 0.0

    contrast_improvement: float = 0.0

    sharpness_original: float = 0.0
    sharpness_restored: float = 0.0

    noise_reduction: float = 0.0

    # -------------------------------------------------
    # 🚀 Main computation function
    # -------------------------------------------------
    @staticmethod
    def compute(
        original: np.ndarray,
        restored: np.ndarray,
        settings: Settings | None = None
    ) -> "ImageMetrics":
        """
        Compute full image quality metrics between original and restored image.
        """

        s = settings or Settings()
        m = ImageMetrics()

        # Convert to grayscale for fair comparison
        g_orig = to_gray(original)
        g_rest = to_gray(restored)

        # =========================
        # 📊 PSNR
        # =========================
        mse = float(
            np.mean((g_orig.astype(np.float64) - g_rest.astype(np.float64)) ** 2)
        )

        m.psnr = float(
            10 * np.log10(s.PSNR_MAX_VAL ** 2 / (mse + 1e-10))
        )

        # =========================
        # 🧠 SSIM (structural similarity)
        # =========================
        if ssim_fn is not None:
            win = min(s.SSIM_WIN_SIZE, min(g_orig.shape) - 1)

            m.ssim = float(
                ssim_fn(g_orig, g_rest, win_size=win)
            )
        else:
            m.ssim = 0.0  # fallback if library not installed

        # =========================
        # 🎨 Contrast improvement
        # =========================
        std_orig = float(np.std(g_orig))
        std_rest = float(np.std(g_rest))

        m.contrast_improvement = std_rest / (std_orig + 1e-10)

        # =========================
        # 🔪 Sharpness (Laplacian variance)
        # =========================
        m.sharpness_original = float(
            cv2.Laplacian(
                g_orig,
                cv2.CV_64F,
                ksize=s.SHARPNESS_KERNEL
            ).var()
        )

        m.sharpness_restored = float(
            cv2.Laplacian(
                g_rest,
                cv2.CV_64F,
                ksize=s.SHARPNESS_KERNEL
            ).var()
        )

        # =========================
        # 🌫 Noise reduction estimate
        # =========================
        noise_orig = float(
            np.std(cv2.absdiff(g_orig, cv2.medianBlur(g_orig, 3)))
        )

        noise_rest = float(
            np.std(cv2.absdiff(g_rest, cv2.medianBlur(g_rest, 3)))
        )

        m.noise_reduction = 1.0 - (noise_rest / (noise_orig + 1e-10))

        return m