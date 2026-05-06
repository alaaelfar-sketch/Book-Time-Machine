"""
Multi-Method Restoration Engine

This module applies multiple restoration techniques in parallel
to enable comparison and automatic best-selection.

Pipeline stages:
1. Denoising
2. Contrast Enhancement
3. Sharpening
4. Inpainting (optional)

Each stage runs multiple algorithms independently.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

from config.constants import (
    DENOISE_METHODS,
    CONTRAST_METHODS,
    SHARPEN_METHODS,
    INPAINT_METHODS,
)

from config.settings import Settings
from src.utils.helpers import to_gray, to_uint8, ensure_3ch
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =========================
# 📦 Restoration Output
# =========================
@dataclass
class RestorationResult:
    """
    Stores all candidate outputs from each restoration stage.
    """

    denoised: dict[str, np.ndarray] = field(default_factory=dict)
    enhanced: dict[str, np.ndarray] = field(default_factory=dict)
    sharpened: dict[str, np.ndarray] = field(default_factory=dict)
    inpainted: dict[str, np.ndarray] = field(default_factory=dict)

    # Best selected outputs
    best_denoised: np.ndarray | None = None
    best_enhanced: np.ndarray | None = None
    best_sharpened: np.ndarray | None = None
    best_inpainted: np.ndarray | None = None

    best_final: np.ndarray | None = None
    best_chain: list[str] = field(default_factory=list)


# =========================
# 🧠 Restoration Engine
# =========================
class RestorationEngine:
    """
    Classical multi-method restoration system.

    Key idea:
    Run multiple algorithms per stage and compare results
    using a sharpness-based heuristic.
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()

    # -------------------------------------------------
    # 🚀 Main Pipeline
    # -------------------------------------------------
    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> RestorationResult:
        """
        Execute full restoration pipeline.

        Steps:
        1. Denoising
        2. Contrast enhancement
        3. Sharpening
        4. Optional inpainting
        """

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        result = RestorationResult()

        # =========================
        # 🔇 Stage 1: Denoising
        # =========================
        result.denoised = self._denoise_all(image)
        result.best_denoised = self._auto_pick(result.denoised)

        # =========================
        # 🎨 Stage 2: Contrast Enhancement
        # =========================
        result.enhanced = self._enhance_all(result.best_denoised)
        result.best_enhanced = self._auto_pick(result.enhanced)

        # =========================
        # 🔪 Stage 3: Sharpening
        # =========================
        result.sharpened = self._sharpen_all(result.best_enhanced)
        result.best_sharpened = self._auto_pick(result.sharpened)

        # =========================
        # 🧩 Stage 4: Inpainting (optional)
        # =========================
        if mask is not None and np.any(mask > 0):
            result.inpainted = self._inpaint_all(result.best_sharpened, mask)
            result.best_inpainted = self._auto_pick(result.inpainted)
            result.best_final = result.best_inpainted

            result.best_chain = [
                self._best_name(result.denoised),
                self._best_name(result.enhanced),
                self._best_name(result.sharpened),
                self._best_name(result.inpainted),
            ]
        else:
            result.best_final = result.best_sharpened

            result.best_chain = [
                self._best_name(result.denoised),
                self._best_name(result.enhanced),
                self._best_name(result.sharpened),
            ]

        logger.info("Best restoration chain: %s", " → ".join(result.best_chain))
        return result

    # -------------------------------------------------
    # 🔇 Denoising Methods
    # -------------------------------------------------
    def _denoise_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run all denoising methods and return results."""
        return {
            m: getattr(self, f"_denoise_{m}")(image)
            for m in DENOISE_METHODS
        }

    def _denoise_median(self, image: np.ndarray) -> np.ndarray:
        """Median filtering for salt & pepper noise removal."""
        k = self.s.MEDIAN_KERNEL
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(image, k)

    def _denoise_gaussian(self, image: np.ndarray) -> np.ndarray:
        """Gaussian blur for general noise smoothing."""
        return cv2.GaussianBlur(image, self.s.GAUSSIAN_KERNEL, self.s.GAUSSIAN_SIGMA)

    def _denoise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filtering."""
        return cv2.bilateralFilter(
            image,
            self.s.BILATERAL_D,
            self.s.BILATERAL_SIGMA_COLOR,
            self.s.BILATERAL_SIGMA_SPACE,
        )

    # -------------------------------------------------
    # 🎨 Contrast Enhancement
    # -------------------------------------------------
    def _enhance_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run all contrast enhancement methods."""
        return {
            m: getattr(self, f"_enhance_{m}")(image)
            for m in CONTRAST_METHODS
        }

    def _enhance_histeq(self, image: np.ndarray) -> np.ndarray:
        """Histogram equalization."""
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)

        channels = cv2.split(image)
        eq = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq)

    def _enhance_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement (best for faded documents)."""
        clahe = cv2.createCLAHE(
            clipLimit=self.s.CLAHE_CLIP_LIMIT,
            tileGridSize=self.s.CLAHE_TILE_SIZE,
        )

        if len(image.shape) == 2:
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -------------------------------------------------
    # 🔪 Sharpening
    # -------------------------------------------------
    def _sharpen_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run all sharpening methods."""
        return {
            m: getattr(self, f"_sharpen_{m}")(image)
            for m in SHARPEN_METHODS
        }

    def _sharpen_unsharp(self, image: np.ndarray) -> np.ndarray:
        """Unsharp masking technique."""
        blurred = cv2.GaussianBlur(image, (0, 0), self.s.UNSHARP_SIGMA)
        result = cv2.addWeighted(
            image,
            1.0 + self.s.UNSHARP_AMOUNT,
            blurred,
            -self.s.UNSHARP_AMOUNT,
            0,
        )
        return to_uint8(result)

    def _sharpen_laplacian(self, image: np.ndarray) -> np.ndarray:
        """Laplacian-based sharpening."""
        gray = to_gray(image)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.s.LAPLACIAN_KERNEL_SIZE)
        lap = to_uint8(np.abs(lap))

        if len(image.shape) == 3:
            lap = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

        result = cv2.addWeighted(image, 1.0, lap, self.s.LAPLACIAN_SCALE, 0)
        return to_uint8(result)

    # -------------------------------------------------
    # 🧩 Inpainting
    # -------------------------------------------------
    def _inpaint_all(self, image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Run all inpainting methods."""
        return {
            m: getattr(self, f"_inpaint_{m}")(image, mask)
            for m in INPAINT_METHODS
        }

    def _inpaint_telea(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Telea inpainting algorithm."""
        return cv2.inpaint(image, mask, self.s.INPAINT_RADIUS, cv2.INPAINT_TELEA)

    def _inpaint_navier_stokes(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Navier-Stokes inpainting algorithm."""
        return cv2.inpaint(image, mask, self.s.INPAINT_RADIUS, cv2.INPAINT_NS)

    # -------------------------------------------------
    # 🧠 Auto-selection (Best Candidate)
    # -------------------------------------------------
    def _auto_pick(self, candidates: dict[str, np.ndarray]) -> np.ndarray:
        """
        Select best result based on sharpness (Laplacian variance).
        Higher variance = sharper image.
        """

        best_img = None
        best_score = -1.0

        for _, img in candidates.items():
            gray = to_gray(img)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if score > best_score:
                best_score = score
                best_img = img

        return best_img

    @staticmethod
    def _best_name(candidates: dict[str, np.ndarray]) -> str:
        """Return method name with highest sharpness score."""
        gray_map = {n: to_gray(img) for n, img in candidates.items()}
        return max(
            gray_map,
            key=lambda n: cv2.Laplacian(gray_map[n], cv2.CV_64F).var()
        )