import cv2
import numpy as np
from dataclasses import dataclass, field

from config.constants import (
    DAMAGE_NOISE,
    DAMAGE_BLUR,
    DAMAGE_FADE,
    DAMAGE_STAIN,
    DAMAGE_TEXTURE,
    SEVERITY_LOW_MAX,
    SEVERITY_MEDIUM_MAX,
    SEVERITY_HIGH_MAX,
)

from config.settings import Settings


# =========================
# 📦 Damage Report
# =========================
@dataclass
class DamageReport:
    scores: dict[str, float] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)

    combined_heatmap: np.ndarray | None = None
    global_severity: float = 0.0
    summary: str = ""


# =========================
# 🧠 Analyzer
# =========================
class DamageAnalyzer:

    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()

    # =========================
    # 🚀 MAIN
    # =========================
    def analyze(self, image: np.ndarray) -> DamageReport:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        report = DamageReport()

        # ===== ANALYSIS =====
        noise_map, noise_score = self._analyze_noise(gray)
        blur_score = self._analyze_blur(gray)
        fade_map, fade_score = self._analyze_fading(gray)
        stain_map, stain_score = self._analyze_stains(image)
        texture_map, texture_score = self._analyze_texture(gray)

        # ===== STORE =====
        report.maps = {
            DAMAGE_NOISE: noise_map,
            DAMAGE_BLUR: self._scalar_map(blur_score, h, w),
            DAMAGE_FADE: fade_map,
            DAMAGE_STAIN: stain_map,
            DAMAGE_TEXTURE: texture_map,
        }

        report.scores = {
            DAMAGE_NOISE: noise_score,
            DAMAGE_BLUR: blur_score,
            DAMAGE_FADE: fade_score,
            DAMAGE_STAIN: stain_score,
            DAMAGE_TEXTURE: texture_score,
        }

        # ===== COMBINED =====
        combined = np.zeros((h, w), dtype=np.float32)

        for m in report.maps.values():
            combined += m.astype(np.float32)

        combined /= len(report.maps)
        report.combined_heatmap = combined.astype(np.uint8)

        report.global_severity = float(np.mean(list(report.scores.values())))

        return report

    # =========================
    # 🎯 REQUIRED BY PIPELINE
    # =========================
    def generate_inpaint_mask(self, report: DamageReport, threshold: int = 80):
        """
        Create mask for restoration engine
        """
        heatmap = report.combined_heatmap

        mask = (heatmap > threshold).astype(np.uint8) * 255

        # clean mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    # =========================
    # 🔍 DAMAGE FUNCTIONS
    # =========================
    def _analyze_noise(self, gray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)

        score = float(np.mean(noise))
        return noise, score

    def _analyze_blur(self, gray):
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = lap.var()

        score = 100 - np.clip(var / 10, 0, 100)
        return float(score)

    def _analyze_fading(self, gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        spread = np.std(hist)

        fade_map = cv2.equalizeHist(gray)
        score = 100 - np.clip(spread / 10, 0, 100)

        return fade_map, float(score)

    def _analyze_stains(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        score = float(np.mean(thresh) / 2.55)

        return thresh, score

    def _analyze_texture(self, gray):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        mag = np.abs(sobel)

        score = float(np.mean(mag))
        return mag.astype(np.uint8), score

    # =========================
    # 🧩 HELPERS
    # =========================
    def _scalar_map(self, value, h, w):
        return np.full((h, w), int(np.clip(value, 0, 255)), dtype=np.uint8)

    def _severity_label(self, score):
        if score <= SEVERITY_LOW_MAX:
            return "LOW"
        if score <= SEVERITY_MEDIUM_MAX:
            return "MEDIUM"
        if score <= SEVERITY_HIGH_MAX:
            return "HIGH"
        return "CRITICAL"