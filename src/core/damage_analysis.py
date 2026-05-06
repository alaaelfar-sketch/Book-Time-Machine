import cv2
import numpy as np
from dataclasses import dataclass, field


# =========================
# 📦 Damage Report
# =========================
@dataclass
class DamageReport:
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    combined_mask: np.ndarray | None = None
    global_severity: float = 0.0


# =========================
# 🧠 Analyzer
# =========================
class DamageAnalyzer:

    def __init__(self):
        pass

    # =========================
    # 🚀 MAIN
    # =========================
    def analyze(self, image: np.ndarray) -> DamageReport:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # =========================
        # 1. STAINS (Improved)
        # =========================
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blur)
        _, stains_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # =========================
        # 2. NOISE
        # =========================
        denoised = cv2.medianBlur(gray, 5)
        noise = cv2.absdiff(gray, denoised)
        _, noise_mask = cv2.threshold(noise, 20, 255, cv2.THRESH_BINARY)

        # =========================
        # 3. FADED TEXT
        # =========================
        clahe = cv2.createCLAHE(2.0, (8, 8))
        enhanced = clahe.apply(gray)
        faded = cv2.absdiff(enhanced, gray)
        _, faded_mask = cv2.threshold(faded, 15, 255, cv2.THRESH_BINARY)

        # =========================
        # 🧠 Protect TEXT (CRITICAL)
        # =========================
        edges = cv2.Canny(gray, 50, 150)

        stains_mask = cv2.bitwise_and(stains_mask, cv2.bitwise_not(edges))
        noise_mask = cv2.bitwise_and(noise_mask, cv2.bitwise_not(edges))
        faded_mask = cv2.bitwise_and(faded_mask, cv2.bitwise_not(edges))

        # =========================
        # 🧩 Combine Masks
        # =========================
        combined = cv2.bitwise_or(stains_mask, noise_mask)
        combined = cv2.bitwise_or(combined, faded_mask)

        # تنظيف الماسك
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # =========================
        # 📊 Severity (simple but stable)
        # =========================
        severity = float(np.mean(combined) / 2.55)

        report = DamageReport()
        report.maps = {
            "stains": stains_mask,
            "noise": noise_mask,
            "faded": faded_mask,
        }
        report.combined_mask = combined
        report.global_severity = severity

        return report

    # =========================
    # 🎯 FINAL MASK
    # =========================
    def generate_inpaint_mask(self, report: DamageReport):
        return report.combined_mask