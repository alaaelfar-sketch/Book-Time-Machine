import cv2
import numpy as np


# =========================
# Helpers
# =========================
def to_gray(image):
    image = np.array(image)

    if len(image.shape) == 2:
        return image.astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)


# =========================
# 1. Noise Detection
# =========================
def detect_noise(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise = np.abs(lap)

    noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    return noise.astype(np.uint8)


# =========================
# 2. Faded Text Detection
# =========================
def detect_faded(gray):
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    faded = cv2.absdiff(gray, blur)

    faded = cv2.normalize(faded, None, 0, 255, cv2.NORM_MINMAX)
    return faded.astype(np.uint8)


# =========================
# 3. Stains / Broken Areas
# =========================
def detect_stains(gray):
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    kernel = np.ones((5, 5), np.uint8)
    stains = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return stains


# =========================
# 4. Full Damage Analysis
# =========================
def analyze_damage(image):

    gray = to_gray(image)

    noise_map = detect_noise(gray)
    faded_map = detect_faded(gray)
    stain_map = detect_stains(gray)

    noise = noise_map / 255.0
    faded = faded_map / 255.0
    stains = stain_map / 255.0

    damage_score_map = (0.3 * noise + 0.4 * faded + 0.3 * stains)

    heatmap = cv2.normalize(damage_score_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    return {
        "noise_map": noise_map,
        "faded_map": faded_map,
        "stain_map": stain_map,
        "damage_score_map": heatmap,
        "damage_score": float(np.mean(damage_score_map))
    }


# =========================
# 5. Heatmap Overlay
# =========================
def generate_heatmap(image, damage_map):

    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    damage_map = np.clip(damage_map, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(damage_map, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        image.astype(np.uint8), 0.6,
        heatmap, 0.4,
        0
    )

    return overlay.astype(np.uint8)