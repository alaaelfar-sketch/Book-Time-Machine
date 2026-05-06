"""
Global constants for Book Time Machine.

This file contains:
- Damage taxonomy
- OCR thresholds
- Visualization settings
- Model-free system constants
"""

# =========================
# 📁 File formats
# =========================
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


# =========================
# 🧩 Damage types
# =========================
DAMAGE_NOISE = "Noise"
DAMAGE_BLUR = "Blur"
DAMAGE_FADE = "Fading"
DAMAGE_STAIN = "Staining"
DAMAGE_TEXTURE = "Texture Degradation"

DAMAGE_TYPES = [
    DAMAGE_NOISE,
    DAMAGE_BLUR,
    DAMAGE_FADE,
    DAMAGE_STAIN,
    DAMAGE_TEXTURE,
]


# =========================
# 📊 Severity thresholds
# =========================
SEVERITY_LOW_MAX = 25
SEVERITY_MEDIUM_MAX = 55
SEVERITY_HIGH_MAX = 80
# >80 = CRITICAL


# =========================
# 📝 OCR thresholds
# =========================
OCR_CONF_HIGH = 80
OCR_CONF_MEDIUM = 50


# =========================
# 🎨 OCR visualization colors (BGR)
# =========================
OCR_COLOR_HIGH = (0, 200, 0)       # Green
OCR_COLOR_MEDIUM = (0, 200, 255)   # Yellow
OCR_COLOR_LOW = (0, 0, 255)        # Red


# =========================
# 🌡 Heatmap settings
# =========================
HEATMAP_COLORMAP = "jet"


# =========================
# 🔧 Restoration methods
# =========================
DENOISE_METHODS = ["median", "gaussian", "bilateral"]
CONTRAST_METHODS = ["histeq", "clahe"]
SHARPEN_METHODS = ["unsharp", "laplacian"]
INPAINT_METHODS = ["telea", "navier_stokes"]


# =========================
# 🧠 Analysis parameters
# =========================
LOCAL_BLOCK_SIZE = 32
DEFAULT_INPAINT_RADIUS = 5