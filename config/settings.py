"""
Configurable runtime settings for Book Time Machine.

This class controls all tunable parameters used across:
- Restoration pipeline
- Damage analysis
- OCR processing
"""

class Settings:
    """Central configuration container."""

    # =========================
    # 🧹 Denoising
    # =========================
    MEDIAN_KERNEL = 3
    GAUSSIAN_KERNEL = (3, 3)
    GAUSSIAN_SIGMA = 0

    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75

    # =========================
    # 🎨 Contrast
    # =========================
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)

    # =========================
    # ✨ Sharpening
    # =========================
    UNSHARP_SIGMA = 1.0
    UNSHARP_AMOUNT = 1.5
    LAPLACIAN_KERNEL_SIZE = 3
    LAPLACIAN_SCALE = 0.3

    # =========================
    # 🧪 Inpainting
    # =========================
    INPAINT_RADIUS = 5
    STAIN_THRESHOLD = 30

    # =========================
    # 🔍 Damage analysis
    # =========================
    NOISE_ESTIMATION_KERNEL = 3
    SALT_PEPPER_THRESHOLD = 10
    BLUR_LAPLACIAN_KERNEL = 3
    STAIN_MORPH_KERNEL = 15

    # =========================
    # 📊 Evaluation
    # =========================
    PSNR_MAX_VAL = 255.0
    SSIM_WIN_SIZE = 7
    SHARPNESS_KERNEL = 3

    # =========================
    # 📝 OCR
    # =========================
    TESSERACT_LANG = "eng"
    TESSERACT_CONFIG = "--psm 6"

    # =========================
    # ⚙️ Pipeline control
    # =========================
    AUTO_SELECT_BEST = True
    MAX_IMAGE_DIM = 4000

    # -------------------------
    # Convert to dict (safe)
    # -------------------------
    def to_dict(self) -> dict:
        return {
            k: getattr(self, k)
            for k in self.__class__.__dict__
            if not k.startswith("_") and not callable(getattr(self, k))
        }

    # -------------------------
    # Safe update
    # -------------------------
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)