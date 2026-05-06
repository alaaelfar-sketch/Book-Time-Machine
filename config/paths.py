"""
Filesystem paths for Book Time Machine project.
"""

class Paths:
    """Centralized path definitions."""

    # =========================
    # 📥 Raw data
    # =========================
    DATA_RAW = "data/raw"
    DATA_RAW_DOCS = "data/raw/historical_docs"
    DATA_RAW_DAMAGE = "data/raw/damage_dataset"
    DATA_RAW_OCR = "data/raw/ocr_dataset"

    # =========================
    # ⚙️ Processed data
    # =========================
    DATA_PROCESSED = "data/processed"
    DATA_PROCESSED_RESTORED = "data/processed/restored"
    DATA_PROCESSED_ENHANCED = "data/processed/enhanced"
    DATA_PROCESSED_HEATMAPS = "data/processed/heatmaps"
    DATA_PROCESSED_TEXT = "data/processed/text"

    # =========================
    # 📤 Outputs
    # =========================
    OUTPUTS = "outputs"
    OUTPUTS_IMAGES = "outputs/images"
    OUTPUTS_TEXT = "outputs/text"
    OUTPUTS_COMPARISONS = "outputs/comparisons"
    OUTPUTS_REPORTS = "outputs/reports"