"""
Master Pipeline — Book Time Machine

Orchestrates the full document restoration workflow:

Image →
Preprocessing →
Damage Analysis →
Restoration →
OCR →
Evaluation →
Visualization Timeline
"""

from dataclasses import dataclass, field
import numpy as np

from config.settings import Settings

from src.core.preprocessing import preprocess_for_restoration
from src.core.damage_analysis import DamageAnalyzer, DamageReport
from src.core.restoration import RestorationEngine, RestorationResult
from src.core.ocr_engine import OCREngine, OCRResult

from src.evaluation.image_metrics import ImageMetrics
from src.evaluation.ocr_metrics import OCRMetrics
from src.evaluation.evaluator import Evaluator, EvaluationReport

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =========================
# 📦 Pipeline Output
# =========================
@dataclass
class PipelineResult:
    """
    Container for all outputs produced by the full pipeline.
    """

    original: np.ndarray = None
    preprocessed: np.ndarray = None

    damage_report: DamageReport = None
    restoration: RestorationResult = None

    inpaint_mask: np.ndarray = None

    ocr_original: OCRResult = None
    ocr_restored: OCRResult = None

    evaluation: EvaluationReport = None

    # Visualization timeline
    stage_images: dict[str, np.ndarray] = field(default_factory=dict)
    stage_labels: list[str] = field(default_factory=list)


# =========================
# 🧠 Master Pipeline
# =========================
class DocumentPipeline:
    """
    End-to-end document restoration system.

    Pipeline stages:
    1. Preprocessing
    2. Damage analysis
    3. Restoration
    4. OCR
    5. Evaluation
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()

        self.analyzer = DamageAnalyzer(self.s)
        self.restorer = RestorationEngine(self.s)
        self.ocr = OCREngine(self.s)
        self.evaluator = Evaluator(self.s)

    # -------------------------------------------------
    # 🚀 Main Execution
    # -------------------------------------------------
    def run(self, image: np.ndarray) -> PipelineResult:
        """
        Execute full pipeline on input document image.
        """

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        result = PipelineResult(original=image)

        # =========================
        # 1️⃣ Preprocessing
        # =========================
        result.preprocessed = preprocess_for_restoration(image)
        logger.info("Stage 1/5 — Preprocessing completed")

        # =========================
        # 2️⃣ Damage Analysis
        # =========================
        result.damage_report = self.analyzer.analyze(image)

        result.inpaint_mask = self.analyzer.generate_inpaint_mask(
            result.damage_report,
            threshold=80,
        )

        logger.info("Stage 2/5 — Damage analysis completed")

        # =========================
        # 3️⃣ Restoration
        # =========================
        result.restoration = self.restorer.run(
            result.preprocessed,
            mask=result.inpaint_mask,
        )

        logger.info("Stage 3/5 — Restoration completed")

        # =========================
        # 4️⃣ OCR
        # =========================
        result.ocr_original = self.ocr.extract(image)
        result.ocr_restored = self.ocr.extract(result.restoration.best_final)

        logger.info("Stage 4/5 — OCR completed")

        # =========================
        # 5️⃣ Evaluation
        # =========================
        img_metrics = ImageMetrics.compute(
            image,
            result.restoration.best_final,
        )

        ocr_metrics = OCRMetrics.compute(
            result.ocr_original,
            result.ocr_restored,
        )

        result.evaluation = self.evaluator.evaluate(
            img_metrics,
            ocr_metrics,
            result.damage_report,
        )

        logger.info(
            "Stage 5/5 — Evaluation completed | Final Score: %.1f/100",
            result.evaluation.final_score,
        )

        # =========================
        # 📊 Build visualization timeline
        # =========================
        self._build_stage_timeline(result)

        return result

    # -------------------------------------------------
    # 📊 Visualization builder
    # -------------------------------------------------
    def _build_stage_timeline(self, r: PipelineResult):
        """
        Create ordered stages for Streamlit visualization slider.
        """

        r.stage_labels = [
            "0 - Original",
            "1 - Preprocessed",
            "2 - Denoised (best)",
            "3 - Enhanced (best)",
            "4 - Sharpened (best)",
        ]

        r.stage_images = {
            r.stage_labels[0]: r.original,
            r.stage_labels[1]: r.preprocessed,
            r.stage_labels[2]: r.restoration.best_denoised,
            r.stage_labels[3]: r.restoration.best_enhanced,
            r.stage_labels[4]: r.restoration.best_sharpened,
        }

        # Optional stages
        if r.restoration.best_inpainted is not None:
            r.stage_labels.append("5 - Inpainted (best)")
            r.stage_images[r.stage_labels[-1]] = r.restoration.best_inpainted

        if r.restoration.best_final is not None:
            r.stage_labels.append("Final Output")
            r.stage_images[r.stage_labels[-1]] = r.restoration.best_final