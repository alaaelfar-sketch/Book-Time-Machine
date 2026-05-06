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
    original: np.ndarray = None
    preprocessed: np.ndarray = None

    damage_report: DamageReport = None
    restoration: RestorationResult = None

    inpaint_mask: np.ndarray = None

    ocr_original: OCRResult = None
    ocr_restored: OCRResult = None

    evaluation: EvaluationReport = None

    stage_images: dict[str, np.ndarray] = field(default_factory=dict)
    stage_labels: list[str] = field(default_factory=list)


# =========================
# 🧠 Master Pipeline
# =========================
class DocumentPipeline:

    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()

        self.analyzer = DamageAnalyzer()
        self.restorer = RestorationEngine(self.s)
        self.ocr = OCREngine(self.s)
        self.evaluator = Evaluator(self.s)

    # =========================
    # 🚀 RUN
    # =========================
    def run(self, image: np.ndarray) -> PipelineResult:

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        result = PipelineResult(original=image)

        # =========================
        # 1️⃣ Preprocessing
        # =========================
        result.preprocessed = preprocess_for_restoration(image)
        logger.info("Stage 1 — Preprocessing done")

        # =========================
        # 2️⃣ Damage Analysis (FIXED)
        # =========================
        result.damage_report = self.analyzer.analyze(result.preprocessed)

        result.inpaint_mask = self.analyzer.generate_inpaint_mask(
            result.damage_report
        )

        logger.info("Stage 2 — Damage analysis done")

        # =========================
        # 3️⃣ Restoration
        # =========================
        result.restoration = self.restorer.run(
            result.preprocessed,
            mask=result.inpaint_mask,
        )

        logger.info("Stage 3 — Restoration done")

        # =========================
        # 4️⃣ OCR
        # =========================
        result.ocr_original = self.ocr.extract(image)
        result.ocr_restored = self.ocr.extract(result.restoration.best_final)

        logger.info("Stage 4 — OCR done")

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
            "Stage 5 — Evaluation done | Score: %.1f",
            result.evaluation.final_score,
        )

        # =========================
        # 📊 Timeline
        # =========================
        self._build_stage_timeline(result)

        return result

    # =========================
    # 📊 Timeline
    # =========================
    def _build_stage_timeline(self, r: PipelineResult):

        r.stage_labels = [
            "0 - Original",
            "1 - Preprocessed",
            "2 - Denoised",
            "3 - Enhanced",
            "4 - Inpainted",
            "5 - Final",
        ]

        r.stage_images = {
            r.stage_labels[0]: r.original,
            r.stage_labels[1]: r.preprocessed,
            r.stage_labels[2]: r.restoration.best_denoised,
            r.stage_labels[3]: r.restoration.best_enhanced,
            r.stage_labels[4]: r.restoration.best_inpainted,
            r.stage_labels[5]: r.restoration.best_final,
        }