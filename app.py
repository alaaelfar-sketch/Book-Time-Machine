"""
📚 Book Time Machine — Streamlit Interface
===========================================
Interactive document restoration + analysis dashboard.
UI layer only (no heavy logic).
"""

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import cv2

from config.settings import Settings
from src.core.pipeline import DocumentPipeline, PipelineResult
from src.visualization.plots import render_image
from src.visualization.heatmaps import (
    build_damage_heatmap_overlay,
    build_ocr_confidence_map,
    build_individual_damage_overlays,
)
from src.visualization.comparison import build_comparison_grid


# =========================
# 📦 Project setup
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# 🌐 Streamlit config
# =========================
st.set_page_config(
    page_title="📚 Book Time Machine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 🎨 UI Styling
# =========================
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }

h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.5rem !important; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #3a3a55;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 🧠 Session state
# =========================
@st.cache_resource
def get_settings():
    return Settings()


@st.cache_resource
def get_pipeline():
    return DocumentPipeline(get_settings())


# =========================
# 📤 Upload image
# =========================
def upload_image():
    file = st.file_uploader(
        "📤 Upload document",
        type=["jpg", "png", "jpeg", "tiff", "bmp"]
    )

    if file is None:
        return None

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image file")
        return None

    # convert BGR → RGB (مهم للعرض)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# =========================
# 📊 Sidebar
# =========================
def sidebar():
    st.sidebar.title("⚙️ Settings")

    settings = get_settings()

    settings.MEDIAN_KERNEL = st.sidebar.selectbox("Median kernel", [3, 5, 7], 0)
    settings.CLAHE_CLIP_LIMIT = st.sidebar.slider("CLAHE", 0.5, 5.0, 2.0)
    settings.INPAINT_RADIUS = st.sidebar.slider("Inpaint radius", 1, 15, 5)

    mask_thresh = st.sidebar.slider("Mask threshold", 40, 200, 80)

    return settings, mask_thresh


# =========================
# 🏠 Overview Tab
# =========================
def tab_overview(r: PipelineResult):
    st.title("📊 Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.image(render_image(r.original), caption="Original")

    with col2:
        st.image(render_image(r.restoration.best_final), caption="Restored")

    st.metric("Final Score", f"{r.evaluation.final_score:.1f}/100")

    st.write("**Pipeline:**", " → ".join(r.restoration.best_chain))


# =========================
# 🔍 Damage Tab
# =========================
def tab_damage(r: PipelineResult):
    st.title("🔍 Damage Analysis")

    st.image(
        render_image(
            build_damage_heatmap_overlay(
                r.original,
                r.damage_report.combined_heatmap
            )
        )
    )

    overlays = build_individual_damage_overlays(
        r.original,
        r.damage_report.maps
    )

    for k, v in overlays.items():
        st.subheader(k)
        st.image(render_image(v))


# =========================
# ⚙️ Stages Tab
# =========================
def tab_stages(r: PipelineResult):
    st.title("⚙️ Pipeline Stages")

    i = st.slider("Stage", 0, len(r.stage_labels) - 1, 0)

    label = r.stage_labels[i]
    st.image(render_image(r.stage_images[label]), caption=label)


# =========================
# 🧪 Comparison Tab
# =========================
def tab_comparison(r: PipelineResult):
    st.title("🧪 Method Comparison")

    for group in [
        r.restoration.denoised,
        r.restoration.enhanced,
        r.restoration.sharpened,
    ]:
        grid = build_comparison_grid(group, cols=3)
        for row in grid:
            st.image(render_image(row))


# =========================
# 📝 OCR Tab
# =========================
def tab_ocr(r: PipelineResult):
    st.title("📝 OCR")

    st.image(
        render_image(build_ocr_confidence_map(r.original, r.ocr_original))
    )

    st.code(r.ocr_original.full_text)


# =========================
# 📊 Evaluation Tab
# =========================
def tab_eval(r: PipelineResult):
    st.title("📊 Evaluation")

    st.json(r.evaluation.details)
    st.text(r.evaluation.text_report)


# =========================
# 🚀 Main App
# =========================
def main():
    st.title("📚 Book Time Machine")

    settings, thresh = sidebar()

    image = upload_image()

    if image is None:
        st.info("📤 Upload an image to start")
        return

    pipeline = get_pipeline()

    with st.spinner("Running pipeline..."):
        result = pipeline.run(image)

    tabs = st.tabs([
        "Overview",
        "Damage",
        "Stages",
        "Comparison",
        "OCR",
        "Evaluation"
    ])

    with tabs[0]:
        tab_overview(result)
    with tabs[1]:
        tab_damage(result)
    with tabs[2]:
        tab_stages(result)
    with tabs[3]:
        tab_comparison(result)
    with tabs[4]:
        tab_ocr(result)
    with tabs[5]:
        tab_eval(result)


if __name__ == "__main__":
    main()