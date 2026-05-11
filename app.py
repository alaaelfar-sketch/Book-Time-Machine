import streamlit as st
import cv2
import numpy as np
import re

from src.pipeline import restore_image
from src.ocr import extract_text_with_confidence
from src.damage import analyze_damage, generate_heatmap


# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="Book Time Machine",
    layout="wide"
)

st.title("📚 Book Time Machine - Enhanced OCR System")


# =========================
# TEXT CLEANER
# =========================
def clean_text(text):

    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^A-Za-z0-9.,;:()'\"\\\- ]+", "", text)

    fixes = {
        "thatthe": "that the",
        "Abd-ul": "Abdul",
        "in- vestigate": "investigate",
    }

    for k, v in fixes.items():
        text = text.replace(k, v)

    return text.strip()


# =========================
# IMAGE LOADER
# =========================
def load_uploaded_image(uploaded):
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "tiff"]
)


# =========================
# MAIN PIPELINE
# =========================
if uploaded:

    img = load_uploaded_image(uploaded)

    # Restore image
    restored = restore_image(img)

    # OCR
    text, confidence_map, avg_conf = extract_text_with_confidence(restored)

    # Clean text
    text = clean_text(text)

    # =========================
    # DAMAGE ANALYSIS 🔥
    # =========================
    damage_result = analyze_damage(restored)
    heatmap = generate_heatmap(restored, damage_result["damage_score_map"])


    # =========================
    # LAYOUT
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original")
        st.image(img, channels="BGR")

    with col2:
        st.subheader("✨ Restored")
        st.image(restored, channels="BGR")


    # =========================
    # BLEND VIEW
    # =========================
    st.subheader("🔀 Before / After Comparison")

    alpha = st.slider("Blend Strength", 0.0, 1.0, 0.7)

    blended = cv2.addWeighted(img, 1 - alpha, restored, alpha, 0)
    st.image(blended, channels="BGR")


    # =========================
    # OCR RESULT
    # =========================
    st.subheader("🧠 Extracted Text")
    st.text_area("", text, height=200)


    # =========================
    # CONFIDENCE MAP
    # =========================
    st.subheader("📊 OCR Confidence Map")
    st.image(confidence_map, channels="BGR")

    st.metric("Average OCR Confidence", f"{avg_conf:.2f}%")


    # =========================
    # DAMAGE ANALYSIS 🔥
    # =========================
    st.subheader("🛠️ Damage Detection Results")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.image(damage_result["noise_map"], caption="Noise Regions")

    with col4:
        st.image(damage_result["faded_map"], caption="Faded Text")

    with col5:
        st.image(damage_result["stain_map"], caption="Stains / Broken Areas")


    # =========================
    # HEATMAP
    # =========================
    st.subheader("🔥 Damage Heatmap")

    st.image(heatmap, channels="BGR")

    st.metric(
        "Damage Score",
        f"{damage_result['damage_score']:.2f}"
    )


    # =========================
    # DEBUG VIEW
    # =========================
    with st.expander("🔍 Debug View (Grayscale)"):
        gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        st.image(gray, clamp=True)