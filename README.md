
---

# 📚 Book Time Machine

### 🧠 AI-Powered Historical Document Restoration System

<p align="center">
  <img src="https://img.shields.io/badge/AI-Computer%20Vision-blue?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/OCR-Tesseract-green?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

---

## 🚀 Project Overview

The **Book Time Machine** is an intelligent system designed to restore and analyze historical documents and old book pages using **Image Processing** and **Computer Vision techniques**.

The goal is to transform **degraded, unclear pages** into **clean, readable digital versions**.

---

## 🧠 Core Idea

This project does not only enhance images, but also provides a full **visual analysis of document condition before and after restoration**.

It acts as both:

* 🖼️ Restoration Tool
* 📊 Analytical System

---

## 📥 Input (Old Historical Documents)

<p align="center">
  <img src="assets/old_book_1.jpg" width="300"/>
  <img src="assets/old_book_2.jpg" width="300"/>
</p>

👉 Real-world degraded documents with noise, fading, and damage

---

## ⚙️ System Pipeline

1️⃣ Upload document image
2️⃣ Damage analysis
3️⃣ Image enhancement (denoise, contrast, inpainting)
4️⃣ OCR text extraction
5️⃣ Visualization dashboard (Streamlit)

---

## 🔥 Core Innovation

---

### 🥇 Comparison Engine (Before → After)

<p align="center">
  <img src="assets/before.jpg" width="300"/>
  <img src="assets/after.jpg" width="300"/>
</p>

👉 Shows full transformation from degraded → restored document

---

### 🥈 Damage Heatmap

<p align="center">
  <img src="assets/heatmap.jpg" width="500"/>
</p>

👉 Highlights damaged regions:

* 🔴 Noise
* 🟡 Fading
* 🟢 Clean areas

---

### 🥉 OCR Output (Text Extraction)

* Extracts readable text from restored image
* Converts image → digital text
* Displays structured output

---

## 💻 Deployment (Streamlit App)

Users can:

* 📤 Upload images
* 🔍 View processing pipeline
* 🔄 Compare results
* 📊 Explore OCR output

---

## 🎯 Key Features

* ✔ End-to-end AI pipeline
* ✔ Image restoration system
* ✔ Damage detection (heatmap)
* ✔ OCR text extraction
* ✔ Visual analytics dashboard
* ✔ Interactive Streamlit UI

---

## 📊 Expected Output

* 🧾 Restored document
* 🔥 Damage heatmap
* ✍️ Extracted text
* 📊 Visual analysis results

---

## 💡 Project Value

✔ Restores historical documents
✔ Explains AI decisions visually
✔ Combines CV + OCR + Visualization
✔ Provides full end-to-end pipeline

👉 Not just enhancement — **intelligent document understanding system**

---

## 🏁 Conclusion

The **Book Time Machine** transforms damaged historical documents into readable digital content while visually explaining every step of the AI process.

---

## 📁 Project Structure

```bash id="k8p1aa"
Book-Time-Machine/
│
├── app.py
├── requirements.txt
├── README.md
│
├── config/
│   ├── paths.py
│   ├── settings.py
│   └── constants.py
│
├── data/
│   ├── raw/
│   │   ├── historical_docs/
│   │   ├── damage_dataset/
│   │   │   ├── blur/
│   │   │   ├── fade/
│   │   │   ├── noise/
│   │   │   └── stain/
│   │   └── ocr_dataset/
│   │       ├── images/
│   │       └── labels/
│   └── processed/
│
├── src/
│   ├── core/
│   │   ├── preprocessing.py
│   │   ├── restoration.py
│   │   ├── damage_analysis.py
│   │   ├── ocr_engine.py
│   │   └── pipeline.py
│
│   ├── evaluation/
│   │   ├── image_metrics.py
│   │   ├── ocr_metrics.py
│   │   └── evaluator.py
│
│   ├── visualization/
│   │   ├── plots.py
│   │   ├── heatmaps.py
│   │   └── comparison.py
│
│   ├── io/
│   │   ├── loader.py
│   │   ├── saver.py
│   │   └── dataset.py
│
│   ├── utils/
│   │   ├── logger.py
│   │   └── helpers.py
│
├── scripts/
│   └── organize_data.py
│
└── tests/
```
---

## 👨‍💻 Developer Notes

> Built with passion for Computer Vision, AI, and Digital Preservation.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub to support development!

---
