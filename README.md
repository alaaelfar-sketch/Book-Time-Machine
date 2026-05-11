# 📚 Book Time Machine

An intelligent document restoration and analysis system for historical books and damaged documents using **Computer Vision**, **Image Processing**, and **OCR** techniques.

The system restores degraded document pages, analyzes damage levels, extracts readable text, and visualizes the entire restoration process interactively.

---

# 🚀 Project Overview

Old books and historical documents often suffer from:

- Noise
- Fading text
- Low contrast
- Physical degradation
- Blurred content

The **Book Time Machine** helps transform damaged pages into cleaner and more readable digital versions while also providing visual insights into the restoration quality.

This project combines:

- Image Restoration
- Damage Analysis
- OCR (Optical Character Recognition)
- Visual Analytics
- Interactive Visualization

into one complete intelligent pipeline.

---

# 🧠 Core Features

## ✨ Document Restoration
Enhances degraded document images using image processing techniques such as:

- Denoising
- Contrast Enhancement
- Sharpening
- Edge Preservation

---

## 🔍 Damage Detection & Analysis
Analyzes document degradation and computes a damage score.

The system highlights:
- Noise regions
- Faded areas
- Damaged sections

using visual damage maps.

---

## 📝 OCR Text Extraction
Extracts readable text from restored documents using OCR.

Features:
- Word-level confidence scores
- OCR confidence visualization
- Extracted text display

---

## 🎨 OCR Confidence Visualization
Each detected word is color-coded based on OCR confidence:

- 🟢 High Confidence
- 🟡 Medium Confidence
- 🔴 Low Confidence

This helps evaluate OCR quality visually instead of relying only on raw text output.

---

## 📊 Interactive Visualization
Built with **Streamlit** to provide an interactive user experience.

Users can:
- Upload document images
- View restoration stages
- Compare original vs restored images
- Explore OCR analysis results
- Inspect damage heatmaps

---

# ⚙️ System Pipeline

```text
Input Image
    ↓
Damage Analysis
    ↓
Image Enhancement
    ↓
OCR Extraction
    ↓
Confidence Evaluation
    ↓
Interactive Visualization
```

---

# 🏗️ Project Structure

```text
Book-Time-Machine/
│
├── app.py                          # Streamlit UI
│
├── src/
│   │
│   ├── pipeline.py                # Main restoration pipeline
│   ├── enhancement.py             # Image enhancement functions
│   ├── damage.py                  # Damage analysis & scoring
│   ├── ocr.py                     # OCR extraction & confidence
│   ├── evaluation.py              # Metrics & evaluation
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Streamlit
- Tesseract OCR
- PIL / Pillow

---

# 📦 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Book-Time-Machine.git
cd Book-Time-Machine
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Install Tesseract OCR

### Windows
Download and install:

https://github.com/UB-Mannheim/tesseract/wiki

Then add Tesseract to your system PATH.

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📸 Example Outputs

The system generates:

- Original document image
- Restored image
- Damage heatmap
- OCR confidence visualization
- Extracted text
- Damage score
- OCR confidence metrics

---

# 📈 Evaluation Metrics

The project evaluates restoration quality using:

- OCR Confidence Score
- Damage Score
- Sharpness Analysis
- Contrast Evaluation

---

# 💡 Project Innovation

Unlike traditional restoration systems that only enhance images, this project provides:

✅ Step-by-step restoration visualization  
✅ Damage analysis before restoration  
✅ OCR confidence mapping  
✅ Interactive visual analytics  
✅ End-to-end intelligent pipeline

---

# 🎯 Future Improvements

Potential future enhancements include:

- Deep Learning based restoration
- GAN-based image enhancement
- Transformer OCR models
- Automatic document segmentation
- Historical handwriting recognition
- Multi-language OCR support

---

# 🖥️ Streamlit Interface

The application allows users to:

- Upload damaged document images
- Compare before/after restoration
- View OCR confidence maps
- Analyze damage visually
- Explore restoration metrics interactively

---

# 📚 Use Cases

- Historical document restoration
- Digital archiving
- Library preservation systems
- OCR preprocessing
- Manuscript digitization
- Research & education

---

# 🏁 Conclusion

The **Book Time Machine** is a complete intelligent system that combines:

- Image Restoration
- OCR
- Damage Analysis
- Visual Analytics

to transform degraded historical documents into readable digital versions while providing deep insight into restoration quality and OCR reliability.

---

