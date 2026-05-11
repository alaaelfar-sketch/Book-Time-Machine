import pytesseract
import cv2
import numpy as np
import re

# IMPORTANT: adjust path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================
# CLEAN TEXT (NEW 🔥)
# =========================
def clean_text(text):

    # fix weird starting symbols
    text = re.sub(r'^[^A-Za-z]+', '', text)

    # fix multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # common OCR fixes (optional but useful)
    text = text.replace("thatthe", "that the")
    text = text.replace("Abd-ul", "Abdul")

    return text.strip()


# =========================
# MAIN OCR FUNCTION
# =========================
def extract_text_with_confidence(image):

    image = np.array(image)

    # =========================
    # SAFE CONVERSION
    # =========================
    if len(image.shape) == 2:
        gray = image
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # =========================
    # 🔥 IMPROVED PREPROCESSING
    # =========================

    # noise reduction (better than bilateral alone)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # contrast normalization (IMPORTANT FIX)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11
    )

    # =========================
    # OCR CONFIG (OPTIMIZED)
    # =========================
    custom_config = r'--oem 3 --psm 4'

    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT,
        config=custom_config
    )

    text = []
    confs = []

    n_boxes = len(data["text"])

    # =========================
    # PARSE OCR RESULT
    # =========================
    for i in range(n_boxes):

        word = data["text"][i].strip()

        try:
            conf = float(data["conf"][i])
        except:
            continue

        # 🔥 stronger filtering
        if conf < 45 or word == "":
            continue

        text.append(word)
        confs.append(conf)

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i]
        )

        # =========================
        # COLOR MAPPING
        # =========================
        if conf >= 80:
            color = (0, 255, 0)      # 🟢 green
        elif conf >= 60:
            color = (0, 255, 255)    # 🟡 yellow
        else:
            color = (0, 0, 255)      # 🔴 red

        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            vis,
            word,
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # =========================
    # FINAL OUTPUT
    # =========================
    full_text = " ".join(text)

    # 🔥 CLEAN TEXT FIX
    full_text = clean_text(full_text)

    avg_conf = float(np.mean(confs)) if confs else 0.0

    return full_text, vis.astype(np.uint8), avg_conf