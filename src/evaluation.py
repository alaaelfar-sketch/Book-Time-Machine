import cv2
import numpy as np


def evaluate_image(image, ocr_conf):

    image = np.array(image)

    # =========================
    # SAFE GRAYSCALE CONVERSION
    # =========================
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = gray.astype(np.uint8)

    # =========================
    # SHARPNESS (LAPLACIAN VARIANCE)
    # =========================
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # =========================
    # CONTRAST (STD DEV)
    # =========================
    contrast = float(np.std(gray))

    # =========================
    # OCR CONFIDENCE SAFE HANDLING
    # =========================
    if ocr_conf is None:
        ocr_conf = 0

    try:
        ocr_conf = float(ocr_conf)
    except:
        ocr_conf = 0.0

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "sharpness": float(sharpness),
        "contrast": contrast,
        "ocr_confidence": ocr_conf
    }