import cv2
import numpy as np


# =========================
# 1. Preprocess
# =========================
def preprocess(image):

    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (900, 1200))

    # 🔥 light smoothing
    return cv2.bilateralFilter(image, 7, 50, 50).astype(np.uint8)


# =========================
# 2. Stain Removal
# =========================
def remove_stains(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


# =========================
# 3. Contrast Enhancement
# =========================
def enhance_contrast(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)


# =========================
# 4. Sharpen
# =========================
def sharpen(image):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)


# =========================
# 5. 🔥 Deskew (NEW FIX)
# =========================
def deskew(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    coords = np.column_stack(np.where(gray > 0))

    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# =========================
# 6. Full Pipeline
# =========================
def restore_document(image):

    stages = {}

    stages["Original"] = image

    step1 = preprocess(image)

    step2 = remove_stains(step1)
    stages["Stain Removal"] = step2

    step3 = enhance_contrast(step2)
    stages["Contrast"] = step3

    # 🔥 IMPORTANT ADDITION
    step4 = deskew(step3)
    stages["Deskew"] = step4

    step5 = sharpen(step4)
    stages["Sharpen"] = step5

    # 🔥 FINAL for OCR (NO binarization)
    final = step5
    stages["Final"] = final

    return final, stages