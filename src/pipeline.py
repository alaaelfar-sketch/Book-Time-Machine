import cv2
import numpy as np
from PIL import Image
import pytesseract


# =========================
# 1. Load Image
# =========================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found or invalid format")
    return img


# =========================
# 2. Denoising 
# =========================
def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)


# =========================
# 3. Contrast Enhancement (CLAHE)
# =========================
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    merged = cv2.merge((l,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# =========================
# 4. 🔥 Deskew (NEW IMPORTANT FIX)
# =========================
def deskew(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords = np.column_stack(np.where(gray > 0))

    if len(coords) == 0:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# =========================
# 5. Sharpen 
# =========================
def sharpen(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)


# =========================
# 6. Full Pipeline (FIXED ORDER)
# =========================
def restore_image(img):

    img = denoise(img)
    img = enhance_contrast(img)

    img = deskew(img)  

    img = sharpen(img)

    return img


# =========================
# 7. OCR (Improved Preprocessing)
# =========================
def extract_text(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 better OCR than hard threshold
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=config)

    return text, thresh