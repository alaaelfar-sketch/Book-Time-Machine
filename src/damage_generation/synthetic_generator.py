import cv2
import numpy as np
import os
import random

input_dir = "data/raw/historical_docs"
output_dir = "data/raw/damage_dataset"

os.makedirs(output_dir, exist_ok=True)

def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def add_blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

def fade_image(img):
    return cv2.convertScaleAbs(img, alpha=0.6, beta=30)

def add_stains(img):
    h, w = img.shape[:2]
    for _ in range(5):
        x, y = random.randint(0, w-50), random.randint(0, h-50)
        cv2.circle(img, (x, y), random.randint(10, 40), (50, 50, 50), -1)
    return img

def corrupt(img, mode):
    if mode == "noise":
        return add_noise(img)
    elif mode == "blur":
        return add_blur(img)
    elif mode == "fade":
        return fade_image(img)
    elif mode == "stain":
        return add_stains(img)

modes = ["noise", "blur", "fade", "stain"]

for file in os.listdir(input_dir):
    img_path = os.path.join(input_dir, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    for mode in modes:
        corrupted = corrupt(img.copy(), mode)

        save_path = os.path.join(output_dir, f"{mode}_{file}")
        cv2.imwrite(save_path, corrupted)

print("Synthetic damage dataset generated successfully!")