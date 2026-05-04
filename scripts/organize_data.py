import os
import shutil

# ===============================
# PATHS
# ===============================
BASE_DIR = "data/raw"

DAMAGE_DIR = os.path.join(BASE_DIR, "damage_dataset")
HIST_DIR = os.path.join(BASE_DIR, "historical_docs")
OCR_DIR = os.path.join(BASE_DIR, "ocr_dataset")

# ===============================
# CREATE FOLDERS
# ===============================
damage_types = ["blur", "fade", "noise", "stain"]

for d_type in damage_types:
    os.makedirs(os.path.join(DAMAGE_DIR, d_type), exist_ok=True)

os.makedirs(os.path.join(OCR_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OCR_DIR, "labels"), exist_ok=True)

# ===============================
# ORGANIZE DAMAGE DATASET
# ===============================
print("🔄 Organizing damage dataset...")

for file in os.listdir(DAMAGE_DIR):
    file_path = os.path.join(DAMAGE_DIR, file)

    if os.path.isfile(file_path):
        for d_type in damage_types:
            if file.startswith(d_type):
                dest = os.path.join(DAMAGE_DIR, d_type, file)
                shutil.move(file_path, dest)
                break

print("✅ Damage dataset organized!")

# ===============================
# PREPARE OCR DATASET
# ===============================
print("🔄 Preparing OCR dataset...")

for file in os.listdir(HIST_DIR):
    if file.endswith(".tiff"):
        src = os.path.join(HIST_DIR, file)
        dst = os.path.join(OCR_DIR, "images", file)

        if not os.path.exists(dst):
            shutil.copy(src, dst)

print("✅ OCR images prepared!")

# ===============================
# FINAL MESSAGE
# ===============================
print("\n🎉 All done successfully!")