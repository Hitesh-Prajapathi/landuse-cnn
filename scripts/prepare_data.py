# scripts/prepare_data.py

import os
import random
import shutil

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_DIR = "data/raw/EuroSAT/2750"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

TRAIN_COUNT = 500
TEST_COUNT = 100

random.seed(42)

# -----------------------------
# CREATE DIRECTORIES
# -----------------------------

def create_dirs(base_dir, class_names):
    for cls in class_names:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)


# -----------------------------
# MAIN LOGIC
# -----------------------------

def prepare_data():
    classes = os.listdir(SOURCE_DIR)

    print("📂 Classes found:", classes)

    create_dirs(TRAIN_DIR, classes)
    create_dirs(TEST_DIR, classes)

    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(cls_path)

        # Shuffle images
        random.shuffle(images)

        # Select images
        train_imgs = images[:TRAIN_COUNT]
        test_imgs = images[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

        print(f"\n🔹 {cls}")
        print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

        # Copy train images
        for img in train_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(TRAIN_DIR, cls, img)
            shutil.copy(src, dst)

        # Copy test images
        for img in test_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(TEST_DIR, cls, img)
            shutil.copy(src, dst)

    print("\n✅ Dataset preparation complete!")


# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    prepare_data()