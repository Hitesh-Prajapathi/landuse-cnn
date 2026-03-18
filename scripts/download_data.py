# scripts/download_data.py

import kagglehub
import shutil
import os

print("🚀 Downloading dataset from Kaggle...")

# Step 1: Download dataset
path = kagglehub.dataset_download("gallo33henrique/sentinel-2-satellite-imagery")

print(f"✅ Downloaded to: {path}")

# Step 2: Define destination
destination = os.path.join("data", "raw")

# Create directory if not exists
os.makedirs(destination, exist_ok=True)

# Step 3: Move dataset
final_path = os.path.join(destination, "EuroSAT")

# Avoid overwriting if already exists
if not os.path.exists(final_path):
    shutil.move(path, final_path)
    print(f"📁 Dataset moved to: {final_path}")
else:
    print("⚠️ Dataset already exists in data/raw/")

# Step 4: Locate actual image folder
dataset_root = None

for root, dirs, files in os.walk(final_path):
    if len(dirs) >= 10:  # expects class folders
        dataset_root = root
        break

# Step 5: Final path output
if dataset_root:
    print(f"\n🎯 USE THIS PATH IN YOUR CODE:\n{dataset_root}")
else:
    print("❌ Could not automatically detect dataset structure. Check manually.")