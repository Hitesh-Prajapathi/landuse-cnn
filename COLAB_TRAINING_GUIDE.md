# 🚀 Colab Training Guide – Land Use CNN

Complete step-by-step guide from zipping to getting `model.pt` back locally.

---

## PART 1: LOCAL PREPARATION (Do this on your Mac)

### Step 1: Clean up local artifacts (Terminal)

```bash
cd /Users/hiteshprajapathi/Desktop/DL_ETE_Hackathon/landuse-cnn

# Remove locally trained model & outputs (we'll retrain on GPU)
rm -rf saved_models/model.pt
rm -rf outputs/plots/*.png
rm -rf outputs/classification_report.txt
rm -rf src/__pycache__
```

### Step 2: Create the ZIP (Terminal)

```bash
cd /Users/hiteshprajapathi/Desktop/DL_ETE_Hackathon

zip -r landuse-cnn.zip landuse-cnn/ \
    -x "landuse-cnn/data/raw/*" \
    -x "landuse-cnn/data/val/*" \
    -x "landuse-cnn/saved_models/*" \
    -x "landuse-cnn/outputs/**/*" \
    -x "landuse-cnn/src/__pycache__/*" \
    -x "landuse-cnn/.DS_Store" \
    -x "landuse-cnn/data/.DS_Store"
```

This keeps only: `data/train/`, `data/test/`, `src/`, `scripts/`, `app/`, `requirements.txt`.

### Step 3: Open Google Colab

Go to: https://colab.research.google.com → New Notebook

**⚡ IMPORTANT: Runtime → Change runtime type → GPU (T4)**

---

## PART 2: COLAB NOTEBOOK CELLS (Copy-paste each cell)

---

### Cell 1: Upload ZIP

```python
from google.colab import files
uploaded = files.upload()  # Select landuse-cnn.zip
```

---

### Cell 2: Unzip & Navigate

```python
!unzip -q landuse-cnn.zip
%cd landuse-cnn
!ls
```

**Expected output:** You should see `data/`, `src/`, `requirements.txt`, etc.

---

### Cell 3: Install Dependencies

```python
!pip install -q -r requirements.txt
```

---

### Cell 4: Fix Import Paths + Verify GPU

```python
import sys
sys.path.insert(0, '/content/landuse-cnn')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

**Expected:** `CUDA available: True`, GPU name shows (e.g., Tesla T4)

---

### Cell 5: Load Data

```python
from src.dataset import get_dataloaders

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    batch_size=32,
    img_size=128,
    num_workers=2
)

print(f"\nClasses ({len(class_names)}): {class_names}")

# Quick shape check
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")   # Should be [32, 3, 128, 128]
print(f"Labels shape: {labels.shape}")  # Should be [32]
```

**Expected:**
- Train: 5000 images
- Test: 1000 images
- Val: may say empty (that's OK — we have test for evaluation)
- Batch shape: `torch.Size([32, 3, 128, 128])`

---

### Cell 6: Load Model

```python
from src.model import get_model

model = get_model(num_classes=len(class_names))
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print(f"Model on: {next(model.parameters()).device}")
```

**Expected:** `Model parameters: 2,492,170`, device: `cuda:0`

---

### Cell 7: Train (3 Epochs)

```python
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 3

train_losses = []
train_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(f"  → Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

print("\n✅ Training complete!")
print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
```

**Expected:** Loss should decrease each epoch, accuracy should increase.

---

### Cell 8: Test Accuracy

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")
```

---

### Cell 9: Confusion Matrix & Classification Report

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
```

---

### Cell 10: Plot Training Curves

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(train_losses) + 1)

ax1.plot(epochs, train_losses, 'b-o', linewidth=2)
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_accs, 'g-o', linewidth=2)
ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

---

### Cell 11: Save Model

```python
import os
os.makedirs('saved_models', exist_ok=True)

# Save full state dict
torch.save(model.state_dict(), 'saved_models/model.pt')

# Verify
size_mb = os.path.getsize('saved_models/model.pt') / (1024 * 1024)
print(f"✅ Model saved: saved_models/model.pt ({size_mb:.1f} MB)")
```

---

### Cell 12: Download Model to Your Mac

```python
from google.colab import files

# Download model weights
files.download('saved_models/model.pt')

# Also download plots if you want
files.download('confusion_matrix.png')
files.download('training_curves.png')
```

---

## PART 3: BACK ON YOUR MAC

### Step 1: Move `model.pt` into the project

```bash
# Move downloaded model.pt from Downloads to project
mv ~/Downloads/model.pt /Users/hiteshprajapathi/Desktop/DL_ETE_Hackathon/landuse-cnn/saved_models/model.pt
```

### Step 2: Move plots (optional)

```bash
mv ~/Downloads/confusion_matrix.png /Users/hiteshprajapathi/Desktop/DL_ETE_Hackathon/landuse-cnn/outputs/plots/
mv ~/Downloads/training_curves.png /Users/hiteshprajapathi/Desktop/DL_ETE_Hackathon/landuse-cnn/outputs/plots/
```

### Step 3: Verify locally

```bash
cd /Users/hiteshprajapathi/Desktop/DL_ETE_Hackathon/landuse-cnn
conda activate landuse-cnn
python src/evaluate.py
```

This loads the Colab-trained `model.pt` and re-evaluates on the local test set.

---

## ⏱️ EXPECTED TIMELINE

| Step | Time |
|---|---|
| Zip + Upload | ~2-3 min |
| Install deps | ~1 min |
| Load data | ~30 sec |
| Train 3 epochs (GPU) | ~1-2 min |
| Evaluate + plots | ~30 sec |
| Download model | ~30 sec |
| **Total** | **~5-7 min** |

---

## 🛑 TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `CUDA not available` | Runtime → Change runtime type → GPU |
| `ModuleNotFoundError` | Re-run Cell 3 (pip install) and Cell 4 (sys.path) |
| `FileNotFoundError: data/train` | Re-run Cell 2, make sure you're in `landuse-cnn/` |
| `No module named 'src'` | Re-run Cell 4 (sys.path.insert) |
| Upload hangs | Use Google Drive instead: upload zip to Drive, then `!cp /content/drive/MyDrive/landuse-cnn.zip .` |

---

## 🎯 AFTER THIS

Once `model.pt` is back in `saved_models/`, you're ready for the **UI/app phase** — tell me when you want to proceed!
