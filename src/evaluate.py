# =============================================================
# src/evaluate.py
# =============================================================
# Model evaluation pipeline: runs inference on the test set and
# computes multiple performance metrics:
#   - Overall Accuracy
#   - Per-class Precision, Recall, F1-score (Classification Report)
#   - Cohen's Kappa (agreement beyond chance)
#   - Jaccard Index (per-class, classification analogue of IoU)
#   - Confusion Matrix (saved as heatmap PNG)
#
# Run standalone:
#   conda activate landuse-cnn
#   python src/evaluate.py
# =============================================================

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────
_this_dir     = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')    # Non-interactive backend — safe for servers/Colab
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,       # Agreement metric for multi-class classification
    jaccard_score            # Per-class Jaccard (classification analogue of IoU)
)
from tqdm import tqdm

from src.utils import get_project_root


def evaluate_model(model, test_loader, class_names, device, save_dir=None):
    """
    Evaluate the trained model on the test set and save all outputs.

    METRICS EXPLAINED:

    1. ACCURACY: fraction of correctly classified images.
       Simple but can be misleading on class-imbalanced datasets.
       Formula: correct / total

    2. PRECISION (per class): of all images predicted as class C,
       what fraction actually are class C?
       Good at measuring false alarm rate.

    3. RECALL (per class): of all images that ARE class C,
       what fraction did the model correctly identify?
       Good at measuring miss rate.

    4. F1-SCORE (per class): harmonic mean of Precision and Recall.
       Balances both false positives and false negatives.
       Formula: 2 × (P × R) / (P + R)

    5. COHEN'S KAPPA: measures agreement between predicted and true
       labels beyond what would be expected by chance alone.
       Range: -1 (total disagreement) to 1 (perfect agreement).
       > 0.80 = excellent; > 0.60 = good; < 0.40 = poor.
       More robust than accuracy for multi-class imbalanced tasks.

    6. JACCARD INDEX (per class): TP / (TP + FP + FN)
       Also known as Jaccard Similarity or Intersection-over-Union (IoU)
       for classification. Measures overlap between predicted and true
       class membership. Useful for assessing per-class quality.

    7. CONFUSION MATRIX: N×N grid where entry [i,j] = number of images
       of true class i predicted as class j. Diagonal = correct; off-diagonal
       = misclassifications. Reveals which classes are confused with each other.

    Args:
        model:       trained LandUseCNN
        test_loader: DataLoader (test set, no augmentation)
        class_names: list of class label strings
        device:      torch.device
        save_dir:    output directory (auto-resolved if None)
    """
    if save_dir is None:
        root     = get_project_root()
        save_dir = root / "outputs"

    plots_dir = os.path.join(str(save_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Inference loop ─────────────────────────────────────────────────
    # We run ALL test images through the model and collect predictions.
    # torch.no_grad() disables gradient tracking — not needed at inference,
    # speeds up computation and reduces memory by ~2×.
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)

            # argmax gives the class with the highest logit (no softmax needed
            # for prediction — relative ordering is the same)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 1. Overall Accuracy ────────────────────────────────────────────
    accuracy = (all_preds == all_labels).mean() * 100.0
    print(f"\n[evaluate] Test Accuracy: {accuracy:.2f}%")

    # ── 2. Classification Report ───────────────────────────────────────
    # sklearn's classification_report gives per-class Precision, Recall,
    # F1, and Support (number of samples) in a formatted table.
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4       # 4 decimal places for finer granularity
    )
    print(f"\n[evaluate] Classification Report:\n{report}")

    # ── 3. Cohen's Kappa ───────────────────────────────────────────────
    # Measures inter-rater agreement after correcting for chance.
    # For a model with 91% accuracy on 10 balanced classes,
    # chance = 10%, so kappa ≈ (0.91 - 0.10) / (1 - 0.10) ≈ 0.90
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"[evaluate] Cohen's Kappa:   {kappa:.4f}")

    # ── 4. Jaccard Index (per class) ───────────────────────────────────
    # jaccard_score returns one value per class: TP / (TP + FP + FN)
    # This is the IoU analogue for classification.
    # average=None returns per-class scores; macro average is the mean.
    jaccard_per_class = jaccard_score(all_labels, all_preds, average=None)
    jaccard_macro     = jaccard_per_class.mean()
    print(f"[evaluate] Jaccard (macro): {jaccard_macro:.4f}")
    print(f"[evaluate] Jaccard per class:")
    for cls, j in zip(class_names, jaccard_per_class):
        print(f"    {cls:<24} {j:.4f}")

    # ── Save metrics to file ───────────────────────────────────────────
    report_path = os.path.join(str(save_dir), "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Jaccard Index (macro avg): {jaccard_macro:.4f}\n\n")
        f.write("Jaccard Index per class:\n")
        for cls, j in zip(class_names, jaccard_per_class):
            f.write(f"  {cls:<24} {j:.4f}\n")
        f.write(f"\n{report}")
    print(f"[evaluate] Report saved to: {report_path}")

    # ── 5. Confusion Matrix ────────────────────────────────────────────
    # confusion_matrix(y_true, y_pred) returns an N×N array.
    # Entry [i,j] = number of samples with true label i predicted as j.
    # Diagonal = correct predictions; off-diagonal = misclassifications.
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',   # 'd' = integer format
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)    # Free memory
    print(f"[evaluate] Confusion matrix saved to: {cm_path}")

    return accuracy, kappa, jaccard_macro


# =============================================================
# Standalone evaluation — run after training:
#   python src/evaluate.py
# =============================================================
if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.model   import get_model

    root       = get_project_root()
    model_path = root / "saved_models" / "model.pt"

    if not model_path.exists():
        print(f"[evaluate] ❌ Model not found at {model_path}")
        print(f"  Run `python src/train.py` first.")
        sys.exit(1)

    # Resolve compute device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[evaluate] Using device: {device}")

    # Load data (test split only)
    _, _, test_loader, class_names = get_dataloaders()

    # Load the saved model weights into a fresh model instance
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.to(device)
    print(f"[evaluate] Model loaded from: {model_path}")

    # Run full evaluation
    evaluate_model(model, test_loader, class_names, device)
    print("\n[evaluate] ✅ Done!")
