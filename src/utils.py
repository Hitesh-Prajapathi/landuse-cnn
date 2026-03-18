# =============================================================
# src/utils.py
# =============================================================
# Shared helper functions used across the training and evaluation
# pipelines. Keeping utilities here avoids code duplication and
# makes every module independently testable.
# =============================================================

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — required for Colab/servers
                        # where no display is available. Must be set BEFORE
                        # importing pyplot.
import matplotlib.pyplot as plt
import torch


def get_project_root():
    """
    Dynamically resolves the project root directory.

    Why: Paths are computed relative to THIS file's location, so the project
    works regardless of where it is placed on disk. We go up TWO levels:
        src/utils.py → src/ → project_root/
    """
    return Path(__file__).resolve().parent.parent


def calculate_accuracy(outputs, labels):
    """
    Compute classification accuracy from raw model outputs.

    Args:
        outputs: Tensor (batch_size, num_classes) — raw logits from the model
        labels:  Tensor (batch_size,) — ground truth integer class indices

    Returns:
        accuracy: float in range [0, 100]

    How it works:
        torch.max(outputs, dim=1) returns the index of the maximum logit for
        each sample — this is the predicted class (argmax). We then compare
        predictions to ground-truth labels and compute the fraction correct.
    """
    _, predicted = torch.max(outputs, dim=1)          # Get predicted class index
    correct = (predicted == labels).sum().item()       # Count correct predictions
    total = labels.size(0)                             # Total number of samples
    return (correct / total) * 100.0                  # Convert to percentage


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Generate and save training/validation loss and accuracy curves.

    Why plot curves? They allow visual diagnosis of:
    - Overfitting: train acc >> val acc (gap keeps growing)
    - Underfitting: both accuracies are low
    - Good generalisation: both curves converge close together

    Args:
        train_losses: list of float — one loss value per epoch (train set)
        val_losses:   list of float — one loss value per epoch (val set)
        train_accs:   list of float — accuracy (%) per epoch (train set)
        val_accs:     list of float — accuracy (%) per epoch (val set)
        save_path:    Path or str — where to save the PNG figure
    """
    # Default save path if not specified
    if save_path is None:
        root = get_project_root()
        save_path = root / "outputs" / "plots" / "training_curves.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create dirs if needed

    epochs = range(1, len(train_losses) + 1)   # Epoch x-axis: 1, 2, 3, ...

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left Panel: Loss ──────────────────────────────────────────────
    # Loss should decrease over epochs. Training loss typically decreases
    # faster; validation loss reveals how well the model generalises.
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',   linewidth=2)
    ax1.set_title('Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right Panel: Accuracy ─────────────────────────────────────────
    # Accuracy increases as the model improves. Ideally both curves
    # plateau close together at a high value (e.g., 90%+).
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs,   'r-o', label='Val Accuracy',   linewidth=2)
    ax2.set_title('Accuracy per Epoch', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)   # Free memory — always close figures after saving

    print(f"[utils] Training curves saved to: {save_path}")


# =============================================================
# Quick sanity test — run: python src/utils.py
# =============================================================
if __name__ == "__main__":
    print(f"Project root: {get_project_root()}")

    # Verify calculate_accuracy with random data
    dummy_out = torch.randn(8, 10)           # 8 images, 10 classes
    dummy_labels = torch.randint(0, 10, (8,))
    acc = calculate_accuracy(dummy_out, dummy_labels)
    print(f"Dummy accuracy: {acc:.2f}%")
    print("utils.py OK ✓")
