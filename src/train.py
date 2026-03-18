# =============================================================
# src/train.py
# =============================================================
# End-to-end training pipeline for the LandUse CNN.
# This script orchestrates: data loading → model construction →
# loss / optimiser setup → training loop → model saving →
# automatic evaluation on the test set.
#
# Run from project root:
#   conda activate landuse-cnn
#   python src/train.py
# =============================================================

import sys
import os
import time

# ── Path setup ────────────────────────────────────────────────────────
# When running `python src/train.py` from the project root, Python's
# sys.path doesn't include the root by default, so imports like
# `from src.model import ...` would fail. We insert the root manually.
_this_dir    = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm   # Progress bars for training loops

from src.dataset import get_dataloaders
from src.model   import get_model
from src.utils   import calculate_accuracy, plot_training_curves, get_project_root


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one full pass over the training dataset.

    HOW TRAINING WORKS:
    1. Set model to train mode (enables Dropout, BatchNorm in training mode)
    2. For each mini-batch:
       a. Move data to device (GPU/CPU)
       b. Zero gradients from last step (PyTorch accumulates by default)
       c. Forward pass — compute predictions
       d. Compute loss — measure prediction error
       e. Backward pass — compute gradients via backpropagation
       f. Optimizer step — update weights using gradients

    Args:
        model:     the LandUseCNN (in train mode)
        loader:    training DataLoader
        criterion: loss function (CrossEntropyLoss)
        optimizer: optimiser (Adam)
        device:    torch.device ('cuda', 'mps', or 'cpu')

    Returns:
        (avg_loss, avg_accuracy) for this epoch
    """
    model.train()          # Enable training-mode layers (Dropout, BatchNorm)
    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        # Move batch to the target device (necessary for GPU acceleration)
        images, labels = images.to(device), labels.to(device)

        # Step 1: Clear old gradients — MUST do this every iteration
        optimizer.zero_grad()

        # Step 2: Forward pass — compute raw logits for each class
        outputs = model(images)

        # Step 3: Compute loss
        # CrossEntropyLoss = Softmax + NegativeLogLikelihood
        # It penalises confident wrong predictions more than uncertain ones.
        loss = criterion(outputs, labels)

        # Step 4: Backward pass — compute ∂loss/∂weight for every parameter
        loss.backward()

        # Step 5: Update weights using the computed gradients
        optimizer.step()

        # Accumulate metrics (weighted by batch size for correct averaging)
        running_loss    += loss.item() * images.size(0)
        _, predicted     = torch.max(outputs, 1)   # Predicted class = argmax
        running_correct += (predicted == labels).sum().item()
        running_total   += labels.size(0)

    avg_loss = running_loss    / running_total
    avg_acc  = (running_correct / running_total) * 100.0
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    """
    Evaluate the model on validation data WITHOUT updating weights.

    WHY NO torch.no_grad()?
    During evaluation we don't call .backward(), so we don't NEED gradients.
    torch.no_grad() disables gradient tracking, reducing memory consumption
    and speeding up the forward pass by ~2×.

    Args:
        model:     trained or partially-trained model
        loader:    validation/test DataLoader
        criterion: loss function (same as training)
        device:    torch.device

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()   # Disable Dropout; use running stats in BatchNorm
    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    with torch.no_grad():    # Disable gradient computation — saves memory
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss    += loss.item() * images.size(0)
            _, predicted     = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total   += labels.size(0)

    avg_loss = running_loss    / running_total
    avg_acc  = (running_correct / running_total) * 100.0
    return avg_loss, avg_acc


def main():
    # ── Hyperparameters ────────────────────────────────────────────────
    # These are the key tuning knobs for training. They were chosen based
    # on standard practice for small-scale image classification:
    EPOCHS        = 3       # Number of full passes over training data
    BATCH_SIZE    = 32      # Mini-batch size: balance of speed and stability
    LEARNING_RATE = 1e-3    # Adam default lr — works well for most CNNs
    IMG_SIZE      = 128     # Input resolution (upscaled from raw 64×64)
    NUM_WORKERS   = 2       # Parallel data-loading threads

    root = get_project_root()

    # ── Device selection ───────────────────────────────────────────────
    # We prioritise GPU (CUDA for Colab/Nvidia, MPS for Apple Silicon),
    # falling back to CPU. GPU training is ~10–20× faster for CNNs.
    if torch.cuda.is_available():
        device = torch.device("cuda")                          # Nvidia GPU (Colab)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")                           # Apple M-chip GPU
    else:
        device = torch.device("cpu")                           # Fallback
    print(f"\n[train] Using device: {device}")

    # ── Data loading ───────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS
    )

    # ── Model instantiation ────────────────────────────────────────────
    # We always use len(class_names) so the architecture auto-adapts
    # if the number of land-use categories changes.
    model = get_model(num_classes=len(class_names))
    model = model.to(device)   # Move all parameters to the target device

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {total_params:,}")

    # ── Loss function ──────────────────────────────────────────────────
    # CrossEntropyLoss is the standard choice for multi-class classification.
    # It combines LogSoftmax + NLLLoss, giving the model a smooth gradient
    # signal that encourages high confidence on the correct class.
    criterion = nn.CrossEntropyLoss()

    # ── Optimiser ──────────────────────────────────────────────────────
    # Adam (Adaptive Moment Estimation) maintains per-parameter learning rates
    # and uses momentum estimates. It converges faster than vanilla SGD
    # and requires less hyperparameter tuning.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ──────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f"\n{'='*60}")
    print(f"  Training for {EPOCHS} epochs | LR={LEARNING_RATE} | Batch={BATCH_SIZE}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train one epoch ────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ── Validate ───────────────────────────────────────────────────
        # Use val_loader if available; if not, we log 0.0 (no penalty).
        # Validation loss monitors generalisation; a large train-val gap
        # signals overfitting.
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch}/{EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
            f"({epoch_time:.1f}s)"
        )

    total_time = time.time() - start_time
    print(f"\n[train] Training completed in {total_time:.1f}s")

    # ── Save model weights ─────────────────────────────────────────────
    # We save only state_dict (not the full model object) — this is the
    # recommended PyTorch practice. It stores only the learned parameter
    # tensors, not the class definition, making it portable.
    save_dir = root / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pt"
    torch.save(model.state_dict(), str(model_path))
    print(f"[train] Model saved to: {model_path}")

    # ── Save training curves ───────────────────────────────────────────
    # Plot loss and accuracy curves for CO4 deliverable.
    plot_path = root / "outputs" / "plots" / "training_curves.png"
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)

    # ── Final test evaluation ──────────────────────────────────────────
    # Run the trained model on the held-out test set to get unbiased
    # performance estimates. This is done ONCE, at the very end.
    print(f"\n{'='*60}")
    print("  Running evaluation on test set...")
    print(f"{'='*60}\n")

    from src.evaluate import evaluate_model
    evaluate_model(model, test_loader, class_names, device)

    print("\n[train] ✅ All done!")


if __name__ == "__main__":
    main()
