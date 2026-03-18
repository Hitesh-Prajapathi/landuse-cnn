# =============================================================
# src/dataset.py
# =============================================================
# Data loading and preprocessing pipeline for the EuroSAT-style
# satellite image dataset. This module handles:
#   - Image transformations (augmentation + normalisation)
#   - Dataset creation using torchvision.ImageFolder
#   - DataLoader wrapping for batching and shuffling
# =============================================================

import os
import sys
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_project_root():
    """Resolve project root dynamically (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def get_transforms(img_size=128):
    """
    Create separate transform pipelines for training vs. testing/validation.

    WHY DIFFERENT TRANSFORMS?
    - Training: We apply augmentations to artificially expand the dataset and
      teach the model to generalise to real-world variations (e.g. a field
      photographed from different angles or in different lighting conditions).
    - Test/Val: We only resize and normalise — no augmentation — because we
      want a fair, deterministic evaluation of model performance.

    AUGMENTATIONS USED:
    - RandomHorizontalFlip/RandomVerticalFlip: Satellite imagery has no fixed
      orientation, so flipping is always a valid transformation.
    - RandomRotation(15°): Cameras may be slightly tilted; small rotations
      simulate this without distorting class-defining features.
    - ColorJitter: Models brightness/contrast variation from different sensors
      or lighting conditions.

    NORMALISATION (ImageNet stats):
    - Even though we train from scratch (no pretrained weights), using
      ImageNet mean/std puts pixel values in a distribution that gradient-based
      optimisers handle well on natural images.

    Args:
        img_size: target square image size (default 128)

    Returns:
        (train_transform, test_transform)
    """
    train_transform = transforms.Compose([
        # Resize all images to a consistent square: 64×64 raw → 128×128
        transforms.Resize((img_size, img_size)),

        # ── Augmentations (training only) ─────────────────────────────
        transforms.RandomHorizontalFlip(),         # 50% chance left-right flip
        transforms.RandomVerticalFlip(),           # 50% chance top-bottom flip
        transforms.RandomRotation(15),             # Random ±15° rotation
        transforms.ColorJitter(                    # Random colour distortion
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),

        # Convert PIL image → float32 Tensor in range [0, 1]
        transforms.ToTensor(),

        # Normalise with ImageNet channel statistics:
        # pixel_normalised = (pixel - mean) / std
        # This zero-centres the distribution for stable gradient flow.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # RGB channel means
            std=[0.229, 0.224, 0.225]     # RGB channel standard deviations
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),   # Resize only (no augmentation)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform


def _validate_folder(path, split_name):
    """
    Validate that a data split folder exists and has class subfolders.

    ImageFolder expects: data_dir/class_name/image.jpg
    This function catches missing or misconfigured directories early,
    before entering the training loop where errors are harder to trace.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"[dataset] ❌ {split_name} folder not found: {path}\n"
            f"  Make sure the data is in the correct location."
        )

    # List non-hidden subdirectories — each should be a class folder
    subfolders = [d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]

    if len(subfolders) == 0:
        raise ValueError(
            f"[dataset] ❌ {split_name} folder is empty (no class subfolders): {path}"
        )

    return subfolders


def get_dataloaders(data_dir=None, batch_size=32, img_size=128, num_workers=2):
    """
    Build and return DataLoaders for the train, validation, and test splits.

    WHY DataLoader?
    PyTorch DataLoader handles:
    - Mini-batching: groups images into batches for gradient computation
    - Shuffling: randomises order each epoch to prevent the model from
      memorising data order
    - Parallel loading: num_workers > 0 loads batches in background threads,
      so GPU is not idle waiting for data

    Args:
        data_dir:    path to the data/ directory (auto-resolved if None)
        batch_size:  number of images per mini-batch (default: 32)
        img_size:    target image resolution (default: 128)
        num_workers: CPU worker threads for data loading (default: 2)

    Returns:
        train_loader, val_loader (may be None), test_loader, class_names
    """
    # ── Resolve paths ──────────────────────────────────────────────────
    if data_dir is None:
        data_dir = str(get_project_root() / "data")   # Default: project/data/
    data_dir = os.path.abspath(data_dir)

    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")
    test_path  = os.path.join(data_dir, "test")

    print(f"[dataset] Resolved data directory: {data_dir}")
    print(f"[dataset]   train: {train_path}")
    print(f"[dataset]   val:   {val_path}")
    print(f"[dataset]   test:  {test_path}")

    # ── Validate folder structure ──────────────────────────────────────
    train_classes = _validate_folder(train_path, "train")
    test_classes  = _validate_folder(test_path,  "test")

    # Validation folder is optional (may be absent / empty)
    has_val = os.path.isdir(val_path) and len(
        [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
    ) > 0

    # Warn if the same classes do not appear in both train and test
    if set(train_classes) != set(test_classes):
        print(f"[dataset] ⚠️ WARNING: train and test have different classes!")
        print(f"  train: {sorted(train_classes)}")
        print(f"  test:  {sorted(test_classes)}")

    # ── Build transforms ───────────────────────────────────────────────
    train_transform, test_transform = get_transforms(img_size)

    # ── Build Datasets ─────────────────────────────────────────────────
    # ImageFolder automatically maps folder names to class indices:
    # AnnualCrop/  → 0,  Forest/ → 1,  HerbaceousVegetation/ → 2, ...
    # Class ordering is alphabetical and consistent across splits.
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_path,  transform=test_transform)

    if has_val:
        val_dataset = datasets.ImageFolder(val_path, transform=test_transform)
    else:
        print("[dataset] ⚠️ Val folder empty — skipping validation split.")
        val_dataset = None

    class_names = train_dataset.classes   # e.g. ['AnnualCrop', 'Forest', ...]

    print(f"\n[dataset] Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    if val_dataset:
        print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes ({len(class_names)}): {class_names}")

    # ── Build DataLoaders ──────────────────────────────────────────────
    # pin_memory=True speeds up CPU→GPU transfer by keeping tensors in
    # pinned (non-pageable) memory. Only beneficial with a CUDA GPU.
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,             # Shuffle every epoch — critical for training
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,        # No shuffle for evaluation — deterministic
            num_workers=num_workers,
            pin_memory=pin_mem
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,            # Deterministic order for reproducible eval
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    return train_loader, val_loader, test_loader, class_names


# =============================================================
# Quick test — run: python src/dataset.py
# =============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print(f"\nClasses: {classes}")

    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}")    # Expected: (32, 3, 128, 128)
    print(f"Labels shape:      {labels.shape}")    # Expected: (32,)
    print("dataset.py OK ✓")