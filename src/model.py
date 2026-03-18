# =============================================================
# src/model.py
# =============================================================
# Defines the CNN architecture for satellite image classification.
# This is the CORE of our deep learning system — a custom 4-block
# Convolutional Neural Network (CNN) trained from scratch (no transfer
# learning), which keeps things lightweight and explainable.
# =============================================================

import torch
import torch.nn as nn


class LandUseCNN(nn.Module):
    """
    Custom CNN for 128×128 RGB satellite image classification into 10 classes.

    Architecture overview:
        4 × [Conv2d → BatchNorm → ReLU → MaxPool]   ← Feature extraction
        AdaptiveAvgPool → Flatten                   ← Spatial aggregation
        Linear(4096 → 512) → ReLU → Dropout(0.5)   ← Fully connected
        Linear(512 → num_classes)                   ← Output logits

    Design decisions:
    - BatchNorm2d: Normalises activations per mini-batch, helping the network
      train faster and more stably. It reduces internal covariate shift.
    - ReLU (inplace): Non-linear activation that allows the network to learn
      complex decision boundaries. inplace=True saves memory.
    - MaxPool2d(2,2): Downsamples spatial dimensions by 2× at each block,
      progressively extracting higher-level features.
    - AdaptiveAvgPool: Collapses any spatial size to a fixed 4×4 grid,
      making the classifier head size independent of input resolution.
    - Dropout(0.5): Randomly zeros 50% of neurons during training to prevent
      co-adaptation (overfitting). Disabled at inference automatically.
    """

    def __init__(self, num_classes=10):
        super(LandUseCNN, self).__init__()

        # ── Feature Extraction Blocks ──────────────────────────────────
        # Each block doubles the number of filters (depth) while halving
        # the spatial resolution via MaxPool. This is the standard CNN
        # design pattern: more channels = richer feature vocabulary.
        self.features = nn.Sequential(

            # Block 1: 3 input channels (RGB) → 32 feature maps
            # Spatial: 128×128 → 64×64 after MaxPool
            # 3×3 kernel with padding=1 keeps spatial size before pooling.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),       # Normalise the 32 feature maps
            nn.ReLU(inplace=True),    # Non-linear activation
            nn.MaxPool2d(2, 2),       # Halve spatial dims: 128 → 64

            # Block 2: 32 → 64 feature maps
            # Spatial: 64×64 → 32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 64 → 32

            # Block 3: 64 → 128 feature maps
            # Spatial: 32×32 → 16×16
            # At this depth, the network starts detecting textures and
            # mid-level patterns (e.g. field edges, tree canopies).
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 32 → 16

            # Block 4: 128 → 256 feature maps
            # Spatial: 16×16 → 8×8
            # At 256 channels, the network learns high-level semantic
            # concepts like "urban grid", "river sinuosity", "forest density".
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 16 → 8
        )

        # ── Spatial Aggregation ────────────────────────────────────────
        # AdaptiveAvgPool ensures the feature map is always 4×4 regardless
        # of input size, making the model resolution-agnostic.
        # Output: (batch, 256, 4, 4) = 4096 values per image.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Classification Head ────────────────────────────────────────
        # Flatten the spatial feature maps and pass through two fully
        # connected layers. The first FC compresses 4096 → 512 features,
        # and Dropout regularises to avoid memorisation.
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # (batch, 256×4×4) = (batch, 4096)
            nn.Linear(256 * 4 * 4, 512),       # Learn global image representation
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                   # 50% dropout for regularisation
            nn.Linear(512, num_classes),        # Output one logit per class
            # NOTE: No Softmax here — CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        """
        Forward pass: feature extraction → pooling → classification.

        Args:
            x: input tensor (batch_size, 3, 128, 128)
        Returns:
            logits: (batch_size, num_classes) — raw, unnormalised scores
        """
        x = self.features(x)    # Extract spatial features
        x = self.pool(x)        # Aggregate to fixed 4×4
        x = self.classifier(x)  # Classify to num_classes logits
        return x


def get_model(num_classes=10):
    """
    Factory function: instantiates and returns the LandUseCNN model.
    Using a factory keeps the interface clean — callers don't need to
    import the class name directly.
    """
    return LandUseCNN(num_classes=num_classes)


# =============================================================
# Smoke test: verify model compiles and forward pass runs
# Run: python src/model.py
# =============================================================
if __name__ == "__main__":
    model = get_model(num_classes=10)
    dummy_input = torch.randn(1, 3, 128, 128)   # Simulated single image
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")       # Expected: (1, 10)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print("model.py OK ✓")
