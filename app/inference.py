# =============================================================
# app/inference.py
# =============================================================
# Direct inference module for Streamlit Cloud deployment.
# Loads the model once and provides a predict() function that
# the Predict page calls directly — no FastAPI server needed.
# =============================================================

import sys
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Path setup
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.model import get_model

# ── Class names (must match training order — alphabetical) ───────────
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# ── Inference transform (same as test/val) ───────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Singleton model loader ───────────────────────────────────────────
_model = None
_device = None


def _load_model():
    """Load model once and cache it in module globals."""
    global _model, _device

    if _model is not None:
        return _model, _device

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    model_path = _project_root / "saved_models" / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    _model = get_model(num_classes=len(CLASS_NAMES))
    _model.load_state_dict(torch.load(str(model_path), map_location=_device))
    _model = _model.to(_device)
    _model.eval()
    return _model, _device


def predict(image: Image.Image) -> dict:
    """
    Run inference on a PIL Image. Returns dict with:
      - predicted_class: str
      - confidence: float (%)
      - all_probabilities: dict[str, float]
    """
    model, device = _load_model()

    input_tensor = inference_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    probs_dict = {
        CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = round(probabilities[predicted_idx].item() * 100, 2)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": probs_dict
    }


def is_model_available() -> bool:
    """Check if model file exists."""
    model_path = _project_root / "saved_models" / "model.pt"
    return model_path.exists()
