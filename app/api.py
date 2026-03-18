# =============================================================
# app/api.py
# =============================================================
# FastAPI backend — the "brain" of the web interface.
# Responsibilities:
#   - Load the trained PyTorch model at startup
#   - Accept image uploads via HTTP POST
#   - Preprocess images identically to test-time transforms
#   - Run inference and return class probabilities
#   - Serve evaluation metrics from saved output files
#
# Run from project root:
#   uvicorn app.api:app --host 0.0.0.0 --port 8000
# =============================================================

import sys
import os
from pathlib import Path
from io import BytesIO   # In-memory file buffer — avoids writing to disk

# ── Path setup ────────────────────────────────────────────────────────
# Needed so we can import `src.model` from the api module.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # Allow cross-origin requests
import torch
import torch.nn.functional as F   # Softmax for probability conversion
from torchvision import transforms
from PIL import Image

from src.model import get_model

# ── FastAPI App Initialisation ────────────────────────────────────────
# FastAPI is an ASGI-based web framework for Python that automatically
# generates OpenAPI (Swagger) documentation at /docs.
app = FastAPI(
    title="LandUse CNN API",
    description="Satellite image land use classification",
    version="1.0.0"
)

# ── CORS Middleware ────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing): browsers block requests from one
# origin (e.g. localhost:8501/Streamlit) to another (localhost:8000/API)
# by default. This middleware adds the necessary response headers so that
# Streamlit (or any frontend) can call the API freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Accept from any origin (OK for local/demo)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class Names ───────────────────────────────────────────────────────
# These MUST match the alphabetical ordering produced by ImageFolder during
# training. Any mismatch causes the API to return wrong class labels even
# if the model's prediction index is correct.
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# ── Inference Transforms ──────────────────────────────────────────────
# CRITICAL: These must exactly match the test_transform used during training.
# If we use different preprocessing at inference, the model receives a
# distribution shift — even a well-trained model will perform poorly.
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),         # Same input size as trained on
    transforms.ToTensor(),                 # PIL → float Tensor [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],        # ImageNet channel means (same as training)
        std=[0.229, 0.224, 0.225]          # ImageNet channel std devs
    )
])

# ── Global Model State ────────────────────────────────────────────────
# These are module-level globals so the same model instance is shared
# across all incoming requests (we don't reload on every request).
device = None
model  = None


def load_model():
    """
    Load the trained model from disk into memory.

    Design choice: load once at startup rather than per-request.
    Loading a model in PyTorch takes ~0.5–2s; doing it per-request
    would make the API very slow at scale.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    global model, device

    # Select the best available compute device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_path = _project_root / "saved_models" / "model.pt"

    if not model_path.exists():
        print(f"[api] ⚠️ Model not found at {model_path}")
        return False

    # Reconstruct the model architecture and load saved weights
    model = get_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(
        torch.load(str(model_path), map_location=device)   # map_location ensures
    )                                                         # CUDA weights load on CPU
    model = model.to(device)
    model.eval()   # Disable Dropout; use BatchNorm in inference mode
    print(f"[api] ✅ Model loaded from {model_path} on {device}")
    return True


@app.on_event("startup")
async def startup():
    """Called once when the FastAPI server boots — loads the model."""
    load_model()


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns the API status and whether the model is loaded.
    Used by the Streamlit frontend to show the green/red API indicator.
    """
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "device":       str(device) if device else "none"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return the land use prediction.

    Processing pipeline:
    1. Validate file type (JPG/PNG only)
    2. Read image bytes into memory
    3. Decode with PIL and convert to RGB (handles grayscale, RGBA, etc.)
    4. Apply inference transforms (resize + normalise)
    5. Add batch dimension: (3,128,128) → (1,3,128,128)
    6. Run forward pass through the model
    7. Apply Softmax to convert logits → probabilities
    8. Return predicted class, confidence, and full probability dict

    Args:
        file: uploaded image file (multipart/form-data)

    Returns:
        JSON with predicted_class, confidence (%), and all_probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Reject non-image files early
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images accepted")

    try:
        # Read raw bytes and decode into a PIL Image
        contents    = await file.read()
        image       = Image.open(BytesIO(contents)).convert("RGB")

        # Apply transforms and add batch dimension (model expects 4D tensor)
        input_tensor = inference_transform(image).unsqueeze(0).to(device)

        # Forward pass — no gradient needed at inference
        with torch.no_grad():
            outputs       = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]   # Convert logits → probs

        # Build human-readable probability dictionary (class_name → %)
        probs_dict = {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        # The predicted class is the one with the highest probability
        predicted_idx   = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(probabilities[predicted_idx].item() * 100, 2)

        return {
            "predicted_class":  predicted_class,
            "confidence":       confidence,
            "all_probabilities": probs_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
def get_metrics():
    """
    Return evaluation metrics from the saved outputs directory.

    This reads the text report and image paths generated by evaluate.py
    and returns them to the Streamlit Analytics page. Keeping metrics
    on disk (rather than recomputing each request) avoids running inference
    on the full test set every time the Analytics page loads.
    """
    report_path = _project_root / "outputs" / "classification_report.txt"

    metrics = {
        "test_accuracy":        None,
        "classification_report": None,
        "confusion_matrix_path": None,
        "training_curves_path":  None,
    }

    if report_path.exists():
        with open(report_path) as f:
            content = f.read()
            metrics["classification_report"] = content
            # Parse the accuracy value from the first line of the report
            for line in content.split("\n"):
                if "Test Accuracy" in line:
                    try:
                        metrics["test_accuracy"] = float(
                            line.split(":")[1].strip().replace("%", "")
                        )
                    except (ValueError, IndexError):
                        pass

    # Serve absolute paths to plot images (Streamlit reads them directly)
    cm_path = _project_root / "outputs" / "plots" / "confusion_matrix.png"
    if cm_path.exists():
        metrics["confusion_matrix_path"] = str(cm_path)

    curves_path = _project_root / "outputs" / "plots" / "training_curves.png"
    if curves_path.exists():
        metrics["training_curves_path"] = str(curves_path)

    return metrics


# ── Run directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
