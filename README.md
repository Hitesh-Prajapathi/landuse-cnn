# 🛰️ LandUse CNN — Satellite Image Land Use Classification

**NeuralHack 2026 | End Trimester Deep Learning Hackathon**

> An end-to-end deep learning solution for classifying satellite imagery into 10 land-use categories using a custom Convolutional Neural Network trained on EuroSAT-style data.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-91%25-brightgreen)]()

---

## 📚 Table of Contents

1. [Problem Definition (CO1)](#1-problem-definition-co1)
2. [Mathematical Modeling (CO2)](#2-mathematical-modeling-co2)
3. [Model Architecture (CO3)](#3-model-architecture-co3)
4. [Data Preparation & Training (CO4)](#4-data-preparation--training-co4)
5. [User Interface Demonstration (CO5)](#5-user-interface-demonstration-co5)
6. [Project Structure](#6-project-structure)
7. [Setup & Installation](#7-setup--installation)
8. [Running Locally](#8-running-locally)
9. [Deployment](#9-deployment)
10. [Results](#10-results)

---

## 1. Problem Definition (CO1)

### Real-World Relevance

Accurate, automated land-use classification from satellite imagery is critical for:
- **Urban planning** — tracking expansion of residential/commercial zones
- **Environmental monitoring** — detecting deforestation and crop changes
- **Disaster response** — rapidly mapping affected regions
- **Agricultural intelligence** — identifying crop types and seasonal change

### Learning Task

| Attribute | Detail |
|---|---|
| **Task type** | Multi-class image classification |
| **Input** | RGB satellite image, resized to 128 × 128 pixels |
| **Output** | One of 10 land-use class labels |
| **Learning paradigm** | Supervised learning |

### Classes (10)

| # | Class | Description |
|---|---|---|
| 0 | AnnualCrop | Seasonal croplands |
| 1 | Forest | Dense tree cover |
| 2 | HerbaceousVegetation | Shrubs, meadows |
| 3 | Highway | Roads and motorways |
| 4 | Industrial | Factories, warehouses |
| 5 | Pasture | Grazing land |
| 6 | PermanentCrop | Orchards, vineyards |
| 7 | Residential | Urban housing |
| 8 | River | Waterways |
| 9 | SeaLake | Open water bodies |

---

## 2. Mathematical Modeling (CO2)

### Architecture Formulation

The network learns a mapping **f(x; θ) → ŷ** where:
- **x** = input image tensor ∈ ℝ^(3×128×128)
- **θ** = learnable parameters (~2.49M)
- **ŷ** = output logit vector ∈ ℝ^10

Each convolutional block applies:
```
h = MaxPool(ReLU(BatchNorm(Conv2d(x))))
```

### Loss Function

**Cross-Entropy Loss** — standard for multi-class classification:

```
L = -(1/N) Σ_i Σ_c y_ic · log(p_ic)
```

where `y_ic` is the one-hot true label and `p_ic = softmax(ŷ_ic)`.

Cross-Entropy combines LogSoftmax with Negative Log-Likelihood (NLLLoss) internally in PyTorch, providing numerically stable gradient computation.

### Optimisation Strategy

**Adam** (Adaptive Moment Estimation):
```
m_t = β₁ m_(t-1) + (1-β₁) g_t          # First moment (momentum)
v_t = β₂ v_(t-1) + (1-β₂) g_t²         # Second moment (variance)
θ_t = θ_(t-1) - α · m̂_t / (√v̂_t + ε) # Parameter update
```
- Learning rate α = 1×10⁻³
- β₁ = 0.9, β₂ = 0.999 (PyTorch defaults)
- Adam adapts the learning rate per parameter — converges faster than SGD and requires less tuning.

### Regularisation

**Dropout (p=0.5)**: randomly zeroes 50% of neurons in the fully connected layer during training. This prevents co-adaptation of neurons and reduces overfitting.

**BatchNorm**: normalises activations within each mini-batch, reducing internal covariate shift and acting as an implicit regulariser.

### Evaluation Metrics

| Metric | Formula | Our Score |
|---|---|---|
| **Accuracy** | correct / total | 91.00% |
| **Macro F1** | mean(2PR/(P+R)) per class | 90.90% |
| **Cohen's Kappa** | (p_o - p_e) / (1 - p_e) | **0.9000** (Excellent) |
| **Jaccard (macro)** | mean(TP/(TP+FP+FN)) per class | **0.8397** |

> **Note on IoU / Jaccard**: The Jaccard Index (TP/(TP+FP+FN)) is the classification analogue of IoU used in segmentation. For our image-level classification task this is the appropriate metric — per-image pixel-level IoU does not apply.

---

## 3. Model Architecture (CO3)

```
Input: (B, 3, 128, 128)
│
├── Conv Block 1: Conv(3→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
│   Output: (B, 32, 64, 64)
│
├── Conv Block 2: Conv(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
│   Output: (B, 64, 32, 32)
│
├── Conv Block 3: Conv(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
│   Output: (B, 128, 16, 16)
│
├── Conv Block 4: Conv(128→256, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
│   Output: (B, 256, 8, 8)
│
├── AdaptiveAvgPool → (B, 256, 4, 4) → Flatten → (B, 4096)
│
├── FC(4096 → 512) → ReLU → Dropout(0.5)
│
└── FC(512 → 10) → [Softmax at inference]
```

**Key design choices:**
- **3×3 kernels** with `padding=1`: preserves spatial size before pooling
- **Doubling filters**: 32→64→128→256 — captures increasingly abstract features
- **AdaptiveAvgPool**: resolution-agnostic — model works on any input size
- **Trained from scratch**: no transfer learning — demonstrates core CNN understanding

**Total Parameters: 2,490,250** | **Trainable: 2,490,250**

---

## 4. Data Preparation & Training (CO4)

### Dataset

| Split | Images | Classes | Source |
|---|---|---|---|
| Train | 5,000 | 10 × 500 | EuroSAT (original 64×64) |
| Val | 200 | 10 × 20 | Sampled from raw set |
| Test | 1,000 | 10 × 100 | EuroSAT held-out |

All images upscaled from **64×64 → 128×128** during training.

### Preprocessing Pipeline

**Training Transforms (with augmentation):**
```python
Resize(128×128) → RandomHorizontalFlip → RandomVerticalFlip
→ RandomRotation(15°) → ColorJitter → ToTensor → Normalize(ImageNet stats)
```

**Test/Val Transforms (no augmentation):**
```python
Resize(128×128) → ToTensor → Normalize(ImageNet stats)
```

Augmentations are applied **only during training** to artificially expand the dataset and teach invariance to orientation and lighting, while keeping evaluation deterministic.

### Training Config

| Setting | Value | Rationale |
|---|---|---|
| Epochs | 3 | Short hackathon — trained on Colab T4 GPU |
| Batch size | 32 | Balance of speed and gradient stability |
| Learning rate | 0.001 | Adam default — converges well for CNNs |
| Optimiser | Adam | Per-parameter adaptive rates |
| Loss | CrossEntropyLoss | Standard for multi-class classification |
| Framework | PyTorch 2.x | Industry standard, full GPU support |

### Training Environment

Trained on **Google Colab T4 GPU** (~15 minutes for 3 epochs).
See [`COLAB_TRAINING_GUIDE.md`](COLAB_TRAINING_GUIDE.md) for step-by-step instructions.

---

## 5. User Interface Demonstration (CO5)

A multi-page web application built with **Streamlit** (frontend) + **FastAPI** (backend API).

### Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Project overview, architecture summary, class gallery |
| 🔍 **Predict** | Upload image → get prediction with confidence chart |
| 📊 **Analytics** | Training curves, confusion matrix, classification report |

### Tech Stack

```
User Browser
    ↕ HTTP
Streamlit Frontend (port 8501)
    ↕ REST API (HTTP POST /predict)
FastAPI Backend (port 8000)
    ↕ torch.load()
Model Weights (saved_models/model.pt)
```

---

## 6. Project Structure

```
landuse-cnn/
├── src/
│   ├── dataset.py       # Data loading, transforms, DataLoaders
│   ├── model.py         # CNN architecture definition
│   ├── train.py         # Training loop and model saving
│   ├── evaluate.py      # Test evaluation, metrics, confusion matrix
│   └── utils.py         # Shared helpers (accuracy, plotting)
│
├── app/
│   ├── api.py           # FastAPI backend (predict, health, metrics)
│   ├── streamlit_app.py # Streamlit landing page + sidebar
│   ├── pages/
│   │   ├── 1_🏠_Home.py      # Home overview page
│   │   ├── 2_🔍_Predict.py   # Image prediction page
│   │   └── 3_📊_Analytics.py  # Model analytics page
│   ├── styles/
│   │   └── custom.css        # Dark theme + animations
│   └── assets/
│       └── class_descriptions.json  # Class metadata
│
├── outputs/
│   ├── classification_report.txt
│   └── plots/
│       ├── confusion_matrix.png
│       └── training_curves.png
│
├── saved_models/
│   └── model.pt            # Trained weights (~9.5MB)
│
├── scripts/
│   ├── download_data.py
│   └── prepare_data.py
│
├── COLAB_TRAINING_GUIDE.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Setup & Installation

### Prerequisites
- **Conda** (Miniconda or Anaconda)
- Python 3.10+

### Create Environment

```bash
conda create -n landuse-cnn python=3.10 -y
conda activate landuse-cnn
pip install -r requirements.txt
```

### Requirements Summary

```
torch, torchvision          # Deep learning framework
fastapi, uvicorn            # API server
streamlit                   # Frontend UI
scikit-learn                # Evaluation metrics
matplotlib, seaborn         # Plotting
plotly                      # Interactive charts
Pillow                      # Image I/O
tqdm                        # Progress bars
python-multipart            # FastAPI file uploads
```

---

## 8. Running Locally

### Step 1: Activate Environment

```bash
conda activate landuse-cnn
cd landuse-cnn
```

### Step 2: Start FastAPI Backend (Terminal 1)

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Verify at: http://localhost:8000/docs (Swagger UI)

### Step 3: Start Streamlit Frontend (Terminal 2)

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open: http://localhost:8501

### Optional: Re-run Evaluation

```bash
python src/evaluate.py
```
Regenerates `outputs/classification_report.txt` and `outputs/plots/*.png`.

---

## 9. Deployment

### Option A: HuggingFace Spaces (Recommended — Free)

HuggingFace Spaces natively supports Streamlit apps with zero configuration.

**Step 1:** Create a new Space at https://huggingface.co/new-space
- SDK: Streamlit
- Visibility: Public

**Step 2:** Push the code

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>

# Copy project files
cp -r /path/to/landuse-cnn/* .

# HuggingFace Spaces reads requirements.txt automatically
git add .
git commit -m "Initial deploy"
git push
```

**Step 3:** Upload model weights via the HuggingFace Hub UI (Settings → Files)

> ⚠️ `model.pt` (~9.5MB) and `data/` are excluded from git via `.gitignore`.  
> Upload `saved_models/model.pt` manually via the HF Files UI or use `huggingface_hub`.

**Note:** For HuggingFace Spaces, the FastAPI backend runs as a subprocess inside the same container. The `app.py` file (Space entry point) should start both servers:

```python
# app.py (Space entry)
import subprocess, time, streamlit_app
subprocess.Popen(["uvicorn", "app.api:app", "--port", "8000"])
time.sleep(2)  # Wait for API to start
```

---

### Option B: Local + ngrok (Quick Public Demo)

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8501   # Exposes Streamlit to the internet
```

---

### Option C: Google Cloud Run / AWS / Azure

Containerise with Docker:

```dockerfile
# Dockerfile (example)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD bash -c "uvicorn app.api:app --port 8000 & streamlit run app/streamlit_app.py --server.port 8501"
```

---

## 10. Results

### Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **91.00%** |
| Macro Precision | 91.34% |
| Macro Recall | 91.00% |
| Macro F1-score | 90.90% |
| Cohen's Kappa | **0.9000** (Excellent) |
| Jaccard Index (macro) | **0.8397** |

### Per-Class Results

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| AnnualCrop | 0.929 | 0.910 | 0.919 |
| Forest | 0.971 | 1.000 | 0.985 |
| HerbaceousVegetation | 0.888 | 0.710 | 0.789 |
| Highway | 0.803 | 0.940 | 0.866 |
| Industrial | 0.989 | 0.920 | 0.953 |
| Pasture | 0.980 | 0.970 | 0.975 |
| PermanentCrop | 0.807 | 0.880 | 0.842 |
| Residential | 0.870 | 1.000 | 0.930 |
| River | 0.908 | 0.790 | 0.845 |
| SeaLake | 0.990 | 0.980 | 0.985 |

Best: **Forest** (F1=0.985), **SeaLake** (F1=0.985)  
Most challenging: **HerbaceousVegetation** (confused with PermanentCrop / Pasture)

---

## Authors

Built for **NeuralHack 2026** — End Trimester Deep Learning Examination  
Framework: **PyTorch** | Training: **Google Colab T4 GPU**
