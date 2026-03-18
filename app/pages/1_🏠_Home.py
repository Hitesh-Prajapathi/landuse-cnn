# app/pages/1_🏠_Home.py
# Project Overview, Technical Summary, and Class Gallery

import streamlit as st
import json
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Home — LandUse CNN", page_icon="🏠", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load class descriptions
assets_path = Path(__file__).parent.parent / "assets" / "class_descriptions.json"
with open(assets_path) as f:
    class_info = json.load(f)

# ── Hero Section ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 40px 20px 20px;">
        <div class="hero-title">🛰️ Satellite Land Use Classification</div>
        <p class="hero-subtitle" style="margin-top: 12px;">
            AI-powered analysis of satellite imagery using Deep Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ── Project Overview ─────────────────────────────────────────────────
st.markdown("## 🌍 About This Project")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        """
        **Land use classification** is one of the most critical applications of
        remote sensing and satellite imagery analysis. Understanding how land is
        utilized — whether for agriculture, urban development, or natural
        preservation — is essential for:

        - 🏙️ **Urban Planning** — Monitor city expansion and infrastructure growth
        - 🌾 **Agriculture** — Track crop patterns and optimize farming decisions
        - 🌳 **Environmental Conservation** — Detect deforestation and land degradation
        - 🌊 **Water Resource Management** — Monitor lakes, rivers, and water bodies
        - 🛣️ **Infrastructure Development** — Plan roads and industrial zones

        This project uses a **Convolutional Neural Network (CNN)** trained on the
        **EuroSAT dataset** — a benchmark collection of Sentinel-2 satellite images
        covering 10 distinct land use categories across Europe.
        """
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h4 style="color: #667eea; margin-bottom: 16px;">📋 Quick Facts</h4>
            <table style="width: 100%; color: #e2e8f0;">
                <tr><td style="padding: 6px 0; color: #a0aec0;">Dataset</td><td style="padding: 6px 0; font-weight: 600;">EuroSAT (Sentinel-2)</td></tr>
                <tr><td style="padding: 6px 0; color: #a0aec0;">Classes</td><td style="padding: 6px 0; font-weight: 600;">10 Land Use Types</td></tr>
                <tr><td style="padding: 6px 0; color: #a0aec0;">Training Images</td><td style="padding: 6px 0; font-weight: 600;">5,000</td></tr>
                <tr><td style="padding: 6px 0; color: #a0aec0;">Test Images</td><td style="padding: 6px 0; font-weight: 600;">1,000</td></tr>
                <tr><td style="padding: 6px 0; color: #a0aec0;">Input Size</td><td style="padding: 6px 0; font-weight: 600;">128 × 128 × 3 (RGB)</td></tr>
                <tr><td style="padding: 6px 0; color: #a0aec0;">Test Accuracy</td><td style="padding: 6px 0; font-weight: 600; color: #48bb78;">91.00%</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ── Technical Summary ────────────────────────────────────────────────
st.markdown("## 🧠 Model Architecture")

st.markdown(
    """
    <div class="info-card" style="font-family: monospace; line-height: 1.8; color: #e2e8f0;">
        <span style="color: #667eea; font-weight: 700;">Input</span> (3, 128, 128) RGB Satellite Image<br>
        &nbsp;&nbsp;→ <span style="color: #48bb78;">Conv2d(3→32)</span> → BatchNorm → ReLU → MaxPool &nbsp; <span style="color: #718096;">→ (32, 64, 64)</span><br>
        &nbsp;&nbsp;→ <span style="color: #48bb78;">Conv2d(32→64)</span> → BatchNorm → ReLU → MaxPool &nbsp; <span style="color: #718096;">→ (64, 32, 32)</span><br>
        &nbsp;&nbsp;→ <span style="color: #48bb78;">Conv2d(64→128)</span> → BatchNorm → ReLU → MaxPool &nbsp; <span style="color: #718096;">→ (128, 16, 16)</span><br>
        &nbsp;&nbsp;→ <span style="color: #48bb78;">Conv2d(128→256)</span> → BatchNorm → ReLU → MaxPool &nbsp; <span style="color: #718096;">→ (256, 8, 8)</span><br>
        &nbsp;&nbsp;→ <span style="color: #f6ad55;">AdaptiveAvgPool</span>(4, 4) &nbsp; <span style="color: #718096;">→ (256, 4, 4)</span><br>
        &nbsp;&nbsp;→ Flatten → <span style="color: #fc8181;">Linear(4096→512)</span> → ReLU → Dropout(0.5)<br>
        &nbsp;&nbsp;→ <span style="color: #667eea; font-weight: 700;">Output</span> <span style="color: #fc8181;">Linear(512→10)</span> → Class Probabilities
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# Key details
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Parameters", "2.49M")
c2.metric("Framework", "PyTorch")
c3.metric("Optimizer", "Adam")
c4.metric("Loss Function", "CrossEntropy")

st.divider()

# ── Training Details ─────────────────────────────────────────────────
st.markdown("## ⚡ Training Pipeline")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Data Preprocessing:**
        - Resized from 64×64 to **128×128** pixels
        - Normalized using ImageNet statistics
        - **Train augmentation**: Random flip, rotation, color jitter

        **Training Configuration:**
        - Trained on **Google Colab GPU** (NVIDIA T4)
        - **15 epochs** with Adam optimizer (lr=0.0005)
        - Batch size: 32
        """
    )

with col2:
    st.markdown(
        """
        **Architecture Highlights:**
        - **4 convolutional blocks** with progressive filter widths
        - **Batch normalization** after each conv layer for stable training
        - **Adaptive pooling** for input size flexibility
        - **Dropout (50%)** to prevent overfitting
        - Custom CNN — **no pretrained weights** used

        **Result:** 91% test accuracy across 10 classes
        """
    )

st.divider()

# ── Class Gallery ────────────────────────────────────────────────────
st.markdown("## 🗂️ Land Use Classes")
st.caption("The model classifies satellite images into these 10 categories:")

st.markdown("")

# Display in a 5x2 grid
rows = [list(class_info.items())[i:i+5] for i in range(0, 10, 5)]

for row in rows:
    cols = st.columns(5)
    for col, (cls_name, info) in zip(cols, row):
        with col:
            st.markdown(
                f"""
                <div class="class-card">
                    <div class="icon">{info['icon']}</div>
                    <div class="name">{cls_name}</div>
                    <p style="color: #718096; font-size: 0.75rem; margin-top: 8px; line-height: 1.4;">
                        {info['description'][:80]}...
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("")

st.divider()

# ── How to Use ───────────────────────────────────────────────────────
st.markdown("## 🚀 How to Use")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card" style="text-align: center; min-height: 140px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📤</div>
            <div style="font-weight: 600; color: #e2e8f0;">Step 1: Upload</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 8px;">
                Go to the <b>Predict</b> page and upload a satellite image
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="info-card" style="text-align: center; min-height: 140px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🧠</div>
            <div style="font-weight: 600; color: #e2e8f0;">Step 2: Analyze</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 8px;">
                The CNN model processes your image through 4 conv blocks
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="info-card" style="text-align: center; min-height: 140px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
            <div style="font-weight: 600; color: #e2e8f0;">Step 3: Results</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 8px;">
                View the predicted land use class with confidence scores
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
