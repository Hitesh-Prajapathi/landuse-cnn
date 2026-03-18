# =============================================================
# app/pages/2_🔍_Predict.py
# =============================================================
# Streamlit page for interactive satellite image classification.
# User flow:
#   1. Upload a satellite image (JPG/PNG)
#   2. Preview is displayed
#   3. Image is sent to FastAPI /predict endpoint
#   4. Result displayed: predicted class, confidence, full
#      probability distribution (Plotly bar chart), and a
#      description card for the predicted land-use type.
# =============================================================

import streamlit as st
import requests
import json
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# ── Page Config ────────────────────────────────────────────────────────
# Each page in the multi-page app sets its own config; this title appears
# in the browser tab when this page is active.
st.set_page_config(page_title="Predict — LandUse CNN", page_icon="🔍", layout="wide")

# ── Load CSS ───────────────────────────────────────────────────────────
# Navigate from pages/ up to app/ to find the shared styles file.
css_path = Path(__file__).parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load Class Metadata ────────────────────────────────────────────────
# JSON file maps each class name to: description, icon emoji, and hex color.
# Used to display rich context after a prediction is made.
assets_path = Path(__file__).parent.parent / "assets" / "class_descriptions.json"
with open(assets_path) as f:
    class_info = json.load(f)

# The FastAPI backend URL — must match the running server
API_URL = "http://localhost:8000"

# ── Page Header ────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0 10px;">
        <div class="hero-title" style="font-size: 2.2rem;">🔍 Image Prediction</div>
        <p class="hero-subtitle" style="font-size: 1rem;">
            Upload a satellite image to classify its land use type
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ── Upload & Preview Section ───────────────────────────────────────────
# Two-column layout: left = upload widget, right = image preview.
# We use equal columns [1, 1] for a balanced look.
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("### 📤 Upload Image")
    # st.file_uploader handles the browser drag-and-drop / file dialog.
    # The returned object has .read(), .getvalue(), .name, .type attributes.
    uploaded_file = st.file_uploader(
        "Choose a satellite image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a 64×64 or 128×128 satellite image for classification"
    )

    # Show placeholder card when no file is uploaded yet
    if uploaded_file is None:
        st.markdown(
            """
            <div class="info-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 3rem; margin-bottom: 12px;">🛰️</div>
                <p style="color: #718096;">
                    Drag and drop or browse to upload a satellite image.<br>
                    Supported formats: <b>JPG, PNG</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_preview:
    # Show image preview as soon as a file is uploaded
    if uploaded_file is not None:
        st.markdown("### 🖼️ Preview")
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=f"{uploaded_file.name} ({image.size[0]}×{image.size[1]})",
            use_container_width=True
        )

# ── Prediction Section ─────────────────────────────────────────────────
# Only runs if a file has been uploaded.
if uploaded_file is not None:
    st.divider()

    # ── Inference: Direct model call (works on Streamlit Cloud) ────────
    # We first try direct inference (no server needed). If that fails,
    # we fall back to calling the FastAPI backend (for local dev).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from app.inference import predict as direct_predict, is_model_available

    result = None

    if is_model_available():
        # Direct inference — no FastAPI server required
        with st.spinner("🧠 Analyzing satellite image..."):
            uploaded_file.seek(0)
            pil_image = Image.open(uploaded_file)
            try:
                result = direct_predict(pil_image)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        # Fallback: try the FastAPI backend (local development)
        try:
            health = requests.get(f"{API_URL}/health", timeout=2)
            api_ok = health.status_code == 200 and health.json().get("model_loaded")
        except Exception:
            api_ok = False

        if not api_ok:
            st.error(
                "🔴 **Model not found and API not available.**\n\n"
                "Place `model.pt` in `saved_models/` or start the FastAPI backend:\n"
                "```\nuvicorn app.api:app --host 0.0.0.0 --port 8000\n```"
            )
        else:
            with st.spinner("🧠 Analyzing satellite image..."):
                uploaded_file.seek(0)
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                )
                if response.status_code == 200:
                    result = response.json()
                else:
                    st.error(f"Prediction failed: {response.text}")

    if result is not None:
        predicted_class = result["predicted_class"]
        confidence      = result["confidence"]
        all_probs       = result["all_probabilities"]

        # ── Result Display ─────────────────────────────────────────
        # Fetch class-specific metadata (icon, description, accent color)
        info        = class_info.get(predicted_class, {})
        icon        = info.get("icon", "🏷️")
        color       = info.get("color", "#667eea")
        description = info.get("description", "")

        # ── Main Prediction Banner ────────────────────────────────
        # Large, styled card showing the decisive result at a glance.
        st.markdown(
            f"""
            <div class="prediction-card">
                <p style="color: #a0aec0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                    Predicted Land Use
                </p>
                <div class="prediction-class">{icon} {predicted_class}</div>
                <div class="prediction-confidence" style="margin-top: 8px;">{confidence}%</div>
                <p style="color: #a0aec0; font-size: 0.85rem; margin-top: 4px;">Confidence Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        # ── Chart & Description Side by Side ──────────────────────
        # 3:2 ratio: chart is wider, description narrower
        col_chart, col_info = st.columns([3, 2])

        with col_chart:
            st.markdown("### 📊 Class Probabilities")

            # Sort ascending so the highest bar appears at the top
            # of the horizontal bar chart (Plotly renders bottom-to-top)
            sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1]))
            classes = list(sorted_probs.keys())
            probs   = list(sorted_probs.values())

            # Highlight the predicted class bar in accent colour;
            # all others shown at 25% opacity for visual contrast.
            colors = [
                "#667eea" if cls == predicted_class else "rgba(102, 126, 234, 0.25)"
                for cls in classes
            ]

            # Plotly horizontal bar chart
            fig = go.Figure(go.Bar(
                x=probs,
                y=classes,
                orientation='h',
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12),
            ))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a0aec0', family='Inter'),
                xaxis=dict(
                    title="Confidence (%)",
                    range=[0, max(probs) * 1.25],
                    gridcolor='rgba(255,255,255,0.05)',
                    showgrid=True,
                ),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                margin=dict(l=0, r=40, t=10, b=40),
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("### 🏷️ About This Class")
            st.markdown(
                f"""
                <div class="info-card">
                    <div style="font-size: 3rem; margin-bottom: 12px;">{icon}</div>
                    <h3 style="color: {color}; margin-bottom: 12px;">{predicted_class}</h3>
                    <p style="color: #e2e8f0; line-height: 1.7;">{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── Top-3 Predictions ─────────────────────────────────
            st.markdown("")
            st.markdown("**Top 3 Predictions:**")
            top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for rank, (cls, prob) in enumerate(top3, 1):
                cls_icon = class_info.get(cls, {}).get("icon", "🏷️")
                medal    = ["🥇", "🥈", "🥉"][rank - 1]
                st.markdown(f"{medal} **{cls}** {cls_icon} — {prob:.1f}%")

