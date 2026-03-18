# =============================================================
# app/streamlit_app.py
# =============================================================
# Main entry point for the Streamlit web application.
# This file sets up:
#   - Global page configuration (title, icon, layout)
#   - Custom CSS theming (dark mode + animations)
#   - Sidebar with branding and API status indicator
#   - Landing page with navigation cards
#
# Streamlit multi-page apps: any .py file in app/pages/ is
# automatically discovered and shown as a separate sidebar page.
#
# Run from project root:
#   streamlit run app/streamlit_app.py
# =============================================================

import streamlit as st
from pathlib import Path

# ── Page Config ────────────────────────────────────────────────────────
# MUST be the first Streamlit call in the script.
# layout="wide" uses the full browser width instead of the default narrow column.
st.set_page_config(
    page_title="LandUse CNN — Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Injection ───────────────────────────────────────────────
# Streamlit renders in a browser, so we can inject raw CSS to override
# the default Streamlit theme. The CSS lives in a separate file for
# maintainability. We use st.markdown with unsafe_allow_html=True to
# inject a <style> block into the page head.
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
# The sidebar is persistent across all pages in the multi-page app.
with st.sidebar:
    st.markdown("# 🛰️ LandUse CNN")
    st.caption("Satellite Image Land Use Classification")
    st.divider()

    # ── Model / API Status Indicator ───────────────────────────────────
    # Check if model file exists locally (works on Streamlit Cloud).
    # Also try the FastAPI backend as a fallback (for local dev).
    from pathlib import Path as _P
    model_exists = (_P(__file__).resolve().parent.parent / "saved_models" / "model.pt").exists()

    if model_exists:
        st.success("🟢 Model Ready (local inference)")
    else:
        import requests
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200 and r.json().get("model_loaded"):
                st.success("🟢 API Online · Model Ready")
            else:
                st.warning("🟡 API Online · No Model")
        except Exception:
            st.error("🔴 No model found")

    st.divider()
    st.markdown(
        """
        **Quick Links**
        - 🏠 Home — Project Overview
        - 🔍 Predict — Classify Images
        - 📊 Analytics — Model Metrics
        """,
    )
    st.divider()
    st.caption("Built with PyTorch · FastAPI · Streamlit")

# ── Landing Page ────────────────────────────────────────────────────────
# The main (root) page shows a hero section and 3 navigation cards.
# Users click through to the specific pages in the sidebar.
st.markdown(
    """
    <div style="text-align: center; padding: 60px 20px;">
        <div class="hero-title">🛰️ LandUse CNN</div>
        <p class="hero-subtitle" style="margin-top: 12px;">
            Deep Learning-powered Satellite Image Classification
        </p>
        <p style="color: #718096; margin-top: 8px; font-size: 1rem;">
            Select a page from the sidebar to get started
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Navigation Cards ───────────────────────────────────────────────────
# Three columns with styled cards, one per page. These are purely visual —
# the actual navigation is done via the sidebar. HTML/CSS cards are used
# because Streamlit's native card component doesn't support custom styling.
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card" style="text-align: center; min-height: 160px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🏠</div>
            <div style="font-weight: 600; color: #e2e8f0; font-size: 1.1rem;">Home</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 8px;">
                Learn about the project, dataset, and model architecture
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card" style="text-align: center; min-height: 160px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🔍</div>
            <div style="font-weight: 600; color: #e2e8f0; font-size: 1.1rem;">Predict</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 8px;">
                Upload satellite images and get instant land use predictions
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card" style="text-align: center; min-height: 160px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
            <div style="font-weight: 600; color: #e2e8f0; font-size: 1.1rem;">Analytics</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 8px;">
                Explore model performance, confusion matrix, and metrics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
