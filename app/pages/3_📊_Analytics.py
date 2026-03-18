# =============================================================
# app/pages/3_📊_Analytics.py
# =============================================================
# Analytics dashboard — displays all evaluation artefacts generated
# by src/evaluate.py after training is complete:
#   - Key metrics (Accuracy, Kappa, Jaccard)
#   - Training curves (loss + accuracy per epoch)
#   - Confusion matrix heatmap
#   - Per-class classification report table
#   - Per-class F1-score bar chart
# =============================================================

import streamlit as st
import requests
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics — LandUse CNN", page_icon="📊", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load class descriptions
assets_path = Path(__file__).parent.parent / "assets" / "class_descriptions.json"
with open(assets_path) as f:
    class_info = json.load(f)

# Project root
project_root = Path(__file__).resolve().parent.parent.parent

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0 10px;">
        <div class="hero-title" style="font-size: 2.2rem;">📊 Model Analytics</div>
        <p class="hero-subtitle" style="font-size: 1rem;">
            Explore model performance, accuracy metrics, and visualizations
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ── Parse classification report ───────────────────────────────────────
# The report is a plain-text file written by src/evaluate.py.
# We parse it line by line to extract structured metrics without
# needing a dedicated storage system.
report_path = project_root / "outputs" / "classification_report.txt"

test_accuracy  = None
kappa          = None
jaccard_macro  = None
class_metrics  = []

if report_path.exists():
    with open(report_path) as f:
        report_text = f.read()

    # Parse scalar metrics from the first few lines of the report
    for line in report_text.split("\n"):
        if "Test Accuracy" in line:
            try:
                test_accuracy = float(line.split(":")[1].strip().replace("%", ""))
            except (ValueError, IndexError):
                pass
        if "Cohen's Kappa" in line:
            try:
                kappa = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        if "Jaccard Index (macro avg)" in line:
            try:
                jaccard_macro = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    # Parse per-class Jaccard scores from the "Jaccard Index per class:" block.
    # Format:  "  ClassName    0.8505"
    jaccard_per_class = {}
    in_jaccard_block = False
    for line in report_text.split("\n"):
        if "Jaccard Index per class" in line:
            in_jaccard_block = True
            continue
        if in_jaccard_block:
            stripped = line.strip()
            if stripped == "":
                in_jaccard_block = False  # blank line ends the block
                continue
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    jaccard_per_class[parts[0]] = float(parts[1])
                except ValueError:
                    pass

    # Parse per-class precision/recall/f1/support from sklearn report block.
    # Each class line has exactly 5 space-separated tokens: name p r f1 support.
    for line in report_text.split("\n"):
        parts = line.split()
        if len(parts) == 5:
            try:
                name = parts[0]
                if name in class_info:
                    class_metrics.append({
                        "class":     name,
                        "precision": float(parts[1]),
                        "recall":    float(parts[2]),
                        "f1":        float(parts[3]),
                        "support":   int(parts[4]),
                        "jaccard":   jaccard_per_class.get(name, None)
                    })
            except (ValueError, IndexError):
                pass

# ── Key Metrics Row ───────────────────────────────────────────────────
# Display the most important numbers at the top of the page so evaluators
# see them immediately. Six cards across two rows.
st.markdown("### 🎯 Key Metrics")
st.markdown("")

# Row 1
c1, c2, c3 = st.columns(3)
c1.metric("Test Accuracy",     f"{test_accuracy:.1f}%" if test_accuracy else "N/A")
c2.metric("Cohen's Kappa",     f"{kappa:.4f}" if kappa else "N/A",
          help="Measures agreement beyond chance. > 0.80 = Excellent.")
c3.metric("Jaccard (macro)",   f"{jaccard_macro:.4f}" if jaccard_macro else "N/A",
          help="TP/(TP+FP+FN) per class, averaged. Analogue of IoU for classification.")

# Row 2
c4, c5, c6 = st.columns(3)
c4.metric("Total Classes", "10")
c5.metric("Model Parameters", "2.49M")
c6.metric("Training Epochs", "15")

st.divider()

# ── Tabs for different visualizations ────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Training Curves",
    "🎯 Confusion Matrix",
    "📋 Classification Report",
    "📊 Per-Class Performance"
])

# ── Tab 1: Training Curves ───────────────────────────────────────────
with tab1:
    st.markdown("### 📈 Training Progress")
    st.caption("Loss and accuracy curves over training epochs")
    st.markdown("")

    curves_path = project_root / "outputs" / "plots" / "training_curves.png"

    if curves_path.exists():
        img = Image.open(curves_path)
        st.image(img, use_container_width=True)
    else:
        st.info("Training curves not found. Run training first to generate plots.")

# ── Tab 2: Confusion Matrix ──────────────────────────────────────────
with tab2:
    st.markdown("### 🎯 Confusion Matrix")
    st.caption("Shows how often each class was predicted correctly vs. misclassified")
    st.markdown("")

    cm_path = project_root / "outputs" / "plots" / "confusion_matrix.png"

    if cm_path.exists():
        img = Image.open(cm_path)
        
        # Use columns to constrain width instead of stretching to full container
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(img, use_container_width=True)

        st.markdown("")
        st.markdown(
            """
            <div class="info-card">
                <p style="color: #e2e8f0; line-height: 1.7;">
                    <b>How to read:</b> Each row represents the <b>true class</b> and each column
                    represents the <b>predicted class</b>. Diagonal cells show correct predictions.
                    Darker cells = more images. Off-diagonal cells reveal common misclassifications.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Confusion matrix not found. Run evaluation first to generate.")

# ── Tab 3: Classification Report ─────────────────────────────────────
with tab3:
    st.markdown("### 📋 Detailed Classification Report")
    st.caption("Per-class precision, recall, F1-score, and Jaccard Index")
    st.markdown("")

    if class_metrics:
        # 7-column table: Class | Icon | Precision | Recall | F1 | Jaccard | Support
        header_cols = st.columns([2, 0.6, 1, 1, 1, 1, 0.8])
        headers = ["Class", "Icon", "Precision", "Recall", "F1-Score", "Jaccard", "Support"]
        for col, header in zip(header_cols, headers):
            col.markdown(f"**{header}**")

        st.divider()

        for m in class_metrics:
            cols = st.columns([2, 0.6, 1, 1, 1, 1, 0.8])
            icon = class_info.get(m["class"], {}).get("icon", "")
            j    = m.get("jaccard")

            cols[0].markdown(f"**{m['class']}**")
            cols[1].markdown(f"{icon}")
            cols[2].markdown(f"`{m['precision']:.4f}`")
            cols[3].markdown(f"`{m['recall']:.4f}`")
            cols[4].markdown(f"`{m['f1']:.4f}`")
            cols[5].markdown(f"`{j:.4f}`" if j is not None else "`—`")
            cols[6].markdown(f"`{m['support']}`")

        st.divider()

        # ── Summary metrics at the bottom of the report ────────────────
        sum_c1, sum_c2, sum_c3 = st.columns(3)
        if test_accuracy:
            sum_c1.markdown(
                f"<div style='text-align:center'><b style='color:#e2e8f0'>Overall Accuracy</b><br>"
                f"<span style='font-size:1.4rem;font-weight:700;color:#667eea'>{test_accuracy:.2f}%</span></div>",
                unsafe_allow_html=True
            )
        if kappa is not None:
            sum_c2.markdown(
                f"<div style='text-align:center'><b style='color:#e2e8f0'>Cohen's Kappa</b><br>"
                f"<span style='font-size:1.4rem;font-weight:700;color:#48bb78'>{kappa:.4f}</span>"
                f"<br><span style='color:#a0aec0;font-size:0.8rem'>Excellent (&gt;0.80)</span></div>",
                unsafe_allow_html=True
            )
        if jaccard_macro is not None:
            sum_c3.markdown(
                f"<div style='text-align:center'><b style='color:#e2e8f0'>Jaccard (macro avg)</b><br>"
                f"<span style='font-size:1.4rem;font-weight:700;color:#f6ad55'>{jaccard_macro:.4f}</span>"
                f"<br><span style='color:#a0aec0;font-size:0.8rem'>IoU analogue for classification</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Classification report not found. Run evaluation first.")

# ── Tab 4: Per-Class Performance ─────────────────────────────────────
with tab4:
    st.markdown("### 📊 Per-Class F1 Score & Jaccard Index")
    st.caption("Compare F1 and Jaccard (IoU analogue) across all 10 land use categories — sorted by F1")
    st.markdown("")

    if class_metrics:
        classes    = [m["class"] for m in class_metrics]
        f1_scores  = [m["f1"] for m in class_metrics]
        jac_scores = [m.get("jaccard") or 0.0 for m in class_metrics]
        icons      = [class_info.get(c, {}).get("icon", "") for c in classes]
        colors_list = [class_info.get(c, {}).get("color", "#667eea") for c in classes]

        # Sort by F1 score ascending (highest at top of horizontal chart)
        sorted_data = sorted(
            zip(classes, f1_scores, jac_scores, icons, colors_list),
            key=lambda x: x[1]
        )
        classes, f1_scores, jac_scores, icons, colors_list = zip(*sorted_data)
        y_labels = [f"{icon} {cls}" for cls, icon in zip(classes, icons)]

        # ── Grouped bar chart: F1 and Jaccard side by side ─────────────
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="F1-Score",
            x=list(f1_scores),
            y=y_labels,
            orientation='h',
            marker_color="#667eea",
            text=[f"{v:.4f}" for v in f1_scores],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
            offsetgroup=0,
        ))

        fig.add_trace(go.Bar(
            name="Jaccard Index",
            x=list(jac_scores),
            y=y_labels,
            orientation='h',
            marker_color="#f6ad55",
            text=[f"{v:.4f}" for v in jac_scores],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
            offsetgroup=1,
        ))

        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0', family='Inter'),
            legend=dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='right',  x=1,
                font=dict(color='#e2e8f0')
            ),
            xaxis=dict(
                title="Score",
                range=[0, 1.2],
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
            ),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=0, r=70, t=30, b=40),
            height=520,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quick insights
        st.markdown("")
        best_class = class_metrics[0]
        worst_class = class_metrics[0]
        for m in class_metrics:
            if m["f1"] > best_class["f1"]:
                best_class = m
            if m["f1"] < worst_class["f1"]:
                worst_class = m

        best_icon = class_info.get(best_class["class"], {}).get("icon", "")
        worst_icon = class_info.get(worst_class["class"], {}).get("icon", "")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4 style="color: #48bb78;">✅ Strongest Class</h4>
                    <p style="color: #e2e8f0; font-size: 1.3rem; font-weight: 700; margin-top: 8px;">
                        {best_icon} {best_class['class']}
                    </p>
                    <p style="color: #a0aec0;">F1: {best_class['f1']:.4f} | Precision: {best_class['precision']:.4f} | Recall: {best_class['recall']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4 style="color: #fc8181;">⚠️ Needs Improvement</h4>
                    <p style="color: #e2e8f0; font-size: 1.3rem; font-weight: 700; margin-top: 8px;">
                        {worst_icon} {worst_class['class']}
                    </p>
                    <p style="color: #a0aec0;">F1: {worst_class['f1']:.4f} | Precision: {worst_class['precision']:.4f} | Recall: {worst_class['recall']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Per-class metrics not available. Run evaluation first.")
