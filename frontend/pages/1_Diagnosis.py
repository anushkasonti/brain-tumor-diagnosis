# frontend/pages/1_Diagnosis.py

import os
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

# -------------------------------------------------------------------
# Make project root importable and bring in backend pipeline
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(  # pages/ -> frontend/
        os.path.dirname(os.path.abspath(__file__))  # frontend/ -> project root
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.pipeline import full_pipeline_from_array  # noqa: E402

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def apply_theme_css():
    theme = st.session_state.get("theme", "Dark")
    if theme == "Dark":
        bg = "#020617"
        card_bg = "rgba(15,23,42,0.96)"
        text = "#e5e7eb"
    else:
        bg = "#f5f5f7"
        card_bg = "rgba(255,255,255,0.96)"
        text = "#111827"

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg};
            color: {text};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .glass-card {{
            background: {card_bg};
            border-radius: 18px;
            padding: 20px 22px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.35);
        }}
        .prob-bar {{
            width: 100%;
            height: 18px;
            border-radius: 999px;
            background: rgba(148,163,184,0.25);
            overflow: hidden;
        }}
        .prob-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg,#38bdf8,#22c55e);
            width: 0%;
            transition: width 900ms ease-out;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return np.array(img)


def apply_overlay(img, mask, color_rgb, opacity=0.5):
    if mask is None:
        return img
    overlay = img.copy().astype(np.float32)
    tumor_mask = mask > 0
    r, g, b = color_rgb
    for c_idx, c_val in enumerate([r, g, b]):
        overlay[tumor_mask, c_idx] = (
            overlay[tumor_mask, c_idx] * (1 - opacity) + c_val * opacity
        )
    return overlay.astype(np.uint8)


apply_theme_css()

# -------------------------------------------------------------------
# Sidebar – controls
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Diagnosis Controls")

opacity = st.sidebar.slider("Overlay opacity", 10, 90, 60, 5) / 100
color_choice = st.sidebar.selectbox("Overlay color", ["Red", "Green", "Blue", "Yellow"])

color_map = {
    "Red": (255, 80, 80),
    "Green": (16, 185, 129),
    "Blue": (59, 130, 246),
    "Yellow": (250, 204, 21),
}

# -------------------------------------------------------------------
# Main layout
# -------------------------------------------------------------------
st.markdown("## 🧠 Diagnosis")
st.markdown(
    "Upload a brain MRI scan to run **detection**, **classification**, and **segmentation**."
)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload MRI (PNG/JPG)", type=["png", "jpg", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded is None:
    st.info("Please upload an MRI scan to start.")
    st.stop()

# -------------------------------------------------------------------
# Run pipeline
# -------------------------------------------------------------------
img_rgb = load_image(uploaded)

with st.spinner("Running AI models on the MRI..."):
    result = full_pipeline_from_array(img_rgb)

has_tumor = bool(result["has_tumor"])
det_prob = float(result["detection_prob"])
label = result.get("predicted_label")
class_probs = result.get("class_probs") or {}
mask = result.get("segmentation_mask")

overlay = apply_overlay(img_rgb, mask, color_map[color_choice], opacity)

# Save into session_state so Report page can use it
st.session_state.last_result = {
    "has_tumor": has_tumor,
    "detection_prob": det_prob,
    "predicted_label": label,
    "class_probs": class_probs,
    "mask": mask,
    "original_image": img_rgb,
    "overlay_image": overlay,
}

# -------------------------------------------------------------------
# Before / After slider
# -------------------------------------------------------------------
st.markdown("### 🩻 Visual Comparison")

image_comparison(
    img1=Image.fromarray(img_rgb),
    img2=Image.fromarray(overlay),
    label1="Original",
    label2=f"Overlay ({color_choice})",
    width=500,
    starting_position=1.0,   # ⭐ Start fully on the RIGHT
)

st.caption(
    f"Overlay opacity: **{int(opacity*100)}%**, color: **{color_choice}**. Slide the divider to compare."
)

st.markdown("---")

# -------------------------------------------------------------------
# Clinical metrics
# -------------------------------------------------------------------
st.markdown("### 📊 AI Findings")
mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Detection")
    if has_tumor:
        st.success("Tumor detected")
    else:
        st.info("No tumor detected")
    st.write(f"Confidence: **{det_prob:.2%}**")
    st.markdown("</div>", unsafe_allow_html=True)

with mc2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Classification")
    if has_tumor and label:
        st.write(f"Predicted type: **{label.upper()}**")
    else:
        st.write("N/A (no tumor detected)")
    st.markdown("</div>", unsafe_allow_html=True)

with mc3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Segmentation")
    if mask is not None:
        tumor_pixels = int(np.sum(mask > 0))
        total_pixels = int(mask.size)
        coverage = tumor_pixels / total_pixels * 100 if total_pixels > 0 else 0.0
        st.write(f"Tumor pixels: **{tumor_pixels:,}**")
        st.write(f"Coverage: **{coverage:.2f}%**")
    else:
        st.write("Segmentation not available.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Animated probability bars
# -------------------------------------------------------------------
if has_tumor and class_probs:
    st.markdown("---")
    st.markdown("### 📈 Class probabilities")

    df_probs = pd.DataFrame(
        [{"class": k, "prob": float(v)} for k, v in class_probs.items()]
    ).sort_values("prob", ascending=False)

    for _, row in df_probs.iterrows():
        c_label = row["class"]
        p = float(row["prob"])
        pct = int(p * 100)
        st.markdown(f"**{c_label}** — {pct}%")
        st.markdown(
            f"""
            <div class="prob-bar">
              <div class="prob-fill" style="width:{pct}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.success(
    "Diagnosis completed. You can now go to the **Report** page to enter doctor notes and export a PDF."
)
