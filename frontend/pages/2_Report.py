# frontend/pages/2_Report.py

import os
import sys
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------------------------------------------------------
# Make project root importable (if you ever need backend here)
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


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
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme_css()

st.markdown("## 📄 Report & Doctor Notes")

if "last_result" not in st.session_state:
    st.warning("No diagnosis found. Please run an analysis on the **Diagnosis** page first.")
    st.stop()

res = st.session_state.last_result
img_rgb = res["original_image"]
overlay = res["overlay_image"]
has_tumor = res["has_tumor"]
det_prob = res["detection_prob"]
label = res["predicted_label"]
class_probs = res["class_probs"]
mask = res["mask"]

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🩺 Doctor Notes")

default_notes = st.session_state.get(
    "doctor_notes",
    "Patient presents with brain MRI scan for automated tumor analysis.",
)
doctor_notes = st.text_area(
    "Enter clinical observations, impressions and follow-up recommendations:",
    value=default_notes,
    height=180,
)
st.session_state.doctor_notes = doctor_notes
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🖼️ Preview")
pc1, pc2 = st.columns(2)
with pc1:
    st.image(img_rgb, caption="Original MRI", use_container_width=True)
with pc2:
    st.image(overlay, caption="Tumor overlay", use_container_width=True)

st.markdown("---")
st.markdown("### 📥 Export as PDF")

def build_pdf() -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, "Brain Tumor Analysis Report")

    # Detection info
    y = height - 100
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "AI Findings")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Tumor detected: {has_tumor}")
    y -= 16
    c.drawString(50, y, f"Detection confidence: {det_prob:.2%}")
    y -= 16
    if has_tumor and label:
        c.drawString(50, y, f"Predicted tumor type: {label.upper()}")
        y -= 20

    if has_tumor and class_probs:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Class probabilities:")
        y -= 16
        c.setFont("Helvetica", 10)
        for cls, p in class_probs.items():
            c.drawString(60, y, f"- {cls}: {float(p):.2%}")
            y -= 14

    # Segmentation stats
    if mask is not None:
        tumor_pixels = int(np.sum(mask > 0))
        total_pixels = int(mask.size)
        coverage = tumor_pixels / total_pixels * 100 if total_pixels > 0 else 0.0
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Segmentation:")
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"Tumor pixels: {tumor_pixels:,}")
        y -= 14
        c.drawString(60, y, f"Coverage: {coverage:.2f}%")
        y -= 24

    # Doctor notes
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Doctor Notes")
    y -= 18
    c.setFont("Helvetica", 10)

    text_obj = c.beginText(50, y)
    wrap_width = 90  # characters per line approx
    for line in doctor_notes.splitlines():
        while len(line) > wrap_width:
            text_obj.textLine(line[:wrap_width])
            line = line[wrap_width:]
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Images at bottom
    img_y = 140
    img_w = 240
    img_h = 240

    img_io = BytesIO()
    Image.fromarray(img_rgb).save(img_io, format="PNG")
    img_io.seek(0)
    c.drawImage(ImageReader(img_io), 50, img_y, width=img_w, height=img_h, preserveAspectRatio=True)

    ov_io = BytesIO()
    Image.fromarray(overlay).save(ov_io, format="PNG")
    ov_io.seek(0)
    c.drawImage(ImageReader(ov_io), 325, img_y, width=img_w, height=img_h, preserveAspectRatio=True)

    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(50, 40, "Generated by Brain Tumor AI Diagnostic System")

    c.save()
    buf.seek(0)
    return buf.getvalue()


pdf_bytes = build_pdf()

st.download_button(
    "⬇️ Download PDF Report",
    data=pdf_bytes,
    file_name="brain_tumor_report.pdf",
    mime="application/pdf",
)

st.caption("The PDF includes AI findings, segmentation statistics, and the doctor notes you entered above.")
