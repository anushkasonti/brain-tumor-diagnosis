# # frontend/pages/3_About.py
#
# import streamlit as st
#
#
# def apply_theme_css():
#     theme = st.session_state.get("theme", "Dark")
#     if theme == "Dark":
#         bg = "#020617"
#         text = "#e5e7eb"
#     else:
#         bg = "#f5f5f7"
#         text = "#111827"
#
#     st.markdown(
#         f"""
#         <style>
#         body {{
#             background-color: {bg};
#             color: {text};
#         }}
#         .block-container {{
#             padding-top: 2rem;
#             padding-bottom: 2rem;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#
#
# apply_theme_css()
#
# st.markdown("## ℹ️ About")
#
# st.write(
#     """
#     This application is part of an academic **Brain Tumour Project** combining:
#
#     - **Tumor detection** using a custom CNN
#     - **Tumor type classification** (e.g., glioma, meningioma, etc.)
#     - **Tumor segmentation** using a U-Net style architecture
#     """
# )
#
# st.markdown("---")
# st.write("Built by - Anushka Sonti, Kunjal Ahuja, Harshal Bankhele, Rohan Sinha, Maithili Mahadik")
# frontend/pages/3_About.py

import streamlit as st

# ---------------------------------------------------------------
# Apply theme (same theme logic as other pages)
# ---------------------------------------------------------------
def apply_theme_css():
    theme = st.session_state.get("theme", "Dark")
    if theme == "Dark":
        bg = "#020617"
        text = "#e5e7eb"
        card_bg = "rgba(15,23,42,0.85)"
    else:
        bg = "#f5f5f7"
        text = "#111827"
        card_bg = "rgba(255,255,255,0.85)"

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

        .gradient-title {{
            font-weight: 900;
            font-size: 2.8rem;
            text-align: center;
            background: linear-gradient(90deg, #60a5fa, #a855f7, #ec4899);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.2rem;
        }}

        .subtitle {{
            text-align:center;
            font-size: 18px;
            color: #94a3b8;
            margin-bottom: 2rem;
        }}

        .glass-card {{
            background: {card_bg};
            padding: 28px;
            border-radius: 20px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.25);
            backdrop-filter: blur(14px);
            margin-bottom: 1.2rem;
            animation: fadeIn 0.8s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .team-card {{
            background: {card_bg};
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            font-size: 15px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
            transition: 0.3s ease;
        }}

        .team-card:hover {{
            transform: translateY(-6px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.35);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme_css()

# ---------------------------------------------------------------
# Premium Header
# ---------------------------------------------------------------
st.markdown('<div class="gradient-title">About This Project</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-powered MRI Analysis</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Project Summary Card
# ---------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown(
    """
### 🧠 Project Overview  

This application is part of a comprehensive academic **Brain Tumor Diagnostic System**, built using cutting-edge AI and medical imaging concepts.  
Our workflow integrates the power of deep learning across **three major modules**:

#### 🔍 Tumor Detection  
A highly optimized **Custom CNN model** identifies whether a brain MRI contains tumor tissues.

#### 🧬 Tumor Classification  
If a tumor is detected, the model classifies it into relevant categories such as:  
- **Glioma**  
- **Meningioma**  
- **Pituitary tumor**

#### 🎯 Tumor Segmentation  
A **U-Net style segmentation model** isolates the tumor boundaries, enabling clinical interpretation and further medical analysis.
    """
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Team Section
# ---------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 👩‍⚕️ Project Team")
st.markdown("<p style='color:#94a3b8;'>The group behind this project</p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c4, c5 = st.columns(2)

with c1:
    st.markdown("<div class='team-card'>Anushka Sonti<br></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='team-card'>Kunjal Ahuja<br><small></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='team-card'>Harshal Bankhele<br></div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='team-card'>Rohan Sinha<br></div>", unsafe_allow_html=True)
with c5:
    st.markdown("<div class='team-card'>Maithili Mahadik<br></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:13px;'>© 2025 Brain Tumor Diagnostic Project</p>",
    unsafe_allow_html=True
)
