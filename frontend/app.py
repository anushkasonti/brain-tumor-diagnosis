# # frontend/app.py
#
# import os
# import sys
# import time
#
# import streamlit as st
#
# # -------------------------------------------------------------------
# # Make project root importable (so "backend" works inside pages)
# # -------------------------------------------------------------------
# PROJECT_ROOT = os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__))  # frontend/ -> project root
# )
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
#
#
# # -------------------------------------------------------------------
# # Page config
# # -------------------------------------------------------------------
# st.set_page_config(
#     page_title="Brain Tumor Analysis",
#     layout="wide",
#     page_icon="🧠",
#     initial_sidebar_state="expanded",
# )
#
# # -------------------------------------------------------------------
# # Theme in session_state (Dark / Light)
# # -------------------------------------------------------------------
# if "theme" not in st.session_state:
#     st.session_state.theme = "Dark"
#
# with st.sidebar:
#     st.markdown("### 🎨 Theme")
#     theme_choice = st.radio("Choose theme", ["Dark", "Light"], index=0)
#     st.session_state.theme = theme_choice
#
#
# # -------------------------------------------------------------------
# # Inject basic CSS according to theme
# # -------------------------------------------------------------------
# def apply_base_css():
#     if st.session_state.theme == "Dark":
#         bg = "#0d1117"
#         card_bg = "rgba(255,255,255,0.06)"
#         text_color = "#f8f9fa"
#     else:
#         bg = "#f5f5f7"
#         card_bg = "rgba(255,255,255,0.9)"
#         text_color = "#111827"
#
#     st.markdown(
#         f"""
#         <style>
#         body {{
#             background-color: {bg};
#             color: {text_color};
#             font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
#         }}
#         .block-container {{
#             padding-top: 2.5rem;
#             padding-bottom: 2.5rem;
#         }}
#         .premium-card {{
#             background: {card_bg};
#             backdrop-filter: blur(18px);
#             border-radius: 20px;
#             padding: 24px 26px;
#             box-shadow: 0 18px 45px rgba(0,0,0,0.25);
#         }}
#         .gradient-title {{
#             font-weight: 800;
#             font-size: 2.5rem;
#             text-align: center;
#             background: linear-gradient(90deg, #60a5fa, #a855f7, #ec4899);
#             -webkit-background-clip: text;
#             color: transparent;
#             margin-bottom: 0.4rem;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#
#
# apply_base_css()
#
# # -------------------------------------------------------------------
# # One-time splash animation
# # -------------------------------------------------------------------
# if "splash_done" not in st.session_state:
#     st.session_state.splash_done = False
#
# if not st.session_state.splash_done:
#     placeholder = st.empty()
#     placeholder.markdown(
#         """
#         <div style="height:100vh;display:flex;flex-direction:column;
#                     align-items:center;justify-content:center;
#                     background:radial-gradient(circle at top,#1f2937,#020617);">
#           <div style="font-size:3rem;font-weight:800;
#                       background:linear-gradient(120deg,#38bdf8,#c4b5fd,#fb7185);
#                       -webkit-background-clip:text;color:transparent;
#                       text-align:center;">
#             Brain Tumor AI Diagnostic Suite
#           </div>
#           <div style="margin-top:1rem;font-size:1.1rem;color:#e5e7eb;">
#             Detection · Classification · Segmentation
#           </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#     time.sleep(2)
#     placeholder.empty()
#     st.session_state.splash_done = True
#
# # -------------------------------------------------------------------
# # Home content
# # -------------------------------------------------------------------
# st.markdown('<div class="gradient-title">Brain Tumor AI Diagnostic</div>', unsafe_allow_html=True)
# st.markdown(
#     "<p style='text-align:center;font-size:18px;color:#9ca3af;'>Upload MRI scans, visualise tumor regions and export structured clinical reports.</p>",
#     unsafe_allow_html=True,
# )
#
# col1, col2 = st.columns([1.4, 1])
#
# with col1:
#     st.markdown('<div class="premium-card">', unsafe_allow_html=True)
#     st.markdown("### 🚀 How to use")
#     st.markdown(
#         """
#         1. Go to the **Diagnosis** page in the left sidebar.
#         2. Upload a brain MRI (PNG/JPG).
#         3. Review:
#            - Tumor detection & probability
#            - Tumor type classification
#            - Segmentation Slider
#         4. Add **doctor notes** and generate a **PDF report** on the **Report** page.
#         """
#     )
#     st.markdown("</div>", unsafe_allow_html=True)
#
# with col2:
#     st.markdown('<div class="premium-card">', unsafe_allow_html=True)
#     st.markdown("### 📂 Latest Session")
#     if "last_result" in st.session_state:
#         res = st.session_state["last_result"]
#         st.write("✔️ Last MRI analysed.")
#         st.write(f"- Tumor detected: **{res['has_tumor']}**")
#         st.write(f"- Detection probability: **{res['detection_prob']:.2%}**")
#         if res["predicted_label"]:
#             st.write(f"- Predicted type: **{res['predicted_label'].upper()}**")
#     else:
#         st.write("No MRI analysed yet. Go to **Diagnosis** to start.")
#     st.markdown("</div>", unsafe_allow_html=True)
#
# st.markdown("---")
# st.markdown(
#     "<p style='text-align:center;color:#6b7280;font-size:13px;'></p>",
#     unsafe_allow_html=True,
# )
# frontend/app.py

import os
import sys
import time

import streamlit as st

# -------------------------------------------------------------------
# Make project root importable (so "backend" works inside pages)
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # frontend/ -> project root
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Analysis",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Base CSS (fixed dark premium theme – no toggle)
# -------------------------------------------------------------------
def apply_base_css():
    bg = "#0d1117"
    card_bg = "rgba(255,255,255,0.06)"
    text_color = "#f8f9fa"

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg};
            color: {text_color};
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .block-container {{
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }}
        .premium-card {{
            background: {card_bg};
            backdrop-filter: blur(18px);
            border-radius: 20px;
            padding: 24px 26px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        }}
        .gradient-title {{
            font-weight: 800;
            font-size: 2.5rem;
            text-align: center;
            background: linear-gradient(90deg, #60a5fa, #a855f7, #ec4899);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_base_css()

# -------------------------------------------------------------------
# One-time splash animation
# -------------------------------------------------------------------
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    placeholder = st.empty()
    placeholder.markdown(
        """
        <div style="height:100vh;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;
                    background:radial-gradient(circle at top,#1f2937,#020617);">
          <div style="font-size:3rem;font-weight:800;
                      background:linear-gradient(120deg,#38bdf8,#c4b5fd,#fb7185);
                      -webkit-background-clip:text;color:transparent;
                      text-align:center;">
            Brain Tumor Diagnosis
          </div>
          <div style="margin-top:1rem;font-size:1.1rem;color:#e5e7eb;">
            Detection · Classification · Segmentation
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(2)
    placeholder.empty()
    st.session_state.splash_done = True

# -------------------------------------------------------------------
# Home content
# -------------------------------------------------------------------
st.markdown('<div class="gradient-title">Brain Tumor AI Diagnostic</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;font-size:18px;color:#9ca3af;'>Upload MRI scans, visualise tumor regions and export structured clinical reports.</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 🚀 How to use")
    st.markdown(
        """
        1. Go to the **Diagnosis** page in the left sidebar.  
        2. Upload a brain MRI (PNG/JPG).  
        3. Review:
           - Tumor detection & probability  
           - Tumor type classification  
           - Segmentation Slider  
        4. Add **doctor notes** and generate a **PDF report** on the **Report** page.  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Latest Session")
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.write("✔️ Last MRI analysed.")
        st.write(f"- Tumor detected: **{res['has_tumor']}**")
        st.write(f"- Detection probability: **{res['detection_prob']:.2%}**")
        if res["predicted_label"]:
            st.write(f"- Predicted type: **{res['predicted_label'].upper()}**")
    else:
        st.write("No MRI analysed yet. Go to **Diagnosis** to start.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:13px;'></p>",
    unsafe_allow_html=True,
)
