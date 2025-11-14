import streamlit as st
from utils.api_helpers import fetch_api, clear_cache
from utils.lottie_helper import show_lottie_loader
import pandas as pd
import plotly.express as px
import json
from streamlit_lottie import st_lottie
from pathlib import Path
import requests, json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import to_rgba
import os
from datetime import datetime

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import altair as alt
from plotnine import  aes, geom_bar
# ==========================================================
# Streamlit Config
# ==========================================================


st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

# Django backend base URL
BASE_URL = "http://bdd_backend:8000"
  # change if hosted remotely

# Utility function to fetch data safely
@st.cache_data(show_spinner=False)
def fetch_data(endpoint: str):
    """
    Fetches JSON data from the Django backend API.
    """
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from {endpoint}: {e}")
        return None

st.set_page_config(
    page_title="Utkarsh BDD100K Dataset Analysis Dashboard",
    layout="wide",
)
import base64





# ==========================================================
# Custom CSS Styling
# ==========================================================

st.markdown("""
<style>
/* =====================================
   Global Application Styling
   ===================================== */

/* ==========================================================
    Dark Theme Background with Gradient + Video Compatibility
   ========================================================== */
[data-testid="stAppViewContainer"] {
    /* Core dark gradient background (keeps dark mode aesthetic) */
    background: linear-gradient(
        180deg,
        rgba(0, 0, 0, 0.95) 0%,
        rgba(10, 10, 15, 0.93) 30%,
        rgba(15, 15, 20, 0.92) 70%,
        rgba(20, 20, 25, 0.9) 100%
    ) !important;
    color: #f8fafc !important;

    font-family: 'Poppins', 'Inter', sans-serif !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    position: relative !important;
    scroll-behavior: smooth;

    /* Smooth transitions between dark shades */
    transition: background 0.8s ease-in-out, color 0.3s ease-in-out;

    /* Subtle glow for white text */
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

/* Subtle glass overlay (for text clarity over video) */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(8px) saturate(130%);
    -webkit-backdrop-filter: blur(8px) saturate(130%);
    z-index: -1;
}

/* Prevent Streamlit sections from creating white areas */
.block-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 2rem 1.8rem;
    box-shadow: 0 4px 22px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px) saturate(120%);
    -webkit-backdrop-filter: blur(10px) saturate(120%);
    z-index: 1;
}

/* ==========================================================
    Static Fullscreen Background Video (Scroll-Locked)
   ========================================================== */
#bg-video {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    object-fit: cover;
    z-index: -5;
    opacity: 0.8;
    filter: brightness(0.55) contrast(1.05) saturate(1.1);
    pointer-events: none;
}








/* ==========================================================
    Transparent Overlay for Readability (Dark Mode Enhanced)
   ========================================================== */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        180deg,
        rgba(0, 0, 0, 0.65) 0%,
        rgba(0, 0, 0, 0.75) 40%,
        rgba(0, 0, 0, 0.85) 100%
    );
    backdrop-filter: blur(8px) saturate(130%);
    -webkit-backdrop-filter: blur(8px) saturate(130%);
    z-index: -1;
    pointer-events: none;
}

/* ==========================================================
    Modern Glassy Gradient Buttons
   ========================================================== */
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    padding: 0.55rem 0.85rem;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.3px;
    font-family: 'Inter', 'Poppins', sans-serif;
    cursor: pointer;
    transition: all 0.25s ease-in-out;
    box-shadow: 0 3px 10px rgba(99, 102, 241, 0.25);
}

/* Hover State */
div.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4);
}

/* Active (Pressed) State */
div.stButton > button:active {
    transform: translateY(0px) scale(0.98);
    background: linear-gradient(90deg, #4f46e5, #3730a3);
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
}

/* Disabled Button */
div.stButton > button:disabled {
    background: rgba(80, 80, 100, 0.4);
    color: rgba(255, 255, 255, 0.5) !important;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
}




/* ==========================================================
    Glassmorphic Main Content Blocks — Dark Theme Edition
   ========================================================== */
.block-container {
    position: relative;
    z-index: 2;
    padding: 2.4rem 2rem;
    border-radius: 18px;
    background: rgba(20, 20, 25, 0.35);  /* darker glass base */
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 6px 28px rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(14px) saturate(130%);
    -webkit-backdrop-filter: blur(14px) saturate(130%);
    transition: all 0.35s ease;
    animation: fadeInUp 0.6s ease-in-out;
}

/* Slight Lift on Hover */
.block-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 32px rgba(99, 102, 241, 0.25);
    border-color: rgba(139, 92, 246, 0.25);
    background: rgba(25, 25, 35, 0.45);
}

/* Subtle Header Highlight (if titles inside cards) */
.block-container h2, 
.block-container h3, 
.block-container h4 {
    color: #e2e8f0 !important;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

/* Smooth Fade Animation */
@keyframes fadeInUp {
    0% {
        opacity: 0;
        transform: translateY(20px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* =====================================
   Sidebar Styling
  /* ==========================================================
   🧭 Sidebar Styling — Glassy Black Gradient
   ========================================================== */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(10, 12, 25, 0.95) 0%,
        rgba(18, 20, 40, 0.9) 50%,
        rgba(25, 28, 55, 0.88) 100%
    );
    color: #f8fafc;
    padding-top: 2rem;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 6px 0 24px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(12px) saturate(130%);
    -webkit-backdrop-filter: blur(12px) saturate(130%);
    z-index: 20;
}

/* Sidebar text elements */
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
    font-weight: 500;
    font-family: 'Inter', 'Poppins', sans-serif;
    letter-spacing: 0.3px;
}

/* Sidebar hover feedback */
[data-testid="stSidebar"] button:hover {
    background: rgba(99, 102, 241, 0.18) !important;
    transform: scale(1.03);
    transition: all 0.25s ease;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.25);
}

/* Active (selected) sidebar item */
[data-testid="stSidebar"] [aria-checked="true"],
[data-testid="stSidebar"] .stRadio > label[data-checked="true"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 10px;
    color: white !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
}

/* Optional: Add slight glow to active sidebar title */
[data-testid="stSidebar"] h2 {
    color: #c7d2fe !important;
    text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
}

/* ==========================================================
   🧩 Typography — Consistent with Dark UI
   ========================================================== */
h1, h2, h3, h4 {
    color: #f5f5f5 !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.3);
    margin-bottom: 0.75rem !important;
}

/* Fancy gradient for main title (hero header) */
h1 {
    font-size: 2.3rem !important;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subheading */
h2 {
    font-size: 1.8rem !important;
    color: #e5e7eb !important;
}

/* Text content */
p, li, span, label, div {
    color: #e2e8f0 !important;
    font-weight: 400;
    line-height: 1.6;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

/* Links (accent highlight) */
a {
    color: #8b5cf6 !important;
    text-decoration: none;
    transition: color 0.25s ease;
}
a:hover {
    color: #a78bfa !important;
    text-decoration: underline;
}








/* ==========================================================
   ⚙️ Buttons and Inputs — Dark Glass Theme
   ========================================================== */

/* Primary Buttons */
.stButton > button {
    color: #ffffff !important;
    border: none !important;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    padding: 0.65rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    font-family: 'Inter', 'Poppins', sans-serif;
    letter-spacing: 0.3px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    cursor: pointer;
}

/* Button Hover */
.stButton > button:hover {
    transform: translateY(-3px) scale(1.03);
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.45);
}

/* Button Active (Pressed) */
.stButton > button:active {
    transform: translateY(0) scale(0.98);
    background: linear-gradient(90deg, #4f46e5, #3730a3);
    box-shadow: 0 3px 12px rgba(79, 70, 229, 0.35);
}

/* Disabled Button */
.stButton > button:disabled {
    background: rgba(100, 100, 120, 0.3) !important;
    color: rgba(255, 255, 255, 0.5) !important;
    box-shadow: none !important;
    cursor: not-allowed;
}

/* ==========================================================
   ✏️ Input Fields, Dropdowns, Text Areas
   ========================================================== */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stTextArea > div > textarea,
.stNumberInput > div > div > input {
    background: rgba(30, 30, 40, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    font-weight: 400;
    transition: all 0.3s ease-in-out;
    backdrop-filter: blur(10px) saturate(120%);
    -webkit-backdrop-filter: blur(10px) saturate(120%);
}

/* Input Focus */
.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus,
.stTextArea > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.4);
    outline: none !important;
}

/* Placeholder Text */
input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

/* ==========================================================
   🎞️ Animations
   ========================================================== */
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ==========================================================
   🦶 Footer & Streamlit Decoration Cleanup
   ========================================================== */
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ==========================================================
   🧭 Scrollbar Styling — Neon Accent for Dark Mode
   ========================================================== */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(20, 20, 30, 0.4);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #4f46e5, #8b5cf6);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: background 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #7c3aed, #6366f1);
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.3);
}

}
</style>
""", unsafe_allow_html=True)

import base64

# Choose path depending on OS
import os

# Dynamically resolve the correct path, regardless of OS or container
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "background.mp4")

with open(video_path, "rb") as f:
    video_bytes = f.read()
video_base64 = base64.b64encode(video_bytes).decode()

# Inject video element
st.markdown(f"""
<video autoplay muted loop id="bg-video">
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
</video>
<div class="video-overlay"></div>
""", unsafe_allow_html=True)


# ==========================================================
# Header with Custom Font
# ==========================================================

st.markdown("""
<!-- Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ==========================================================
    Dashboard Hero Title + Subtitle Styling
   ========================================================== */
[data-testid="stAppViewContainer"] h1 {
    text-align: center;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 2.8rem;
    letter-spacing: 0.6px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 2rem;
    margin-bottom: 0.8rem;
    text-shadow: 0 0 14px rgba(99, 102, 241, 0.3);
    animation: fadeInDown 1s ease-in-out;
}

/* Subtitle / Caption */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] .stCaption {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 1.05rem;
    color: rgba(235, 235, 235, 0.88);
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.65;
    letter-spacing: 0.25px;
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.5);
    animation: fadeInUp 1.2s ease-in-out;
}

/* Fade Animations */
@keyframes fadeInDown {
    0% { opacity: 0; transform: translateY(-20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)




# ==========================================================
# Sidebar Navigation
# ==========================================================
# ==========================================================
# Sidebar Section (Refined & Professional)


# ==========================================================



# ==========================================================
#  Sidebar Background Setup (Cross-Platform)
# ==========================================================
def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Choose correct path depending on OS
import os

# Dynamically resolve the correct directory based on where the file is
current_dir = os.path.dirname(os.path.abspath(__file__))
sidebar_bg_path = os.path.join(current_dir, "image.jpg")


# Check and encode image
if os.path.exists(sidebar_bg_path):
    bg_image_base64 = get_base64_of_image(sidebar_bg_path)

    st.markdown(
        f"""
        <style>
        /* ==========================================================
            Sidebar with Dark Overlay + Background Image
           ========================================================== */
        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg,
                    rgba(5, 5, 10, 0.92) 0%,
                    rgba(10, 10, 25, 0.92) 40%,
                    rgba(20, 20, 45, 0.9) 100%
                ),
                url("data:image/jpg;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #f9fafb;
            padding-top: 2rem;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 4px 0 15px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(10px) saturate(130%);
            -webkit-backdrop-filter: blur(10px) saturate(130%);
            z-index: 20;
        }}

        /* Sidebar text elements */
        [data-testid="stSidebar"] * {{
            color: #e2e8f0 !important;
            font-family: 'Inter', sans-serif;
        }}

        /* Sidebar Header */
        .sidebar-title {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 0.5px;
            color: #ffffff;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1.8rem;
            text-shadow: 0 2px 6px rgba(0,0,0,0.5);
        }}

        /* Radio Buttons */
        div[role="radiogroup"] label {{
            font-weight: 500;
            color: #f1f1f1 !important;
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            transition: all 0.25s ease-in-out;
            background-color: rgba(255, 255, 255, 0.05);
            cursor: pointer;
        }}

        div[role="radiogroup"] label:hover {{
            background-color: rgba(99, 102, 241, 0.12);
            transform: translateX(4px);
        }}

        /* Active Radio Button */
        div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{
            background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
            border-radius: 6px;
        }}

        /* Recompute Button */
        .stButton > button {{
            width: 100%;
            color: #ffffff !important;
            border: none;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            padding: 0.6rem;
            border-radius: 10px;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            margin-top: 2rem;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
        }}
        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 18px rgba(139, 92, 246, 0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning(f" Sidebar background image not found at: {sidebar_bg_path}")












# Sidebar Header
st.sidebar.markdown("<div class='sidebar-title'>Analysis Modules</div>", unsafe_allow_html=True)

# Define navigation options with icons
menu_options = {
    "Home": "Home",
    "Dataset Summary": "Dataset Summary",
    "Class Distribution": "Class Distribution",
    "Visual Distributions": "Visual Distributions",
    "Data Quality per Class": "Data Quality per Class",
    "Object Density": "Object Density",
    "Class Balance Insights": "Class Balance Insights",
    "Sample Visualizations": "Sample Visualizations",
}

# Sidebar Radio Navigation
selected = st.sidebar.radio(
    "Navigation",  # label hidden
    list(menu_options.keys()),
    format_func=lambda x: menu_options[x],
    label_visibility="collapsed",
)

# Recompute Button
recompute = st.sidebar.button(" Recompute (Ignore Cache)")
params = {"recompute": 1} if recompute else {}

# Small active-item glow (optional feedback)
st.markdown(f"""
<style>
/* Highlight the currently active radio label */
div[role="radiogroup"] label[data-baseweb="radio"]:has(input[checked]) {{
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.35);
    border-radius: 10px;
    transform: translateX(4px);
}}

/* Recompute Button Glow when Active */
.stButton > button:active {{
    transform: scale(0.97);
    background: linear-gradient(90deg, #4f46e5, #3730a3);
    box-shadow: 0 0 18px rgba(99, 102, 241, 0.5);
}}
</style>
""", unsafe_allow_html=True)



# ==========================================================
#  Home Section (Interactive Landing)
# ==========================================================
if selected == "Home":



    st.markdown("""
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">

    <style>
    /* ==========================================================
     Centered Hero Section — Dark Glass Aesthetic
    ========================================================== */

    /* Outer wrapper to center hero both horizontally and vertically */
    .hero-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        width: 100%;
        margin-top: 2.5rem;
    }

    /* Glassy card around the hero title */
    .hero-container {
        background: rgba(0, 0, 0, 0.45);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        backdrop-filter: blur(12px) saturate(140%);
        -webkit-backdrop-filter: blur(12px) saturate(140%);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        display: inline-block;
        animation: fadeInUp 1s ease-in-out;
    }

    /* Hero Title */
    .hero-container h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        letter-spacing: 0.6px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        text-shadow: 0 0 14px rgba(99, 102, 241, 0.3);
    }

    /* Hero Subtitle */
    .hero-container p {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        color: rgba(235, 235, 235, 0.88);
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.65;
        letter-spacing: 0.25px;
        text-shadow: 0 1px 6px rgba(0, 0, 0, 0.5);
    }

    /* Smooth Fade Animation */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>

    <div class="hero-wrapper">
    <div class="hero-container">
        <h1>BDD100K Visual Analytics Dashboard</h1>
        <p>An interactive platform for comprehensive exploration and evaluation of the Berkeley DeepDrive dataset</p>
    </div>
    </div>
    """, unsafe_allow_html=True)



    # ==========================================================
    # Dataset Overview Section
    # ==========================================================
    st.markdown("""
    <style>
    .section-title {
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: 0.5px;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        color: #ffffff;
    }

    .dataset-image {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }

    .dataset-caption {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: rgba(230, 230, 230, 0.85);
        margin-bottom: 1.5rem;
    }

    .dataset-description {
        text-align: justify;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 900px;
        margin: 0 auto;
        color: rgba(240, 240, 240, 0.92);
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease-in-out;
    }

    .dataset-description:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)

    # Image (centered)
    st.markdown("""
    <div class='dataset-image'>
        <img src='https://bair.berkeley.edu/static/blog/bdd/geo_distribution.jpg' width='85%' 
             style='border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.35);'>
    </div>
    <div class='dataset-caption'>
        Geographical distribution of sample driving videos in the BDD100K dataset
    </div>
    """, unsafe_allow_html=True)



    st.markdown("""
    <style>
    /* ==========================================================
    🧩 Dataset Description Block — Polished Dark Glass Style
    ========================================================== */
    .dataset-description {
        text-align: justify;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        line-height: 1.75;
        max-width: 900px;
        margin: 0 auto;
        color: rgba(245, 245, 245, 0.95);
        background: rgba(255, 255, 255, 0.05);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(14px) saturate(130%);
        -webkit-backdrop-filter: blur(14px) saturate(130%);
        transition: all 0.35s ease-in-out;
        animation: fadeInUp 1.2s ease-in-out;
    }

    /* Subtle border glow accent */
    .dataset-description::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25));
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: -1;
    }

    .dataset-description:hover {
        transform: scale(1.01);
        box-shadow: 0 10px 36px rgba(139, 92, 246, 0.25);
    }

    /* Fade-in Animation */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Bold text highlight */
    .dataset-description b {
        color: #a78bfa;
        font-weight: 600;
    }
    </style>

 <div class='dataset-description'>

  <!-- Section 1: Annotation Richness -->
  <h4 style="font-family:Montserrat; color:#e0e7ff; margin-bottom:0.4rem;">
    1. Rich Multi-Task Annotations
  </h4>
  <p>
    Beyond scale, BDD100K is distinguished by its <b>multi-task annotation depth</b>. 
    Each video includes <b>frame-level labels</b> for several core perception tasks—
    <b>object detection</b>, <b>semantic segmentation</b>, <b>lane marking detection</b>, 
    <b>drivable area estimation</b>, and <b>multi-object tracking</b>. 
    A curated subset also provides <b>instance segmentation</b> and <b>MOTS</b> annotations, 
    enabling unified research across both image-level and sequence-level understanding.
  </p>

  <hr style="border:0; height:1px; background:rgba(255,255,255,0.07); margin:14px 0;">

  <!-- Section 2: Geographic Diversity -->
  <h4 style="font-family:Montserrat; color:#e0e7ff; margin-bottom:0.4rem;">
    2. Geographic and Environmental Breadth
  </h4>
  <p>
    BDD100K captures driving scenes across multiple U.S. states, spanning <b>urban centers</b>, 
    <b>highways</b>, <b>suburbs</b>, and <b>rural environments</b>.  
    This geographic diversity introduces natural variation in architecture, vegetation, 
    road design, signage, and traffic density—making the dataset ideal for 
    <b>domain adaptation</b> and <b>cross-city generalization</b> studies.
  </p>

  <hr style="border:0; height:1px; background:rgba(255,255,255,0.07); margin:14px 0;">

  <!-- Section 3: Photometric & Weather Diversity -->
  <h4 style="font-family:Montserrat; color:#e0e7ff; margin-bottom:0.4rem;">
    3. Weather, Lighting & Photometric Variation
  </h4>
  <p>
    The dataset exhibits extensive <b>photometric variability</b>, covering bright daylight, 
    long-shadow evenings, <b>night-time low light</b>, fog, rain, and overcast conditions. 
    These natural changes introduce meaningful <b>domain shift</b>, making BDD100K a realistic 
    benchmark for robustness under <b>weather-induced</b> and <b>illumination-induced</b> challenges.
  </p>

  <hr style="border:0; height:1px; background:rgba(255,255,255,0.07); margin:14px 0;">

  <!-- Section 4: Long-Tailed Class Distribution -->
  <h4 style="font-family:Montserrat; color:#e0e7ff; margin-bottom:0.4rem;">
    4. Long-Tailed Class Distribution
  </h4>
  <p>
    BDD100K naturally follows a <b>long-tailed</b> object distribution: 
    common classes like <b>car</b> and <b>person</b> appear frequently, while 
    <b>train</b>, <b>trailer</b>, <b>motorcycle</b>, and <b>rider</b> occur far less often.  
    This imbalance mirrors real-world driving scenarios and motivates techniques such as 
    <b>class-balanced sampling</b>, <b>focal loss</b>, and <b>re-weighting</b> to prevent bias 
    toward high-frequency categories.
  </p>

  <hr style="border:0; height:1px; background:rgba(255,255,255,0.07); margin:14px 0;">

  <!-- Section 5: Why BDD100K Matters for Research -->
  <h4 style="font-family:Montserrat; color:#e0e7ff; margin-bottom:0.4rem;">
    5. Why BDD100K is a Foundational Research Dataset
  </h4>
  <p>
    Its combination of <b>scale</b>, <b>task diversity</b>, and <b>visual variability</b> 
    makes BDD100K one of the most influential datasets for autonomous-driving research. 
    It supports unified training pipelines where a single model can jointly learn detection, 
    segmentation, tracking, and high-level scene understanding from a coherent dataset. 
    This multi-modal richness enables robust, real-world generalization.
  </p>

</div>

    """, unsafe_allow_html=True)



# ==========================================================
# Dataset Summary Section (Dynamic + Recompute)
# ==========================================================

BASE_URL = "http://bdd_backend:8000"



import time

        

# ---- Load Lottie animation once ----
@st.cache_resource
def load_lottie_url(url: str):
    """Load a Lottie animation JSON from URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# You can use any animation from https://lottiefiles.com
LOTTIE_LOADING = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_usmfx6bp.json")

# ---- Fetch API with Lottie Loader ----
def fetch_dataset_summary(recompute=False):
    """Fetch dataset summary with visible Lottie loader."""
    url = f"{BASE_URL}/api/analysis/dataset-summary/"
    if recompute:
        url += "?recompute=1"

    placeholder = st.empty()  # placeholder to show loader dynamically

    with placeholder:
        st_lottie(
            LOTTIE_LOADING,
            height=180,
            key="loading",
            speed=1,
            loop=True,
            quality="high"
        )
        st.markdown(
            "<p style='text-align:center;font-family:Inter;font-size:1.05rem;color:rgba(230,230,230,0.85);margin-top:1rem;'>Recomputing dataset summary... please wait.</p>",
            unsafe_allow_html=True
        )

    try:
        response = requests.get(url, timeout=None)  # wait indefinitely
        response.raise_for_status()
        data = response.json()
        time.sleep(0.4)
        placeholder.empty()  # remove loader
        st.toast(" Dataset summary successfully updated!")
        return data
    except requests.exceptions.RequestException as e:
        placeholder.empty()
        st.error(f" Failed to fetch dataset summary: {e}")
        return None


if selected == "Dataset Summary":
    st.markdown("""
    <style>
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.4rem;
        margin-top: 2.2rem;
        justify-items: center;
    }

    .summary-card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        color: #ffffff;
        width: 100%;
        backdrop-filter: blur(10px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .summary-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.5px;
        margin-top: 0.4rem;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: rgba(235, 235, 235, 0.85);
        margin-top: 0.2rem;
    }

    .meta-info {
        text-align: center;
        font-size: 0.9rem;
        color: rgba(220,220,220,0.7);
        font-family: 'Inter', sans-serif;
        margin-top: 2rem;
    }

    .recompute-btn {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center;font-family:Montserrat;font-weight:700;'>Dataset Summary Overview</h3>", unsafe_allow_html=True)

    # === Recompute button ===

    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        recompute = st.button("Recompute Summary", use_container_width=True)


    data = fetch_dataset_summary(recompute=recompute)

    if data:
        # Exclude path
        metrics = {k: v for k, v in data.items() if "Root" not in k and "meta" not in k}

        # Create grid layout
        st.markdown("<div class='summary-grid'>", unsafe_allow_html=True)

        icons = {
            "Training Images": '<i class="fa-solid fa-image"></i>',
            "Validation Images": '<i class="fa-regular fa-square-check"></i>',
            "Training Annotations": '<i class="fa-solid fa-pen-nib"></i>',
            "Avg Objects/Image": '<i class="fa-solid fa-chart-line"></i>',
            "Last Computed": '<i class="fa-regular fa-clock"></i>'
        }

        for key, value in metrics.items():
            icon = icons.get(key, "📊")
            st.markdown(
                f"""
                <div class='summary-card'>
                    <div style="font-size:2rem;">{icon}</div>
                    <div class='metric-value'>{value}</div>
                    <div class='metric-label'>{key}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Meta info section
        last_computed = data.get("Last Computed", "N/A")
        cache_status = data.get("meta", {}).get("cache_status", "unknown").capitalize()

        st.markdown(
            f"""
            <div class='meta-info'>
                <b>Last Computed:</b> {last_computed} &nbsp; | &nbsp;
                <b>Cache:</b> {cache_status}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("No dataset summary data available.")










# =====================================
# Class Distribution Section
# =====================================










BASE_URL = "http://bdd_backend:8000"


def fetch_class_distribution():
    try:
        response = requests.get(f"{BASE_URL}/api/analysis/class-distribution/", timeout=None)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching class distribution: {e}")
        return None


# ==========================================================
# CLASS DISTRIBUTION SECTION
# ==========================================================


if selected == "Class Distribution":

    # === CSS: Enhanced Styling, Typography, Animations ===
    st.markdown("""
                    <style>
                    /* ===== Header Styling ===== */
                    .class-header {
                        text-align: center;
                        font-family: 'Poppins', sans-serif;
                        font-weight: 800;
                        font-size: 2rem;
                        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        letter-spacing: 1px;
                        margin-bottom: 0.6rem;
                        animation: fadeInDown 1s ease-in-out;
                    }

                    .class-subtext {
                        text-align: center;
                        font-family: 'Inter', sans-serif;
                        color: rgba(230,230,230,0.9);
                        font-size: 1.05rem;
                        max-width: 950px;
                        margin: 0 auto 2rem auto;
                        animation: fadeInUp 1.3s ease;
                    }

                    /* ===== Button Group Styling ===== */
                    div[data-testid="stHorizontalBlock"] button {
                        background: linear-gradient(90deg, #6366f1, #8b5cf6);
                        color: white !important;
                        border: none;
                        border-radius: 10px;
                        padding: 0.7rem 1rem;
                        font-weight: 500;
                        transition: all 0.3s ease;
                        font-family: 'Inter', sans-serif;
                        box-shadow: 0 3px 10px rgba(99, 102, 241, 0.25);
                    }
                    div[data-testid="stHorizontalBlock"] button:hover {
                        transform: translateY(-3px);
                        background: linear-gradient(90deg, #7c3aed, #4f46e5);
                        box-shadow: 0 5px 18px rgba(99, 102, 241, 0.35);
                    }
                    .active-btn {
                        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
                        box-shadow: 0 0 20px rgba(139,92,246,0.6) !important;
                        transform: scale(1.03);
                    }

                    /* ===== Chart Container ===== */
                    .chart-container {
                        background: rgba(255, 255, 255, 0.04);
                        padding: 2rem 1.5rem;
                        border-radius: 16px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.35);
                        backdrop-filter: blur(12px) saturate(130%);
                        -webkit-backdrop-filter: blur(12px) saturate(130%);
                        max-width: 900px;
                        margin: 0 auto;
                        animation: fadeInUp 1.1s ease-in-out;
                    }

                    /* Fade Animations */
                    @keyframes fadeInDown {
                        0% { opacity: 0; transform: translateY(-20px); }
                        100% { opacity: 1; transform: translateY(0); }
                    }
                    @keyframes fadeInUp {
                        0% { opacity: 0; transform: translateY(20px); }
                        100% { opacity: 1; transform: translateY(0); }
                    }

                /* ==========================================================
                📦 Summary Box Styling — Unified Dark Glass Design
                ========================================================== */
                .summary-box {
                    background: rgba(255, 255, 255, 0.06);
                    border-radius: 18px;
                    padding: 2.2rem 2.6rem;
                    margin: 3rem auto;
                    font-family: 'Inter', sans-serif;
                    color: rgba(240, 240, 240, 0.92);
                    line-height: 1.75;
                    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
                    max-width: 1200px;
                    backdrop-filter: blur(12px) saturate(130%);
                    -webkit-backdrop-filter: blur(12px) saturate(130%);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    transition: all 0.45s ease;
                    animation: fadeInUp 1.3s ease-in-out;
                }

                .summary-box:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
                    border: 1px solid rgba(167, 139, 250, 0.25);
                }

                /* ==========================================================
                🏷️ Section Titles
                ========================================================== */
                .summary-box h3 {
                    color: #a78bfa;
                    font-family: 'Poppins', sans-serif;
                    font-weight: 600;
                    margin-top: 1.4rem;
                    margin-bottom: 0.8rem;
                    text-shadow: 0 0 10px rgba(167, 139, 250, 0.25);
                    transition: color 0.3s ease;
                    letter-spacing: 0.4px;
                }

                .summary-box h3:hover {
                    color: #c084fc;
                }

                /* ==========================================================
                📊 Table Styling
                ========================================================== */
                .summary-box table {
                    margin: 1rem auto;
                    border-collapse: collapse;
                    width: 96%;
                    border-radius: 10px;
                    overflow: hidden;
                    animation: fadeIn 1.2s ease;
                }

                .summary-box th, .summary-box td {
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    padding: 10px 14px;
                    text-align: center;
                    font-size: 0.95rem;
                }

                .summary-box th {
                    background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.2));
                    color: #e0e7ff;
                    font-weight: 600;
                    letter-spacing: 0.3px;
                    text-transform: uppercase;
                }

                .summary-box td {
                    color: #f3f4f6;
                    background: rgba(255, 255, 255, 0.03);
                    transition: background 0.3s ease, transform 0.2s ease;
                }

                /* Row hover */
                .summary-box tr:hover td {
                    background: rgba(139, 92, 246, 0.12);
                    transform: scale(1.01);
                }

                /* ==========================================================
                ✨ Animations
                ========================================================== */
                @keyframes fadeInUp {
                    0% { opacity: 0; transform: translateY(20px); }
                    100% { opacity: 1; transform: translateY(0); }
                }

                @keyframes fadeInDown {
                    0% { opacity: 0; transform: translateY(-20px); }
                    100% { opacity: 1; transform: translateY(0); }
                }

                @keyframes fadeIn {
                    0% { opacity: 0; }
                    100% { opacity: 1; }
                }

                    </style>
        """, unsafe_allow_html=True)

        # === Header ===
    st.markdown("<div class='class-header'>Class Distribution Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='class-subtext'>
        A detailed comparison of <b>BDD100K</b> class frequency across training and validation sets — visualized through research-driven insights, 
        statistical distributions, and interactive exploration tools.
    </div>
    """, unsafe_allow_html=True)





    data = fetch_class_distribution()

    if data and "train" in data:

        # === Radio Buttons Inside Dashboard ===
        st.markdown("<h5 style='text-align:center; font-family:Montserrat;'>Select Distribution Scale</h5>", unsafe_allow_html=True)
        scale_col1, scale_col2, scale_col3 = st.columns([1, 1, 1])
        with scale_col1:
            count_selected = st.button("Count")
        with scale_col2:
            log_selected = st.button("Log Count")
        with scale_col3:
            percent_selected = st.button("Percentage")

        if "mode" not in st.session_state:
            st.session_state["mode"] = "Count"
        if count_selected:
            st.session_state["mode"] = "Count"
        elif log_selected:
            st.session_state["mode"] = "Log Count"
        elif percent_selected:
            st.session_state["mode"] = "Percentage"
        mode = st.session_state["mode"]

        # === Tabs ===
        tab_train, tab_val = st.tabs(["Training Set", "Validation Set"])

        def make_dataframe(split_data):
            return pd.DataFrame({
                "Class": list(split_data["class_counts"].keys()),
                "Count": list(split_data["class_counts"].values()),
                "Log Count": list(split_data["log_class_counts"].values()),
                "Percentage": list(split_data["class_percent"].values())
            })

        # ================= TRAINING =================
        
        with tab_train:
            df_train = make_dataframe(data["train"])
            y = mode
            st.subheader(f"Training Set — {mode}")

            

            # =======================
            # 🎨 BEAUTIFIED PLOTLY BAR CHART (TRAIN)
            # =======================
            fig = px.bar(
                df_train,
                x="Class",
                y=y,
                title=f"✨ Training Class Distribution ({mode})",
                color="Class",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                text_auto=True,
            )

            # =======================
            # 🧩 STYLING FIXES + VISUALS
            # =======================
            fig.update_traces(
                marker=dict(line=dict(width=1.2, color="rgba(255,255,255,0.6)")),
                opacity=0.93,
                textfont=dict(size=11, color="rgba(40,40,40,0.9)", family="Poppins"),
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            )

            # ✅ Fixed axis formatting (no `titlefont` anymore)
            fig.update_layout(
                plot_bgcolor="rgba(255,255,255,0.05)",
                paper_bgcolor="rgba(255,255,255,0.0)",
                font=dict(family="Poppins, sans-serif", size=13, color="rgba(235,235,235,0.9)"),

                xaxis=dict(
                    tickangle=-35,
                    showgrid=False,
                    zeroline=False,
                    title=dict(text="Class", font=dict(size=12, color="#93c5fd")),  # ✅ fixed
                    tickfont=dict(size=11, color="rgba(230,230,230,0.9)")
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zeroline=False,
                    title=dict(text="Frequency", font=dict(size=12, color="#93c5fd")),  # ✅ fixed
                    tickfont=dict(size=11, color="rgba(230,230,230,0.9)")
                ),

                title=dict(
                    x=0.5,
                    font=dict(size=16, color="#60a5fa", family="Poppins", weight="bold"),
                ),
                margin=dict(t=60, b=60, l=60, r=40),
                hoverlabel=dict(
                    bgcolor="rgba(96,165,250,0.9)",
                    font_size=12,
                    font_family="Poppins",
                    font_color="white"
                ),
                bargap=0.25,
                transition_duration=500,
            )

            # =======================
            # ✨ ADD SUBTLE HIGHLIGHT EFFECTS
            # =======================
            fig.update_traces(
                marker_line_width=1.2,
                marker_line_color="rgba(255,255,255,0.5)",
                selector=dict(type='bar')
            )

            # =======================
            # 🌈 GLASSMORPHIC CARD CONTAINER
            # =======================
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05);
                        border-radius:15px;
                        padding:1.2rem;
                        box-shadow:0 4px 20px rgba(0,0,0,0.35);
                        margin-bottom:1rem;
                        animation:fadeInUp 0.8s ease-in-out;'>
            """, unsafe_allow_html=True)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)



        # ================= VALIDATION =================
        with tab_val:
            df_val = make_dataframe(data["val"])
            y = mode
            st.subheader(f" Validation Set — {mode}")

            

            # =======================
            # 🎨 BEAUTIFIED PLOTLY BAR CHART (VALIDATION)
            # =======================
            fig = px.bar(
                df_val,
                x="Class",
                y=y,
                title=f" Validation Class Distribution ({mode})",
                color="Class",
                color_discrete_sequence=px.colors.qualitative.Pastel,  # soft pastel tones
                text_auto=True,
            )

            # =======================
            # 🧩 STYLE FIXES + MODERN LAYOUT
            # =======================
            fig.update_layout(
                plot_bgcolor="rgba(255,255,255,0.05)",
                paper_bgcolor="rgba(255,255,255,0.0)",
                font=dict(family="Poppins, sans-serif", size=13, color="rgba(235,235,235,0.9)"),

                xaxis=dict(
                    tickangle=-35,
                    showgrid=False,
                    zeroline=False,
                    title=dict(text="Class", font=dict(size=12, color="#93c5fd")),  # ✅ fixed
                    tickfont=dict(size=11, color="rgba(230,230,230,0.9)")
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zeroline=False,
                    title=dict(text="Frequency", font=dict(size=12, color="#93c5fd")),  # ✅ fixed
                    tickfont=dict(size=11, color="rgba(230,230,230,0.9)")
                ),

                title=dict(
                    x=0.5,
                    font=dict(size=16, color="#60a5fa", family="Poppins", weight="bold"),
                ),
                margin=dict(t=60, b=60, l=60, r=40),
                hoverlabel=dict(
                    bgcolor="rgba(96,165,250,0.9)",
                    font_size=12,
                    font_family="Poppins",
                    font_color="white"
                ),
                bargap=0.25,
                transition_duration=500,
            )

            # =======================
            # ✨ ADD SUBTLE HIGHLIGHT EFFECTS
            # =======================
            fig.update_traces(
                marker_line_width=1.2,
                marker_line_color="rgba(255,255,255,0.5)",
                selector=dict(type='bar')
            )

            # =======================
            # 🌫️ GLASSMORPHIC CARD CONTAINER
            # =======================
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05);
                        border-radius:15px;
                        padding:1.2rem;
                        box-shadow:0 4px 20px rgba(0,0,0,0.35);
                        margin-bottom:1rem;
                        animation:fadeInUp 0.8s ease-in-out;'>
            """, unsafe_allow_html=True)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)



        # === Meta Info ===
        st.markdown(
            f"<div style='text-align:center;font-family:Inter;font-size:0.9rem;color:rgba(220,220,220,0.7);margin-top:1.2rem;'>"
            f"<b>Last Computed:</b> {data.get('Last Computed', 'N/A')} &nbsp; | &nbsp;"
            f"<b>Cache:</b> {data.get('meta', {}).get('cache_status', 'unknown').capitalize()}</div>",
            unsafe_allow_html=True
        )

        # === Summary Section (Animated) ===
        st.markdown("""
      




    <style>
    .summary-box {
        background: rgba(255,255,255,0.06);
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        line-height: 1.55;
        font-family: 'Inter', sans-serif;
        color: #e5e7eb;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.15);
        animation: fadeInUp 0.8s ease-in-out;
    }
    .summary-box h3 {
        font-family: 'Poppins', sans-serif;
        color: #93c5fd;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }
    .summary-box table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.6rem 0;
    }
    .summary-box table, .summary-box th, .summary-box td {
        border: 1px solid rgba(255,255,255,0.15);
        padding: 6px;
        text-align: center;
    }
    .summary-box th {
        background: rgba(255,255,255,0.15);
        color: #f9fafb;
    }
    .summary-box ul {
        margin-left: 1rem;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>

    <div class='summary-box'>

    <h3>1. Overall Dataset Scale</h3>
    <ul>
        <li><b>Training set:</b> 69,863 images containing ~1.94M annotated objects.</li>
        <li><b>Validation set:</b> 10,000 images containing ~279K annotated objects.</li>
        <li>Each image holds on average <b>27–28 objects</b>, reflecting the complexity of real-world traffic scenes.</li>
    </ul>

    <h3>2. Major Class Distribution</h3>
    <p>
        <b>Dominant:</b> Car (36–37%) and Lane (27%) dominate, covering ~63% of all annotations.<br>
        <b>Moderate:</b> Drivable area (6%), Person (4.7%).<br>
        <b>Minority:</b> Truck, Bus, Bike, Rider, Motor, Train (<3% combined).<br><br>
        <b>Interpretation:</b> The dataset exhibits a natural <i>long-tailed imbalance</i> common in urban driving environments.
    </p>

    <h3>3. Distribution Consistency (Train vs Validation)</h3>
    <table>
        <tr><th>Class</th><th>Train (%)</th><th>Validation (%)</th><th>Difference</th></tr>
        <tr><td>Car</td><td>36.74</td><td>36.71</td><td>0.03</td></tr>
        <tr><td>Lane</td><td>27.23</td><td>27.12</td><td>0.11</td></tr>
        <tr><td>Traffic Sign</td><td>12.35</td><td>12.50</td><td>-0.15</td></tr>
        <tr><td>Traffic Light</td><td>9.59</td><td>9.63</td><td>-0.04</td></tr>
        <tr><td>Drivable Area</td><td>6.48</td><td>6.44</td><td>0.04</td></tr>
        <tr><td>Person</td><td>4.71</td><td>4.75</td><td>-0.04</td></tr>
    </table>
    <i>→ The nearly identical train/val percentages confirm unbiased sampling and excellent dataset consistency.</i>

    <h3>4. Long-Tail Behavior & Frequency Drop</h3>
    <p>
        Object frequencies fall from <b>log(5.85)</b> for cars to <b>log(2.13)</b> for trains — a <b>3.7 log-unit drop (~5000× difference)</b>.<br>
        This is a textbook <b>Zipf/power-law distribution</b>, verified in the BDD100K research paper.<br>
        <b>Implication:</b> Rare classes (train, rider, motor) need focal loss, class rebalancing, or targeted augmentation.
    </p>

    <h3>5. Rare Classes Are Safety-Critical</h3>
    <ul>
        <li>Rider (0.23%), Motor (0.15%), Train (0.01%) are extremely rare.</li>
        <li>These appear mostly in <b>safety-critical scenarios</b> like intersections.</li>
        <li>Low frequency → low recall unless treated with balancing strategies.</li>
    </ul>

    <h3>6. Correlation with Scene Types</h3>
    <p>Class frequencies correlate with geography and scene-type:</p>
    <ul>
        <li><b>Urban</b> → more persons, buses, riders, bikes.</li>
        <li><b>Highways</b> → more trucks, fewer pedestrians.</li>
        <li><b>Night</b> → traffic-light visibility dominates.</li>
    </ul>

    <h3>7. Why Car & Lane Dominate (Sensor Geometry)</h3>
    <p>
        BDD100K uses <b>roof-mounted forward-facing cameras</b>:
    </p>
    <ul>
        <li><b>Cars (~37%)</b> consistently appear ahead.</li>
        <li><b>Lanes (~27%)</b> always visible beneath the camera.</li>
        <li><b>Traffic signs/lights (~22%)</b> frequently fall in the camera’s field of view.</li>
    </ul>

    <h3>8. Drivable-Area Underrepresentation (~6%)</h3>
    <p>
        Despite its importance, drivable-area forms only ~6% of annotations.<br>
        Weather, nighttime lighting, and occlusion reduce road visibility — also noted in the BDD100K paper.
    </p>

    <h3>9. Occlusion & Scale Challenges</h3>
    <ul>
        <li>Small classes (bike, rider) suffer from <b>heavy occlusion</b>.</li>
        <li>They appear at <b>very small scale</b>, increasing detection difficulty.</li>
        <li>Underrepresentation + small scale = low AP without targeted training.</li>
    </ul>

    <h3>10. Multi-Task Label Influence</h3>
    <ul>
        <li>Lane & drivable-area → dominate segmentation tasks.</li>
        <li>Vehicles & persons → dominate detection tasks.</li>
        <li>Thus, long-tail imbalance differs per task — crucial for multi-task learning.</li>
    </ul>

    <h3>Summary</h3>
    <p>
        BDD100K shows a strong <b>long-tailed distribution</b> with excellent train/val consistency.<br>
        Rare classes are safety-critical and require <b>focal loss</b>, <b>oversampling</b>, and <b>augmentation</b>.<br>
        The dataset’s scale and diversity make it ideal for robust autonomous driving perception models.
    </p>

    </div>









        """, unsafe_allow_html=True)

    else:
        st.warning("No class distribution data available.")

# ==========================================================
# VISUAL ATTRIBUTE DISTRIBUTIONS (Brightness / Contrast / Saturation)
# ==========================================================





# ==========================================================
#  VISUAL DISTRIBUTIONS SECTION (with Annotations)
# ==========================================================
elif selected == "Visual Distributions":

    import requests, json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
    import altair as alt

    # -----------------------------------------------
    # 🔹 Section Header and Style
    # -----------------------------------------------


    st.markdown("""
    <style>
    /* ==========================================
       🌈 Animated Visual Distribution Header
       ========================================== */

    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes textGlow {
        0% { text-shadow: 0 0 8px rgba(96,165,250,0.3), 0 0 15px rgba(139,92,246,0.2); }
        50% { text-shadow: 0 0 14px rgba(167,139,250,0.55), 0 0 25px rgba(96,165,250,0.35); }
        100% { text-shadow: 0 0 8px rgba(96,165,250,0.3), 0 0 15px rgba(139,92,246,0.2); }
    }

    .visual-header {
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        letter-spacing: 0.5px;
        animation: fadeInDown 1s ease-out, textGlow 3s ease-in-out infinite alternate;
    }

    .visual-header:hover {
        transform: scale(1.02);
        filter: brightness(1.1);
        transition: all 0.3s ease;
    }

    .visual-subtext {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: rgba(235,235,235,0.88);
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 850px;
        margin: 0 auto 2rem auto;
        animation: fadeInUp 1.2s ease-in-out;
        background: rgba(250,250,250,0.03);
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        backdrop-filter: blur(6px);
    }

    .visual-subtext:hover {
        background: rgba(255,255,255,0.06);
        box-shadow: 0 6px 20px rgba(99,102,241,0.25);
        transform: translateY(-2px);
        transition: all 0.35s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='visual-header'>Visual Feature Distributions Across Classes</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='visual-subtext'>
        Visual analysis of <b>Brightness</b>, <b>Contrast</b>, and <b>Saturation</b> distributions in the <b>BDD100K</b> dataset.<br>
        These visual metrics help quantify <b>illumination diversity</b> and <b>sensor variation</b> across different categories.
    </div>
    """, unsafe_allow_html=True)


    # -----------------------------------------------
    # 🔹 Load data from backend API
    # -----------------------------------------------
    data = fetch_api("visual-distributions")

    if not data or "train" not in data:
        st.warning(" No visual distribution data found. Please ensure API is available.")
    else:
        train_data = data["train"]
        val_data = data["val"]

        # Sidebar / Radio inside dashboard
 # -----------------------------------------------

        st.markdown("""
        <style>
        /* ====== Radio Button Container Styling ====== */
        div[data-testid="stRadio"] {
            background: rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin: 1.2rem auto 2rem auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            width: fit-content;
            display: flex;
            justify-content: center;
            align-items: center;
            animation: fadeInUp 0.9s ease-in-out;
        }

        /* ====== Radio Labels ====== */
        div[role="radiogroup"] label {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: rgba(235,235,235,0.9);
            border-radius: 10px;
            padding: 0.5rem 1rem;
            margin: 0 0.3rem;
            transition: all 0.25s ease-in-out;
            cursor: pointer;
            border: 1px solid rgba(255,255,255,0.08);
        }

        /* ====== Hover Effect ====== */
        div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.1);
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(139,92,246,0.2);
        }

        /* ====== Active (Selected) Radio Option ====== */
        div[role="radiogroup"] label[data-testid="stMarkdownContainer"] div[style*="background-color: rgb(49, 51, 63)"],
        div[role="radiogroup"] label:has(input[checked]) {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white !important;
            box-shadow: 0 0 14px rgba(99,102,241,0.5);
            transform: scale(1.03);
            border: 1px solid rgba(255,255,255,0.15);
        }

        /* ====== Title Styling ====== */
        .visual-radio-title {
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 1.3rem;
            letter-spacing: 0.3px;
            color: #a5b4fc;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 8px rgba(99,102,241,0.35);
            animation: fadeInDown 0.9s ease-in-out;
        }

        /* ====== Animation Keyframes ====== */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)

        # Title above the selector
        st.markdown("<div class='visual-radio-title'>Choose Visualization Type</div>", unsafe_allow_html=True)

        # Radio Buttons
        view_type = st.radio(
            "Choose Visualization Type",
            ["Histogram Distribution", "Boxplot "],
            horizontal=True,
        )

        # ---------------------------------------------------
        # 🔹 1. HISTOGRAM DISTRIBUTION (Train vs Val)
        # ---------------------------------------------------
        if view_type == "Histogram Distribution":
            # ---------------------------------------------------
            # 🔹 HISTOGRAMS: Brightness, Contrast, Saturation
            # ---------------------------------------------------
            st.markdown("""
            <style>
            /* ====== Section Title Animation ====== */
            .hist-section-title {
                text-align: center;
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                font-size: 1.6rem;
                letter-spacing: 0.4px;
                background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 8px rgba(96,165,250,0.4);
                margin-bottom: 0.8rem;
                animation: fadeInDown 0.9s ease-in-out;
            }

            /* ====== Plot Card Animation ====== */
            .plot-card {
                background: rgba(255,255,255,0.05);
                border-radius: 16px;
                padding: 1.5rem 2rem;
                margin-top: 1rem;
                box-shadow: 0 6px 24px rgba(0,0,0,0.35);
                backdrop-filter: blur(10px);
                animation: fadeInUp 1s ease-in-out;
            }

            /* ====== Descriptive Box Styling ====== */
            .analysis-box {
                text-align: justify;
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: rgba(230,230,230,0.9);
                margin-top: 0.8rem;
                background: rgba(255,255,255,0.06);
                border-radius: 14px;
                padding: 1.4rem 1.8rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.35);
                backdrop-filter: blur(8px);
                animation: fadeInUp 1.3s ease-in-out;
                line-height: 1.7;
            }

            /* ====== Keyframe Animations ====== */
            @keyframes fadeInUp {
                from {opacity: 0; transform: translateY(25px);}
                to {opacity: 1; transform: translateY(0);}
            }
            @keyframes fadeInDown {
                from {opacity: 0; transform: translateY(-25px);}
                to {opacity: 1; transform: translateY(0);}
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("<div class='hist-section-title'>Histogram Distributions — Brightness, Contrast, and Saturation</div>", unsafe_allow_html=True)

            sns.set_style("whitegrid")
            plt.rcParams.update({
                "axes.facecolor": (1, 1, 1, 0.03),
                "figure.facecolor": "none",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": (1, 1, 1, 0.1),        
                "axes.titlecolor": "#a5b4fc"
            })

            # --- Helper for consistent plotting ---
            def plot_hist_comparison(train_dict, val_dict, metric_name, bins, x_label):
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                fig.patch.set_alpha(0.0)

                for cls, dist in train_dict.items():
                    axes[0].plot(bins[:len(dist)], dist, label=cls, alpha=0.8)
                axes[0].set_title(f"{metric_name} — Train Set", fontsize=13)
                axes[0].set_xlabel(x_label)
                axes[0].set_ylabel("Frequency")

                for cls, dist in val_dict.items():
                    axes[1].plot(bins[:len(dist)], dist, label=cls, alpha=0.8)
                axes[1].set_title(f"{metric_name} — Validation Set", fontsize=13)
                axes[1].set_xlabel(x_label)

                for ax in axes:
                    ax.grid(True, linestyle="--", alpha=0.3)
                    from matplotlib.colors import to_rgba
                    ax.set_facecolor(to_rgba("white", 0.03))  # ✅ Same effect, more readable

                    ax.tick_params(colors="white", labelsize=10)
                axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
                plt.tight_layout()

                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

            # --- Brightness ---
            plot_hist_comparison(
                train_data["brightness_distribution_per_class"],
                val_data["brightness_distribution_per_class"],
                "Brightness",
                np.linspace(0, 255, 20),
                "Brightness (0–255)"
            )

            st.markdown("""
            <div class='analysis-box'>
                <b><u>Brightness Distribution Analysis</u></b><br><br>
                - The histograms show a strong <b>mid-brightness dominance (70–90 range)</b>, typical of well-lit urban scenes.<br>
                - Classes like <b>person</b>, <b>truck</b>, and <b>bus</b> exhibit higher mean brightness due to reflective surfaces.<br>
                - The <b>train</b> class peaks near <b>107</b>, possibly from metallic glare or headlights.<br>
                - Consistency across splits indicates <b>excellent photometric calibration</b> and low exposure bias.<br><br>
                <b>Model Implication:</b> Brightness normalization is optional — only minor tone-mapping may help in nighttime scenes.
            </div>
            """, unsafe_allow_html=True)



            # --- Contrast ---
            plot_hist_comparison(
                train_data["contrast_distribution_per_class"],
                val_data["contrast_distribution_per_class"],
                "Contrast",
                np.linspace(0, 128, 20),
                "Contrast (0–128)"
            )

            st.markdown("""
                            <div style='text-align:justify; font-family:Inter; font-size:0.95rem;
                                        color:rgba(230,230,230,0.9); margin-top:0.5rem;
                                        background:rgba(255,255,255,0.06);
                                        border-radius:14px; padding:1.3rem 1.6rem;
                                        box-shadow:0 4px 16px rgba(0,0,0,0.35);
                                        animation:fadeInUp 1.4s ease-in-out;'>

                            <b><u>Contrast Distribution Analysis</u></b><br><br>

                            - The majority of classes cluster within the <b>45–55 contrast range</b>, suggesting well-balanced dynamic range across lighting conditions.  
                            - <b>Human and large vehicle</b> classes (person, truck, bus) exhibit slightly higher contrast, reflecting clear edges and textured boundaries — beneficial for object localization models like YOLO or DETR.  
                            - Classes such as <b>lane</b> and <b>drivable area</b> show lower contrast, which aligns with smoother surfaces and uniform illumination.  
                            - The absence of extremely low-contrast samples indicates the dataset’s strong handling of fog and low-visibility frames.  
                            - Validation contrast follows a nearly identical pattern to training, reinforcing **temporal consistency** and uniform sensor capture.  
                            <br>
                            <b>Model Implication:</b> High contrast uniformity ensures <b>stable feature extraction</b> during convolution, reducing the need for histogram equalization or CLAHE preprocessing.

                            </div>
                            """, unsafe_allow_html=True)



            # --- Saturation ---
            plot_hist_comparison(
                train_data["saturation_distribution_per_class"],
                val_data["saturation_distribution_per_class"],
                "Saturation",
                np.linspace(0, 255, 20),
                "Saturation (0–255)"
            )

            
            st.markdown("""
                        <div style='text-align:justify; font-family:Inter; font-size:0.95rem;
                                    color:rgba(230,230,230,0.9); margin-top:0.5rem;
                                    background:rgba(255,255,255,0.06);
                                    border-radius:14px; padding:1.3rem 1.6rem;
                                    box-shadow:0 4px 16px rgba(0,0,0,0.35);
                                    animation:fadeInUp 1.6s ease-in-out;'>

                        <b><u>Saturation Distribution Analysis</u></b><br><br>

                        - Most classes exhibit <b>moderate saturation (60–75)</b>, signifying naturally balanced color tones — neither overexposed nor washed out.  
                        - The <b>person</b> and <b>bike</b> categories show slightly lower saturation, typically captured under shaded regions or less vibrant surroundings.  
                        - Classes like <b>traffic signs</b> and <b>traffic lights</b> display stronger color intensity peaks, consistent with their inherently vivid colors (red, green, yellow).  
                        - The smooth decline toward high-saturation bins indicates <b>minimal color clipping</b>, a strong indicator of good image quality and proper encoding.  
                        - The uniform alignment between training and validation saturation implies no color domain shift.  
                        <br>
                        <b>Model Implication:</b> Color-based detectors (e.g., semantic segmentation or traffic signal recognition) will benefit from the dataset’s <b>natural color variance</b>, improving generalization across weather and daylight conditions.

                        </div>
                        """, unsafe_allow_html=True)




            st.subheader("Mean Visual Metrics — Train vs Validation")

            # Collect all unique class names from both sets
            all_classes = sorted(
                set(train_data["mean_brightness_per_class"].keys())
                | set(val_data["mean_brightness_per_class"].keys())
            )

            # Helper function for safe dictionary access
            def safe_get(d, k):
                return d.get(k, 0)

            # Build dataframe
            df_mean = pd.DataFrame({
                "Class": all_classes,
                "Brightness (Train)": [safe_get(train_data["mean_brightness_per_class"], c) for c in all_classes],
                "Brightness (Val)": [safe_get(val_data["mean_brightness_per_class"], c) for c in all_classes],
                "Contrast (Train)": [safe_get(train_data["mean_contrast_per_class"], c) for c in all_classes],
                "Contrast (Val)": [safe_get(val_data["mean_contrast_per_class"], c) for c in all_classes],
                "Saturation (Train)": [safe_get(train_data["mean_saturation_per_class"], c) for c in all_classes],
                "Saturation (Val)": [safe_get(val_data["mean_saturation_per_class"], c) for c in all_classes],
            })

            # Round for readability
            df_mean = df_mean.round(2)

            # Add difference columns
            df_mean["Δ Brightness"] = df_mean["Brightness (Val)"] - df_mean["Brightness (Train)"]
            df_mean["Δ Contrast"] = df_mean["Contrast (Val)"] - df_mean["Contrast (Train)"]
            df_mean["Δ Saturation"] = df_mean["Saturation (Val)"] - df_mean["Saturation (Train)"]

            # Styled dataframe for Streamlit
            st.dataframe(
                df_mean.style
                .background_gradient(
                    subset=["Brightness (Train)", "Brightness (Val)"], cmap="Blues"
                )
                .background_gradient(
                    subset=["Contrast (Train)", "Contrast (Val)"], cmap="Purples"
                )
                .background_gradient(
                    subset=["Saturation (Train)", "Saturation (Val)"], cmap="Greens"
                )
                .bar(
                    subset=["Δ Brightness", "Δ Contrast", "Δ Saturation"],
                    align="mid",
                    color=["#34d399", "#f87171"]
                )
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#1f2937"), ("color", "white"), ("font-weight", "600")]},
                    {"selector": "td", "props": [("color", "#e5e7eb"), ("font-family", "'Inter', sans-serif")]}
                ])
                .format(precision=2)
            )

            # ---------------------------------------------------

            st.markdown("""
            <div style="margin-top:2.5rem; padding:2rem; border-radius:18px;
                        background:rgba(255,255,255,0.05);
                        box-shadow:0 8px 30px rgba(0,0,0,0.25);
                        backdrop-filter:blur(10px);
                        font-family:'Inter',sans-serif;
                        color:rgba(230,230,230,0.92);
                        animation:fadeIn 1s ease-in-out;">

            <h2 style="text-align:center; font-family:'Montserrat',sans-serif; 
                    background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    font-size:1.8rem; letter-spacing:1px; margin-bottom:1.2rem;">
             Visual Distribution Analysis — Inference
            </h2>

            <div style="animation:fadeInUp 1.5s ease;">
            <h3 style="color:#a78bfa;">1️ Brightness Patterns</h3>
            <ul>
            <li>Brightness values peak around mid-range intensity (70–80), indicating balanced lighting conditions across driving scenes.</li>
            <li>Person, bus, truck, and rider classes show slightly higher brightness means (85–89), reflecting exposure to open daylight or reflective surfaces.</li>
            <li>The <b>train class (107.3)</b> shows exceptional brightness — possibly due to metallic reflections or sensor glare in high-contrast railway frames.</li>
            <li>Validation and training means differ marginally (<1–2%), confirming <b>consistent photometric conditions</b> across splits.</li>
            </ul>
            </div>

            <div style="animation:fadeInUp 1.8s ease;">
            <h3 style="color:#8b5cf6;">2️ Contrast Distribution</h3>
            <ul>
            <li>Contrast distributions cluster around 50–52 for dominant classes (car, lane, traffic sign), suggesting <b>moderate texture richness</b>.</li>
            <li>Human-centric and heavy vehicle classes (person, truck, bus) exhibit higher contrast (55–57), aligning with edge-dense silhouettes and reflective materials.</li>
            <li>Validation contrasts are nearly identical to training — a sign of <b>uniform image sharpness and sensor calibration</b>.</li>
            <li>The overall contrast range (49–58) indicates a <b>well-normalized dataset</b> suitable for CNN-based feature extraction.</li>
            </ul>
            </div>

            <div style="animation:fadeInUp 2s ease;">
            <h3 style="color:#f472b6;">3️ Saturation Behavior</h3>
            <ul>
            <li>Average saturation lies between 60–74 across most classes — consistent with <b>natural daylight color balance</b>.</li>
            <li>Lower saturation for <b>bike and person</b> categories (≈60–63) indicates dull or shadowed regions, whereas traffic-related classes are more vibrant.</li>
            <li>Train and drivable area classes have moderate saturation (~69–74), implying <b>scene color uniformity</b> without dominant hue biases.</li>
            <li>The stability across train/val sets suggests <b>no chromatic domain shift</b> — ensuring color robustness for generalization.</li>
            </ul>
            </div>

            <div style="animation:fadeInUp 2.3s ease;">
            <h3 style="color:#60a5fa;">4️ Summary Insights</h3>
            <ul>
            <li>The dataset exhibits <b>high photometric stability</b> — mean values across brightness, contrast, and saturation are consistent within a ±2% margin.</li>
            <li>Minor brightness elevation in person/truck scenes aligns with <b>outdoor illumination variance</b>.</li>
            <li>Uniform tone and exposure confirm <b>excellent dataset quality</b> for vision tasks like object detection and segmentation.</li>
            <li>No significant visual bias observed across training and validation splits.</li>
            </ul>
            </div>

            <hr style="border:1px solid rgba(255,255,255,0.1); margin:1.8rem 0;">

            <div style="text-align:center; font-size:1rem; color:rgba(220,220,220,0.85);">
            <b>Conclusion:</b> The BDD100K dataset demonstrates <b>consistent visual characteristics</b> across classes and splits.
            This uniformity strengthens model generalization and ensures reliable validation under diverse real-world lighting.
            </div>

            <style>
            @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
            @keyframes fadeInUp { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
            </style>

            </div>
            """, unsafe_allow_html=True)










            st.info("""
             **Insights:**
            - **Brightness:** Well-balanced across splits; ideal for photometric consistency.  
            - **Contrast:** Slightly higher variance for smaller objects (signs/lights).  
            - **Saturation:** Stable across splits — confirming consistent scene color calibration.  
            """)

           
        else:

            st.markdown("""
            <style>
            .boxplot-section-title {
                text-align: center;
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                font-size: 1.6rem;
                letter-spacing: 0.4px;
                background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 10px rgba(96,165,250,0.4);
                margin-bottom: 1rem;
                animation: fadeInDown 1s ease-in-out;
            }
            .plot-card {
                background: rgba(255,255,255,0.05);
                border-radius: 16px;
                padding: 1.5rem;
                margin-top: 1rem;
                box-shadow: 0 6px 22px rgba(0,0,0,0.4);
                backdrop-filter: blur(10px);
                animation: fadeInUp 1.1s ease-in-out;
            }
            .box-info {
                text-align: justify;
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: rgba(230,230,230,0.9);
                margin-top: 1rem;
                background: rgba(255,255,255,0.06);
                border-radius: 14px;
                padding: 1.2rem 1.6rem;
                box-shadow: 0 4px 18px rgba(0,0,0,0.35);
                backdrop-filter: blur(6px);
                animation: fadeInUp 1.4s ease-in-out;
                line-height: 1.7;
            }
            @keyframes fadeInUp {
                from {opacity: 0; transform: translateY(25px);}
                to {opacity: 1; transform: translateY(0);}
            }
            @keyframes fadeInDown {
                from {opacity: 0; transform: translateY(-25px);}
                to {opacity: 1; transform: translateY(0);}
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("<div class='boxplot-section-title'>Boxplot Analysis — Brightness, Contrast, and Saturation (Train Set)</div>", unsafe_allow_html=True)

            # ------------------------------------------
            # 🧩 Helper to expand histogram into values
            # ------------------------------------------
            def expand_histogram_to_values(hist_dict, bins=20, value_range=(0, 255)):
                all_data = {}
                bin_edges = np.linspace(value_range[0], value_range[1], bins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                for cls, hist in hist_dict.items():
                    expanded = []
                    for freq, val in zip(hist, bin_centers):
                        expanded.extend([val] * int(freq))
                    all_data[cls] = expanded
                return all_data

            brightness_vals = expand_histogram_to_values(train_data["brightness_distribution_per_class"])
            contrast_vals = expand_histogram_to_values(train_data["contrast_distribution_per_class"], value_range=(0, 128))
            saturation_vals = expand_histogram_to_values(train_data["saturation_distribution_per_class"])

            # ------------------------------------------
            # 🎨 Boxplot Function (Glassmorphic Style)
            # ------------------------------------------
            def plot_boxplot(metric_data, metric_name, color):
                sns.set(style="whitegrid", context="talk")
                plt.figure(figsize=(12, 6))
                sorted_classes = sorted(metric_data.keys(), key=lambda k: np.median(metric_data[k]) if metric_data[k] else 0)

                box = sns.boxplot(
                    data=[metric_data[c] for c in sorted_classes],
                    color=color,
                    linewidth=1.2,
                    fliersize=3
                )
                plt.xticks(
                    ticks=range(len(sorted_classes)),
                    labels=sorted_classes,
                    rotation=45,
                    ha="right",
                    fontsize=9,
                    color="white"
                )
                plt.title(f"{metric_name} Distribution per Class (Train Set)", fontsize=14, color="#93c5fd", pad=15)
                plt.ylabel(metric_name, color="white")
                plt.xlabel("Class", color="white")
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.gca().set_facecolor((1, 1, 1, 0.03))  # ✅ fixed RGBA
                plt.tight_layout()

                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.pyplot(plt.gcf())
                st.markdown("</div>", unsafe_allow_html=True)

            # ------------------------------------------
            # 📊 Plot the three metrics
            # ------------------------------------------
            plot_boxplot(brightness_vals, "Brightness", "#38bdf8")
            plot_boxplot(contrast_vals, "Contrast", "#f59e0b")
            plot_boxplot(saturation_vals, "Saturation", "#10b981")

            # ------------------------------------------
            # 🧠 Insight Box
            # ------------------------------------------
            st.markdown("""
            <div class='box-info'>
                <b> Boxplot Insights</b><br><br>
                - Objects like <b>traffic signs</b> and <b>traffic lights</b> show wider brightness and contrast spread,  
                indicating diverse illumination and reflective surfaces.<br>
                - <b>Person</b> and <b>bike</b> classes show lower saturation — consistent with shaded captures.<br>
                - Most other classes cluster tightly, confirming consistent lighting and texture balance.<br><br>
                <b>Model Implication:</b> Brightness/contrast normalization not strictly needed —  
                dataset has <b>natural photometric diversity</b> ideal for robust learning.
            </div>
            """, unsafe_allow_html=True)

            # ------------------------------------------
            # 🕒 Meta Footer
            # ------------------------------------------
            meta = data.get("meta", {})
            st.markdown(f"""
            <div style='text-align:center;
                        font-family:Inter;
                        font-size:0.9rem;
                        color:rgba(220,220,220,0.75);
                        margin-top:1.5rem;
                        background:rgba(255,255,255,0.03);
                        padding:0.6rem 0.8rem;
                        border-radius:10px;
                        width:fit-content;
                        margin-left:auto;
                        margin-right:auto;
                        box-shadow:0 0 10px rgba(139,92,246,0.25);
                        backdrop-filter:blur(6px);'>
                <b> Sample Size:</b> {train_data.get("sample_size", "N/A")} images &nbsp; | &nbsp;
                <b> Last Computed:</b> {meta.get("Last Computed", "N/A")} &nbsp; | &nbsp;
                <b> Cache:</b> {meta.get("cache_status", "unknown").capitalize()}
            </div>
            """, unsafe_allow_html=True)
        










elif selected == "Data Quality per Class":

    import requests, json
    
    lottie_url = "https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json"
    lottie_json = requests.get(lottie_url).json()
    st_lottie(lottie_json, height=130, speed=1.1, key="loading_quality")

    st.markdown("""
    <style>
    /* =============================
    ✨ Animated Gradient Header
    ============================= */
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .data-quality-title {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: 0.6px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6, #60a5fa);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 18px rgba(96,165,250,0.4);
        animation: gradientFlow 5s ease infinite, fadeInUp 1.2s ease;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    /* Optional subtitle style */
    .data-quality-subtext {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: rgba(230,230,230,0.85);
        font-size: 1.05rem;
        margin: 0 auto 2rem auto;
        max-width: 800px;
        line-height: 1.6;
        animation: fadeInUp 1.6s ease;
    }

    /* Gradient divider for accent */
    .gradient-divider {
        width: 70%;
        height: 1px;
        margin: 1.2rem auto;
        background: linear-gradient(90deg, rgba(96,165,250,0.4), rgba(244,114,182,0.4));
        border-radius: 2px;
        animation: fadeInUp 2s ease;
    }
    </style>

    <div class='data-quality-title'>
        Data Quality per Class (BDD100K)
    </div>
    <div class='data-quality-subtext'>
        Evaluating per-class <b>exposure</b>, <b>contrast</b>, <b>sharpness</b>, and <b>color balance</b> 
        to assess dataset photometric consistency.
    </div>
    <div class='gradient-divider'></div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:#ccc;'>Evaluating per-class exposure, contrast, sharpness, and color balance.</p>", unsafe_allow_html=True)

   

    cache_path = Path("precomputed_backend") / "data_quality_per_class.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        url = "http://bdd_backend:8000/api/analysis/data-quality-per-class/"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ===============================================
    # ✅ Data Quality Per Class — Visualized Section
    # ===============================================
    if not data or "train" not in data or "val" not in data:
        st.warning(" No data found — check precomputed JSON or backend API.")
        st.stop()

    train = data["train"]["per_class_quality"]
    val = data["val"]["per_class_quality"]

    # ---- Step 2: Convert JSON to DataFrames ----
    train_df = pd.DataFrame(train).T.reset_index().rename(columns={'index': 'class'})
    train_df["subset"] = "Train"
    val_df = pd.DataFrame(val).T.reset_index().rename(columns={'index': 'class'})
    val_df["subset"] = "Validation"
    combined_df = pd.concat([train_df, val_df])

    # ---- Seaborn Theme ----
    sns.set_theme(style="whitegrid", context="talk")

    # ===============================================
    #  Helper: Glassy Compact Bar Plot
    # ===============================================
    def compact_plot(df, metric, title, ylabel, color_palette=["#60a5fa", "#a78bfa"]):
        fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)

        # Draw improved bar chart
        sns.barplot(
            data=df,
            x="class",
            y=metric,
            hue="subset",
            palette=color_palette,
            ax=ax,
            edgecolor="white",
            linewidth=1.3,
            alpha=0.95
        )

        # === Title ===
        ax.set_title(
            f" {title}",
            fontsize=12,
            fontweight="bold",
            color="#a5b4fc",
            pad=14
        )

        # === Axes + Labels ===
        ax.set_xlabel("Class", fontsize=9, color="#e5e7eb", labelpad=8)
        ax.set_ylabel(ylabel, fontsize=10, color="#e5e7eb", labelpad=8)
        ax.tick_params(axis="x", colors="#d1d5db", labelsize=8, rotation=35)
        ax.tick_params(axis="y", colors="#d1d5db", labelsize=8)

        # === Background + Grid ===
        ax.set_facecolor("#0f172a")  # dark navy background
        fig.patch.set_facecolor("#0f172a")
        ax.grid(True, linestyle="--", color=(1, 1, 1, 0.1), alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")

        # === Legend ===
        leg = ax.legend(
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            labelcolor="#f9fafb"
        )
        if leg.get_title() is not None:
            leg.get_title().set_color("#93c5fd")

        # === Glow effect around bars ===
        for patch in ax.patches:
            patch.set_alpha(0.92)
            patch.set_linewidth(0.8)
            patch.set_edgecolor((1, 1, 1, 0.25))


        plt.tight_layout()
        return fig


    # ===============================================
    # 📊 Step 3: Display in Columns with Fade Animations
    # ===============================================
    st.markdown("""
    <style>
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-container {
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 25px rgba(0,0,0,0.35);
        animation: fadeUp 0.9s ease forwards;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.pyplot(compact_plot(combined_df, "exposure_mean", "Exposure (Train vs Validation)", "Mean Exposure"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.pyplot(compact_plot(combined_df, "contrast_mean", "Contrast (Train vs Validation)", "Mean Contrast"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.pyplot(compact_plot(combined_df, "sharpness_mean", "Sharpness (Train vs Validation)", "Mean Sharpness"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.pyplot(compact_plot(combined_df, "colorfulness_mean", "Colorfulness (Train vs Validation)", "Mean Colorfulness"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================================
    # 💬 Insight Block — Styled with Gradient Border
    # ===============================================
    


    st.markdown("""
        <style>
        @keyframes fadeUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .insight-card {
            background: rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 1.6rem 2rem;
            margin-top: 1.2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.45);
            border: 1px solid rgba(147,197,253,0.25);
            backdrop-filter: blur(10px);
            font-family: 'Inter', sans-serif;
            color: rgba(235,235,235,0.92);
            animation: fadeUp 1.2s ease-in-out forwards;
            transition: all 0.3s ease;
        }
        .insight-card:hover {
            transform: scale(1.01);
            box-shadow: 0 12px 40px rgba(0,0,0,0.55);
            border-color: rgba(147,197,253,0.45);
        }
        .insight-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.25rem;
            text-align: center;
            background: linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            letter-spacing: 0.5px;
        }
        .insight-list {
            margin-left: 0.6rem;
            line-height: 1.75;
            font-size: 0.97rem;
        }
        .insight-list li::marker {
            color: #a78bfa;
        }
        </style>

        <div class="insight-card">
            <h4 class="insight-title">Observations & Insights</h4>
            <ul class="insight-list">
                <li><b>Exposure</b> and <b>contrast</b> show remarkable stability — a sign of consistent sensor calibration.</li>
                <li><b>Human-centric</b> classes (person, rider, bike) exhibit slightly higher exposure due to daylight captures.</li>
                <li><b>Sharpness</b> peaks for small-object classes, enabling precise visual feature extraction.</li>
                <li><b>Colorfulness</b> remains balanced — ensuring minimal hue or saturation bias across urban scenes.</li>
                <li>Both <b>train</b> and <b>validation</b> sets display strong <b>photometric alignment</b>, confirming dataset reliability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



    # ===============================================
    # 🔥 Correlation Heatmap
    # ===============================================
    
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    # ======================================================
    # 💡 STEP 1 — Compute Correlation Matrix
    # ======================================================
    metrics = ["exposure_mean", "contrast_mean", "sharpness_mean", "colorfulness_mean"]
    corr = train_df[metrics].corr()

    # ======================================================
    # 🎨 STEP 2 — Compact, Beautified Heatmap
    # ======================================================
    fig, ax = plt.subplots(figsize=(6, 6))  # compact but readable

    # Blue–violet–pink gradient
    cmap = sns.blend_palette(["#4F46E5", "#8B5CF6", "#EC4899"], as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        square=True,
        cbar=True,
        linewidths=0.9,
        linecolor=(1, 1, 1, 0.18),
        annot_kws={"fontsize": 8.5, "fontweight": "600", "color": "#111827"},
        ax=ax
    )

    ax.set_title(
        "Correlation Between Data Quality Metrics",
        fontsize=9.5,
        color="#C4B5FD",
        fontweight="bold",
        pad=8,
        fontname="Poppins"
    )
    ax.tick_params(axis='x', colors="#E5E7EB", labelsize=7.5, rotation=20)
    ax.tick_params(axis='y', colors="#E5E7EB", labelsize=7.5)
    ax.set_facecolor((1, 1, 1, 0.04))
    fig.patch.set_alpha(0.0)
    plt.tight_layout(pad=1.0)

    # ======================================================
    # 🪩 Streamlit Animated Glassmorphic Container
    # ======================================================
    st.markdown("""
    <style>
    @keyframes fadeSlideUp {
        0% {opacity: 0; transform: translateY(25px) scale(0.96);}
        60% {opacity: 1; transform: translateY(-2px) scale(1.02);}
        100% {opacity: 1; transform: translateY(0) scale(1);}
    }
    @keyframes pulseGlow {
        0% {text-shadow: 0 0 6px rgba(147,197,253,0.25);}
        50% {text-shadow: 0 0 16px rgba(147,197,253,0.5);}
        100% {text-shadow: 0 0 6px rgba(147,197,253,0.25);}
    }

    .heatmap-glass {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        margin: 1rem auto;
        max-width: 420px; /* ✅ Slightly larger for centering */
        box-shadow: 0 8px 26px rgba(0,0,0,0.55);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(167,139,250,0.25);
        animation: fadeSlideUp 1.1s ease forwards;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        align-items: center; /* ✅ centers inner content */
        justify-content: center;
    }

    .heatmap-glass:hover {
        transform: scale(1.03);
        box-shadow: 0 10px 34px rgba(147,197,253,0.35);
        border-color: rgba(167,139,250,0.45);
    }

    .heatmap-title {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.4rem;
        color: #C4B5FD;
        animation: pulseGlow 3s infinite ease-in-out;
    }
    </style>

    <div class="heatmap-glass">
        <div class="heatmap-title">Correlation Between Data Quality Metrics</div>
        <div style="display:flex; justify-content:center; align-items:center;">
    """, unsafe_allow_html=True)

    # ✅ Center heatmap figure
    st.pyplot(fig, use_container_width=False)

    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)



    # ===============================================
    # 📉 Variance Error Bars (Compact)
    # ===============================================
    def variance_plot(df, mean_col, std_col, title, ylabel):
        fig, ax = plt.subplots(figsize=(5, 3))
        classes = df["class"]
        ax.errorbar(classes, df[mean_col], yerr=df[std_col], fmt='o-', capsize=3,
                    ecolor="#a78bfa", color="#6366f1", markersize=4, linewidth=1)
        ax.set_facecolor("#f9fafb")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Class", fontsize=9)
        plt.xticks(rotation=40, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        return fig

    col5, col6 = st.columns(2)
    with col5:
        st.pyplot(variance_plot(train_df, "exposure_mean", "exposure_std", "Exposure Variability", "Exposure Value"), use_container_width=True)
    with col6:
        st.pyplot(variance_plot(train_df, "contrast_mean", "contrast_std", "Contrast Variability", "Contrast Value"), use_container_width=True)

    st.markdown("""
    <div style='background:rgba(255,255,255,0.08); border-radius:14px;
    padding:1.1rem 1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.4);
    font-family:Inter; color:#e8e8e8; animation:fadeInUp 1.2s ease-in-out;'>

     <b>Variance Insights:</b><br>
    - Minimal intra-class standard deviation → strong <b>exposure and sharpness consistency</b>.<br>
    - <b>Truck</b> and <b>bus</b> show slightly larger exposure deviation (likely outdoor glare).<br>
    - Indicates <b>well-curated, stable data capture conditions</b> ideal for model pre-training.
    </div>
    """, unsafe_allow_html=True)

    # ===============================================
    # 🔬 Scatterplot (Sharpness vs Contrast)
    # ===============================================
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    # ======================================================
    # 📊 Sharpness vs Contrast — Enhanced Visualization
    # ======================================================
    fig, ax = plt.subplots(figsize=(5.2, 3.2))  # compact height for Streamlit layout

    # Modern gradient palette (blue → violet → rose)
    palette = sns.blend_palette(["#60A5FA", "#8B5CF6", "#EC4899"], n_colors=len(train_df["class"].unique()))

    sns.scatterplot(
        data=train_df,
        x="contrast_mean",
        y="sharpness_mean",
        hue="class",
        s=65,
        alpha=0.9,
        edgecolor=to_rgba("#ffffff", 0.7),
        linewidth=0.7,
        palette=palette,
        ax=ax
    )

    # Subtle gridlines + dark glass background
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.6)
    ax.set_facecolor(to_rgba("#0F172A", 0.6))  # transparent navy tone

    # Typography — cohesive with your theme
    ax.set_title(
        "Sharpness vs Contrast (Train Set)",
        fontsize=11,
        weight="bold",
        color="#A78BFA",
        pad=8,
    )
    ax.set_xlabel("Contrast Mean", fontsize=9.5, labelpad=6, color="#E5E7EB")
    ax.set_ylabel("Sharpness Mean", fontsize=9.5, labelpad=6, color="#E5E7EB")
    plt.xticks(fontsize=8, color="#E5E7EB")
    plt.yticks(fontsize=8, color="#E5E7EB")

    # Legend — simplified, outside the chart
    leg = ax.legend(
        title="Class",
        title_fontsize=9,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02),
        frameon=False,
    )
    for text in leg.get_texts():
        text.set_color("#F3F4F6")

    fig.patch.set_facecolor("#0B1120")  # full dark background
    plt.tight_layout(pad=1.3)

    # ======================================================
    # 🪩 Glassmorphic Animated Container (Streamlit)
    # ======================================================
    st.markdown("""
    <style>
    @keyframes fadeUp {
        0% {opacity:0; transform:translateY(25px);}
        100% {opacity:1; transform:translateY(0);}
    }

    .scatter-glass {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin: 1.4rem auto;
        max-width: 720px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(167,139,250,0.28);
        animation: fadeUp 1.2s ease;
        transition: all 0.3s ease;
    }

    .scatter-glass:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 42px rgba(139,92,246,0.35);
        border-color: rgba(139,92,246,0.45);
    }
    </style>

    <div class="scatter-glass">
        <div style="text-align:center; font-family:'Poppins',sans-serif; font-weight:600;
                    font-size:1rem; color:#C4B5FD; margin-bottom:0.8rem;">
            s Sharpness vs Contrast (Train Set)
        </div>
        <div style="display:flex; justify-content:center; align-items:center;">
    """, unsafe_allow_html=True)

    st.pyplot(fig, use_container_width=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ======================================================
    # 🧠 Insight Card (below chart)
    # ======================================================
    st.markdown("""
    <div style='background:rgba(255,255,255,0.08);
                border-radius:14px;
                padding:1.1rem 1.5rem;
                box-shadow:0 4px 20px rgba(0,0,0,0.4);
                font-family:Poppins;
                color:#e8e8e8;
                animation:fadeUp 1s ease;'> 

    <h4 style='margin-bottom:0.4rem;
            background:linear-gradient(90deg,#a78bfa,#60a5fa);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;'>
     Sharpness–Contrast Dynamics
    </h4>

    <ul style='list-style-type:" "; margin-left:0.5rem; line-height:1.6;'>
    <li>Most classes follow a <b>positive linear trend</b> — higher contrast yields better edge clarity.</li>
    <li><b>Rider</b> and <b>bike</b> form high-sharpness clusters — useful for motion and texture learning.</li>
    <li><b>Lane</b> and <b>drivable area</b> remain smooth — confirming low-frequency textures.</li>
    <li>Overall <b>photometric stability</b> ensures consistent feature learning.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # 🌟 Summary Block
    # ======================================================
    st.markdown("""
    <div style='text-align:center;
                font-size:0.95rem;
                font-family:Inter;
                color:#E5E7EB;
                margin-top:1.5rem;
                background:linear-gradient(90deg,rgba(99,102,241,0.15),rgba(139,92,246,0.15));
                border-radius:16px;
                padding:1.2rem 1.5rem;
                box-shadow:0 6px 20px rgba(0,0,0,0.45);
                animation:fadeUp 1.2s ease;'> 

    <b>Summary:</b><br>
    Consistent imaging metrics validate <b>photometric balance</b> across BDD100K classes.<br>
    This ensures <b>robust feature learning</b>, <b>low sensor bias</b>, and <b>stable model generalization</b>.<br>
    Ideal dataset for high-performance <b>vision pretraining</b> and deployment-ready models.
    </div>
    """, unsafe_allow_html=True)

    
   








# ==========================================================
# 3️⃣ Object Density
# ==========================================================
elif selected == "Object Density":
  
    # ============================================
    # 1️⃣ Load Data from API\
    # ============================================
    cache_path = os.path.join("precomputed_backend", "object_density.json")

    if os.path.exists(cache_path):
        print(" Loading precomputed data from cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(" Fetching data from API...")
        url = "http://bdd_backend:8000/api/analysis/object-density/"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(" Cached API response for future runs.")

    # ============================================


    train = data["train"]
    val = data["val"]

    # ============================================
    # 2️ Summary Overview Table
    # ============================================
    summary_df = pd.DataFrame({
        "Subset": ["Train", "Validation"],
        "Total Images": [train["total_images"], val["total_images"]],
        "Total Objects": [train["total_objects"], val["total_objects"]],
        "Mean Objects/Image": [train["objects_per_image_mean"], val["objects_per_image_mean"]],
        "Std Dev": [train["objects_per_image_std"], val["objects_per_image_std"]],
    })

    st.markdown("""
    <h2 style='text-align:center; font-family:Poppins;
    background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
     Object Density Analysis
    </h2>
    """, unsafe_allow_html=True)

    st.dataframe(
        summary_df.style.format(precision=3)
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#1e293b'), ('color', 'white'),
                                         ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('background-color', '#111827'), ('color', '#e5e7eb'),
                                         ('text-align', 'center')]}
        ])
    )

    st.markdown("""
    <div style='background:rgba(255,255,255,0.07); border-radius:12px;
    padding:1rem 1.5rem; margin-top:1rem; font-family:Inter; color:#e8e8e8;
    box-shadow:0 4px 16px rgba(0,0,0,0.4); animation:fadeInUp 1s ease-in-out;'>
    <b>Summary Insights:</b><br>
    - Both subsets contain <b>5000 images</b> — ensuring balanced sampling.<br>
    - Average objects per image ≈ <b>28</b>, confirming consistent scene complexity.<br>
    - Slightly higher standard deviation in <b>train set</b> (12.76) → slightly more varied density distribution.<br>
    </div>
    """, unsafe_allow_html=True)

  

    # ============================================================
    # 3️ Objects per Image Histogram — Enhanced Gradient Bars
    # ============================================================

    bins = np.arange(len(train["objects_per_image_histogram"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.5))

    # Gradient-style bars
    bar_colors_train = sns.color_palette("cool", len(bins))
    bar_colors_val = sns.color_palette("rocket", len(bins))

    ax.bar(bins, train["objects_per_image_histogram"], 
        alpha=0.9, label="Train", color=bar_colors_train)
    ax.bar(bins, val["objects_per_image_histogram"], 
        alpha=0.6, label="Validation", color=bar_colors_val)

    ax.set_title(" Object Count Distribution per Image", fontsize=12, weight="bold", color="#7c3aed", pad=6)
    ax.set_xlabel("Objects per Image (bin index)", fontsize=9)
    ax.set_ylabel("Number of Images", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # Subtle styling
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_facecolor(to_rgba("#f8fafc", 0.8))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Insight Card
    st.markdown("""
    <div style='background:rgba(255,255,255,0.07); border-radius:12px;
    padding:1rem 1.5rem; margin-top:0.8rem; font-family:Inter; color:#e8e8e8;
    box-shadow:0 4px 16px rgba(0,0,0,0.4); animation:fadeInUp 1s ease-in-out;'>
    <b> Histogram Insights:</b><br>
    • Both subsets exhibit a <b>bell-shaped density</b>, peaking near 25–30 objects per frame.<br>
    • The <b>validation split</b> shows slightly smoother uniformity, reflecting random sampling.<br>
    • Outliers with 80+ objects likely correspond to <b>dense urban traffic frames</b>.<br>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    #  Per-Class Object Density — Dual-Tone Horizontal Bars
    # ============================================================

    train_density = pd.Series(train["per_class_density"], name="Train")
    val_density = pd.Series(val["per_class_density"], name="Validation")
    combined_density = pd.concat([train_density, val_density], axis=1).fillna(0)
    combined_density = combined_density.sort_values(by="Train", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(combined_density.index, combined_density["Train"], color="#60a5fa", alpha=0.9, label="Train")
    ax.barh(combined_density.index, combined_density["Validation"], color="#f472b6", alpha=0.8, label="Validation")

    ax.set_title(" Per-Class Object Density (Avg Objects per Image)", fontsize=12, color="#6366f1", weight="bold", pad=6)
    ax.set_xlabel("Objects per Image", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # Background styling
    ax.set_facecolor(to_rgba("#f8fafc", 0.85))
    ax.grid(True, linestyle="--", alpha=0.3, axis='x')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div style='background:rgba(255,255,255,0.07); border-radius:12px;
    padding:1rem 1.5rem; margin-top:0.8rem; font-family:Inter; color:#e8e8e8;
    box-shadow:0 4px 16px rgba(0,0,0,0.4); animation:fadeInUp 1.1s ease-in-out;'>
    <b> Class Density Insights:</b><br>
    • <b>Cars</b> dominate with ~10 per frame, followed by <b>lanes</b> and <b>traffic signs</b>.<br>
    • <b>Rider</b> and <b>motorcycle</b> classes remain rare yet stable across subsets.<br>
    • Balanced density confirms <b>multi-object realism</b> typical in driving scenes.<br>
    </div>
    """, unsafe_allow_html=True)


    # ============================================================
    #  Summary
    # ============================================================
    st.markdown("""
    <div style='text-align:center; font-size:1rem; font-family:Inter;
    color:#e8e8e8; margin-top:1.2rem;
    background:linear-gradient(90deg,rgba(99,102,241,0.25),rgba(139,92,246,0.25));
    border-radius:18px; padding:1.2rem 1.8rem; box-shadow:0 6px 20px rgba(0,0,0,0.45);
    animation:fadeInUp 1.4s ease-in-out;'>

    <b> Summary:</b><br>
    The <b>BDD100K</b> dataset maintains excellent <b>object density equilibrium</b> across subsets.<br>
    Combined distributions confirm <b>realistic urban complexity</b> without overrepresentation of any class.<br>
    Such balance ensures high-quality supervision and stable convergence during model training.<br>
    </div>
    """, unsafe_allow_html=True)



# ==========================================================
# 5️⃣ Class Balance Insights
# ==========================================================
elif selected == "Class Balance Insights":

    import json, pandas as pd, streamlit as st
    from datetime import datetime

    # ==============================
    # Load Data (from JSON or API)
    # ==============================
    try:
        cache_path = os.path.join("/app/precomputed_backend", "class_balance_insights.json")
        print(" Loading class balance insights from cache..." , cache_path , "****((((((()))))))")
        with open(cache_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        url = "http://bdd_backend:8000/api/analysis/class-balance-insights/"
        import requests
        data = requests.get(url).json()

    summary = data["summary"]
    coverage = data["coverage_stats"]
    drift = data["drift_analysis"]
    sampling = data["sampling_suggestions"]
    cumulative = data["cumulative_distribution"]

    # ==============================
    # Section Header
    # ==============================
    st.markdown("""
    <h2 style='text-align:center; font-family:Poppins;
    background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:1rem;'>
     Class Balance Insights
    </h2>
    """, unsafe_allow_html=True)

    # ==============================
    # 📊 Summary Metrics (Cards)
    # ==============================
    st.markdown("""
    <div style='display:flex; justify-content:space-around; flex-wrap:wrap; gap:1rem;'>
        <div style='background:rgba(255,255,255,0.08); padding:1.2rem 1.8rem; border-radius:14px;
        flex:1; min-width:200px; text-align:center; color:#f3f4f6; box-shadow:0 4px 20px rgba(0,0,0,0.4);'>
            <h4 style='color:#93c5fd;'> Total Train Objects</h4>
            <h2 style='margin:0; color:#60a5fa;'>{:,}</h2>
        </div>
        <div style='background:rgba(255,255,255,0.08); padding:1.2rem 1.8rem; border-radius:14px;
        flex:1; min-width:200px; text-align:center; color:#f3f4f6; box-shadow:0 4px 20px rgba(0,0,0,0.4);'>
            <h4 style='color:#c084fc;'> Total Val Objects</h4>
            <h2 style='margin:0; color:#a78bfa;'>{:,}</h2>
        </div>
        <div style='background:rgba(255,255,255,0.08); padding:1.2rem 1.8rem; border-radius:14px;
        flex:1; min-width:200px; text-align:center; color:#f3f4f6; box-shadow:0 4px 20px rgba(0,0,0,0.4);'>
            <h4 style='color:#f9a8d4;'> Most Common Class</h4>
            <h2 style='margin:0; color:#f472b6;'>{}</h2>
            <p style='font-size:0.9rem; opacity:0.8;'>{:,} samples</p>
        </div>
        <div style='background:rgba(255,255,255,0.08); padding:1.2rem 1.8rem; border-radius:14px;
        flex:1; min-width:200px; text-align:center; color:#f3f4f6; box-shadow:0 4px 20px rgba(0,0,0,0.4);'>
            <h4 style='color:#67e8f9;'> Least Common Class</h4>
            <h2 style='margin:0; color:#22d3ee;'>{}</h2>
            <p style='font-size:0.9rem; opacity:0.8;'>{:,} samples</p>
        </div>
    </div>
    """.format(
        summary["total_train_objects"],
        summary["total_val_objects"],
        summary["most_common_class"].title(),
        summary["most_common_count"],
        summary["least_common_class"].title(),
        summary["least_common_count"],
    ), unsafe_allow_html=True)

    # ==============================
    #  Balance Summary Metrics
    # ==============================
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.08); border-radius:16px;
    padding:1.3rem 1.6rem; margin-top:1.2rem; font-family:Inter; color:#f3f4f6;
    box-shadow:0 6px 20px rgba(0,0,0,0.45);'>

    <b> Balance Summary:</b><br>
    • <b>Class Imbalance Ratio:</b> {summary["class_imbalance_ratio"]}  
    • <b>Gini Index:</b> {summary["gini_index"]:.3f} — measures inequality (0 = perfect balance).  
    • <b>Shannon Entropy:</b> {summary["shannon_entropy"]:.3f} — indicates diversity in distribution.  
    • <b>Effective Classes:</b> ≈ {summary["effective_classes"]} active classes contributing meaningfully.  
    • <b>Train–Val Drift:</b> {summary["train_val_drift"]:.3f} — minimal distributional drift (<3% ideal).  

    <br>These metrics show that while <b>class imbalance exists</b> (cars dominate),  
    the overall entropy and drift values suggest <b>good coverage stability</b> across training and validation sets.
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    #  Coverage Statistics
    # ==============================
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.06); border-radius:14px;
    padding:1.1rem 1.5rem; margin-top:1rem; font-family:Inter; color:#f3f4f6;
    box-shadow:0 4px 16px rgba(0,0,0,0.35);'>

    <b> Coverage Overview:</b><br>
    • Classes covering >1% of dataset: <b>{coverage["classes_over_1_percent"]}</b>  
    • Classes under 1% contribution: <b>{coverage["classes_under_1_percent"]}</b>  
    • Overall Coverage Ratio: <b>{coverage["coverage_ratio"]}</b>  

    <br>This confirms that roughly <b>58% of the dataset mass</b> is captured by just 7 dominant classes —  
    while the remaining long-tail ensures <b>contextual variety</b> for rare event learning.
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    #  Drift Analysis Summary
    # ==============================
    drift_df = pd.DataFrame(list(drift.items()), columns=["Class", "Drift Δ (Train-Val)"])
    drift_df["Drift Δ (Train-Val)"] = drift_df["Drift Δ (Train-Val)"].round(3)

    st.markdown("""
    <h4 style='margin-top:1.5rem; text-align:center; color:#a78bfa; font-family:Poppins;'>
     Distribution Drift Analysis
    </h4>
    """, unsafe_allow_html=True)

    st.dataframe(
        drift_df.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#4f46e5"), ("color", "white"), ("font-size", "0.9rem")]},
            {"selector": "tbody td", "props": [("background-color", "#111827"), ("color", "#e5e7eb"), ("text-align", "center")]}
        ]).format(precision=3),
        use_container_width=True, height=300
    )

    st.markdown("""
    <div style='background:rgba(255,255,255,0.06); border-radius:14px;
    padding:1rem 1.4rem; margin-top:0.8rem; font-family:Inter; color:#f3f4f6;
    box-shadow:0 4px 14px rgba(0,0,0,0.4);'>
    <b>Insight:</b>  
    The <b>lane</b> class shows the largest drift (≈11%),  
    while <b>traffic sign</b> and <b>person</b> slightly underrepresent in validation.  
    Overall drift (<4%) is minimal — confirming robust data split consistency.
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    #  Sampling Recommendations
    # ==============================
    sample_df = pd.DataFrame([
        {"Class": k.title(), "Action": v["sampling_action"], "Weight": v["weight"]}
        for k, v in sampling.items()
    ])

    st.markdown("""
    <h4 style='margin-top:1.6rem; text-align:center; color:#60a5fa; font-family:Poppins;'>
     Sampling Recommendations
    </h4>
    """, unsafe_allow_html=True)

    st.dataframe(
        sample_df.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#6366f1"), ("color", "white"), ("font-size", "0.9rem")]},
            {"selector": "tbody td", "props": [("background-color", "#1e293b"), ("color", "#e5e7eb"), ("text-align", "center")]}
        ]).format({"Weight": "{:.4f}"}),
        use_container_width=True, height=360
    )

    st.markdown("""
    <div style='background:rgba(255,255,255,0.07); border-radius:16px;
    padding:1.2rem 1.6rem; margin-top:1rem; font-family:Inter; color:#f3f4f6;
    box-shadow:0 6px 18px rgba(0,0,0,0.45); animation:fadeInUp 1.2s ease-in-out;
    backdrop-filter: blur(6px);'>

    <b> Sampling Strategy Insights:</b><br>
    • Majority classes like <b>car</b> and <b>lane</b> → <span style='color:#fca5a5;'>undersample</span> recommended.  
    • Minority classes (<b>train, motor, rider, bike</b>) → <span style='color:#86efac;'>oversample</span> for better recall.  
    • Balanced mid-tier classes (<b>sign, light, person</b>) need no adjustment.  
    • Weight coefficients suggest <b>adaptive loss scaling</b> can replace naive rebalancing.  
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    #  Summary
    # ==============================
    st.markdown(f"""
    <div style='text-align:center; font-size:1rem; font-family:Inter;
    color:#e8e8e8; margin-top:1.5rem;
    background:linear-gradient(90deg,rgba(99,102,241,0.25),rgba(139,92,246,0.25));
    border-radius:18px; padding:1.4rem 2rem; box-shadow:0 6px 20px rgba(0,0,0,0.45);
    animation:fadeInUp 1.3s ease-in-out;'>

    <b> Summary:</b><br>
    The dataset shows <b>moderate class imbalance</b> (Gini = {summary["gini_index"]:.3f})  
    dominated by <b>{summary["most_common_class"].title()}</b> instances.  
    Yet entropy (≈{summary["shannon_entropy"]:.2f}) and coverage ratio ({coverage["coverage_ratio"]})  
    indicate strong representation diversity and low overfit risk.  
    The train–val drift of <b>{summary["train_val_drift"]:.3f}</b> validates consistent sampling strategies —  
    making the dataset <b>highly reliable for deep model pretraining</b>.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 6️⃣ Sample Visualizations
# ==========================================================
elif selected == "Sample Visualizations":
    import os
    import random
    import streamlit as st
    from PIL import Image

    # =====================================================
    # 🌍 Path Handling — Cross-platform
    # =====================================================
    BASE_SAMPLE_DIR = os.path.join("samples")  # Works on Windows & Linux
    sample_categories = {
        "Basic Samples": "1_basic_samples",
        "Extreme Density": "2_extreme_density",
        "Bounding Box Size Extremes": "3_bbox_size_extremes",
        "Class Representatives": "4_class_representatives",
        "Diversity Samples": "5_diversity_samples",
        "Occlusion Samples": "6_occlusion_samples"
    }

    # =====================================================
    # ✨ Page Header with Gradient Animation
    # =====================================================
    st.markdown("""
    <style>
    @keyframes fadeUp {
        0% {opacity:0; transform:translateY(25px);}
        100% {opacity:1; transform:translateY(0);}
    }
    .sample-header {
        text-align:center;
        font-family:'Poppins',sans-serif;
        font-weight:600;
        font-size:1.8rem;
        background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        animation:fadeUp 1s ease;
        margin-bottom:0.4rem;
    }
    .sample-subtext {
        text-align:center;
        font-family:'Inter',sans-serif;
        color:#d1d5db;
        font-size:1rem;
        margin-bottom:1.5rem;
        animation:fadeUp 1.2s ease;
    }
    .glass-card {
        background:rgba(255,255,255,0.05);
        border-radius:18px;
        padding:1.2rem;
        margin-bottom:1.8rem;
        box-shadow:0 6px 25px rgba(0,0,0,0.55);
        backdrop-filter:blur(10px);
        border:1px solid rgba(167,139,250,0.25);
        animation:fadeUp 1s ease;
    }
    .sample-title {
        text-align:center;
        font-family:'Poppins',sans-serif;
        font-weight:600;
        color:#C4B5FD;
        font-size:1.2rem;
        margin-bottom:0.6rem;
    }
    </style>

    <div class='sample-header'> Sample Visualizations (BDD100K)</div>
    <div class='sample-subtext'>
        Explore representative samples across multiple visual categories — from basic frames to dense, occluded, and diverse scenarios.
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    #  Display 2 Images per Category
    # =====================================================
    for title, folder in sample_categories.items():
        folder_path = os.path.join(BASE_SAMPLE_DIR, folder)
        if not os.path.exists(folder_path):
            st.warning(f"⚠️ Folder not found: {folder_path}")
            continue

        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            st.info(f"No images found in {title}")
            continue

        # Pick up to 2 random samples (or all if fewer)
        selected_images = random.sample(image_files, min(2, len(image_files)))

        # Display the section with glass-style box
        st.markdown(f"<div class='glass-card'><div class='sample-title'> {title}</div>", unsafe_allow_html=True)

        cols = st.columns(len(selected_images))
        for idx, img_name in enumerate(selected_images):
            img_path = os.path.join(folder_path, img_name)
            try:
                image = Image.open(img_path)
                with cols[idx]:
                    st.image(image, use_container_width=True, caption=f"{img_name}", output_format="PNG")
            except Exception as e:
                st.error(f"Error loading {img_path}: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # 🌟 Summary Footer
    # =====================================================
    st.markdown("""
    <div style='text-align:center; font-family:Inter; font-size:0.95rem;
                color:rgba(235,235,235,0.85); margin-top:1.8rem;
                background:linear-gradient(90deg,rgba(99,102,241,0.2),rgba(139,92,246,0.2));
                border-radius:16px; padding:1.2rem 1.6rem;
                box-shadow:0 6px 20px rgba(0,0,0,0.45);
                animation:fadeUp 1.3s ease;'> 
         Displaying 2 representative samples per category.  
        Visual diversity in BDD100K supports robust training for all environmental and density conditions.
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<hr style='border:1px solid rgba(255,255,255,0.1); margin-top:2rem;'>
<p style='text-align:center; color:rgba(200,200,200,0.7); font-size:0.85rem;'>
  © 2025 Utkarsh Suryaman | BDD100K Visual Analytics Suite | Robert Bosch | Bosch Bangalore R&D Center
</p>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")
cache_target = st.sidebar.text_input("Enter cache name to clear:")
if st.sidebar.button(" Clear Cache"):
    clear_cache(cache_target)
