import streamlit as st
from streamlit_lottie import st_lottie
import json
import time
from pathlib import Path


def load_lottie_file(filepath: Path):
    """Load Lottie JSON animation safely."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load Lottie animation: {e}")
        return None


def show_lottie_loader(lottie_path: str = "loading.json"):
    """Display fullscreen, centered Lottie animation (no scroll, fully visible)."""

    # --- Remove Streamlit padding and add fullscreen loader ---
    st.markdown(
        """
        <style>
        /* Remove Streamlit's default spacing */
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
        }
        #MainMenu, header, footer {visibility: hidden;}

        /* Fullscreen centered loader */
        .loader-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: white;
            z-index: 9999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Load the animation ---
    base_dir = Path(__file__).resolve().parent.parent
    lottie_file = base_dir / lottie_path
    animation = load_lottie_file(lottie_file)

    # --- Display loader ---
    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<div class="loader-container">', unsafe_allow_html=True)
        if animation:
            st_lottie(animation, speed=1, height=400, key="loading")
        else:
            st.write("⏳ Loading BDD100K Dashboard...")
        st.markdown("<h4 style='margin-top:20px;'>🚀 Initializing Dashboard...</h4>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Wait to let user see animation ---
    time.sleep(3)

    # --- Clear loader ---
    placeholder.empty()
