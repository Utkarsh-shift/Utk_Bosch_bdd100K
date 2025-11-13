import json
from streamlit_lottie import st_lottie
import streamlit as st

st.title("Test Lottie Animation")

with open("loading.json", "r", encoding="utf-8") as f:
    anim = json.load(f)

st_lottie(anim, speed=1, height=300, key="test")
