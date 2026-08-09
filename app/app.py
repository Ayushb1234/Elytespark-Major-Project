import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="IBM Smart Healthcare Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom style
style_path = Path("app/styles/style.css")
if style_path.exists():
    with open(style_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.title("IBM Healthcare AI")
st.sidebar.caption("Disease Prediction & Analytics")

st.title("🏥 IBM Smart Healthcare Analytics Platform")
st.markdown(
    """
    Welcome to the AI-powered healthcare analytics dashboard.

    Use the left sidebar to explore:
    - Dataset analytics
    - EDA charts
    - Disease prediction
    - Model performance
    - Explainability
    - Business insights
    """
)

st.info("Start from the Home page and move through each module from the sidebar.")