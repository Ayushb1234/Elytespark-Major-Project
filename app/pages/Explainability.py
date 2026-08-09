import streamlit as st
from pathlib import Path

st.title("🧠 Explainable AI")

st.subheader("SHAP summary")
if Path("outputs/explainability/shap_summary.png").exists():
    st.image("outputs/explainability/shap_summary.png", use_container_width=True)
else:
    st.warning("SHAP summary not found.")

st.subheader("SHAP bar plot")
if Path("outputs/explainability/shap_bar.png").exists():
    st.image("outputs/explainability/shap_bar.png", use_container_width=True)
else:
    st.warning("SHAP bar plot not found.")

st.subheader("LIME explanation")
lime_path = Path("outputs/explainability/lime_patient_explanation.html")
if lime_path.exists():
    st.markdown("Open the HTML file from the outputs/explainability folder.")
else:
    st.warning("LIME HTML explanation not found.")