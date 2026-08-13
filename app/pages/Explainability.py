import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.utils import PROJECT_ROOT


st.title("Explainable AI")


def relative_output_path(path):
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def show_output_image(path, missing_message):
    path = Path(path)

    if not path.exists():
        st.warning(missing_message)
        st.caption(f"Expected file: {relative_output_path(path)}")
        return

    try:
        st.image(path.read_bytes(), use_container_width=True)
    except TypeError:
        st.image(path.read_bytes(), use_column_width=True)


# ============================================================
# SHAP SUMMARY
# ============================================================

st.header("SHAP Summary")

shap_summary = PROJECT_ROOT / "outputs" / "explainability" / "shap_summary.png"

show_output_image(shap_summary, "SHAP summary not found.")


# ============================================================
# SHAP BAR
# ============================================================

st.header("SHAP Feature Importance")

shap_bar = PROJECT_ROOT / "outputs" / "explainability" / "shap_bar.png"

show_output_image(shap_bar, "SHAP bar plot not found.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("Random Forest Feature Importance")

feature_importance = PROJECT_ROOT / "outputs" / "figures" / "feature_importance.png"

show_output_image(feature_importance, "Feature importance plot not found.")


# ============================================================
# LIME
# ============================================================

st.header("LIME Patient Explanation")

lime_file = PROJECT_ROOT / "outputs" / "explainability" / "lime_patient_explanation.html"

if lime_file.exists():

    st.success(
        "LIME explanation generated successfully."
    )

    components.html(
        lime_file.read_text(encoding="utf-8"),
        height=750,
        scrolling=True,
    )

else:

    st.warning(
        "LIME explanation not found."
    )
    st.caption(f"Expected file: {relative_output_path(lime_file)}")
