import streamlit as st
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.utils import PROJECT_ROOT


st.title("Explainable AI")


# ============================================================
# SHAP SUMMARY
# ============================================================

st.header("SHAP Summary")

shap_summary = PROJECT_ROOT / "outputs" / "explainability" / "shap_summary.png"

if shap_summary.exists():

        st.image(
            str(shap_summary),
            use_column_width=True
        )

else:

    st.warning(
        "SHAP summary not found."
    )


# ============================================================
# SHAP BAR
# ============================================================

st.header("SHAP Feature Importance")

shap_bar = PROJECT_ROOT / "outputs" / "explainability" / "shap_bar.png"

if shap_bar.exists():

        st.image(
            str(shap_bar),
            use_column_width=True
        )

else:

    st.warning(
        "SHAP bar plot not found."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("Random Forest Feature Importance")

feature_importance = PROJECT_ROOT / "outputs" / "figures" / "feature_importance.png"

if feature_importance.exists():

        st.image(
            str(feature_importance),
            use_column_width=True
        )

else:

    st.warning(
        "Feature importance plot not found."
    )


# ============================================================
# LIME
# ============================================================

st.header("LIME Patient Explanation")

lime_file = PROJECT_ROOT / "outputs" / "explainability" / "lime_patient_explanation.html"

if lime_file.exists():

    st.success(
        "LIME explanation generated successfully."
    )

    st.write(
        "LIME HTML file:"
    )

    st.code(
        str(lime_file)
    )

else:

    st.warning(
        "LIME explanation not found."
    )
