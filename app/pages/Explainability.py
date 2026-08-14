import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.utils import PROJECT_ROOT, load_bundle, sample_data_notice

st.title("Explainable AI")
sample_data_notice()


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
        return False

    try:
        st.image(path.read_bytes(), use_container_width=True)
    except TypeError:
        st.image(path.read_bytes(), use_column_width=True)
    return True


st.header("Model Feature Importance")

try:
    bundle = load_bundle()
    model = bundle["model"]

    if hasattr(model, "feature_importances_"):
        feature_names = bundle.get("transformed_feature_names")
        if not feature_names or len(feature_names) != len(
            model.feature_importances_
        ):
            feature_names = [
                f"feature_{i}"
                for i in range(len(model.feature_importances_))
            ]

        importance_df = (
            pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": model.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )

        st.dataframe(importance_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = importance_df.iloc[::-1]
        ax.barh(plot_df["Feature"], plot_df["Importance"])
        ax.set_title("Top Model Feature Importances")
        ax.set_xlabel("Importance")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        st.info("Loaded model does not expose feature importances.")
except Exception as exc:
    st.warning("Could not compute live feature importance.")
    st.caption(str(exc))

st.caption(
    "SHAP and LIME plots below are optional offline artifacts. "
    "They are not required for Streamlit Cloud deployment."
)

st.header("SHAP Summary")
show_output_image(
    PROJECT_ROOT / "outputs" / "explainability" / "shap_summary.png",
    "SHAP summary not found (optional offline artifact).",
)

st.header("SHAP Feature Importance")
show_output_image(
    PROJECT_ROOT / "outputs" / "explainability" / "shap_bar.png",
    "SHAP bar plot not found (optional offline artifact).",
)

st.header("Random Forest Feature Importance (saved)")
show_output_image(
    PROJECT_ROOT / "outputs" / "figures" / "feature_importance.png",
    "Feature importance plot not found (optional offline artifact).",
)

st.header("LIME Patient Explanation")
lime_file = PROJECT_ROOT / "outputs" / "explainability" / "lime_patient_explanation.html"

if lime_file.exists():
    st.success("LIME explanation generated successfully.")
    components.html(
        lime_file.read_text(encoding="utf-8"),
        height=750,
        scrolling=True,
    )
else:
    st.warning("LIME explanation not found (optional offline artifact).")
    st.caption(f"Expected file: {relative_output_path(lime_file)}")
