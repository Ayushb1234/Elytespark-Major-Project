import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.utils import list_image_files, load_bundle, sample_data_notice

st.title("Model Performance")
sample_data_notice()

try:
    bundle = load_bundle()
except Exception as exc:
    st.error("Unable to prepare the model.")
    st.exception(exc)
    st.stop()

model = bundle["model"]
metrics = bundle.get("metrics") or {}

st.subheader("Hold-out Metrics")

if metrics:
    cols = st.columns(len(metrics))
    for column, (name, value) in zip(cols, metrics.items()):
        column.metric(name.replace("_", " ").title(), f"{float(value):.4f}")
else:
    st.info(
        "Using saved model artifacts from `models/`. "
        "Hold-out metrics are only shown when the model is trained in-app."
    )

if bundle.get("trained_in_app"):
    st.caption(
        f"Trained in-app on {bundle['training_rows']:,} rows "
        "(Random Forest). Cached for this session."
    )

st.subheader("Feature Importance")

if hasattr(model, "feature_importances_"):
    feature_names = bundle.get("transformed_feature_names")
    if not feature_names or len(feature_names) != len(model.feature_importances_):
        feature_names = [
            f"feature_{i}" for i in range(len(model.feature_importances_))
        ]

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    st.dataframe(importance_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = importance_df.iloc[::-1]
    ax.barh(plot_df["Feature"], plot_df["Importance"])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
else:
    st.info("The loaded model does not expose feature_importances_.")

st.subheader("Saved Evaluation Figures")
figures_dir = PROJECT_ROOT / "outputs" / "figures"
images = list_image_files(figures_dir)

preferred = [
    "confusion_matrix",
    "roc_curve",
    "pr_curve",
    "learning_curve",
    "feature_importance",
]
ordered = []
for stem in preferred:
    ordered.extend([path for path in images if path.stem == stem])
ordered.extend([path for path in images if path not in ordered])

if not ordered:
    st.warning(
        "No saved figures found in outputs/figures. "
        "Generate them locally with the evaluation scripts, or rely on the "
        "in-app metrics above."
    )
else:
    for index in range(0, len(ordered), 2):
        cols = st.columns(2)
        for target, image_path in zip(cols, ordered[index:index + 2]):
            with target:
                st.markdown(f"**{image_path.stem.replace('_', ' ').title()}**")
                try:
                    st.image(image_path.read_bytes(), use_container_width=True)
                except TypeError:
                    st.image(image_path.read_bytes(), use_column_width=True)
