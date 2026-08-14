import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import streamlit as st

from components.utils import (
    load_bundle,
    load_data,
    model_features,
    sample_data_notice,
)
from src.features.preprocessing import HealthcarePreprocessor

st.title("Disease Prediction")
sample_data_notice()

st.write(
    """
    Enter patient information below to generate a machine-learning
    prediction.
    """
)

try:
    df = load_data()
    bundle = load_bundle()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    label_encoder = bundle["label_encoder"]
except Exception as e:
    st.error(f"Unable to load model or data: {e}")
    st.stop()

if bundle.get("trained_in_app") and bundle.get("metrics"):
    metrics = bundle["metrics"]
    st.info(
        "Model trained in-app on "
        f"{bundle['training_rows']:,} rows "
        f"(hold-out accuracy {metrics['accuracy']:.3f}, "
        f"F1 {metrics['f1']:.3f}"
        + (
            f", ROC-AUC {metrics['roc_auc']:.3f}"
            if "roc_auc" in metrics
            else ""
        )
        + "). Place saved `.pkl` files under `models/` to skip training."
    )

processor = HealthcarePreprocessor()

try:
    working_df = processor.remove_leakage(df.copy())
    working_df = processor.engineer_features(working_df)
except Exception as e:
    st.error(f"Feature engineering failed: {e}")
    st.stop()

model_feature_list = model_features(bundle)
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]

ENGINEERED_FEATURES = {
    "age_normalized",
    "pulse_pressure",
    "mean_arterial_pressure",
    "cholesterol_hdl_ratio",
    "ldl_hdl_ratio",
    "bmi_age",
    "glucose_hba1c",
}

input_features = [
    col for col in model_feature_list if col not in ENGINEERED_FEATURES
]

sample_row = df.dropna(axis=0, how="all").iloc[0].to_dict()

numeric_input_features = []
categorical_input_features = []

for col in input_features:
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_input_features.append(col)
        else:
            categorical_input_features.append(col)

st.info(
    "Enter the patient information and click Predict. "
    "Engineered features are calculated automatically."
)

with st.form("prediction_form"):
    input_data = {}

    st.subheader("Patient Measurements")
    numeric_columns = st.columns(2)

    for index, col in enumerate(numeric_input_features):
        with numeric_columns[index % 2]:
            if col in sample_row and pd.notna(sample_row[col]):
                try:
                    default_value = float(sample_row[col])
                except Exception:
                    default_value = 0.0
            else:
                default_value = 0.0

            input_data[col] = st.number_input(
                col.replace("_", " ").title(),
                value=default_value,
                format="%.4f",
            )

    st.subheader("Patient Information")
    categorical_columns = st.columns(2)

    for index, col in enumerate(categorical_input_features):
        with categorical_columns[index % 2]:
            options = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )
            if not options:
                options = ["Unknown"]

            if col in sample_row and pd.notna(sample_row[col]):
                default_value = str(sample_row[col])
            else:
                default_value = options[0]

            if default_value not in options:
                default_value = options[0]

            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                options,
                index=options.index(default_value),
            )

    submit = st.form_submit_button("Predict Disease Risk")

if submit:
    try:
        st.divider()
        st.subheader("Processing Patient Data...")

        patient_df = pd.DataFrame([input_data])

        for col in input_features:
            if col not in patient_df.columns:
                if col in df.columns:
                    patient_df[col] = (
                        df[col].median()
                        if pd.api.types.is_numeric_dtype(df[col])
                        else df[col].mode().iloc[0]
                    )
                else:
                    patient_df[col] = 0

        patient_engineered = processor.engineer_features(patient_df.copy())

        for col in model_feature_list:
            if col not in patient_engineered.columns:
                patient_engineered[col] = 0

        patient_engineered = patient_engineered[model_feature_list]
        patient_processed = preprocessor.transform(patient_engineered)
        prediction = model.predict(patient_processed)[0]

        try:
            predicted_label = label_encoder.inverse_transform([prediction])[0]
        except Exception:
            predicted_label = str(prediction)

        confidence = None
        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(patient_processed)[0]
            confidence = float(np.max(probabilities))

        st.subheader("Prediction Result")

        if str(predicted_label).lower() == "abnormal":
            st.error(f"Prediction: {predicted_label}")
        else:
            st.success(f"Prediction: {predicted_label}")

        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence:.2%}")
            st.progress(min(max(confidence, 0.0), 1.0))

        if probabilities is not None:
            st.subheader("Class Probabilities")
            probability_data = [
                {"Class": class_name, "Probability": f"{probability:.2%}"}
                for class_name, probability in zip(
                    label_encoder.classes_, probabilities
                )
            ]
            st.dataframe(
                pd.DataFrame(probability_data),
                use_container_width=True,
            )

        with st.expander("View Patient Input"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Feature": list(input_data.keys()),
                        "Value": list(input_data.values()),
                    }
                ),
                use_container_width=True,
            )

        with st.expander("View Automatically Generated Features"):
            engineered_display = {
                feature: patient_engineered[feature].iloc[0]
                for feature in ENGINEERED_FEATURES
                if feature in patient_engineered.columns
            }
            if engineered_display:
                st.dataframe(
                    pd.DataFrame([engineered_display]).T.rename(
                        columns={0: "Value"}
                    )
                )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
