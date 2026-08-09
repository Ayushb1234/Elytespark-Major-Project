import streamlit as st
import pandas as pd
import numpy as np
import joblib

from app.components.utils import (
    load_data,
    load_model,
    load_preprocessor,
    load_label_encoder,
)

from src.features.preprocessing import HealthcarePreprocessor

st.title("🤖 Disease Prediction")

df = load_data()
model = load_model()
preprocessor = load_preprocessor()
label_encoder = load_label_encoder()

# Build raw input from the same dataset columns
processor = HealthcarePreprocessor()

# Use a sample row to create default values
sample_row = df.dropna(axis=0).iloc[0].to_dict()

# Remove leakage columns and label from user-visible features
leakage_cols = {
    "composite_key", "source_dataset", "sublabel", "disease_flags",
    "diabetes", "hypertension", "heart_disease", "label"
}

raw_feature_cols = [c for c in df.columns if c not in leakage_cols]

# Try to preserve same feature engineering pipeline
working_df = df.copy()
working_df = processor.remove_leakage(working_df)
working_df = processor.engineer_features(working_df)

# After feature engineering, some new columns exist
all_input_cols = [c for c in working_df.columns if c != "label"]

st.info("Fill the patient details below and press Predict.")

with st.form("prediction_form"):
    input_data = {}

    # numeric columns
    numeric_cols = [
        c for c in all_input_cols
        if pd.api.types.is_numeric_dtype(working_df[c]) and c != "label"
    ]

    categorical_cols = [
        c for c in all_input_cols
        if c not in numeric_cols and c != "label"
    ]

    st.subheader("Numeric fields")
    for col in numeric_cols:
        default_value = float(sample_row[col]) if col in sample_row and pd.notna(sample_row[col]) else 0.0
        input_data[col] = st.number_input(
            col,
            value=default_value,
            format="%.4f"
        )

    st.subheader("Categorical fields")
    for col in categorical_cols:
        options = sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else ["Unknown"]
        default = str(sample_row[col]) if col in sample_row else options[0]
        if default not in options:
            default = options[0]
        input_data[col] = st.selectbox(col, options, index=options.index(default))

    submit = st.form_submit_button("Predict")

if submit:
    try:
        patient_df = pd.DataFrame([input_data])

        # Same feature engineering as training
        patient_df = processor.engineer_features(patient_df)

        # Align columns with training data
        train_cols = [c for c in working_df.columns if c != "label"]
        for col in train_cols:
            if col not in patient_df.columns:
                patient_df[col] = 0

        patient_df = patient_df[train_cols]

        # Transform with saved preprocessor
        patient_processed = preprocessor.transform(patient_df)

        pred_encoded = model.predict(patient_processed)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(patient_processed)[0]
            confidence = float(np.max(prob))
        else:
            confidence = None

        st.success(f"Prediction: {pred_label}")

        if confidence is not None:
            st.write(f"Confidence: {confidence:.2%}")

        st.write("Encoded prediction:", int(pred_encoded))

    except Exception as e:
        st.error(f"Prediction failed: {e}")