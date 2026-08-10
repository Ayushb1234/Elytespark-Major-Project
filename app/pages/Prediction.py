import sys
from pathlib import Path

# ============================================================
# FIX IMPORT PATH
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from components.utils import (
    load_data,
    load_model,
    load_preprocessor,
    load_label_encoder,
)

from src.features.preprocessing import HealthcarePreprocessor


st.title("Disease Prediction")

st.write(
    """
    Enter patient information below to generate a machine-learning
    prediction.
    """
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================

try:

    df = load_data()

    model = load_model()

    preprocessor = load_preprocessor()

    label_encoder = load_label_encoder()

except Exception as e:

    st.error(
        f"Unable to load model or data: {e}"
    )

    st.stop()


# ============================================================
# INITIALIZE PROCESSOR
# ============================================================

processor = HealthcarePreprocessor()


# ============================================================
# PREPARE DATA STRUCTURE
# ============================================================

try:

    # Remove columns that should never be supplied by user
    # or used as prediction features.

    working_df = processor.remove_leakage(
        df.copy()
    )

    # Apply exactly the same feature engineering used
    # during model training.

    working_df = processor.engineer_features(
        working_df
    )

except Exception as e:

    st.error(
        f"Feature engineering failed: {e}"
    )

    st.stop()


# ============================================================
# GET FEATURES EXPECTED BY SAVED PREPROCESSOR
# ============================================================

try:

    numeric_features = (
        preprocessor
        .transformers_[0][2]
    )

    categorical_features = (
        preprocessor
        .transformers_[1][2]
    )

    model_features = (
        list(numeric_features)
        + list(categorical_features)
    )

except Exception as e:

    st.error(
        f"Could not read features from preprocessor: {e}"
    )

    st.stop()


# ============================================================
# ENGINEERED FEATURES
# ============================================================

# These are calculated automatically.
# The user should NOT enter them manually.

ENGINEERED_FEATURES = {
    "age_normalized",
    "pulse_pressure",
    "mean_arterial_pressure",
    "cholesterol_hdl_ratio",
    "ldl_hdl_ratio",
    "bmi_age",
}


# ============================================================
# RAW INPUT FEATURES
# ============================================================

# Features that should actually be entered by the user.

input_features = [
    col
    for col in model_features
    if col not in ENGINEERED_FEATURES
]


# ============================================================
# SAMPLE DEFAULT VALUES
# ============================================================

sample_row = (
    df.dropna(
        axis=0,
        how="all"
    )
    .iloc[0]
    .to_dict()
)


# ============================================================
# SPLIT INPUT TYPES
# ============================================================

numeric_input_features = []

categorical_input_features = []


for col in input_features:

    if col in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            numeric_input_features.append(
                col
            )

        else:

            categorical_input_features.append(
                col
            )


# ============================================================
# DISPLAY INFORMATION
# ============================================================

st.info(
    "Enter the patient information and click Predict. "
    "Engineered features are calculated automatically."
)


# ============================================================
# PREDICTION FORM
# ============================================================

with st.form(
    "prediction_form"
):

    input_data = {}


    # ========================================================
    # NUMERIC INPUTS
    # ========================================================

    st.subheader(
        "Patient Measurements"
    )

    numeric_columns = st.columns(2)

    for index, col in enumerate(
        numeric_input_features
    ):

        with numeric_columns[index % 2]:

            if (
                col in sample_row
                and pd.notna(
                    sample_row[col]
                )
            ):

                try:

                    default_value = float(
                        sample_row[col]
                    )

                except Exception:

                    default_value = 0.0

            else:

                default_value = 0.0


            input_data[col] = st.number_input(

                col.replace(
                    "_",
                    " "
                ).title(),

                value=default_value,

                format="%.4f"
            )


    # ========================================================
    # CATEGORICAL INPUTS
    # ========================================================

    st.subheader(
        "Patient Information"
    )

    categorical_columns = st.columns(2)


    for index, col in enumerate(
        categorical_input_features
    ):

        with categorical_columns[index % 2]:

            # Get possible values from original dataset

            options = sorted(
                df[col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            if not options:

                options = [
                    "Unknown"
                ]


            # Default value

            if (
                col in sample_row
                and pd.notna(
                    sample_row[col]
                )
            ):

                default_value = str(
                    sample_row[col]
                )

            else:

                default_value = options[0]


            if default_value not in options:

                default_value = options[0]


            input_data[col] = st.selectbox(

                col.replace(
                    "_",
                    " "
                ).title(),

                options,

                index=options.index(
                    default_value
                )
            )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    submit = st.form_submit_button(
        "Predict Disease Risk"
    )


# ============================================================
# PREDICTION
# ============================================================

if submit:

    try:

        st.divider()

        st.subheader(
            "Processing Patient Data..."
        )


        # ====================================================
        # CREATE RAW PATIENT DATAFRAME
        # ====================================================

        patient_df = pd.DataFrame(
            [input_data]
        )


        # ====================================================
        # MAKE SURE RAW COLUMNS EXIST
        # ====================================================

        for col in input_features:

            if col not in patient_df.columns:

                if col in df.columns:

                    patient_df[col] = (
                        df[col]
                        .median()
                        if pd.api.types.is_numeric_dtype(
                            df[col]
                        )
                        else df[col]
                        .mode()
                        .iloc[0]
                    )

                else:

                    patient_df[col] = 0


        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        patient_engineered = (
            processor
            .engineer_features(
                patient_df.copy()
            )
        )


        # ====================================================
        # ENSURE ALL MODEL FEATURES EXIST
        # ====================================================

        for col in model_features:

            if col not in patient_engineered.columns:

                patient_engineered[col] = 0


        # ====================================================
        # SELECT EXACT MODEL FEATURES
        # ====================================================

        patient_engineered = (
            patient_engineered[
                model_features
            ]
        )


        # ====================================================
        # TRANSFORM USING SAVED PREPROCESSOR
        # ====================================================

        patient_processed = (
            preprocessor
            .transform(
                patient_engineered
            )
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            patient_processed
        )[0]


        # ====================================================
        # DECODE LABEL
        # ====================================================

        try:

            predicted_label = (
                label_encoder
                .inverse_transform(
                    [prediction]
                )[0]
            )

        except Exception:

            predicted_label = str(
                prediction
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        confidence = None

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model
                .predict_proba(
                    patient_processed
                )[0]
            )

            confidence = float(
                np.max(
                    probabilities
                )
            )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader(
            "Prediction Result"
        )


        if str(
            predicted_label
        ).lower() == "abnormal":

            st.error(
                f"Prediction: {predicted_label}"
            )

        else:

            st.success(
                f"Prediction: {predicted_label}"
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        if confidence is not None:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

            st.progress(
                min(
                    max(
                        confidence,
                        0.0
                    ),
                    1.0
                )
            )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        if probabilities is not None:

            st.subheader(
                "Class Probabilities"
            )


            probability_data = []

            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities
            ):

                probability_data.append({

                    "Class":
                        class_name,

                    "Probability":
                        f"{probability:.2%}"

                })


            probability_df = pd.DataFrame(
                probability_data
            )


            st.dataframe(
                probability_df,
                use_container_width=True
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander(
            "View Patient Input"
        ):

            input_display = pd.DataFrame({

                "Feature": list(
                    input_data.keys()
                ),

                "Value": list(
                    input_data.values()
                )

            })


            st.dataframe(
                input_display,
                use_container_width=True
            )


        # ====================================================
        # ENGINEERED FEATURES
        # ====================================================

        with st.expander(
            "View Automatically Generated Features"
        ):

            engineered_display = {}

            for feature in ENGINEERED_FEATURES:

                if feature in patient_engineered.columns:

                    engineered_display[
                        feature
                    ] = patient_engineered[
                        feature
                    ].iloc[0]


            if engineered_display:

                st.dataframe(
                    pd.DataFrame(
                        [
                            engineered_display
                        ]
                    ).T.rename(
                        columns={
                            0: "Value"
                        }
                    )
                )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)
