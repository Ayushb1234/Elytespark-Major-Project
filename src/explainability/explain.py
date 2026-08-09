from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.features.preprocessing import HealthcarePreprocessor
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.explanation_report import ExplanationReport


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/clean_data.csv"

MODEL_PATH = "models/best_model.pkl"

PREPROCESSOR_PATH = "models/preprocessor.pkl"

LABEL_ENCODER_PATH = "models/label_encoder.pkl"

OUTPUT_DIR = "outputs/explainability"


# ============================================================
# EXPLANATION SAMPLE SIZE
# ============================================================

# Your dataset has 280,985 rows.
# We do NOT need to explain all ~56,000 test rows.

SHAP_SAMPLE_SIZE = 1000

LIME_BACKGROUND_SIZE = 1000


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor):

    """
    Rebuild transformed feature names manually.

    This works with older versions of scikit-learn where
    Pipeline.get_feature_names_out() may fail because of
    SimpleImputer.
    """

    feature_names = []

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    num_transformer = preprocessor.transformers_[0]

    num_features = num_transformer[2]

    for col in num_features:

        feature_names.append(
            f"num__{col}"
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    cat_transformer = preprocessor.transformers_[1]

    cat_features = cat_transformer[2]

    cat_pipeline = cat_transformer[1]

    encoder = (
        cat_pipeline
        .named_steps["encoder"]
    )

    for feature, categories in zip(
        cat_features,
        encoder.categories_
    ):

        for category in categories:

            feature_names.append(
                f"cat__{feature}_{category}"
            )

    return feature_names


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("MODULE 8 - EXPLAINABLE AI")

    print("=" * 70)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "Dataset shape:",
        df.shape
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model:",
        type(model).__name__
    )


    # ========================================================
    # LOAD SAVED PREPROCESSOR
    # ========================================================

    print("\nLoading saved preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        "Preprocessor loaded successfully."
    )


    # ========================================================
    # LOAD LABEL ENCODER
    # ========================================================

    print("\nLoading label encoder...")

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    print(
        "Label classes:",
        list(label_encoder.classes_)
    )


    # ========================================================
    # RECREATE SAME DATA PREPARATION
    # ========================================================

    print(
        "\nPreparing data using the same "
        "feature engineering pipeline..."
    )

    processor = HealthcarePreprocessor()


    # --------------------------------------------------------
    # Remove leakage
    # --------------------------------------------------------

    df_for_prep = (
        processor
        .remove_leakage(
            df.copy()
        )
    )


    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df_for_prep = (
        processor
        .engineer_features(
            df_for_prep
        )
    )


    # --------------------------------------------------------
    # Split X / y
    # --------------------------------------------------------

    X, y = (
        processor
        .split_features_target(
            df_for_prep
        )
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print(
        "\nCreating train/test split..."
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=processor.test_size,

        random_state=processor.random_state,

        stratify=y
    )


    print(
        "Train rows:",
        X_train.shape[0]
    )

    print(
        "Test rows:",
        X_test.shape[0]
    )


    # ========================================================
    # USE SAVED PREPROCESSOR
    # ========================================================

    print(
        "\nTransforming data using saved preprocessor..."
    )

    # IMPORTANT:
    # Do NOT use fit_transform here.
    #
    # The preprocessor was already fitted during training.
    #
    # We only transform new data.

    X_train_processed = (
        preprocessor
        .transform(X_train)
    )

    X_test_processed = (
        preprocessor
        .transform(X_test)
    )


    print(
        "Processed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed test shape:",
        X_test_processed.shape
    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = get_feature_names(
        preprocessor
    )


    print(
        "\nNumber of transformed features:",
        len(feature_names)
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if (
        X_train_processed.shape[1]
        != len(feature_names)
    ):

        raise ValueError(
            "Feature-name count does not match "
            "processed feature count."
        )


    if (
        X_train_processed.shape[1]
        != model.n_features_in_
    ):

        raise ValueError(
            "Model feature count does not match "
            "processed feature count.\n"
            f"Model expects: {model.n_features_in_}\n"
            f"Processed data has: {X_train_processed.shape[1]}"
        )


    # ========================================================
    # SAVE FEATURE NAMES
    # ========================================================

    Path(
        "models"
    ).mkdir(
        exist_ok=True
    )

    joblib.dump(
        feature_names,
        "models/feature_names.pkl"
    )

    print(
        "\nFeature names saved:"
        " models/feature_names.pkl"
    )


    # ========================================================
    # CREATE EXPLANATION SAMPLES
    # ========================================================

    print(
        "\nPreparing explanation samples..."
    )

    # --------------------------------------------------------
    # SHAP sample
    # --------------------------------------------------------

    shap_size = min(
        SHAP_SAMPLE_SIZE,
        X_test_processed.shape[0]
    )

    X_shap = (
        X_test_processed[
            :shap_size
        ]
    )


    # --------------------------------------------------------
    # LIME background sample
    # --------------------------------------------------------

    lime_size = min(
        LIME_BACKGROUND_SIZE,
        X_train_processed.shape[0]
    )

    X_lime = (
        X_train_processed[
            :lime_size
        ]
    )


    print(
        "SHAP samples:",
        shap_size
    )

    print(
        "LIME background samples:",
        lime_size
    )


    # ========================================================
    # SHAP
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING SHAP"
    )

    print(
        "=" * 70
    )

    shap_explainer = SHAPExplainer(

        model=model,

        feature_names=feature_names

    )


    explainer, shap_values = (
        shap_explainer.explain(

            X_train=X_lime,

            X_test=X_shap,

            output_dir=OUTPUT_DIR

        )
    )


    print(
        "\nSHAP completed successfully."
    )


    # ========================================================
    # LIME
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING LIME"
    )

    print(
        "=" * 70
    )


    # LIME works better with dense arrays.

    if hasattr(
        X_lime,
        "toarray"
    ):

        X_lime_dense = (
            X_lime
            .toarray()
        )

    else:

        X_lime_dense = np.asarray(
            X_lime
        )


    lime_explainer = LIMEExplainer(

        X_train=X_lime_dense,

        feature_names=feature_names,

        class_names=list(
            label_encoder.classes_
        )

    )


    # --------------------------------------------------------
    # Explain first test patient
    # --------------------------------------------------------

    sample = X_shap[0]


    if hasattr(
        sample,
        "toarray"
    ):

        sample = (
            sample
            .toarray()
            .ravel()
        )

    else:

        sample = np.asarray(
            sample
        ).ravel()


    lime_path = (
        Path(OUTPUT_DIR)
        / "lime_patient_explanation.html"
    )


    lime_explainer.explain_instance(

        model=model,

        sample=sample,

        output_path=lime_path

    )


    print(
        "\nLIME explanation saved:"
    )

    print(
        lime_path
    )


    # ========================================================
    # FEATURE IMPORTANCE REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING FEATURE IMPORTANCE REPORT"
    )

    print(
        "=" * 70
    )


    if hasattr(
        model,
        "feature_importances_"
    ):

        fi_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                model.feature_importances_

        })


        fi_df.sort_values(

            "Importance",

            ascending=False,

            inplace=True

        )


        fi_df.reset_index(

            drop=True,

            inplace=True

        )


        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        metrics_dir = Path(
            "outputs/metrics"
        )

        metrics_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        fi_path = (
            metrics_dir
            / "feature_importance.csv"
        )


        fi_df.to_csv(

            fi_path,

            index=False

        )


        print(
            "\nFeature importance CSV saved:"
        )

        print(
            fi_path
        )


        # ----------------------------------------------------
        # Text report
        # ----------------------------------------------------

        report = ExplanationReport(

            output_dir="outputs/reports"

        )


        report_path = (
            report.save_top_features(

                fi_df,

                filename=
                "explainability_report.txt",

                top_n=15

            )
        )


        print(
            "\nExplainability report saved:"
        )

        print(
            report_path
        )


    else:

        print(
            "Model does not support "
            "feature_importances_."
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MODULE 8 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated outputs:"
    )

    print(
        "1. outputs/explainability/shap_summary.png"
    )

    print(
        "2. outputs/explainability/shap_bar.png"
    )

    print(
        "3. outputs/explainability/lime_patient_explanation.html"
    )

    print(
        "4. outputs/reports/explainability_report.txt"
    )

    print(
        "5. outputs/metrics/feature_importance.csv"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()