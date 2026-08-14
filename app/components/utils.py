"""Shared data and model access for the Streamlit app.

Neither the processed dataset nor the trained .pkl artifacts are committed (they
are git-ignored and too large for a repository), so on Streamlit Community Cloud
the app has only the bundled sample CSV to work with. Everything the pages need
is therefore derived at runtime: the sample is loaded, the preprocessing
pipeline is fitted and the model is trained on first use, then cached for the
lifetime of the process.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for _path in (APP_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Full dataset when running locally, bundled sample when deployed.
FULL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "healthcare_sample.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Keeps in-app training inside Community Cloud's memory and CPU budget.
MAX_TRAINING_ROWS = 20000
RANDOM_STATE = 42


def data_source():
    """Return the dataset path in use and whether it is the bundled sample."""
    if FULL_DATA_PATH.exists():
        return FULL_DATA_PATH, False
    return SAMPLE_DATA_PATH, True


@st.cache_data(show_spinner="Loading healthcare dataset...")
def load_data():
    path, _ = data_source()

    if not path.exists():
        raise FileNotFoundError(
            f"No dataset available. Expected {FULL_DATA_PATH} or {SAMPLE_DATA_PATH}."
        )

    return pd.read_csv(path)


def using_sample_data():
    return data_source()[1]


def sample_data_notice():
    """Render a one-line caption explaining which dataset the page is showing."""
    path, is_sample = data_source()

    if is_sample:
        st.caption(
            "Running on the bundled sample dataset "
            f"(`{path.relative_to(PROJECT_ROOT).as_posix()}`). "
            "Place the full `data/processed/clean_data.csv` in the project to use it instead."
        )
    else:
        st.caption(f"Using the full dataset (`{path.relative_to(PROJECT_ROOT).as_posix()}`).")


@st.cache_resource(show_spinner="Preparing model (training on first run)...")
def load_bundle():
    """Fit the preprocessing pipeline and model, or load saved artifacts.

    Returns a dict with the model, fitted preprocessor, label encoder, the
    feature lists the preprocessor expects and hold-out metrics.
    """
    # Imported lazily so pages that only show tables do not pay for sklearn.
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    from src.features.preprocessing import HealthcarePreprocessor

    def transformed_feature_names(preprocessor):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            return None

    saved = [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]
    if all(path.exists() for path in saved):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return {
            "model": joblib.load(MODEL_PATH),
            "preprocessor": preprocessor,
            "label_encoder": joblib.load(LABEL_ENCODER_PATH),
            "numeric_features": list(preprocessor.transformers_[0][2]),
            "categorical_features": list(preprocessor.transformers_[1][2]),
            "transformed_feature_names": transformed_feature_names(preprocessor),
            "metrics": None,
            "trained_in_app": False,
            "training_rows": None,
        }

    df = load_data()
    if len(df) > MAX_TRAINING_ROWS:
        df = df.sample(MAX_TRAINING_ROWS, random_state=RANDOM_STATE)

    processor = HealthcarePreprocessor(random_state=RANDOM_STATE)
    (
        _X_train,
        _X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
    ) = processor.prepare(df)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average="weighted"),
    }

    if len(processor.label_encoder.classes_) == 2:
        probabilities = model.predict_proba(X_test_processed)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return {
        "model": model,
        "preprocessor": processor.preprocessor,
        "label_encoder": processor.label_encoder,
        "numeric_features": list(processor.numeric_features),
        "categorical_features": list(processor.categorical_features),
        "transformed_feature_names": transformed_feature_names(
            processor.preprocessor
        ),
        "metrics": metrics,
        "trained_in_app": True,
        "training_rows": len(df),
    }


def model_features(bundle):
    return bundle["numeric_features"] + bundle["categorical_features"]


def load_model():
    return load_bundle()["model"]


def load_preprocessor():
    return load_bundle()["preprocessor"]


def load_label_encoder():
    return load_bundle()["label_encoder"]


def safe_metric(value):
    try:
        return round(float(value), 4)
    except Exception:
        return value


def list_image_files(folder):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])
