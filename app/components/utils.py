from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"


@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_resource
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_PATH)


@st.cache_resource
def load_feature_names():
    if FEATURE_NAMES_PATH.exists():
        return joblib.load(FEATURE_NAMES_PATH)
    return None


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
