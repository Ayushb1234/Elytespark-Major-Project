from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

DATA_PATH = "data/processed/clean_data.csv"
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"


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
    p = Path(FEATURE_NAMES_PATH)
    if p.exists():
        return joblib.load(p)
    return None


def safe_metric(value):
    try:
        return round(float(value), 4)
    except Exception:
        return value


def list_image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])