import sys
from pathlib import Path

# Add app directory to Python path
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import streamlit as st

from components.utils import load_data, safe_metric


st.title("Home")

df = load_data()


# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{df.shape[0]:,}"
)

col2.metric(
    "Features",
    f"{df.shape[1]:,}"
)

col3.metric(
    "Missing Values",
    f"{df.isnull().sum().sum():,}"
)

col4.metric(
    "Duplicate Rows",
    f"{df.duplicated().sum():,}"
)


st.divider()


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.header(
    "IBM Smart Healthcare Disease Prediction & Analytics Platform"
)

st.write(
    """
    This platform uses machine learning to analyze healthcare data,
    predict disease risk, evaluate model performance, and provide
    explainable AI insights.
    """
)


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)
