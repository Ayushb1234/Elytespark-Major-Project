import sys
from pathlib import Path
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import streamlit as st
from components.utils import load_data, sample_data_notice


st.title("Dataset Analytics")

try:
    df = load_data()
except Exception as exc:
    st.error("Unable to load the healthcare dataset.")
    st.exception(exc)
    st.stop()

sample_data_notice()


# ============================================================
# DATASET SIZE
# ============================================================

st.subheader("Dataset Shape")

col1, col2 = st.columns(2)

col1.metric(
    "Rows",
    f"{df.shape[0]:,}"
)

col2.metric(
    "Columns",
    f"{df.shape[1]:,}"
)


# ============================================================
# COLUMNS
# ============================================================

st.subheader("Dataset Columns")

st.write(
    df.columns.tolist()
)


# ============================================================
# DATA TYPES
# ============================================================

st.subheader("Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(
    dtype_df,
    use_container_width=True
)


# ============================================================
# MISSING VALUES
# ============================================================

st.subheader("Missing Values")

missing_df = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

missing_df.columns = [
    "Column",
    "Missing Values"
]

st.dataframe(
    missing_df,
    use_container_width=True
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

if "label" in df.columns:

    st.subheader("Disease Prediction Target")

    target_counts = (
        df["label"]
        .value_counts()
        .reset_index()
    )

    target_counts.columns = [
        "Label",
        "Count"
    ]

    st.dataframe(
        target_counts,
        use_container_width=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("Data Preview")

st.dataframe(
    df.head(50),
    use_container_width=True
)
