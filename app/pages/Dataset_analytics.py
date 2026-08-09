import streamlit as st
import pandas as pd
from app.components.utils import load_data

st.title("📊 Dataset Analytics")

df = load_data()

st.subheader("Dataset shape")
st.write(df.shape)

st.subheader("Columns")
st.write(df.columns.tolist())

st.subheader("Data types")
st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Dtype"}), use_container_width=True)

st.subheader("Missing values")
missing = df.isnull().sum().sort_values(ascending=False)
st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing Values"}), use_container_width=True)

st.subheader("Target distribution")
if "label" in df.columns:
    st.dataframe(df["label"].value_counts().reset_index().rename(columns={"index": "Label", "label": "Count"}), use_container_width=True)

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)