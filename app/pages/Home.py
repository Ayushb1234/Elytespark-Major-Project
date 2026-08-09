import streamlit as st
from app.components.utils import load_data, safe_metric

st.title("🏠 Home")

df = load_data()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", safe_metric(df.shape[0]))
c2.metric("Columns", safe_metric(df.shape[1]))
c3.metric("Missing Values", safe_metric(df.isnull().sum().sum()))
c4.metric("Duplicate Rows", safe_metric(df.duplicated().sum()))

st.divider()

st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Project summary")
st.write(
    """
    This platform predicts disease risk from healthcare data and provides analytics,
    explainability, and business insights for clinical-style decision support.
    """
)