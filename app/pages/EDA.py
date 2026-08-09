import streamlit as st
from pathlib import Path
from app.components.utils import list_image_files

st.title("📈 Exploratory Data Analysis")

folder = "outputs/figures"
images = list_image_files(folder)

if not images:
    st.warning("No figures found in outputs/figures yet.")
else:
    st.caption("Generated charts from your EDA and evaluation modules.")
    for img_path in images:
        st.subheader(img_path.name)
        st.image(str(img_path), use_container_width=True)