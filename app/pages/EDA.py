import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import streamlit as st

from components.utils import list_image_files


st.title("Exploratory Data Analysis")


images = list_image_files(
    "outputs/figures"
)


if not images:

    st.warning(
        "No EDA figures found in outputs/figures."
    )

else:

    st.success(
        f"{len(images)} figures found."
    )

    for image_path in images:

        st.subheader(
            image_path.name
        )

        # Compatible with older Streamlit
        st.image(
            str(image_path),
            use_column_width=True
        )
