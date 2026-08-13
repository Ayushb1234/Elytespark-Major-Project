import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.utils import list_image_files, load_data


st.title("Exploratory Data Analysis")

sns.set_theme(style="whitegrid")

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PLOT_SAMPLE_SIZE = 10000

NUMERIC_CANDIDATES = [
    ("age", ["age"]),
    ("bmi", ["bmi"]),
    ("hba1c_level", ["hba1c_level", "HbA1c_level"]),
    ("glucose", ["glucose"]),
    ("cholesterol", ["cholesterol"]),
    ("sleep_hours", ["sleep_hours"]),
    ("triglycerides", ["triglycerides"]),
    ("crp_level", ["crp_level"]),
    ("homocysteine_level", ["homocysteine_level"]),
    ("systolic_bp", ["systolic_bp"]),
    ("diastolic_bp", ["diastolic_bp"]),
    ("alcohol_intake", ["alcohol_intake"]),
    ("salt_intake", ["salt_intake"]),
    ("heart_rate", ["heart_rate"]),
    ("hdl", ["hdl"]),
    ("ldl", ["ldl"]),
]

RISK_CANDIDATES = [
    ("smoking", ["smoking"]),
    ("diabetes", ["diabetes"]),
    ("hypertension", ["hypertension"]),
    ("heart_disease", ["heart_disease"]),
    ("family_history", ["family_history"]),
    ("low_hdl_cholesterol", ["low_hdl_cholesterol"]),
    ("high_ldl_cholesterol", ["high_ldl_cholesterol"]),
    ("high_blood_pressure", ["high_blood_pressure"]),
]


def column_lookup(df):
    return {column.lower(): column for column in df.columns}


def find_column(df, candidates):
    lookup = column_lookup(df)
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        match = lookup.get(candidate.lower())
        if match:
            return match
    return None


def display_pyplot(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def display_image(path):
    try:
        st.image(str(path), use_container_width=True)
    except TypeError:
        st.image(str(path), use_column_width=True)


def clean_title(value):
    return value.replace("_", " ").title()


def plot_frame(df):
    if len(df) <= PLOT_SAMPLE_SIZE:
        return df
    return df.sample(PLOT_SAMPLE_SIZE, random_state=42)


def render_dataset_summary(df):
    st.subheader("Dataset Overview")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{df.shape[0]:,}")
    metric_cols[1].metric("Columns", f"{df.shape[1]:,}")
    metric_cols[2].metric("Duplicate Rows", f"{df.duplicated().sum():,}")
    metric_cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Column Quality")
    quality_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(dropna=True).values,
        }
    )
    st.dataframe(quality_df, use_container_width=True)

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        st.subheader("Numerical Summary")
        sample = plot_frame(numeric_df)
        if len(sample) < len(numeric_df):
            st.caption(
                f"Summary shown on a {len(sample):,}-row sample to keep the app responsive."
            )
        st.dataframe(sample.describe().T, use_container_width=True)


def render_target_distribution(df):
    label_col = find_column(df, ["label"])
    if not label_col:
        st.info("Target column 'label' was not found.")
        return

    st.subheader("Disease Label Distribution")

    counts = df[label_col].value_counts(dropna=False)
    target_df = pd.DataFrame(
        {
            "Label": counts.index.astype(str),
            "Count": counts.values,
            "Percentage": (counts.values / len(df) * 100).round(2),
        }
    )

    st.dataframe(target_df, use_container_width=True)

    sampled_df = plot_frame(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=sampled_df, x=label_col, ax=ax)
    ax.set_title("Distribution of Disease Labels")
    ax.set_xlabel("Disease Label")
    ax.set_ylabel("Number of Patients")
    ax.tick_params(axis="x", rotation=20)
    display_pyplot(fig)


def render_numeric_distributions(df):
    st.subheader("Numerical Distributions")
    st.caption(
        f"Charts use up to {PLOT_SAMPLE_SIZE:,} rows for faster Streamlit rendering. "
        "Tables use the full dataset."
    )
    sampled_df = plot_frame(df)

    available = [
        (display_name, find_column(df, candidates))
        for display_name, candidates in NUMERIC_CANDIDATES
    ]
    available = [(display_name, column) for display_name, column in available if column]

    if not available:
        st.info("No configured numerical columns were found.")
        return

    selected_names = st.multiselect(
        "Select numerical columns",
        options=[display_name for display_name, _ in available],
        default=[display_name for display_name, _ in available[:6]],
    )

    selected = [(display_name, column) for display_name, column in available if display_name in selected_names]

    for index in range(0, len(selected), 2):
        cols = st.columns(2)
        for target, (display_name, column) in zip(cols, selected[index:index + 2]):
            with target:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                values = pd.to_numeric(sampled_df[column], errors="coerce").dropna()
                sns.histplot(values, bins=30, kde=True, ax=ax)
                ax.set_title(f"{clean_title(display_name)} Distribution")
                ax.set_xlabel(clean_title(display_name))
                display_pyplot(fig)


def render_correlation_heatmap(df):
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        st.info("At least two numerical columns are required for a correlation heatmap.")
        return

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Healthcare Feature Correlation Matrix")
    display_pyplot(fig)

    with st.expander("Correlation Table"):
        st.dataframe(corr, use_container_width=True)


def render_disease_by_gender(df):
    gender_col = find_column(df, ["gender"])
    label_col = find_column(df, ["label"])

    if not gender_col or not label_col:
        st.info("Gender and label columns are required for disease-by-gender analysis.")
        return

    st.subheader("Disease Distribution by Gender")

    table = pd.crosstab(df[gender_col], df[label_col], normalize="index") * 100
    st.dataframe(table.round(2), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot(kind="bar", ax=ax)
    ax.set_title("Disease Distribution by Gender")
    ax.set_ylabel("Percentage of Patients")
    ax.set_xlabel("Gender")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Disease Label")
    display_pyplot(fig)


def as_binary_rate(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.mean() * 100

    normalized = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "true", "yes", "y", "high", "smoker", "current", "former"}
    negative_values = {"0", "false", "no", "n", "low", "never", "none"}
    mapped = normalized.map(
        lambda value: 1 if value in positive_values else 0 if value in negative_values else pd.NA
    )
    mapped = pd.to_numeric(mapped, errors="coerce")
    if mapped.notna().any():
        return mapped.mean() * 100

    return None


def render_risk_factor_prevalence(df):
    st.subheader("Risk Factor Prevalence")

    values = {}
    for display_name, candidates in RISK_CANDIDATES:
        column = find_column(df, candidates)
        if not column:
            continue

        rate = as_binary_rate(df[column])
        if rate is not None:
            values[display_name] = rate

    if not values:
        st.info("No configured risk factor columns were found.")
        return

    risk_df = pd.Series(values).sort_values(ascending=False)

    st.dataframe(
        risk_df.rename("Patients (%)").round(2).reset_index().rename(columns={"index": "Risk Factor"}),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    risk_df.plot(kind="bar", ax=ax)
    ax.set_title("Risk Factor Prevalence")
    ax.set_ylabel("Patients (%)")
    ax.set_xlabel("Risk Factor")
    ax.tick_params(axis="x", rotation=45)
    display_pyplot(fig)


def render_outlier_boxplots(df):
    st.subheader("Outlier Boxplots")
    sampled_df = plot_frame(df)

    available = [
        (display_name, find_column(df, candidates))
        for display_name, candidates in NUMERIC_CANDIDATES
    ]
    available = [(display_name, column) for display_name, column in available if column]

    if not available:
        st.info("No configured numerical columns were found.")
        return

    selected_names = st.multiselect(
        "Select boxplot columns",
        options=[display_name for display_name, _ in available],
        default=[display_name for display_name, _ in available[:6]],
    )
    selected = [(display_name, column) for display_name, column in available if display_name in selected_names]

    for index in range(0, len(selected), 2):
        cols = st.columns(2)
        for target, (display_name, column) in zip(cols, selected[index:index + 2]):
            with target:
                fig, ax = plt.subplots(figsize=(6, 3))
                values = pd.to_numeric(sampled_df[column], errors="coerce").dropna()
                sns.boxplot(x=values, ax=ax)
                ax.set_title(f"{clean_title(display_name)} Outlier Analysis")
                ax.set_xlabel(clean_title(display_name))
                display_pyplot(fig)


def render_saved_figures():
    st.subheader("Saved Output Figures")

    images = list_image_files(FIGURES_DIR)
    if not images:
        st.warning("No saved figures found in outputs/figures.")
        return

    st.caption(f"{len(images)} saved diagrams found in outputs/figures.")

    for index in range(0, len(images), 2):
        cols = st.columns(2)
        for target, image_path in zip(cols, images[index:index + 2]):
            with target:
                st.markdown(f"**{clean_title(image_path.stem)}**")
                display_image(image_path)


try:
    df = load_data()
except Exception as exc:
    st.error("Unable to load data/processed/clean_data.csv.")
    st.exception(exc)
    st.stop()


section = st.radio(
    "EDA Section",
    [
        "Summary",
        "Distributions",
        "Relationships",
        "Risk Factors",
        "Saved Diagrams",
    ],
    horizontal=True,
)

if section == "Summary":
    render_dataset_summary(df)
    render_target_distribution(df)
elif section == "Distributions":
    render_numeric_distributions(df)
    render_outlier_boxplots(df)
elif section == "Relationships":
    render_correlation_heatmap(df)
    render_disease_by_gender(df)
elif section == "Risk Factors":
    render_risk_factor_prevalence(df)
else:
    render_saved_figures()
