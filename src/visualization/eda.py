from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class HealthcareEDA:

    def __init__(self, df: pd.DataFrame, output_dir="outputs/figures"):

        self.df = df.copy()

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # -------------------------------------------
    # Save plot
    # -------------------------------------------

    def save_plot(self, name):

        path = self.output_dir / name

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        plt.close()

    # -------------------------------------------
    # Dataset summary
    # -------------------------------------------

    def dataset_summary(self):

        print("=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)

        print("Shape:", self.df.shape)

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nMissing Values:")
        print(
            self.df
            .isnull()
            .sum()
            .sort_values(ascending=False)
        )

        print("\nDuplicate Rows:")
        print(self.df.duplicated().sum())

    # -------------------------------------------
    # Target distribution
    # -------------------------------------------

    def target_distribution(self):

        if "label" not in self.df.columns:
            return

        print(
            self.df["label"]
            .value_counts(dropna=False)
        )

        print(
            self.df["label"]
            .value_counts(
                normalize=True,
                dropna=False
            ) * 100
        )

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=self.df,
            x="label"
        )

        plt.title(
            "Distribution of Disease Labels"
        )

        plt.xlabel("Disease Label")

        plt.ylabel("Number of Patients")

        self.save_plot(
            "target_distribution.png"
        )

    # -------------------------------------------
    # Age distribution
    # -------------------------------------------

    def age_distribution(self):

        if "age" not in self.df.columns:
            return

        plt.figure(figsize=(9, 5))

        sns.histplot(
            self.df["age"],
            bins=30,
            kde=True
        )

        plt.title(
            "Patient Age Distribution"
        )

        plt.xlabel("Age")

        self.save_plot(
            "age_distribution.png"
        )

    # -------------------------------------------
    # BMI distribution
    # -------------------------------------------

    def bmi_distribution(self):

        if "bmi" not in self.df.columns:
            return

        plt.figure(figsize=(9, 5))

        sns.histplot(
            self.df["bmi"],
            bins=30,
            kde=True
        )

        plt.title(
            "BMI Distribution"
        )

        plt.xlabel("BMI")

        self.save_plot(
            "bmi_distribution.png"
        )

    # -------------------------------------------
    # Glucose
    # -------------------------------------------

    def glucose_distribution(self):

        if "glucose" not in self.df.columns:
            return

        plt.figure(figsize=(9, 5))

        sns.histplot(
            self.df["glucose"],
            bins=30,
            kde=True
        )

        plt.title(
            "Blood Glucose Distribution"
        )

        plt.xlabel("Glucose")

        self.save_plot(
            "glucose_distribution.png"
        )

    # -------------------------------------------
    # Cholesterol
    # -------------------------------------------

    def cholesterol_distribution(self):

        if "cholesterol" not in self.df.columns:
            return

        plt.figure(figsize=(9, 5))

        sns.histplot(
            self.df["cholesterol"],
            bins=30,
            kde=True
        )

        plt.title(
            "Cholesterol Distribution"
        )

        plt.xlabel("Cholesterol")

        self.save_plot(
            "cholesterol_distribution.png"
        )

    # -------------------------------------------
    # Numerical distributions
    # -------------------------------------------

    def numerical_distributions(self):

        cols = [
            "age",
            "bmi",
            "hba1c_level",
            "glucose",
            "cholesterol",
            "sleep_hours",
            "triglycerides",
            "crp_level",
            "homocysteine_level",
            "systolic_bp",
            "diastolic_bp",
            "alcohol_intake",
            "salt_intake",
            "heart_rate",
            "hdl",
            "ldl"
        ]

        for col in cols:

            if col not in self.df.columns:
                continue

            plt.figure(figsize=(8, 5))

            sns.histplot(
                self.df[col].dropna(),
                kde=True,
                bins=30
            )

            plt.title(
                f"{col.replace('_', ' ').title()} Distribution"
            )

            self.save_plot(
                f"{col}_distribution.png"
            )

    # -------------------------------------------
    # Correlation heatmap
    # -------------------------------------------

    def correlation_heatmap(self):

        numeric_df = self.df.select_dtypes(
            include="number"
        )

        if numeric_df.empty:
            return

        corr = numeric_df.corr()

        plt.figure(
            figsize=(18, 14)
        )

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0
        )

        plt.title(
            "Healthcare Feature Correlation Matrix"
        )

        self.save_plot(
            "correlation_heatmap.png"
        )

    # -------------------------------------------
    # Disease by gender
    # -------------------------------------------

    def disease_by_gender(self):

        if not {
            "gender",
            "label"
        }.issubset(self.df.columns):

            return

        table = pd.crosstab(
            self.df["gender"],
            self.df["label"],
            normalize="index"
        ) * 100

        print(table)

        table.plot(
            kind="bar",
            figsize=(10, 6)
        )

        plt.title(
            "Disease Distribution by Gender"
        )

        plt.ylabel(
            "Percentage of Patients"
        )

        plt.xticks(
            rotation=0
        )

        self.save_plot(
            "disease_by_gender.png"
        )

    # -------------------------------------------
    # Risk factor prevalence
    # -------------------------------------------

    def risk_factor_prevalence(self):

        risk_cols = [
            "smoking",
            "diabetes",
            "hypertension",
            "heart_disease",
            "family_history",
            "low_hdl_cholesterol",
            "high_ldl_cholesterol",
            "high_blood_pressure"
        ]

        values = {}

        for col in risk_cols:

            if col not in self.df.columns:
                continue

            numeric = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

            if numeric.notna().any():

                values[col] = (
                    numeric.mean() * 100
                )

        if not values:
            return

        risk_df = (
            pd.Series(values)
            .sort_values(ascending=False)
        )

        print(risk_df)

        plt.figure(figsize=(11, 6))

        risk_df.plot(
            kind="bar"
        )

        plt.ylabel(
            "Patients (%)"
        )

        plt.title(
            "Risk Factor Prevalence"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        self.save_plot(
            "risk_factor_prevalence.png"
        )

    # -------------------------------------------
    # Boxplots for outliers
    # -------------------------------------------

    def outlier_analysis(self):

        cols = [
            "age",
            "bmi",
            "glucose",
            "cholesterol",
            "triglycerides",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "hdl",
            "ldl"
        ]

        for col in cols:

            if col not in self.df.columns:
                continue

            plt.figure(figsize=(8, 4))

            sns.boxplot(
                x=self.df[col]
            )

            plt.title(
                f"{col.replace('_', ' ').title()} Outlier Analysis"
            )

            self.save_plot(
                f"{col}_boxplot.png"
            )

    # -------------------------------------------
    # Full EDA
    # -------------------------------------------

    def run_all(self):

        self.dataset_summary()

        self.target_distribution()

        self.age_distribution()

        self.bmi_distribution()

        self.glucose_distribution()

        self.cholesterol_distribution()

        self.numerical_distributions()

        self.correlation_heatmap()

        self.disease_by_gender()

        self.risk_factor_prevalence()

        self.outlier_analysis()