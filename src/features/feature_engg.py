import numpy as np
import pandas as pd


class HealthcareFeatureEngineer:

    def __init__(self):
        pass

    # --------------------------------------------
    # Pulse pressure
    # --------------------------------------------

    def add_pulse_pressure(self, df):

        if {
            "systolic_bp",
            "diastolic_bp"
        }.issubset(df.columns):

            df["pulse_pressure"] = (
                df["systolic_bp"]
                - df["diastolic_bp"]
            )

        return df


    # --------------------------------------------
    # Mean arterial pressure
    # --------------------------------------------

    def add_mean_arterial_pressure(self, df):

        if {
            "systolic_bp",
            "diastolic_bp"
        }.issubset(df.columns):

            df["mean_arterial_pressure"] = (

                df["diastolic_bp"]

                + (

                    df["systolic_bp"]
                    - df["diastolic_bp"]

                ) / 3

            )

        return df


    # --------------------------------------------
    # Cholesterol ratio
    # --------------------------------------------

    def add_cholesterol_ratio(self, df):

        if {
            "cholesterol",
            "hdl"
        }.issubset(df.columns):

            hdl = df["hdl"].replace(0, np.nan)

            df["cholesterol_hdl_ratio"] = (
                df["cholesterol"] / hdl
            )

        return df


    # --------------------------------------------
    # LDL / HDL ratio
    # --------------------------------------------

    def add_ldl_hdl_ratio(self, df):

        if {
            "ldl",
            "hdl"
        }.issubset(df.columns):

            hdl = df["hdl"].replace(0, np.nan)

            df["ldl_hdl_ratio"] = (
                df["ldl"] / hdl
            )

        return df


    # --------------------------------------------
    # Glucose / HbA1c interaction
    # --------------------------------------------

    def add_glucose_hba1c_interaction(self, df):

        hba1c_col = next(
            (
                column
                for column in ("HbA1c_level", "hba1c_level")
                if column in df.columns
            ),
            None,
        )

        if hba1c_col and "glucose" in df.columns:
            df["glucose_hba1c"] = df["glucose"] * df[hba1c_col]

        return df


    # --------------------------------------------
    # BMI age interaction
    # --------------------------------------------

    def add_bmi_age_interaction(self, df):

        if {
            "bmi",
            "age"
        }.issubset(df.columns):

            df["bmi_age"] = (
                df["bmi"] * df["age"]
            )

        return df


    # --------------------------------------------
    # Run feature engineering
    # --------------------------------------------

    def transform(self, df):

        df = df.copy()

        df = self.add_pulse_pressure(df)

        df = self.add_mean_arterial_pressure(df)

        df = self.add_cholesterol_ratio(df)

        df = self.add_ldl_hdl_ratio(df)

        df = self.add_glucose_hba1c_interaction(df)

        df = self.add_bmi_age_interaction(df)

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        return df