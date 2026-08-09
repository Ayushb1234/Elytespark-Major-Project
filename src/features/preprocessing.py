from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.features.feature_engg import HealthcareFeatureEngineer


DATA_PATH = r"D:\programming\programming\python\age&genderdetection\yolov9\Resume History\Elytespark Major Project\IBM-Healthcare-AI\data\processed\clean_data.csv"


class HealthcarePreprocessor:

    def __init__(
        self,
        target="label",
        test_size=0.20,
        random_state=42
    ):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
        self.label_encoder = LabelEncoder()

    # --------------------------------------------
    # Remove leakage columns
    # --------------------------------------------
    def remove_leakage(self, df):
        leakage_cols = [
            "composite_key",
            "sublabel",
            "disease_flags",
            "source_dataset"
        ]

        existing = [
            col for col in leakage_cols
            if col in df.columns
        ]

        return df.drop(columns=existing, errors="ignore")

    # --------------------------------------------
    # Feature engineering
    # --------------------------------------------
    def engineer_features(self, df):
        engineer = HealthcareFeatureEngineer()
        return engineer.transform(df)

    # --------------------------------------------
    # Split X / y
    # --------------------------------------------
    def split_features_target(self, df):
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not found.")

        X = df.drop(columns=[self.target])
        y = df[self.target].copy()

        # Encode target labels so XGBoost and other models work properly
        y = self.label_encoder.fit_transform(y.astype(str))

        print("\nTarget Encoding:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name} -> {i}")

        return X, y

    # --------------------------------------------
    # Identify feature types
    # --------------------------------------------
    def identify_features(self, X):
        self.numeric_features = (
            X.select_dtypes(include=["number", "bool"])
            .columns
            .tolist()
        )

        self.categorical_features = (
            X.select_dtypes(include=["object", "string", "category"])
            .columns
            .tolist()
        )

        print("\nNumerical Features:", len(self.numeric_features))
        print(self.numeric_features)

        print("\nCategorical Features:", len(self.categorical_features))
        print(self.categorical_features)

    # --------------------------------------------
    # Build preprocessing pipeline
    # --------------------------------------------
    def build_preprocessor(self):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features)
            ],
            remainder="drop"
        )

        return self.preprocessor

    # --------------------------------------------
    # Complete preparation
    # --------------------------------------------
    def prepare(self, df):
        df = self.remove_leakage(df)
        df = self.engineer_features(df)

        X, y = self.split_features_target(df)

        self.identify_features(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.build_preprocessor()

        # Fit only on training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_processed,
            X_test_processed
        )

    # --------------------------------------------
    # Save preprocessor + label encoder
    # --------------------------------------------
    def save_preprocessor(self, path="models/preprocessor.pkl"):
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.preprocessor, path)
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")

        print(f"Preprocessor saved: {path}")
        print("Label encoder saved: models/label_encoder.pkl")


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print(df.head())

    processor = HealthcarePreprocessor()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed
    ) = processor.prepare(df)

    print("\n========== SUCCESS ==========")
    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)
    print("Processed Train:", X_train_processed.shape)
    print("Processed Test:", X_test_processed.shape)

    processor.save_preprocessor()

    print("Preprocessor Saved!")


if __name__ == "__main__":
    main()