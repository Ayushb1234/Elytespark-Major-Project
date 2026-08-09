from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/clean_data.csv"

MODEL_PATH = "models/best_model.pkl"

PREPROCESSOR_PATH = "models/preprocessor.pkl"

OUTPUT_PATH = "outputs/figures/feature_importance.png"


def get_feature_names(preprocessor):

    feature_names = []

    # --------------------------------------------
    # Numerical features
    # --------------------------------------------

    numeric_features = preprocessor.transformers_[0][2]

    for feature in numeric_features:

        feature_names.append(
            f"num__{feature}"
        )

    # --------------------------------------------
    # Categorical features
    # --------------------------------------------

    categorical_pipeline = (
        preprocessor
        .transformers_[1][1]
    )

    categorical_features = (
        preprocessor
        .transformers_[1][2]
    )

    # Get fitted OneHotEncoder
    encoder = (
        categorical_pipeline
        .named_steps["encoder"]
    )

    # Build one-hot feature names manually
    for feature, categories in zip(
        categorical_features,
        encoder.categories_
    ):

        for category in categories:

            feature_names.append(
                f"cat__{feature}_{category}"
            )

    return feature_names


def main():

    print("Loading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model:",
        type(model).__name__
    )


    # --------------------------------------------
    # Load EXACT saved preprocessor
    # --------------------------------------------

    print("\nLoading saved preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        "Preprocessor loaded successfully."
    )


    # --------------------------------------------
    # Get feature names
    # --------------------------------------------

    print(
        "\nBuilding feature names..."
    )

    feature_names = get_feature_names(
        preprocessor
    )


    # --------------------------------------------
    # Get model importance
    # --------------------------------------------

    if not hasattr(
        model,
        "feature_importances_"
    ):

        raise ValueError(
            "This model does not support "
            "feature_importances_."
        )


    importance = (
        model.feature_importances_
    )


    print(
        "\nNumber of feature names:",
        len(feature_names)
    )

    print(
        "Number of importance values:",
        len(importance)
    )


    # --------------------------------------------
    # Safety check
    # --------------------------------------------

    if len(feature_names) != len(importance):

        raise ValueError(
            f"\nFeature mismatch!\n"
            f"Feature names: {len(feature_names)}\n"
            f"Importances: {len(importance)}"
        )


    # --------------------------------------------
    # Create dataframe
    # --------------------------------------------

    imp = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })


    # --------------------------------------------
    # Sort
    # --------------------------------------------

    imp.sort_values(
        "Importance",
        ascending=False,
        inplace=True
    )


    # --------------------------------------------
    # Print top 20
    # --------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "TOP 20 IMPORTANT FEATURES"
    )

    print(
        "=" * 65
    )

    print(
        imp.head(20).to_string(
            index=False
        )
    )


    # --------------------------------------------
    # Save CSV
    # --------------------------------------------

    csv_path = Path(
        "outputs/metrics/feature_importance.csv"
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    imp.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nFeature importance CSV saved:"
    )

    print(
        csv_path
    )


    # --------------------------------------------
    # Plot top 20
    # --------------------------------------------

    top = (
        imp
        .head(20)
        .sort_values(
            "Importance",
            ascending=True
        )
    )


    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.figure(
        figsize=(11, 8)
    )


    plt.barh(
        top["Feature"],
        top["Importance"]
    )


    plt.xlabel(
        "Feature Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "Top 20 Feature Importance - Random Forest"
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nFeature importance plot saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    main()