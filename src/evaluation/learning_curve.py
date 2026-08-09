from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from src.features.preprocessing import HealthcarePreprocessor


# --------------------------------------------
# Paths
# --------------------------------------------

DATA_PATH = "data/processed/clean_data.csv"

MODEL_PATH = "models/best_model.pkl"

OUTPUT_PATH = "outputs/figures/learning_curve.png"


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        "Dataset shape:",
        df.shape
    )


    # --------------------------------------------
    # Load trained model
    # --------------------------------------------

    print("\nLoading best model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model:",
        type(model).__name__
    )


    # --------------------------------------------
    # Prepare data
    # --------------------------------------------

    print("\nPreparing data...")

    processor = HealthcarePreprocessor()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed
    ) = processor.prepare(df)


    print(
        "Training data:",
        X_train_processed.shape
    )


    # --------------------------------------------
    # Learning curve
    # --------------------------------------------

    print(
        "\nGenerating learning curve..."
    )

    train_sizes, train_scores, val_scores = learning_curve(

        model,

        X_train_processed,

        y_train,

        cv=3,

        train_sizes=[
            0.10,
            0.30,
            0.50,
            0.70,
            1.00
        ],

        scoring="f1_weighted",

        n_jobs=1,

        shuffle=True,

        random_state=42

    )


    # --------------------------------------------
    # Calculate mean/std
    # --------------------------------------------

    train_mean = train_scores.mean(
        axis=1
    )

    train_std = train_scores.std(
        axis=1
    )

    val_mean = val_scores.mean(
        axis=1
    )

    val_std = val_scores.std(
        axis=1
    )


    # --------------------------------------------
    # Print results
    # --------------------------------------------

    print(
        "\nLearning Curve Results"
    )

    print(
        "=" * 50
    )

    for size, train, val in zip(

        train_sizes,

        train_mean,

        val_mean

    ):

        print(
            f"Samples: {size:,} | "
            f"Train F1: {train:.4f} | "
            f"Validation F1: {val:.4f}"
        )


    # --------------------------------------------
    # Create output directory
    # --------------------------------------------

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------
    # Plot
    # --------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        train_sizes,

        train_mean,

        marker="o",

        label="Training F1"

    )


    plt.plot(

        train_sizes,

        val_mean,

        marker="o",

        label="Validation F1"

    )


    # --------------------------------------------
    # Standard deviation
    # --------------------------------------------

    plt.fill_between(

        train_sizes,

        train_mean - train_std,

        train_mean + train_std,

        alpha=0.15

    )


    plt.fill_between(

        train_sizes,

        val_mean - val_std,

        val_mean + val_std,

        alpha=0.15

    )


    plt.xlabel(
        "Number of Training Samples"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Random Forest Learning Curve"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()


    # --------------------------------------------
    # Save
    # --------------------------------------------

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


    print(
        "\nLearning curve saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    main()