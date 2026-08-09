import pandas as pd


DATA_PATH = "data/processed/clean_data.csv"


def check_feature(feature, df):

    if feature not in df.columns:
        print(f"\n{feature}: NOT FOUND")
        return

    print("\n" + "=" * 70)

    print(f"{feature} vs label")

    print("=" * 70)

    table = pd.crosstab(
        df[feature],
        df["label"],
        normalize="index"
    ) * 100

    print(
        table.round(2)
    )


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "Dataset:",
        df.shape
    )

    print(
        "\nOverall Label Distribution:"
    )

    print(
        df["label"]
        .value_counts()
    )

    print(
        "\nPercentage:"
    )

    print(
        df["label"]
        .value_counts(
            normalize=True
        ).mul(100).round(2)
    )

    # -----------------------------------------
    # Important disease features
    # -----------------------------------------

    features = [
        "hypertension",
        "diabetes",
        "heart_disease",
        "high_blood_pressure",
        "smoking",
        "family_history",
        "low_hdl_cholesterol",
        "high_ldl_cholesterol"
    ]

    for feature in features:

        check_feature(
            feature,
            df
        )

    # -----------------------------------------
    # Sublabel
    # -----------------------------------------

    if "sublabel" in df.columns:

        print(
            "\n" + "=" * 70
        )

        print(
            "SUBLABEL DISTRIBUTION"
        )

        print(
            "=" * 70
        )

        print(
            df["sublabel"]
            .value_counts(
                dropna=False
            )
        )

    # -----------------------------------------
    # Disease flags
    # -----------------------------------------

    if "disease_flags" in df.columns:

        print(
            "\n" + "=" * 70
        )

        print(
            "DISEASE FLAGS DISTRIBUTION"
        )

        print(
            "=" * 70
        )

        print(
            df["disease_flags"]
            .value_counts(
                dropna=False
            )
            .head(20)
        )


if __name__ == "__main__":

    main()