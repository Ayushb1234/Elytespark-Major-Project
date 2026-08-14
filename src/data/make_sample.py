"""Build the small sample dataset that ships with the repository.

The full processed dataset (data/processed/clean_data.csv) is ~90 MB and is
git-ignored, so the deployed Streamlit app has no access to it. This script
draws a stratified sample small enough to commit, which the app falls back to
when the full file is absent.

Run from the project root after regenerating the processed data:

    python -m src.data.make_sample
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FULL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "healthcare_sample.csv"

SAMPLE_SIZE = 6000
RANDOM_STATE = 42

# Kept coarse enough that the committed CSV stays small without changing the
# distributions the app reports.
FLOAT_PRECISION = 3

STRATIFY_COLUMNS = ["label", "source_dataset"]

# composite_key is a concatenation of columns already present in the frame and
# is dropped as leakage before training, so it only inflates the committed file.
DROP_COLUMNS = ["composite_key"]


def stratified_sample(df, size, random_state=RANDOM_STATE):
    if len(df) <= size:
        return df.copy()

    groups = [column for column in STRATIFY_COLUMNS if column in df.columns]
    if not groups:
        return df.sample(size, random_state=random_state)

    fraction = size / len(df)

    sampled = (
        df.groupby(groups, group_keys=False, observed=True)
        .apply(
            lambda group: group.sample(
                max(1, round(len(group) * fraction)),
                random_state=random_state,
            )
        )
    )

    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def main():
    if not FULL_DATA_PATH.exists():
        raise SystemExit(f"Full dataset not found: {FULL_DATA_PATH}")

    df = pd.read_csv(FULL_DATA_PATH)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    sample = stratified_sample(df, SAMPLE_SIZE)
    sample = sample.drop(columns=DROP_COLUMNS, errors="ignore")

    float_columns = sample.select_dtypes(include="float").columns
    sample[float_columns] = sample[float_columns].round(FLOAT_PRECISION)

    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_DATA_PATH, index=False)

    size_kb = SAMPLE_DATA_PATH.stat().st_size / 1024
    print(f"Wrote {len(sample):,} rows to {SAMPLE_DATA_PATH} ({size_kb:,.0f} KB)")
    print("\nLabel distribution:")
    print(sample["label"].value_counts())


if __name__ == "__main__":
    main()
