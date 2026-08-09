from pathlib import Path
import pandas as pd


class ExplanationReport:
    def __init__(self, output_dir="outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_top_features(self, feature_importance_df, filename="explainability_report.txt", top_n=15):
        path = self.output_dir / filename

        top_df = feature_importance_df.head(top_n).copy()

        lines = []
        lines.append("Explainability Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Top {top_n} Features Influencing Disease Prediction")
        lines.append("")

        for idx, row in top_df.iterrows():
            lines.append(f"{idx + 1}. {row['Feature']}  ->  Importance: {row['Importance']:.6f}")

        lines.append("")
        lines.append("Interpretation:")
        lines.append("- Higher importance means the feature has stronger influence on predictions.")
        lines.append("- Use this report with SHAP/LIME plots for clinical interpretation.")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path