from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)

    def explain(self, X_train, X_test, output_dir="outputs/explainability"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # SHAP TreeExplainer works best for tree-based models like RandomForest
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, shap_values may be:
        # - list of arrays [class0, class1]
        # - or single array depending on shap version/model
        if isinstance(shap_values, list):
            # Use positive class if available
            shap_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_to_plot = shap_values

        # Summary plot
        plt.figure()
        shap.summary_plot(
            shap_to_plot,
            X_test,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Bar plot
        plt.figure()
        shap.summary_plot(
            shap_to_plot,
            X_test,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

        return explainer, shap_values

    def explain_single(self, explainer, sample, output_dir="outputs/explainability", sample_name="patient"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shap_values_single = explainer.shap_values(sample)

        if isinstance(shap_values_single, list):
            shap_to_plot = shap_values_single[1] if len(shap_values_single) > 1 else shap_values_single[0]
        else:
            shap_to_plot = shap_values_single

        # Waterfall/force style explanation
        plt.figure()
        shap.initjs()
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            shap_to_plot[0] if shap_to_plot.ndim == 2 else shap_to_plot,
            sample,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{sample_name}_force.png", dpi=300, bbox_inches="tight")
        plt.close()