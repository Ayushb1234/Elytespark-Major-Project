from pathlib import Path
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


class LIMEExplainer:
    def __init__(self, X_train, feature_names, class_names):
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)

        # LimeTabularExplainer expects numpy arrays
        self.explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True
        )

    def explain_instance(self, model, sample, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sample = np.array(sample).reshape(1, -1)[0]

        explanation = self.explainer.explain_instance(
            data_row=sample,
            predict_fn=model.predict_proba,
            num_features=min(10, len(self.feature_names))
        )

        explanation.save_to_file(str(output_path))
        return explanation