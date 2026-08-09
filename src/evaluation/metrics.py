from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


class MetricsCalculator:

    def calculate(self, model, X_test, y_test):

        y_pred = model.predict(X_test)

        metrics = {

            "Accuracy": accuracy_score(y_test, y_pred),

            "Precision": precision_score(
                y_test,
                y_pred,
                average="weighted"
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                average="weighted"
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred,
                average="weighted"
            )

        }

        try:

            y_prob = model.predict_proba(X_test)

            metrics["ROC-AUC"] = roc_auc_score(

                y_test,

                y_prob,

                multi_class="ovr"

            )

        except:

            pass

        return metrics, y_pred