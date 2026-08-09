import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


class PlotGenerator:

    def confusion_plot(
        self,
        model,
        X_test,
        y_test,
        save_path
    ):

        cm = confusion_matrix(

            y_test,

            model.predict(X_test)

        )

        plt.figure(figsize=(6,5))

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues"

        )

        plt.title("Confusion Matrix")

        plt.savefig(save_path)

        plt.close()

    def roc_plot(
        self,
        model,
        X_test,
        y_test,
        save_path
    ):

        RocCurveDisplay.from_estimator(

            model,

            X_test,

            y_test

        )

        plt.savefig(save_path)

        plt.close()

    def pr_curve(
        self,
        model,
        X_test,
        y_test,
        save_path
    ):

        PrecisionRecallDisplay.from_estimator(

            model,

            X_test,

            y_test

        )

        plt.savefig(save_path)

        plt.close()