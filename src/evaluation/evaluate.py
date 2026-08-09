import joblib

from sklearn.metrics import classification_report

import pandas as pd

from src.features.preprocessing import HealthcarePreprocessor

from src.evaluation.metrics import MetricsCalculator

from src.evaluation.plots import PlotGenerator

from src.evaluation.report import ReportGenerator


df = pd.read_csv(

    "data/processed/clean_data.csv"

)

processor = HealthcarePreprocessor()

(

    X_train,

    X_test,

    y_train,

    y_test,

    X_train_processed,

    X_test_processed

) = processor.prepare(df)


model = joblib.load(

    "models/best_model.pkl"

)


calculator = MetricsCalculator()

metrics, y_pred = calculator.calculate(

    model,

    X_test_processed,

    y_test

)


print(metrics)


report = classification_report(

    y_test,

    y_pred

)


plot = PlotGenerator()

plot.confusion_plot(

    model,

    X_test_processed,

    y_test,

    "outputs/figures/confusion_matrix.png"

)

try:

    plot.roc_plot(

        model,

        X_test_processed,

        y_test,

        "outputs/figures/roc_curve.png"

    )

except:

    pass

try:

    plot.pr_curve(

        model,

        X_test_processed,

        y_test,

        "outputs/figures/pr_curve.png"

    )

except:

    pass


reporter = ReportGenerator()

reporter.save_metrics(

    metrics,

    "outputs/metrics/model_metrics.csv"

)

reporter.save_classification_report(

    report,

    "outputs/reports/classification_report.txt"

)

print()

print("Evaluation Complete")