import pandas as pd


class ReportGenerator:

    def save_metrics(

        self,

        metrics,

        path

    ):

        df = pd.DataFrame(

            metrics,

            index=[0]

        )

        df.to_csv(

            path,

            index=False

        )

    def save_classification_report(

        self,

        report,

        path

    ):

        with open(

            path,

            "w"

        ) as f:

            f.write(report)