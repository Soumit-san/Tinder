import os

import mlflow
import pandas as pd


def main():
    if os.path.exists("ml/mlruns"):
        os.chdir("ml")

    mlflow.set_experiment("Sentix_Tinder_Baselines")
    experiment = mlflow.get_experiment_by_name("Sentix_Tinder_Baselines")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    cols = [
        "tags.mlflow.runName",
        "metrics.accuracy",
        "metrics.macro_f1",
        "metrics.roc_auc_macro",
    ]
    df = runs[cols]
    df.columns = ["Model", "Accuracy", "Macro F1", "ROC AUC (Macro)"]

    report = "\n## Baseline Results\n" + df.to_string(index=False)
    with open("metrics_report.txt", "w") as f:
        f.write(report)
    print("Report written to metrics_report.txt")


if __name__ == "__main__":
    main()
