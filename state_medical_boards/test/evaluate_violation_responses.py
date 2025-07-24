## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This file looks at responses produced by model
# and compares to hand-coding
# to produce confusion matrices 
# on testing data 

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Cleans responses
def clean_responses(responses: pd.DataFrame) -> pd.DataFrame:

    # Filters empty textdata
    responses = responses[~responses["textdata"].isna()]

    # Drops unnecessary columns
    responses = responses.drop(columns=["iddoc", "textdata", "year", "state"])

    # Converts all columns to str dtype
    responses = responses.astype(str)

    return responses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Version
    parser.add_argument(
        "-ver", "--version", help="Specify model version", required=True
    )

    args = parser.parse_args()

    # Loads responses
    true_response_path = "./violation/true/seed_2_50_violations_true_bool.csv"
    y_true = pd.read_csv(true_response_path)

    pred_response_path = (
        f"./violation/pred/seed_2_50_violations_pred_v{args.version}.csv"
    )
    y_pred = pd.read_csv(pred_response_path)

    # Cleans responses
    y_true = clean_responses(responses=y_true)
    y_pred = clean_responses(responses=y_pred)

    # Creates confusion matrices for each column
    violations = y_true.columns
    for violation in violations:
        cm = confusion_matrix(
            y_true=y_true[violation].tolist(),
            y_pred=y_pred[violation].tolist(),
            labels=["1", "0", "-1"],
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["1", "0", "-1"],
            yticklabels=["1", "0", "-1"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{violation} Confusion Matrix")
        plt.savefig(
            f"./violation/v{args.version}/figures/{violation}_confusion_matrix.png"
        )
        plt.close()
