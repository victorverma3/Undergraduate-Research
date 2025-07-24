## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# Run program to summarize text 
# After DSPY training

import argparse
import os
import pandas as pd
import sys
import time
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from setup import configure_dspy, SimpleSummaryModel
from utils import truncate_text_by_max_tokens


# Runs summary model on documents
def summary_helper(row: pd.DataFrame, summary_model, verbose: bool) -> pd.DataFrame:

    try:
        truncated_case_text = truncate_text_by_max_tokens(
            text=row["textdata"], max_tokens=12000
        )
        response = summary_model(case_document=truncated_case_text)
        row["trouble_summary"] = response.trouble_summary

        if verbose:
            print("\nTrouble summary:", response.trouble_summary)
    except Exception as e:
        print(f"\nError processing document {row.name}: {str(e)}")
        row["trouble_summary"] = "Failed to process"

    return row


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase summary program verbosity",
        action="store_true",
    )
    args = parser.parse_args()

    start = time.perf_counter()

    # Defines relative input and output paths
    input_documents_path = (
        "./training/seed_1_50_training_documents.csv"  # NOTE adjust as necessary
    )
    output_summaries_path = (
        "./output/seed_1_50_summaries.csv"  # NOTE adjust as necessary
    )

    # Configures DSPy summary program
    dspy = configure_dspy()
    summary_model = dspy.ChainOfThought(SimpleSummaryModel)

    # Runs summary program on documents
    documents = pd.read_csv(input_documents_path)
    tqdm.pandas(desc="Summarizing case documents")
    summaries = documents.apply(
        summary_helper,
        model=summary_model,
        verbose=args.verbose,
        axis=1,
    )
    summaries.to_csv(output_summaries_path, index=False)

    finish = time.perf_counter()
    print(f"\nGenerated summaries in {finish - start} seconds")
