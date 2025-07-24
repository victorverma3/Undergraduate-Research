## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This program uses trained dspy model 
# to check whether documents contain 
# allegations of particular violations 

import argparse
import os
import pandas as pd
import sys
import time
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from setup import configure_dspy, load_dspy_program
from utils import truncate_text_by_max_tokens


# Runs violation program on documents
def violations_helper(row: pd.DataFrame, program) -> pd.DataFrame:

    if row["textdata"] == "":
        row["patient_mentioned"] = "empty"
        row["fraud_case"] = "empty"
        row["malpractice_case"] = "empty"
        row["dea_case"] = "empty"
        row["improper_opioid_prescription"] = "empty"
        row["improper_drug_prescription"] = "empty"
        row["unfit_to_practice"] = "empty"
        row["bad_medical_records"] = "empty"
        row["license_issues"] = "empty"
        row["miscellaneous_violation"] = "empty"
        row["other_state_action"] = "empty"
        row["no_substantative_information"] = "empty"
        row["proactive"] = "empty"

        return row

    try:
        truncated_case_text = truncate_text_by_max_tokens(
            text=row["textdata"], max_tokens=12000
        )
        response = program(case_document=truncated_case_text)

        row["patient_mentioned"] = response.patient_mentioned
        row["fraud_case"] = response.fraud_case
        row["malpractice_case"] = response.malpractice_case
        row["dea_case"] = response.dea_case
        row["improper_opioid_prescription"] = response.improper_opioid_prescription
        row["improper_drug_prescription"] = response.improper_opioid_prescription
        row["unfit_to_practice"] = response.unfit_to_practice
        row["bad_medical_records"] = response.bad_medical_records
        row["license_issues"] = response.license_issues
        row["miscellaneous_violation"] = response.miscellaneous_violation
        row["other_state_action"] = response.other_state_action
        row["no_substantive_information"] = response.no_substantive_information
        row["proactive"] = response.proactive

    except Exception as e:
        print(f"Error processing document {row.name}: {str(e)}")

        row["patient_mentioned"] = "failed to process"
        row["fraud_case"] = "failed to process"
        row["malpractice_case"] = "failed to process"
        row["dea_case"] = "failed to process"
        row["improper_opioid_prescription"] = "failed to process"
        row["improper_drug_prescription"] = "failed to process"
        row["unfit_to_practice"] = "failed to process"
        row["bad_medical_records"] = "failed to process"
        row["license_issues"] = "failed to process"
        row["miscellaneous_violation"] = "failed to process"
        row["other_state_action"] = "failed to process"
        row["no_substantative_information"] = "failed to process"
        row["proactive"] = "failed to process"

    return row


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase violation program verbosity",
        action="store_true",
    )
    args = parser.parse_args()

    start = time.perf_counter()

    # Defines relative input and output paths
    input_documents_path = "../samples/documents/seed_2_50_sample_documents.csv"  # NOTE adjust as necessary
    output_summaries_path = "../test/violation/pred/seed_2_50_violations_pred_v2.csv"  # NOTE adjust as necessary

    # Loads DSPy program
    if args.verbose:
        print("Loading violation program")
    program_path = "../training/programs/violation/violation_program_v2.pkl"
    dspy = configure_dspy()
    violation_program = load_dspy_program(dspy=dspy, program_path=program_path)

    # Runs violation program on documents
    documents = pd.read_csv(input_documents_path)
    tqdm.pandas(desc="Identifying violations")
    violations = documents.progress_apply(
        violations_helper,
        program=violation_program,
        axis=1,
    )
    if args.verbose:
        print(f"Saving violations output to {output_summaries_path}")
    print(violations.head())
    # violations.to_csv(output_summaries_path, index=False)

    finish = time.perf_counter()
    print(f"Identified violations in {finish - start} seconds")
