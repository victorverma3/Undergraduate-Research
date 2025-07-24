## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This file trains the DSPY model
# feeding in input and output paths 
# Also includes optimizer settings and config 

import argparse
import contextlib
from dspy import Example, Evaluate, Program, SIMBA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
import io
import os
import pandas as pd
import sys
import time
from tqdm import tqdm
from typing import Sequence

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from setup import configure_dspy, doctor_violation_metric, DoctorViolationModel
from utils import truncate_text_by_max_tokens


# Prepares the DSPy training data
def prepare_training_data(
    training_documents_path: str,
    training_responses_path: str,
    training_inputs: set,
    verbose: bool = False,
) -> Sequence[Example]:

    training_documents = pd.read_csv(training_documents_path)
    training_responses = pd.read_csv(training_responses_path)
    training_data = pd.concat(objs=[training_documents, training_responses], axis=1)
    training_data.drop(columns=["iddoc", "year", "state"], inplace=True)
    training_data.dropna(subset=["textdata"], inplace=True)
    tqdm.pandas(desc="Truncating training data")
    training_data["textdata"] = training_data["textdata"].progress_apply(
        lambda case_text: truncate_text_by_max_tokens(text=case_text, max_tokens=500)
    )

    if verbose:
        print(f"Training columns: {list(training_data.columns)}")

    trainset = []
    for row in training_data.itertuples():
        trainset.append(
            Example(
                case_document=row.textdata,
                patient_mentioned=row.patient_mentioned,
                fraud_case=row.fraud_case,
                malpractice_case=row.malpractice_case,
                dea_case=row.dea_case,
                improper_opioid_prescription=row.improper_opioid_prescription,
                improper_drug_prescription=row.improper_drug_prescription,
                unfit_to_practice=row.unfit_to_practice,
                bad_medical_records=row.bad_medical_records,
                license_issues=row.license_issues,
                miscellaneous_violation=row.miscellaneous_violation,
                other_state_action=row.other_state_action,
                no_substantive_information=row.no_substantive_information,
                proactive=row.proactive,
            ).with_inputs(training_inputs)
        )

    length = len(trainset)
    cutoff = int(0.8 * length)
    if verbose:
        print(f"Prepared {length} training examples")

    return trainset[:cutoff], trainset[cutoff:]


def optimize_program(
    trainset: Sequence[Example], optimizer: str, save_path: str, verbose: bool
) -> Program:

    dspy = configure_dspy()
    violation_program = dspy.ChainOfThought(DoctorViolationModel)

    # Saves baseline prompt
    violation_program(case_document="Placeholder for baseline prompt")
    history_output = io.StringIO()
    with contextlib.redirect_stdout(history_output):
        dspy.inspect_history()
    history_text = history_output.getvalue()
    with open("./prompts/baseline_prompt.txt", "w") as file:
        file.write(history_text)

    # Optimizes program
    if optimizer == "bfsrs":
        if verbose:
            print("Optimizing program with Bootstrap Few Shot Random Search")

    elif optimizer == "miprov2":
        if verbose:
            print("Optimizing program with MIPROv2...")

        teleprompter = MIPROv2(
            metric=doctor_violation_metric,
            auto="light",
        )
        optimized_violation_program = teleprompter.compile(
            violation_program.deepcopy(),
            trainset=trainset,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            requires_permission_to_run=False,
            seed=0,
        )

    elif optimizer == "simba":
        if verbose:
            print("Optimizing program with SIMBA...")
        simba = SIMBA(
            metric=doctor_violation_metric,
            max_steps=1,
            max_demos=0,
            demo_input_field_maxlen=500,
        )
        optimized_violation_program = simba.compile(
            violation_program, trainset=trainset, seed=0
        )

    # Saves optimized program to disk
    try:
        optimized_violation_program.save(path=save_path)
        if args.verbose:
            print(f"Saved violation program to {save_path}")
    except Exception as e:
        raise e

    # Evaluates optimized program
    if verbose:
        print(f"Evaluating optimized program...")
    evaluate = Evaluate(
        devset=devset[:],
        metric=doctor_violation_metric,
        num_threads=8,
        display_progress=True,
        display_table=False,
    )
    evaluate(program=optimized_violation_program, devset=devset[:])

    # Saves optimized prompt
    violation_program(case_document="Placeholder for optimized prompt.")
    history_output = io.StringIO()
    with contextlib.redirect_stdout(history_output):
        dspy.inspect_history()
    history_text = history_output.getvalue()
    with open(f"./prompts/{optimizer}_optimized_prompt.txt", "w") as file:
        file.write(history_text)

    return optimized_violation_program


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Optimizer
    parser.add_argument(
        "-o",
        "--optimizer",
        choices=["bfsrs", "miprov2", "simba"],
        default="miprov2",
        help="Specify the optimizer (bfsrs, miprov2, simba)",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose", help="Increase training verbosity", action="store_true"
    )

    args = parser.parse_args()

    start = time.perf_counter()

    # Defines training configurations
    training_documents_path = (
        "./seed_1_50_training_documents.csv"  # NOTE adjust as necessary
    )
    training_responses_path = (
        "./seed_1_50_training_responses.csv"  # NOTE adjust as necessary
    )
    training_inputs = "case_document"

    # Prepares the training data
    if args.verbose:
        print("Preparing training data...")
    trainset, devset = prepare_training_data(
        training_documents_path=training_documents_path,
        training_responses_path=training_responses_path,
        training_inputs=training_inputs,
        verbose=args.verbose,
    )

    # Optimizes the DSPy program
    optimized_violation_program = optimize_program(
        trainset=trainset,
        optimizer=args.optimizer,
        save_path="./programs/violation/violation_program_v3.pkl",
        verbose=args.verbose,
    )

    finish = time.perf_counter()
    print(f"Trained DSPy program in {finish - start} seconds")
