## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# Randomly sample a subset of documents
# for testing or training
# By specifying seed, # docs 

import argparse
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import sample_random_documents

if __name__ == "__main__":

    # Initializes the sample parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        choices=[str(i) for i in range(0, 10)],
        default="0",
        help="Set the random seed for the sampling process (0-9).",
    )
    parser.add_argument(
        "--num_samples",
        choices=["1", "2", "5", "10", "20", "25", "50", "100"],
        default="10",
        help="Choose the number of samples.",
    )
    parser.add_argument(
        "-v", "--verbose", help="Increase chatbot verbosity", action="store_true"
    )
    args = parser.parse_args()

    start = time.perf_counter()

    # Defines relative input and output paths
    input_documents_path = "./../../text/fltxt.csv"  # NOTE adjust as necessary
    output_sample_documents_path = f"./documents/seed_{args.seed}_{args.num_samples}_sample_documents.csv"  # NOTE adjust as necessary

    # Samples random documents
    sample_random_documents(
        input_documents_path=input_documents_path,
        output_sample_documents_path=output_sample_documents_path,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        verbose=args.verbose,
    )

    finish = time.perf_counter()
    print(
        f"\nSampled {args.num_samples} documents with random seed {args.seed} in {finish - start} seconds"
    )
