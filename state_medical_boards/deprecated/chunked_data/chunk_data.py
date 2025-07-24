## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This file turns long documents into shorter documents
# by chunking, to create smaller versions for use with 
# limited LLMs 

# Not necessary in GPT-4

import argparse
import os
import pandas as pd
import time
import sys
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import chunk_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Verbosity
    parser.add_argument(
        "-v", "--verbose", help="Increase preprocessing verbosity.", action="store_true"
    )
    # Input data source
    parser.add_argument(
        "-ds",
        "--data-source",
        choices=["sample", "production"],
        default="sample",
        help="The input data source.",
    )
    # Input file name
    parser.add_argument(
        "-f",
        "--filename",
        help="The input filename. Not used when data source is production.",
    )
    # Case document chunk size
    parser.add_argument(
        "-cs",
        "--chunk-size",
        help="The size of case document chunks.",
        choices=[1000, 2500, 5000],
        default=2500,
    )
    args = parser.parse_args()

    if args.data_source != "production" and not args.filename:
        raise ValueError(
            "Filename argument is required for non-production data sources"
        )

    start = time.perf_counter()

    # Loads case documents
    if args.data_source == "sample":
        documents_path = f"../samples/documents/{args.filename}"
    else:
        documents_path = "../../text/fltext.csv"

    if args.verbose:
        print(f"Read case documents from {documents_path}")

    documents_df = pd.read_csv(documents_path)

    # Chunks case documents
    chunk_size = 2500
    tqdm.pandas(desc="Chunking case documents")
    documents_df["textdata"] = documents_df["textdata"].progress_apply(
        lambda case_text: chunk_text(case_text, chunk_size=chunk_size)
    )
    chunked_documents_df = documents_df.explode("textdata", ignore_index=True)
    if args.verbose:
        print(
            f"Split {len(documents_df)} case documents into {len(chunked_documents_df)} chunks"
        )

    # Saves chunked documents
    processed_path = f"./chunked_{args.filename}"
    chunked_documents_df.to_csv(processed_path, index=False)
    if args.verbose:
        print(f"Saved chunked case documents to {processed_path}")

    finish = time.perf_counter()
    print(f"Chunked case documents in {finish - start} seconds")
