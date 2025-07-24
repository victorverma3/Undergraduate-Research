## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This file reads in data from florida
# to scope whether text files have too 
# many tokens/breaks context window in
# early GPT versions

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import count_tokens, truncate_text_by_max_tokens

if __name__ == "__main__":

    # Loads Florida case text
    cases = pd.read_csv("../text/fltxt.csv")
    cases.drop(columns=["iddoc", "year", "state"], inplace=True)
    num_cases = len(cases)
    print(f"Number of FL cases:", num_cases)

    # Counts total tokens
    tqdm.pandas(desc="Counting total tokens")
    cases["token_count"] = cases["textdata"].progress_apply(
        lambda case_text: count_tokens(text=case_text)
    )
    token_count = cases["token_count"].sum()
    print(f"Total tokens:", token_count)
    print(f"Average tokens per case:", np.round(token_count // num_cases))

    # Counts total tokens in truncated case text
    tqdm.pandas(desc="Truncating case text")
    cases["truncated_case_text"] = cases["textdata"].progress_apply(
        lambda case_text: truncate_text_by_max_tokens(text=case_text, max_tokens=12000)
    )
    tqdm.pandas(desc="Counting total tokens (truncated)")
    cases["truncated_token_count"] = cases["truncated_case_text"].progress_apply(
        lambda case_text: count_tokens(text=case_text)
    )
    truncated_token_count = cases["truncated_token_count"].sum()
    print(f"Total tokens (truncated):", truncated_token_count)
    print(f"Average tokens per case (truncated):", truncated_token_count // num_cases)
