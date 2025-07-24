## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This file defines helper functions
# for the code base 

from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import tiktoken
from typing import Sequence

load_dotenv()


# Samples random documents
def sample_random_documents(
    input_documents_path: str,
    output_sample_documents_path: str,
    num_samples: int,
    seed: int,
    verbose: bool,
) -> None:

    df = pd.read_csv(input_documents_path)
    np.random.seed(seed)
    sample_indices = np.random.choice(len(df), size=num_samples, replace=False)

    sample = df.iloc[sample_indices]
    sample.to_csv(output_sample_documents_path, index=False)

    documents = sample["textdata"]

    if verbose:
        print("\nSample indices:", sample_indices)
        print("\nSample documents:", documents.values)


# Tokenizes text
def tokenize_text(text: str) -> str:

    encoding = tiktoken.get_encoding(os.getenv("OPENAI_MODEL_ENCODING"))
    tokenized_text = encoding.encode(text)

    return tokenized_text


# Detokenizes text
def detokenize_text(tokenized_text: str) -> str:

    encoding = tiktoken.get_encoding(os.getenv("OPENAI_MODEL_ENCODING"))
    detokenized_text = encoding.decode(tokenized_text)

    return detokenized_text


# Filters text longer than a maximum number of tokens
def filter_text_by_max_tokens(text: str, max_tokens: int) -> str | None:

    # Encodes the case text
    encoded_text = tokenize_text(text)

    if len(encoded_text) > max_tokens:

        return None
    else:

        return text


# Truncates text to fit a maximum number of tokens
def truncate_text_by_max_tokens(text: str, max_tokens: int) -> str:

    # Encodes text
    encoded_text = tokenize_text(text)

    # Truncates text as necessary
    if len(encoded_text) > max_tokens:
        truncated_encoded_text = encoded_text[:max_tokens]
        truncated_text = detokenize_text(truncated_encoded_text)

        return truncated_text
    else:

        return text


# Chunks text
def chunk_text(text: str, chunk_size: int = 2500) -> Sequence[str]:

    # Encodes the text
    encoded_text = tokenize_text(text)

    # Chunks the text
    chunks = []
    for i in range(0, len(encoded_text), chunk_size):
        chunks.append(detokenize_text(encoded_text[i : i + chunk_size]))

    return chunks


# Counts number of tokens in text
def count_tokens(text: str) -> int:

    # Encodes the text
    encoded_text = tokenize_text(text)

    return len(encoded_text)
