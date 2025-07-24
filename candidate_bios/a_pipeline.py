"""
CandidateBios Data Pipeline

Created by Victor Verma
Last edited May 7, 2024

This file was used to run the data processing pipeline from end to end, starting 
with the Google search for information and ending with parsing the ChatGPT response 
and exporting the information into an output CSV file.
"""

# Imports
from b_search import search, searchRow
from c_retrieval import retrieve
from d_extraction import extract
import glob
import os
import pandas as pd
import time

# Setup
testRows = {"link test": 126073, "content test": 24526, "length test": 142634}

outputFiles = [
    "b1_searches.csv",
    "c1_retrievals.csv",
    "c2_scrapingTimeouts.csv",
    "d1_extractions.csv",
    "d2_promptErrors.csv",
    "d3_parseErrors.csv",
    "d4_reruns.csv",
    "errors.txt",
]

# groups of candidates
groups = [
    "0-9699",
    "9700-19399",
    "19400-29099",
    "29100-38799",
    "38800-48499",
    "48500-58199",
    "58200-67899",
    "67900-77599",
    "77600-87299",
    "87300-96999",
    "97000-106699",
    "106700-116399",
    "116400-126099",
    "126100-135799",
    "135800-145499",
    "145500-end",
    "all",
]


# Pipeline
def gatherData(n, r=4, read="random"):
    """
    Description
        - Wrapper function used to run the program.
    Parameters
        - n: an integer that indicates the number of unique candidates for whom
        to gather biodata.
        - r: an integer that specifies the number of Google API search results
        to use during the gathering process. 1 <= r <= 4, and r is set to 4 by
        default.
        - read: a string that defines how the n unique candidates should be
        chosen. If read is set to "random", then n unique random candidates are
        used, and if read = "order", then the first n unique candidates in order
        are used. read is set to random by default.
    Return
        - A dataframe containing each candidate’s full name, state, min year,
        candid, college major, undergraduate institution, highest degree, work
        history, sources, and ChatGPT confidence. This dataframe is also output
        to extractions.csv.
    """

    start = time.perf_counter()

    # verifies parameters
    assert n > 0
    assert 1 <= r <= 4
    assert read in ["random", "order"]

    # deletes existing output files to avoid confusion
    deleteOutput(outputFiles)
    doneDelete = time.perf_counter()
    print(f"deleted old output files: {doneDelete - start} seconds")

    # starts data search
    search(n, r, read)

    # starts data retrieval
    retrieve()

    # starts data extraction
    extractions = extract()

    end = time.perf_counter()
    print(f"pipeline: {end - start} seconds")
    return extractions


def gatherRow(r=4, rows=[126073, 0, 27, 44, 46]):
    """
    Description
        - Wrapper function used to run the program on specified candidates.
    Parameters
        - r: an integer that specifies the number of Google API search results
        to use during the gathering process. 1 <= r <= 4, and r is set to 4 by
        default.
        - rows: an integer array containing the candidates' row numbers for whom
        to gather biodata. The row number passed into the array for a candidate
        should be equal to the row number for that candidate in
        ldata_R_unique.csv - 2 in order to account for the indexing in pandas
        dataframes.
    Return
        - A dataframe containing each candidate’s full name, state, min year,
        candid, college major, undergraduate institution, highest degree, work
        history, sources, and ChatGPT confidence. This dataframe is also output
        to extractions.csv.
    """

    start = time.perf_counter()

    # verifies parameters
    assert 1 <= r <= 4

    # deletes existing output files to avoid confusion
    deleteOutput(outputFiles)
    doneDelete = time.perf_counter()
    print(f"deleted old output files: {doneDelete - start} seconds")

    # starts data search
    searchRow(r, rows)

    # starts data retrieval
    retrieve()

    # starts data extraction
    extractions = extract()

    end = time.perf_counter()
    print(f"pipeline: {end - start} seconds")
    return extractions


def deleteOutput(outputFiles):
    """
    Description
        - Deletes existing program output files from the local directory.
    Parameters
        - outputFiles: a string array that contains the names of all output files
        that should be deleted. Globally defined as ['errors.txt',
        'extractions.csv', 'parseErrors.csv', 'promptErrors.csv',  'reruns.csv',
        'retrievals.csv', 'scrapingTimeouts.csv', 'searches.csv'].
    Return
        - No return value.
    """

    for file in outputFiles:
        if os.path.exists(file):
            os.remove(file)


def combineCSV(comboType, group):
    """
    Description
        - Combines the specified set of CSVs into one large CSV.
    Parameters
        - comboType: a string that indicates whether the prompt, timeout, or
        output CSVs should be combined.
        - group: a string that specifies which group’s candidates should be used
        for pathing purposes.
    Return
        - No return value.

    """

    combinationPaths = {
        "retrievals": f"../Results/Scraping/Prompts/{group}",
        "timeouts": f"../Results/Scraping/Timeouts/{group}",
        "outputs": f"../Results/ChatGPT/{group}",
        "allSearches": "../Results/GoogleSearch",
        "allRetrievals": "../Results/Scraping/Prompts/all",
        "allTimeouts": "../Results/Scraping/Timeouts/all",
        "allOutputs": "../Results/Outputs",
        "allParseErrors": "../Results/ChatGPT/ParseErrors",
        "allPromptErrors": "../Results/ChatGPT/PromptErrors",
        "summaryOutputs": "../Results/Temp",
        "promptErrorReruns": "../Results/Missing/Summary/promptErrorReruns",
        "all": "../Results/All",
    }

    # verifies parameters
    assert comboType in combinationPaths
    assert group in groups

    path = combinationPaths[comboType]
    files = glob.glob(f"{path}/*.csv")
    df = pd.concat(
        [pd.read_csv(file, index_col=None, encoding="latin-1") for file in files]
    )
    df.to_csv(f"{path}/{comboType}{group}.csv", escapechar="/")
    print(f"\n{comboType}\n{df}")
