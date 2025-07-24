# Analyzing Medical Violations with DSPy

## Table of Contents

-   [Code Flow](#code-flow)
-   [Directory Structure](#directory-structure)
    -   [Codebase Layout](#codebase-layout)
    -   [code/](#code)
    -   [code/deprecated/](#codedeprecated)
    -   [code/models/](#codemodels)
    -   [code/output/](#codeoutput)
    -   [code/samples/](#codesamples)
    -   [code/test/](#codetest)
    -   [code/training/](#codetraining)

## Code Flow

-   Setup.py initalizes the DSPY model parameters.
-   optimize_violation_program.py runs DSPY on training data to optimize prompt,
    output model.
-   evaluate_violation_responses.py compares output to hand-coded data for
    confusion matrices & check correctness.
-   run_summary_program.py creates summary files (not optimized).
-   run_violation_program.py uses optimized DSPY model on new output.
-   sample_documents.py creates random samples for training and testing.

## Directory Structure

### Codebase Layout

```
code
|__ deprecated
    |__ chunked_data
        |__ __init__.py
        |__ chunk_data.py
        |__ chunked_seed_0_1_sample_documents.csv
        |__ ...
    |__ misc
        |__ SMB_Overview_DSPy_Discord.pdf
    |__ analyze_tokens.py
    |__ sample_pipeline.ipynb
    |__ sample_pipeline.py
|__ models
    |__ __init__.py
    |__ run_summary_program.py
    |__ run_violation_program.py
|__ output
|__ samples
    |__ documents
        |__ seed_0_1_sample_documents.csv
        |__ ...
    |__ responses
        |__ seed_0_1_sample_responses.csv
        |__ ...
    |__ __init__.py
    |__ sample_documents.py
|__ test
    |__ summary
        |__ seed_1_50_summaries.py
        |__ ...
    |__ violation
        |__ pred
            |__ seed_2_50_violations_pred_v1.csv
            |__ ...
        |__ true
            |__ seed_2_50_violations_true_bool.csv
            |__ ...
        |__ v1
            |__ figures
                |__ bad_medical_records_confusion_matrix.png
                |__ ...
        |__ v2
            |__ figures
                |__ bad_medical_records_confusion_matrix.png
                |__ ...
    |__ __init__.py
    |__ evaluate_violation_responses.py
|__ training
    |__ programs
        |__ summary
        |__ violation
            |__ descriptions.txt
            |__ violation_program_v1.json
            |__ violation_program_v2.pkl
    |__ prompts
        |__ baseline_prompt.txt
        |__ miprov2_optimized_prompt.txt
    |__ __init__.py
    |__ optimize_violation_program.py
    |__ seed_1_50_training_documents.csv
    |__ seed_1_50_training_responses.csv
|__ venv
|__ __init__.py
|__ .env
|__ .gitignore
|__ README.MD
|__ requirements.txt
|__ setup.py
|__ utils.py
```

### code/

The directory contains a script that was used to analyze the size of the case
documents, and a file defining utility functions for the entire codebase. There
is also a `.env` and `requirements.txt` file to set up the necessary codebase
environment. `setup.py` is used to configure all of the `DSPy` settings and
programs.

### code/deprecated/

This directory contains deprecated code and files used to gain insights about
the project and explore the feasibility of using DSPy.

### code/models

This directory contains scripts to run the summary program and optimized
violation program.

### code/output

This directory is meant to contain the final output. It is currently empty
because no final output has been generated.

### code/samples

This directory contains sample case documents, sample responses, and a script to
randomly generate sample documents.

### code/test

This directory contains a script that can be used to evaluate the accuracy of
the violation program. It compares the predicted and true values, which are also
stored in this directory, and creates confusion matrices for each column. There
is also one set of predicted summary program responses, but no infrastructure
has been created to evaluate it (yet).

### code/training

This directory contains different program versions saved to disk, as well as the
prompts produced by the different optimizers. It also contains feature and
target data for training, as well as a script that can be used to optimize the
violation program with different optimizers.

## Setup Guidelines

-   Navigate to the `ai_code/code` directory and run `source venv/bin/activate`
    in terminal to start the virtual environment necessary to run any code.
-   The current OpenAI API key is for the `StateMedicalBoards` project under the
    `JetsonResearch` organization. Ideally, we want to keep this API key because
    it has reached usage tier 4 for the OpenAI API, which is particularly
    advantageous because it has higher rate limits. Reach out to Victor if a new
    member needs to be added as an owner for the organization.
