from dotenv import load_dotenv
import dspy
import os
import pandas as pd
import time

from utils import sample_documents


# Configures DSPy
def configure_dspy(model, organization):

    load_dotenv()

    lm = dspy.LM(
        model=model,
        organization=organization,
    )
    dspy.configure(lm=lm)

    return dspy


# Defines input and output structure of DSPy pipeline
class DoctorTrouble(dspy.Signature):
    """Extract boolean information regarding the contents of a document.
    Respond 'Not Sure' if any answer is unclear."""

    # Inputs
    document: str = dspy.InputField()

    # Outputs
    trouble_summary: str = dspy.OutputField()
    patient_mentioned: str = dspy.OutputField(
        desc="'Yes' if harmed patients were mentioned, 'No' otherwise."
    )
    fraud_case: str = dspy.OutputField(
        desc="'Yes' if related to fraud, 'No' otherwise. Fraud is any behavior where the doctor lied to the government or an insurance company in order to increase their profits. This includes billing for services not rendered, billing for patients not seen, sending inflated invoices to receive higher reimbursement, and lying about a patient's health to get insurance to pay for a service.  Mark as yes if the doctor has been subject to any form of anti-fraud enforcement, including litigation under the false claims act, the federal health care fraud statute, the anti-kickback statute, the Stark Law; arrest or indictment for fraud; or exclusion from the Medicare program due to fraud."
    )
    malpractice_case: str = dspy.OutputField(
        desc="'Yes' if related to malpractice, 'No' otherwise."
    )
    dea_case: str = dspy.OutputField(
        desc="'Yes' if related to the DEA, 'No' otherwise."
    )
    inappropriate_opioid_prescription: str = dspy.OutputField(
        desc="'Yes' if the doctor inappropriately prescribed opioids, 'No' otherwise."
    )
    doctor_drug_abuse: str = dspy.OutputField(
        desc="'Yes' if the doctor was themselves abusing drugs, 'No' otherwise."
    )
    proactive: str = dspy.OutputField(
        desc="'Yes' if board initiated investigation independently, 'No' if responding to another agency. \
        Reactive indicators: other agency investigations, regulatory complaints, post-criminal charges, post-enforcement actions. \
        Proactive indicators: internal monitoring, patient complaints to board, audits, no prior investigations."
    )


# Calls DSPy pipeline on each row
def summarize_doctor_trouble(row, pipeline, verbose):

    try:
        response = pipeline(document=row["textdata"])
        row["trouble_summary"] = response.trouble_summary
        row["patient_mentioned"] = response.patient_mentioned
        row["fraud_case"] = response.fraud_case
        row["malpractice_case"] = response.malpractice_case
        row["dea_case"] = response.dea_case
        row["inappropriate_opioid_prescription"] = (
            response.inappropriate_opioid_prescription
        )
        row["doctor_drug_abuse"] = response.doctor_drug_abuse
        row["proactive"] = response.proactive

        if verbose:
            print("\nTrouble summary:", response.trouble_summary)
            print("Patient mentioned:", response.patient_mentioned)
            print("Fraud case:", response.fraud_case)
            print("Malpractice case:", response.malpractice_case)
            print("DEA case:", response.dea_case)
            print(
                "Inappropriate opioid prescription:",
                response.inappropriate_opioid_prescription,
            )
            print("Doctor drug abuse:", response.doctor_drug_abuse)
            print("Proactive:", response.proactive)

    except Exception as e:
        print(f"\nError processing document {row.name}: {str(e)}")
        row["trouble_summary"] = "Failed to process"
        row["patient_mentioned"] = "Failed to process"
        row["fraud_case"] = "Failed to process"
        row["malpractice_case"] = "Failed to process"
        row["dea_case"] = "Failed to process"
        row["inappropriate_opioid_prescription"] = "Failed to process"
        row["doctor_drug_abuse"] = "Failed to process"
        row["proactive"] = "Failed to process"

    return row


# Runs DSPy pipeline on the sample documents
def run_pipeline(sample_documents_path, sample_responses_path, dspy, verbose):

    start = time.perf_counter()

    # Instantiates DSPy pipeline
    ExtractDocumentCaseInfo = dspy.ChainOfThought(DoctorTrouble)

    # Runs pipeline on each document
    sample_documents = pd.read_csv(sample_documents_path)
    sample_responses = sample_documents.apply(
        summarize_doctor_trouble,
        pipeline=ExtractDocumentCaseInfo,
        verbose=verbose,
        axis=1,
    )
    sample_responses.to_csv(sample_responses_path, index=False)

    finish = time.perf_counter()
    print(f"\nRan DSPy pipeline on {num_samples} documents in {finish - start} seconds")


if __name__ == "__main__":

    # Defines sample parameters
    num_samples = 100
    seed = 0

    # Defines relative input and output paths
    cwd = os.path.dirname(os.path.abspath(__file__))
    input_documents_path = os.path.join(cwd, "../text/fltxt.csv")
    sample_documents_path = os.path.join(
        cwd, f"./samples/documents/seed_{seed}_{num_samples}_sample_documents.csv"
    )
    sample_responses_path = os.path.join(
        cwd, f"./samples/responses/seed_{seed}_{num_samples}_sample_responses.csv"
    )

    # Defines OpenAI model parameters
    model_codes = ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo-0125"]
    model = "openai/gpt-3.5-turbo-0125"
    organization = "org-PemKciJ4sgMCN79S7pGVK3Fc"

    # Samples documents
    documents = sample_documents(
        file_path=input_documents_path,
        sample_documents_path=sample_documents_path,
        num_samples=num_samples,
        seed=seed,
        verbose=False,
    )

    # Configures DSPy
    dspy = configure_dspy(model=model, organization=organization)

    # Runs DSPy pipeline on the sample documents
    run_pipeline(
        sample_documents_path=sample_documents_path,
        sample_responses_path=sample_responses_path,
        dspy=dspy,
        verbose=False,
    )
