## Last edited: May 7, 2025
## Created by: Victor Verma
## State Medical Boards Project


# This file configures DSPY and helper functions
# and defines DSPY modules 

from dotenv import load_dotenv
import dspy
import os

load_dotenv()


# Configures DSPy
def configure_dspy() -> dspy:

    lm = dspy.LM(
        model=os.environ.get("OPENAI_MODEL"),
        organization=os.environ.get("OPENAI_ORG"),
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    return dspy


# Loads saved DSPy program
def load_dspy_program(dspy: dspy, program_path: str) -> dspy.Program:

    violation_program = dspy.ChainOfThought(DoctorViolationModel)
    violation_program.load(path=program_path)

    return violation_program


# Defines the input and output structure of the violation DSPy model
class DoctorViolationModel(dspy.Signature):
    """Extract boolean information regarding the contents of a medical board case.
    Respond -1 for any output that is unclear."""

    # Inputs
    case_document: str = dspy.InputField()

    # Outputs
    patient_mentioned: int = dspy.OutputField(
        desc="1 if the patient is mentioned in the case text (including by initials or as 'patient'), 0 otherwise. -1 if unsure."
    )
    fraud_case: int = dspy.OutputField(
        desc="1 if the case is related to fraud, 0 otherwise. -1 if unsure. Fraud is any behavior where the doctor lied to the government or an insurance company in order to increase their profits. This includes billing for services not rendered, billing for patients not seen, sending inflated invoices to receive higher reimbursement, and lying about a patient's health to get insurance to pay for a service.  Mark 1 if the doctor has been subject to any form of anti-fraud enforcement, including litigation under the false claims act, the federal health care fraud statute, the anti-kickback statute, the Stark Law; arrest or indictment for fraud; or exclusion from the Medicare program due to fraud. "
    )
    malpractice_case: int = dspy.OutputField(
        desc="1 if the case is related to medical malpractice, 0 otherwise. -1 if unsure."
    )
    dea_case: int = dspy.OutputField(
        desc="1 if the case involves the Drug Enforcement Agency, 0 otherwise. -1 if unsure."
    )
    improper_opioid_prescription: int = dspy.OutputField(
        desc="1 if the doctor improperly prescribed opioids, 0 otherwise. -1 if unsure."
    )
    improper_drug_prescription: int = dspy.OutputField(
        desc="1 if the doctor improperly prescribed any drugs (including opioids), 0 otherwise. -1 if unsure."
    )
    unfit_to_practice: int = dspy.OutputField(
        desc="1 if the doctor got into unrelated legal trouble resulting in action being taken against their license, 0 otherwise. -1 if unsure."
    )
    bad_medical_records: int = dspy.OutputField(
        desc="1 if the doctor failed to maintain adequate medical records, 0 otherwise. -1 if unsure."
    )
    license_issues: int = dspy.OutputField(
        desc="1 if the doctor faced administrative issues with their license, 0 otherwise. -1 if unsure. Voluntary relinquishment should be marked as 0."
    )
    miscellaneous_violation: int = dspy.OutputField(
        desc="1 if the doctor committed a violation that is not already specified, 0 otherwise. -1 if unsure."
    )
    other_state_action: int = dspy.OutputField(
        desc="1 if the doctor committed a violation in another state, 0 otherwise. -1 if unsure."
    )
    no_substantive_information: int = dspy.OutputField(
        desc="1 if there is no substantive information in the case text regarding the doctor's violation, 0 otherwise. -1 if unsure."
    )
    proactive: int = dspy.OutputField(
        desc="1 if someone else (agency, lawsuit, etc) got the doctor in trouble first, before the state medical board, 0 otherwise. -1 if unsure."
    )


# Evaluation metric for the doctor violation program.
def doctor_violation_metric(example, pred, trace=None) -> int:

    score = 0
    score += int(pred.patient_mentioned == example.patient_mentioned)
    score += int(pred.fraud_case == example.fraud_case)
    score += int(pred.malpractice_case == example.malpractice_case)
    score += int(pred.dea_case == example.dea_case)
    score += int(
        pred.improper_opioid_prescription == example.improper_opioid_prescription
    )
    score += int(pred.improper_drug_prescription == example.improper_drug_prescription)
    score += int(pred.unfit_to_practice == example.unfit_to_practice)
    score += int(pred.bad_medical_records == example.bad_medical_records)
    score += int(pred.license_issues == example.license_issues)
    score += int(pred.miscellaneous_violation == example.miscellaneous_violation)
    score += int(pred.other_state_action == example.other_state_action)
    score += int(pred.no_substantive_information == example.no_substantive_information)
    score += int(pred.proactive == example.proactive)

    return score


# Defines the input and output structure of the simple summary DSPy model
class SimpleSummaryModel(dspy.Signature):
    """Create a 1-2 sentence summary describing why the doctor faced disciplinary action
    from the state medical board."""

    # Inputs
    case_document: str = dspy.InputField()

    # Outputs
    trouble_summary: str = dspy.OutputField(
        desc="A 1-2 sentence summary describing why the doctor faced disciplinary action from the state medical board. Respond 'Not Sure' if the answer is unclear. If the trouble is related to violations in another state, describe those violations as well."
    )
