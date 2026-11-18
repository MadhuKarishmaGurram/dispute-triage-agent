"""
apply_adjudication.py
Applies human adjudication decisions to the golden set, with a recorded reason
for each change. This file IS the record of what changed and why.

Context: the first classification eval (71%) showed that 7 of the 20 CFPB
labels did not match what the consumer actually wrote. Before re-reading the
cases, we wrote a tie-break rule:

  "If the consumer says the problem was caused by identity theft or fraud
   (e.g. an identity theft report, police report, or accounts opened without
   their knowledge), use fraud_and_identity, even if they also mention
   incorrect information or an investigation."

NOTE: these labels were changed AFTER seeing the model's answers, so scores on
them are optimistic. A blind held-out set is used for the trustworthy score.

Run: uv run python evals/apply_adjudication.py
"""

import pandas as pd

GOLDEN_PATH = "evals/golden_set.csv"
DATE = "2026-11-17"

EXPECTED_SECTIONS = {
    "incorrect_information": "1681i;1681s-2;1681e",
    "investigation_problem": "1681i",
    "improper_use": "1681b",
    "fraud_and_identity": "1681c-1;1681c-2",
    "cannot_access_report": "1681g;1681j",
    "other": "",
}

# case_id -> (new label, reason)
DECISIONS = {
    "incorrect_information_1": ("fraud_and_identity",
        "Cites an FTC identity theft report and FCRA 605B blocking; accounts are wrong due to fraud."),
    "investigation_problem_1": ("fraud_and_identity",
        "Consumer reports identity theft; no earlier dispute or investigation is mentioned."),
    "investigation_problem_2": ("other",
        "Asks a bank for bank statements; not a credit reporting issue."),
    "investigation_problem_4": ("fraud_and_identity",
        "Borderline: complains about investigations, but the cause is fraud accounts and FTC "
        "reports, so the tie-break rule applies."),
    "improper_use_3": ("fraud_and_identity",
        "Data breach and a request to block information under 1681c-2; no improper pull described."),
    "improper_use_4": ("incorrect_information",
        "Wants address and phone updated and negative accounts disputed; no unauthorized use."),
    "cannot_access_report_1": ("incorrect_information",
        "Can see the report; wants a student loan default corrected."),
}


def main() -> None:
    try:
        golden = pd.read_csv(GOLDEN_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        golden = pd.read_csv(GOLDEN_PATH, encoding="cp1252")
    golden["notes"] = golden["notes"].fillna("").astype(str)
    golden["expected_sections"] = golden["expected_sections"].fillna("").astype(str)

    changed = 0
    for case_id, (new_label, reason) in DECISIONS.items():
        row = golden["case_id"] == case_id
        old_label = golden.loc[row, "label"].iloc[0]
        if old_label == new_label:
            print(f"  already applied: {case_id}")
            continue
        golden.loc[row, "label"] = new_label
        golden.loc[row, "expected_sections"] = EXPECTED_SECTIONS[new_label]
        golden.loc[row, "notes"] = f"ADJUDICATED {DATE}: was {old_label}. {reason}"
        changed += 1
        print(f"  {case_id}: {old_label} -> {new_label}")

    golden.to_csv(GOLDEN_PATH, index=False, encoding="utf-8-sig")
    print(f"\nChanged {changed} labels. New counts:")
    print(golden["label"].value_counts().to_string())


if __name__ == "__main__":
    main()