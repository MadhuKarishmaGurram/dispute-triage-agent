"""
clean_categories.py
Merges the CFPB's messy issue names into 6 clean categories.

Why: the CFPB renamed its categories over the years, so the same problem
appears under several names. Our agent needs ONE consistent label per
problem so we can classify complaints and measure accuracy.

Reads and updates: data/processed/complaints.parquet (adds a 'category' column)
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/complaints.parquet")

# Each CFPB issue name (left) -> our clean category (right).
CATEGORY_MAP = {
    # Wrong information on the credit report
    "Incorrect information on your report": "incorrect_information",
    "Incorrect information on credit report": "incorrect_information",

    # The company's investigation of a dispute was inadequate
    "Problem with a company's investigation into an existing problem": "investigation_problem",
    "Problem with a credit reporting company's investigation into an existing problem": "investigation_problem",
    "Problem with a company's investigation into an existing issue": "investigation_problem",
    "Credit reporting company's investigation": "investigation_problem",

    # Someone pulled or used the report without a valid reason
    "Improper use of your report": "improper_use",
    "Improper use of my credit report": "improper_use",

    # Fraud alerts, security freezes, identity theft, monitoring services
    "Problem with fraud alerts or security freezes": "fraud_and_identity",
    "Credit monitoring or identity theft protection services": "fraud_and_identity",
    "Credit monitoring or identity protection": "fraud_and_identity",
    "Identity theft protection or other monitoring services": "fraud_and_identity",
    "Fraud or scam": "fraud_and_identity",

    # Consumer couldn't get their own report or score
    "Unable to get your credit report or credit score": "cannot_access_report",
    "Unable to get credit report/credit score": "cannot_access_report",
}


def main() -> None:
    # STEP 1: Load the 5,000 complaints we saved earlier.
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} complaints.")

    # STEP 2: Look up each complaint's issue in the map.
    # Anything not in the map becomes "other".
    df["category"] = df["issue"].map(CATEGORY_MAP).fillna("other")

    # STEP 3: Show which issue names went to "other", so nothing
    # important gets hidden there by accident.
    unmapped = df.loc[~df["issue"].isin(CATEGORY_MAP.keys()), "issue"].value_counts()
    print("\nIssue names sent to 'other':")
    print(unmapped.to_string() if len(unmapped) else "  (none)")

    # STEP 4: Save the file back with the new 'category' column.
    df.to_parquet(DATA_PATH, index=False)

    # STEP 5: Print the final counts and percentages.
    counts = df["category"].value_counts()
    percents = (counts / len(df) * 100).round(1)
    print("\nComplaints per clean category:")
    for category, count in counts.items():
        print(f"  {category:<25} {count:>5}  ({percents[category]}%)")


if __name__ == "__main__":
    main()