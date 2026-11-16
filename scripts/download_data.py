"""
download_data.py
Fetches credit-reporting complaints WITH consumer narratives from an archived
public snapshot of the CFPB Consumer Complaint Database on Hugging Face.

Why an archive? The CFPB removed narratives from its live database in
September 2026, so the current CFPB downloads no longer contain them.

Output: data/processed/complaints.parquet
"""

import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---- Settings ----------------------------------------------------------
DATASET_NAME = "buildpaths/complaint_Data_Finance"  # archived CFPB snapshot
TARGET_ROWS = 5000                                   # how many complaints we want
OUTPUT_PATH = Path("data/processed/complaints.parquet")

# Column names are cleaned to snake_case first (see clean_name below).
# This maps cleaned source names -> the names we use in our project.
COLUMNS = {
    "complaint_id": "complaint_id",
    "date_received": "date_received",
    "product": "product",
    "sub_product": "sub_product",
    "issue": "issue",
    "sub_issue": "sub_issue",
    "consumer_complaint_narrative": "narrative",
    "company": "company",
    "state": "state",
    "company_response_to_consumer": "company_response",
}


def clean_name(name: str) -> str:
    """Turn 'Consumer complaint narrative' or 'Sub-issue' into
    'consumer_complaint_narrative' / 'sub_issue', so small differences in
    how a dataset spells its columns don't break our code."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    # STEP 1: Connect in streaming mode.
    # Streaming reads rows one at a time over the internet, so we never
    # download the whole multi-gigabyte dataset.
    print(f"Connecting to {DATASET_NAME} (streaming)...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as error:
        print(f"Could not load the dataset: {error}")
        sys.exit(1)

    kept = []        # complaints that pass our filter
    rows_seen = 0    # total rows we looked at

    # STEP 2: Walk through the rows and keep only the ones we want.
    for raw_row in dataset:
        row = {clean_name(key): value for key, value in raw_row.items()}

        # On the very first row, show the columns and check for narratives.
        if rows_seen == 0:
            print("Columns found:", list(raw_row.keys()))
            if "consumer_complaint_narrative" not in row:
                print("No narrative column in this dataset. Stopping.")
                sys.exit(1)

        rows_seen += 1

        product = str(row.get("product") or "")
        narrative = str(row.get("consumer_complaint_narrative") or "").strip()

        # Keep it only if it's a credit-reporting complaint with a real narrative.
        is_credit_reporting = "credit reporting" in product.lower()
        has_narrative = narrative and narrative.lower() not in ("none", "nan")

        if is_credit_reporting and has_narrative:
            kept.append({new: row.get(old) for old, new in COLUMNS.items()})
            if len(kept) % 500 == 0:
                print(f"  kept {len(kept)} complaints (scanned {rows_seen} rows)")
            if len(kept) >= TARGET_ROWS:
                break

    if not kept:
        print(f"Scanned {rows_seen} rows but found no matching complaints.")
        sys.exit(1)

    # STEP 3: Save to a Parquet file (a compact, fast table format).
    df = pd.DataFrame(kept)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    # STEP 4: Print a summary so we can check the data makes sense.
    print("\n===== SUMMARY =====")
    print(f"Rows scanned:      {rows_seen}")
    print(f"Complaints saved:  {len(df)}  ->  {OUTPUT_PATH}")
    dates = df["date_received"].astype(str)
    print(f"Date range:        {dates.min()}  to  {dates.max()}")
    print("\nSample narrative (first 300 characters):")
    print("  " + str(df["narrative"].iloc[0])[:300])
    print("\nComplaints per issue:")
    print(df["issue"].value_counts().to_string())


if __name__ == "__main__":
    main()