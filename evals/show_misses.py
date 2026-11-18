"""
show_misses.py
Prints every complaint the classifier got wrong in the last evaluation run,
with the human label and Gemini's answer side by side, for adjudication.

Run: uv run python evals/show_misses.py
"""

import json
from pathlib import Path

import pandas as pd

golden = pd.read_csv("evals/golden_set.csv", encoding="utf-8-sig").set_index("case_id")
results = json.loads(Path("evals/results/classification_results.json").read_text(encoding="utf-8"))

for case in results["cases"]:
    if case["correct"]:
        continue
    print("=" * 80)
    print(f"{case['case_id']}   |   your label: {case['expected']}   |   Gemini: {case['predicted']}")
    print("-" * 80)
    print(golden.loc[case["case_id"], "complaint"])
    print(f"\nGemini's reasoning: {case['reasoning']}\n")