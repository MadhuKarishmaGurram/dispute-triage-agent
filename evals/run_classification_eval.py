"""
run_classification_eval.py
Measures how accurately the agent classifies complaints, against the
hand-reviewed golden set (evals/golden_set.csv).

Uses the SAME prompt and output format as the agent's classify step.
Reports overall accuracy, accuracy per category, a baseline to compare
against, and every miss with Gemini's reasoning.

Run: uv run python evals/run_classification_eval.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from triage.agent.prompts import CLASSIFY_PROMPT  # after load_dotenv
from triage.agent.state import Classification
from triage.gcp import make_chat_model

GOLDEN_PATH = Path("evals/golden_set.csv")
RESULTS_PATH = Path("evals/results/classification_results.json")


def load_golden_set() -> pd.DataFrame:
    """Read the golden set. Excel sometimes re-saves CSVs in an older text
    encoding, so fall back to that if the normal one fails."""
    try:
        return pd.read_csv(GOLDEN_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        print("(note: golden_set.csv was re-saved by Excel in an older encoding)")
        return pd.read_csv(GOLDEN_PATH, encoding="cp1252")


def classify(classifier, complaint: str):
    """Classify one complaint, retrying once if Google has a hiccup."""
    for attempt in range(2):
        try:
            return classifier.invoke(CLASSIFY_PROMPT.format(complaint=complaint))
        except Exception as error:
            if attempt == 1:
                raise
            print(f"  retrying after error: {str(error)[:80]}")
            time.sleep(3)


def main() -> None:
    golden = load_golden_set()
    classifier = make_chat_model().with_structured_output(Classification)
    print(f"Classifying {len(golden)} golden cases...\n")

    results = []
    start = time.time()
    for _, case in golden.iterrows():
        output = classify(classifier, case["complaint"])
        correct = output.category == case["label"]
        results.append({
            "case_id": case["case_id"],
            "expected": case["label"],
            "predicted": output.category,
            "correct": correct,
            "reasoning": output.reasoning,
        })
        print(f"  {'OK  ' if correct else 'MISS'}  {case['case_id']:<28} -> {output.category}")

    # ---- Scores ----------------------------------------------------------
    total = len(results)
    correct_count = sum(r["correct"] for r in results)

    per_category = defaultdict(lambda: [0, 0])  # category -> [correct, total]
    for r in results:
        per_category[r["expected"]][1] += 1
        per_category[r["expected"]][0] += r["correct"]

    # Baseline: always guess the most common label in the golden set.
    most_common = golden["label"].value_counts()
    baseline = most_common.iloc[0] / total

    print("\n===== CLASSIFICATION RESULTS =====")
    print(f"Overall accuracy: {correct_count}/{total} = {correct_count / total:.0%}")
    print(f"Baseline (always guess '{most_common.index[0]}'): {baseline:.0%}")
    print("\nPer category:")
    for category, (good, count) in sorted(per_category.items()):
        print(f"  {category:<25} {good}/{count}  ({good / count:.0%})")

    misses = [r for r in results if not r["correct"]]
    print(f"\nMisses ({len(misses)}):")
    for r in misses:
        print(f"  {r['case_id']}: expected {r['expected']}, got {r['predicted']}")
        print(f"     Gemini's reasoning: {r['reasoning']}")
    print(f"\nTime taken: {time.time() - start:.0f} seconds")

    # ---- Save, so future runs can be compared ----------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "accuracy": correct_count / total,
        "baseline": baseline,
        "per_category": {c: {"correct": g, "total": n} for c, (g, n) in per_category.items()},
        "cases": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()