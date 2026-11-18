"""
pii.py  (guardrails)
Plain-code checks that run on every draft BEFORE it can be saved.
No AI is involved, so these rules can't be talked out of their decisions.

check_draft() returns a list of problems. An empty list means the draft passed.
Each problem is written as an instruction the agent can act on when it retries.
"""

import re

# A Social Security number pattern, like 123-45-6789
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 9 or more digits in a row: could be an account or card number
LONG_NUMBER = re.compile(r"\b\d{9,}\b")
# Promising an outcome, like "will be removed" or "we will correct"
PROMISE = re.compile(
    r"\bwill (?:be )?(?:remove|removed|delete|deleted|correct|corrected|fix|fixed)\b",
    re.IGNORECASE,
)
MIN_WORDS, MAX_WORDS = 60, 350


def find_pii(text: str) -> list[str]:
    """Return a description of each kind of personal data found in the text."""
    found = []
    if SSN.search(text):
        found.append("contains what looks like a Social Security number")
    if LONG_NUMBER.search(text):
        found.append("contains a long number that could be an account or card number")
    return found


def check_draft(response: str, citations: list[str], allowed_citations: set[str]) -> list[str]:
    """Check a draft against all rules. Returns a list of problems (empty = passed)."""
    problems = [f"Personal data: the response {issue}. Remove it." for issue in find_pii(response)]

    if PROMISE.search(response):
        problems.append('The response promises an outcome (e.g. "will be removed"). '
                        "Describe the investigation process instead.")

    words = len(response.split())
    if words < MIN_WORDS:
        problems.append(f"The response is too short ({words} words). Aim for 120 to 250 words.")
    if words > MAX_WORDS:
        problems.append(f"The response is too long ({words} words). Aim for 120 to 250 words.")

    if not citations:
        problems.append("There are no citations. Cite at least one allowed FCRA passage.")
    unknown = [c for c in citations if c not in allowed_citations]
    if unknown:
        problems.append(f"These citations were not in the research results: {unknown}. "
                        "Use only citations from the allowed list.")
    return problems