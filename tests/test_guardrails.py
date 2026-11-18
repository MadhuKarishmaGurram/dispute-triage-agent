"""
test_guardrails.py
Tests for the plain-code guardrails in pii.py. No AI involved, so these are
instant, free, and run on every push in CI.

Each rule is tested in BOTH directions: bad drafts are caught, and good
drafts are not flagged (a guardrail that blocks everything is useless).

Run: uv run pytest tests -v
"""

from triage.guardrails.pii import check_draft, find_pii

ALLOWED = {"15 U.S.C. §1681i(a)", "15 U.S.C. §1681b(a)"}

# A realistic draft that should pass every rule (about 130 words).
CLEAN_DRAFT = (
    "Thank you for contacting us about the collection account on your credit report. "
    "We understand you paid this account in full in March and that the report still shows "
    "a balance. Under federal law, when a consumer disputes the accuracy of information in "
    "their file, the consumer reporting agency must conduct a free, reasonable "
    "reinvestigation, generally within 30 days of receiving the dispute. As part of that "
    "process, the company that reported the account is asked to verify the information. "
    "If the information is found to be inaccurate or cannot be verified, the law requires "
    "that it be corrected or deleted. You will receive written notice of the results when "
    "the reinvestigation is complete. You may also add a brief statement to your file "
    "explaining the dispute. Please keep your payment receipt, as it may help the review."
)


def test_clean_draft_passes():
    assert check_draft(CLEAN_DRAFT, ["15 U.S.C. §1681i(a)"], ALLOWED) == []


# ---- Personal data ------------------------------------------------------
def test_ssn_is_caught():
    assert find_pii("My SSN is 123-45-6789.")


def test_long_account_number_is_caught():
    assert find_pii("Account number 4111111111111111 is wrong.")


def test_ordinary_numbers_are_not_flagged():
    # Dates, years, day counts and short numbers are normal in responses.
    assert find_pii("On 03/15/2026 we received it and will reply within 30 days, by 2026.") == []


def test_pii_in_draft_is_reported():
    problems = check_draft(CLEAN_DRAFT + " Your SSN 123-45-6789 is on file.",
                           ["15 U.S.C. §1681i(a)"], ALLOWED)
    assert any("Personal data" in p for p in problems)


# ---- Promises -------------------------------------------------------------
def test_promise_of_removal_is_caught():
    draft = CLEAN_DRAFT + " This item will be removed from your report."
    assert any("promises an outcome" in p for p in check_draft(draft, ["15 U.S.C. §1681i(a)"], ALLOWED))


def test_promise_we_will_delete_is_caught():
    draft = CLEAN_DRAFT + " We will delete the account this week."
    assert any("promises an outcome" in p for p in check_draft(draft, ["15 U.S.C. §1681i(a)"], ALLOWED))


def test_conditional_legal_duty_is_allowed():
    # Describing what the LAW requires, conditionally, is not a promise.
    # (CLEAN_DRAFT already contains "it be corrected or deleted".)
    assert check_draft(CLEAN_DRAFT, ["15 U.S.C. §1681i(a)"], ALLOWED) == []


# ---- Length -------------------------------------------------------------
def test_too_short_is_caught():
    problems = check_draft("Thanks, we are looking into it.", ["15 U.S.C. §1681i(a)"], ALLOWED)
    assert any("too short" in p for p in problems)


def test_too_long_is_caught():
    problems = check_draft(CLEAN_DRAFT * 4, ["15 U.S.C. §1681i(a)"], ALLOWED)
    assert any("too long" in p for p in problems)


# ---- Citations ------------------------------------------------------------
def test_missing_citations_are_caught():
    assert any("no citations" in p for p in check_draft(CLEAN_DRAFT, [], ALLOWED))


def test_citation_not_retrieved_is_caught():
    # Real FCRA section, but it was NOT in this complaint's research results.
    problems = check_draft(CLEAN_DRAFT, ["15 U.S.C. §1681e(b)"], ALLOWED)
    assert any("not in the research results" in p for p in problems)


def test_invented_citation_is_caught():
    problems = check_draft(CLEAN_DRAFT, ["15 U.S.C. §1681z(a)"], ALLOWED)
    assert any("not in the research results" in p for p in problems)
