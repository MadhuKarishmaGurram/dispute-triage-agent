"""
test_db.py
Tests that try to BREAK the draft-approval safety rules.
Each test passes only if the system blocks the unsafe action.

Run: uv run pytest tests -v
"""

import sqlite3

import pytest

from triage.api import db


@pytest.fixture(autouse=True)
def temporary_database(tmp_path, monkeypatch):
    """Point the code at a throwaway database for every test,
    so the real data/triage.db is never touched."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


def make_draft() -> dict:
    """Create a sample draft the way the agent would."""
    return db.create_draft(
        complaint="I paid this collection but it still shows as unpaid.",
        category="incorrect_information",
        draft_response="Thank you for contacting us. We are reinvestigating ...",
        citations=["15 U.S.C. §1681i(a)"],
    )


def test_new_draft_is_always_pending():
    draft = make_draft()
    assert draft["status"] == "pending_approval"


def test_agent_cannot_choose_status():
    # create_draft has no 'status' option, so trying to pass one fails.
    with pytest.raises(TypeError):
        db.create_draft("complaint", "improper_use", "response", [], status="approved")


def test_database_rejects_invalid_status():
    # Even raw SQL that bypasses our functions cannot set an unknown status.
    draft = make_draft()
    connection = db._connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE drafts SET status = 'sent' WHERE id = ?", (draft["id"],))
    finally:
        connection.close()


def test_invalid_decision_is_rejected():
    draft = make_draft()
    with pytest.raises(ValueError):
        db.review_draft(draft["id"], decision="sent", reviewer="alex")
    assert db.get_draft(draft["id"])["status"] == "pending_approval"


def test_reviewer_name_is_required():
    draft = make_draft()
    with pytest.raises(ValueError):
        db.review_draft(draft["id"], decision="approved", reviewer="   ")


def test_draft_cannot_be_reviewed_twice():
    draft = make_draft()
    db.review_draft(draft["id"], decision="approved", reviewer="alex")
    with pytest.raises(ValueError):
        db.review_draft(draft["id"], decision="rejected", reviewer="sam")
    assert db.get_draft(draft["id"])["status"] == "approved"


def test_actions_are_recorded_in_audit_log():
    draft = make_draft()
    db.review_draft(draft["id"], decision="approved", reviewer="alex", note="Looks correct")
    log = db.get_audit_log(draft["id"])
    assert [entry["action"] for entry in log] == ["draft_created", "draft_approved"]
    assert [entry["actor"] for entry in log] == ["agent", "human:alex"]
