"""
db.py
The case database: stores the agent's draft responses and a full audit log.

Safety design:
  - create_draft() ALWAYS saves a draft as 'pending_approval'. There is no way
    to pass a different status, so the agent can never approve its own work.
  - review_draft() is the ONLY way to approve or reject a draft. It is meant
    for the human-facing API (Phase 4) and is never exposed as an agent tool.
  - Every change is written to the audit_log table: who did what, and when.

Storage: a single SQLite file at data/triage.db (SQLite is built into Python).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "triage.db"

VALID_DECISIONS = ("approved", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint      TEXT NOT NULL,
    category       TEXT NOT NULL,
    draft_response TEXT NOT NULL,
    citations      TEXT NOT NULL,          -- JSON list of law citations
    status         TEXT NOT NULL DEFAULT 'pending_approval'
                   CHECK (status IN ('pending_approval', 'approved', 'rejected')),
    created_at     TEXT NOT NULL,
    reviewed_at    TEXT,
    reviewer       TEXT,
    review_note    TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor     TEXT NOT NULL,               -- 'agent' or 'human:<name>'
    action    TEXT NOT NULL,               -- e.g. 'draft_created', 'draft_approved'
    draft_id  INTEGER,
    details   TEXT                         -- JSON with extra information
);
"""


def _now() -> str:
    """Current time in UTC, in a standard text format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """Open the database (creating it and its tables if needed)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row  # rows behave like dictionaries
    connection.executescript(SCHEMA)
    return connection


def _audit(connection, actor: str, action: str, draft_id: int | None, details: dict) -> None:
    """Record one action in the audit log."""
    connection.execute(
        "INSERT INTO audit_log (timestamp, actor, action, draft_id, details) "
        "VALUES (?, ?, ?, ?, ?)",
        (_now(), actor, action, draft_id, json.dumps(details)),
    )


# ---- Used by the agent (through the actions MCP server) ----------------

def create_draft(complaint: str, category: str, draft_response: str,
                 citations: list[str]) -> dict:
    """Save a new draft. Its status is ALWAYS 'pending_approval'."""
    connection = _connect()
    try:
        with connection:  # saves everything together, or nothing if an error happens
            cursor = connection.execute(
                "INSERT INTO drafts (complaint, category, draft_response, citations, "
                "status, created_at) VALUES (?, ?, ?, ?, 'pending_approval', ?)",
                (complaint, category, draft_response, json.dumps(citations), _now()),
            )
            draft_id = cursor.lastrowid
            _audit(connection, "agent", "draft_created", draft_id,
                   {"category": category, "citations": citations})
    finally:
        connection.close()
    return get_draft(draft_id)


def get_draft(draft_id: int) -> dict | None:
    """Return one draft as a dictionary, or None if it doesn't exist."""
    connection = _connect()
    try:
        row = connection.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    draft = dict(row)
    draft["citations"] = json.loads(draft["citations"])
    return draft


def list_drafts(status: str | None = None) -> list[dict]:
    """Return all drafts, newest first, optionally only those with one status."""
    connection = _connect()
    try:
        if status:
            rows = connection.execute(
                "SELECT id FROM drafts WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = connection.execute("SELECT id FROM drafts ORDER BY id DESC").fetchall()
    finally:
        connection.close()
    return [get_draft(row["id"]) for row in rows]


# ---- Used ONLY by humans (through the API in Phase 4) ------------------

def review_draft(draft_id: int, decision: str, reviewer: str, note: str = "") -> dict:
    """Approve or reject a pending draft. Never exposed to the agent."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Decision must be one of {VALID_DECISIONS}, not '{decision}'.")
    if not reviewer.strip():
        raise ValueError("A reviewer name is required.")

    draft = get_draft(draft_id)
    if draft is None:
        raise ValueError(f"Draft {draft_id} does not exist.")
    if draft["status"] != "pending_approval":
        raise ValueError(f"Draft {draft_id} was already {draft['status']}.")

    connection = _connect()
    try:
        with connection:
            connection.execute(
                "UPDATE drafts SET status = ?, reviewed_at = ?, reviewer = ?, review_note = ? "
                "WHERE id = ?",
                (decision, _now(), reviewer, note, draft_id),
            )
            _audit(connection, f"human:{reviewer}", f"draft_{decision}", draft_id,
                   {"note": note})
    finally:
        connection.close()
    return get_draft(draft_id)


def get_audit_log(draft_id: int | None = None) -> list[dict]:
    """Return audit log entries (for one draft, or all), oldest first."""
    connection = _connect()
    try:
        if draft_id is None:
            rows = connection.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM audit_log WHERE draft_id = ? ORDER BY id", (draft_id,)).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]