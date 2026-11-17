"""
actions_server.py  (MCP server)

Gives the agent its only WRITE tool, with strict limits:
  1. submit_draft_for_approval - saves a draft response as 'pending_approval'
  2. get_draft_status          - checks the status of a draft

Safety design:
  - The agent can PROPOSE a response but can never approve or send one.
    There is no approve tool, and db.review_draft is deliberately not imported.
  - Every citation must belong to an FCRA section that exists in our library,
    so the agent cannot save a draft with an invented legal citation.
  - Category must be one of our 6 known categories.

Logs go to stderr and logs/actions_server.log (never to stdout).
"""

import json
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from triage.api.db import create_draft, get_draft  # NOT review_draft, on purpose

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "regulations.jsonl"
LOG_PATH = PROJECT_ROOT / "logs" / "actions_server.log"

VALID_CATEGORIES = {
    "incorrect_information", "investigation_problem", "improper_use",
    "fraud_and_identity", "cannot_access_report", "other",
}


def log(message: str) -> None:
    """Write a timestamped message to stderr and to the log file."""
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    print(line, file=sys.stderr)
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def load_known_sections() -> set[str]:
    """Read which FCRA sections exist in our library, e.g. '15 U.S.C. §1681i'."""
    sections = set()
    with open(CHUNKS_PATH, encoding="utf-8") as chunks_file:
        for line in chunks_file:
            sections.add(f"15 U.S.C. §{json.loads(line)['section']}")
    return sections


KNOWN_SECTIONS = load_known_sections()
log(f"Loaded {len(KNOWN_SECTIONS)} known FCRA sections.")

mcp = FastMCP("actions")


def _is_known_citation(citation: str) -> bool:
    """True if the citation is a known section, optionally followed by a
    subsection like '(a)'. '§1681c-1' must not be mistaken for '§1681c'."""
    for section in KNOWN_SECTIONS:
        if citation == section or citation.startswith(section + "("):
            return True
    return False


# ---- Tool 1 ------------------------------------------------------------
@mcp.tool()
def submit_draft_for_approval(complaint: str, category: str, draft_response: str,
                              citations: list[str]) -> dict:
    """Save a drafted response to a consumer complaint FOR HUMAN REVIEW.

    This does NOT send anything to the consumer. The draft is stored with
    status 'pending_approval', and only a human reviewer can approve it.

    Arguments:
      complaint      - the consumer's complaint text
      category       - one of: incorrect_information, investigation_problem,
                       improper_use, fraud_and_identity, cannot_access_report, other
      draft_response - the proposed response to the consumer
      citations      - FCRA citations used, exactly as returned by
                       search_regulations, e.g. ["15 U.S.C. §1681i(a)"]

    If anything is invalid (unknown category, empty response, or a citation
    not in the FCRA library), nothing is saved and an error explains what to fix.
    """
    log(f"[submit] called: category={category}, citations={citations}")

    if category not in VALID_CATEGORIES:
        log("[submit] REJECTED: unknown category")
        raise ValueError(f"Unknown category '{category}'. Use one of: {sorted(VALID_CATEGORIES)}")
    if not draft_response.strip():
        log("[submit] REJECTED: empty response")
        raise ValueError("draft_response is empty. Write the proposed response first.")
    if not citations:
        log("[submit] REJECTED: no citations")
        raise ValueError("At least one citation is required. Use search_regulations to find one.")

    unknown = [c for c in citations if not _is_known_citation(c)]
    if unknown:
        log(f"[submit] REJECTED: unknown citations {unknown}")
        raise ValueError(
            f"These citations are not in the FCRA library: {unknown}. "
            "Only cite passages returned by search_regulations."
        )

    draft = create_draft(complaint, category, draft_response, citations)
    log(f"[submit] saved draft #{draft['id']} as {draft['status']}")
    return {
        "draft_id": draft["id"],
        "status": draft["status"],
        "message": "Draft saved for human review. It has NOT been sent.",
    }


# ---- Tool 2 ------------------------------------------------------------
@mcp.tool()
def get_draft_status(draft_id: int) -> dict:
    """Check the review status of a draft: pending_approval, approved, or rejected."""
    log(f"[status] called for draft #{draft_id}")
    draft = get_draft(draft_id)
    if draft is None:
        raise ValueError(f"Draft {draft_id} does not exist.")
    return {
        "draft_id": draft["id"],
        "status": draft["status"],
        "reviewer": draft["reviewer"],
        "review_note": draft["review_note"],
    }


# ---- Start the server --------------------------------------------------
if __name__ == "__main__":
    log("Actions MCP server starting (stdio)...")
    mcp.run()
