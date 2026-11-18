"""
main.py  (FastAPI service)
The backend for the triage system. It is the ONLY place a draft can be
approved or rejected; the agent itself has no way to do that.

Run:  uv run uvicorn triage.api.main:app --port 8000
Docs: http://localhost:8000/docs
"""

import json
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from triage.agent.graph import triage_complaint
from triage.api import db

app = FastAPI(
    title="Consumer Dispute Triage API",
    description="An AI agent drafts responses to credit-reporting complaints; "
                "a human reviewer approves or rejects every draft.",
    version="0.1.0",
)

Status = Literal["pending_approval", "approved", "rejected"]


# ---- Request formats (FastAPI rejects anything that doesn't match) ------
class TriageRequest(BaseModel):
    complaint: str = Field(min_length=20, max_length=10000,
                           description="The consumer's complaint text.")


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=100, description="Who is reviewing.")
    note: str = Field(default="", max_length=2000, description="Optional reason.")


# ---- Endpoints ---------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Simple check that the service is running."""
    return {"status": "ok"}


@app.post("/triage")
async def triage(request: TriageRequest) -> dict:
    """Run the agent on a complaint. Any draft it creates is saved as
    'pending_approval', and the agent's trace is stored in the audit log."""
    try:
        result = await triage_complaint(request.complaint)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"The agent failed: {error}")

    if result.get("draft_id"):
        db.record_agent_trace(result["draft_id"], result.get("trace", []))

    fields = ("draft_id", "status", "category", "key_facts",
              "draft_response", "citations", "trace")
    return {field: result.get(field) for field in fields}


@app.get("/drafts")
def list_drafts(status: Status | None = None) -> list[dict]:
    """List drafts, newest first. Optionally filter by status."""
    return db.list_drafts(status)


@app.get("/drafts/{draft_id}")
def get_draft(draft_id: int) -> dict:
    """One draft, with the agent's trace and the full audit log."""
    draft = db.get_draft(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found.")

    audit_log = db.get_audit_log(draft_id)
    for entry in audit_log:
        entry["details"] = json.loads(entry["details"]) if entry["details"] else {}
    trace = next((entry["details"].get("trace", []) for entry in reversed(audit_log)
                  if entry["action"] == "agent_trace"), [])
    return {"draft": draft, "trace": trace, "audit_log": audit_log}


@app.post("/drafts/{draft_id}/review")
def review_draft(draft_id: int, request: ReviewRequest) -> dict:
    """A human approves or rejects a pending draft. The ONLY way to change a status."""
    if db.get_draft(draft_id) is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found.")
    try:
        return db.review_draft(draft_id, request.decision, request.reviewer, request.note)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))