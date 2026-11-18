"""
state.py
The agent's shared "notepad" (state), plus the exact formats Gemini must
answer in (structured output).

Each step of the graph READS what it needs from the state and WRITES its
results back. At any point we can print the state to see what the agent knew.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# The 6 categories from Phase 1. Gemini must pick exactly one of these.
Category = Literal[
    "incorrect_information",
    "investigation_problem",
    "improper_use",
    "fraud_and_identity",
    "cannot_access_report",
    "other",
]


# ---- The notepad that travels through the graph ------------------------
class TriageState(TypedDict, total=False):
    # Input
    complaint: str

    # Written by CLASSIFY
    category: Category
    key_facts: list[str]

    # Written by RESEARCH: Gemini's tool calls and the tools' results.
    # add_messages means new messages are ADDED to the list, not replacing it.
    messages: Annotated[list[AnyMessage], add_messages]

    # Written by COLLECT_EVIDENCE: the FCRA passages the draft may rely on
    passages: list[dict]

    # Written by DRAFT
    draft_response: str
    citations: list[str]
    draft_attempts: int

    # Written by CHECK: what is wrong with the draft (empty list = all good)
    problems: list[str]

    # Written by SUBMIT
    draft_id: int | None
    status: str

    # Written by EVERY step: a running log of what the agent did and why.
    # operator.add means each step's lines are APPENDED to the log.
    trace: Annotated[list[str], operator.add]


# ---- Formats Gemini must answer in (structured output) -----------------
class Classification(BaseModel):
    """What the CLASSIFY step must return."""
    category: Category = Field(description="The single best category for this complaint.")
    key_facts: list[str] = Field(
        description="2 to 5 short, specific facts from the complaint that matter "
                    "for the response (dates, amounts, what the consumer says is wrong).")
    reasoning: str = Field(description="One sentence explaining why this category fits.")


class Draft(BaseModel):
    """What the DRAFT step must return."""
    response: str = Field(
        description="The proposed response to the consumer: polite, plain English, "
                    "120 to 250 words. Must not promise any outcome.")
    citations: list[str] = Field(
        description="Every FCRA citation the response relies on, copied EXACTLY "
                    "as it appeared in the research results, e.g. '15 U.S.C. §1681i(a)'.")