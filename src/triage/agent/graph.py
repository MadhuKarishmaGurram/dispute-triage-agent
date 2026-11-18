"""
graph.py
The triage agent, built as a LangGraph flowchart:

  classify -> research <-> tools -> collect_evidence -> draft -> check -> submit
                                                          ^        |
                                                          +- fix --+   (at most 2 retries)
                                                                   +-> needs_human

Only RESEARCH lets Gemini choose what to do, and only among read-only tools.
COLLECT_EVIDENCE, CHECK and SUBMIT are plain code with fixed rules.
"""

import json
import os
import sys

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from triage.agent.prompts import CLASSIFY_PROMPT, DRAFT_PROMPT, RESEARCH_PROMPT, REVISION_NOTES
from triage.agent.state import Classification, Draft, TriageState
from triage.gcp import make_chat_model, message_text
from triage.guardrails.pii import check_draft

MAX_TOOL_CALLS = 4        # research may call tools at most this many times
MAX_DRAFT_ATTEMPTS = 3    # the first draft + 2 retries
READ_ONLY_TOOLS = {
    "search_similar_complaints", "get_category_stats",
    "search_regulations", "list_regulation_sections",
}


def _server(name: str) -> dict:
    """How to start one of our MCP servers (same way as the test client)."""
    return {
        "command": sys.executable,
        "args": ["-m", f"triage.mcp_servers.{name}_server"],
        "transport": "stdio",
        "env": dict(os.environ),
    }


def _json_items(content) -> list:
    """Tool results arrive as JSON text. Turn them back into Python dicts,
    whatever exact shape (a string, or a list of text blocks) they come in."""
    pieces = []
    if isinstance(content, str):
        pieces = [content]
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict) and "text" in block:
                pieces.append(block["text"])
    items = []
    for piece in pieces:
        try:
            value = json.loads(piece)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("result"), list):
            value = value["result"]
        items.extend(value if isinstance(value, list) else [value])
    return items


def _passages_from(content) -> list[dict]:
    """Keep only the items that look like FCRA passages (they have a citation)."""
    return [item for item in _json_items(content) if isinstance(item, dict) and item.get("citation")]


def _bullets(items) -> str:
    """Format a list as '- item' lines for a prompt."""
    return "\n".join(f"- {item}" for item in items) or "- (none)"


async def build_agent():
    """Connect to the 3 MCP servers, load their tools, and build the graph."""
    client = MultiServerMCPClient({name: _server(name) for name in ("complaints", "regulations", "actions")})
    tools = await client.get_tools()
    by_name = {tool.name: tool for tool in tools}
    read_tools = [tool for tool in tools if tool.name in READ_ONLY_TOOLS]
    search_regulations = by_name["search_regulations"]
    submit_tool = by_name["submit_draft_for_approval"]

    model = make_chat_model()
    classifier = model.with_structured_output(Classification)  # must answer in this format
    researcher = model.bind_tools(read_tools)                    # may call ONLY read-only tools
    drafter = model.with_structured_output(Draft)

    # ---- Nodes (the boxes of the flowchart) ----------------------------

    async def classify(state: TriageState) -> dict:
        result = await classifier.ainvoke(CLASSIFY_PROMPT.format(complaint=state["complaint"]))
        return {
            "category": result.category,
            "key_facts": result.key_facts,
            "trace": [f"classify: {result.category} - {result.reasoning}"],
        }

    async def research(state: TriageState) -> dict:
        messages = state.get("messages", [])
        new_messages = []
        if not messages:  # first time: give Gemini its research instructions
            new_messages.append(HumanMessage(RESEARCH_PROMPT.format(
                category=state["category"],
                key_facts=_bullets(state["key_facts"]),
                complaint=state["complaint"],
            )))
        response = await researcher.ainvoke(messages + new_messages)
        trace = [f"research: calls {call['name']}({json.dumps(call['args'])})"
                 for call in response.tool_calls]
        if not response.tool_calls:
            trace.append("research: finished - " + message_text(response)[:200])
        return {"messages": new_messages + [response], "trace": trace}

    async def collect_evidence(state: TriageState) -> dict:
        passages = []
        for message in state.get("messages", []):
            if message.type == "tool" and message.name == "search_regulations":
                passages.extend(_passages_from(message.content))
        trace = []
        if not passages:  # safety net: the law search is required, so run it ourselves
            query = "consumer rights and agency duties: " + "; ".join(state["key_facts"])
            passages = _passages_from(await search_regulations.ainvoke({"query": query, "limit": 3}))
            trace.append("collect_evidence: research found no law, so ran a fallback FCRA search")
        unique = {}
        for passage in passages:  # remove duplicates found by repeated searches
            unique.setdefault(passage["text"], passage)
        passages = list(unique.values())
        citations = sorted({p["citation"] for p in passages})
        trace.append(f"collect_evidence: {len(passages)} FCRA passages ({', '.join(citations)})")
        return {"passages": passages, "trace": trace}

    async def draft(state: TriageState) -> dict:
        attempts = state.get("draft_attempts", 0) + 1
        passages_text = "\n\n".join(
            f"[{p['citation']}] {p['title']}\n{p['text']}" for p in state["passages"])
        allowed = sorted({p["citation"] for p in state["passages"]})
        revision = (REVISION_NOTES.format(problems=_bullets(state["problems"]))
                    if state.get("problems") else "")
        result = await drafter.ainvoke(DRAFT_PROMPT.format(
            category=state["category"],
            key_facts=_bullets(state["key_facts"]),
            complaint=state["complaint"],
            passages=passages_text or "(none found)",
            allowed_citations=_bullets(allowed),
            revision_notes=revision,
        ))
        return {
            "draft_response": result.response,
            "citations": result.citations,
            "draft_attempts": attempts,
            "trace": [f"draft: attempt {attempts}, cites {', '.join(result.citations) or 'nothing'}"],
        }

    def check(state: TriageState) -> dict:
        allowed = {p["citation"] for p in state.get("passages", [])}
        problems = check_draft(state["draft_response"], state["citations"], allowed)
        if problems:
            line = f"check: {len(problems)} problem(s) - " + " | ".join(problems)
        else:
            line = "check: passed all guardrails"
        return {"problems": problems, "trace": [line]}

    async def submit(state: TriageState) -> dict:
        try:
            result = await submit_tool.ainvoke({
                "complaint": state["complaint"],
                "category": state["category"],
                "draft_response": state["draft_response"],
                "citations": state["citations"],
            })
        except Exception as error:
            return {"draft_id": None, "status": "submit_failed",
                    "trace": [f"submit: FAILED - {error}"]}
        saved = next((item for item in _json_items(result)
                      if isinstance(item, dict) and "draft_id" in item), {})
        return {
            "draft_id": saved.get("draft_id"),
            "status": saved.get("status", "unknown"),
            "trace": [f"submit: saved draft #{saved.get('draft_id')} as {saved.get('status')}"],
        }

    def needs_human(state: TriageState) -> dict:
        return {"draft_id": None, "status": "needs_human_review",
                "trace": [f"needs_human: still failing checks after {MAX_DRAFT_ATTEMPTS} drafts; "
                          "not saved, flagged for a person"]}

    # ---- Routing (the arrows that make decisions) ----------------------

    def after_research(state: TriageState) -> str:
        last = state["messages"][-1]
        tool_calls_made = sum(1 for m in state["messages"] if m.type == "tool")
        if getattr(last, "tool_calls", None) and tool_calls_made < MAX_TOOL_CALLS:
            return "tools"
        return "collect_evidence"

    def after_check(state: TriageState) -> str:
        if not state["problems"]:
            return "submit"
        if state["draft_attempts"] < MAX_DRAFT_ATTEMPTS:
            return "draft"
        return "needs_human"

    # ---- Wire the flowchart together -----------------------------------
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify)
    graph.add_node("research", research)
    graph.add_node("tools", ToolNode(read_tools))
    graph.add_node("collect_evidence", collect_evidence)
    graph.add_node("draft", draft)
    graph.add_node("check", check)
    graph.add_node("submit", submit)
    graph.add_node("needs_human", needs_human)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "research")
    graph.add_conditional_edges("research", after_research, ["tools", "collect_evidence"])
    graph.add_edge("tools", "research")
    graph.add_edge("collect_evidence", "draft")
    graph.add_edge("draft", "check")
    graph.add_conditional_edges("check", after_check, ["submit", "draft", "needs_human"])
    graph.add_edge("submit", END)
    graph.add_edge("needs_human", END)
    return graph.compile()


async def triage_complaint(complaint: str) -> dict:
    """Run the whole agent on one complaint and return the final state."""
    agent = await build_agent()
    return await agent.ainvoke(
        {"complaint": complaint, "trace": []},
        {"recursion_limit": 30},  # hard stop on total steps, as a last safety net
    )