"""
app.py  (Streamlit review dashboard)
The screen a human compliance reviewer uses. It talks ONLY to the API,
never to the database directly, so the API's rules always apply.

Run (in a second terminal, while the API is running):
  uv run streamlit run src/triage/ui/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Dispute Triage Review", layout="wide")
st.title("Consumer Dispute Triage")
st.caption("An AI agent drafts each response. Nothing is sent until a human reviewer approves it.")


def api_get(path: str, **params):
    """Call a GET endpoint of our API and return its JSON."""
    response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def error_text(response) -> str:
    """Pull a readable error message out of an API error response."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text
    return detail if isinstance(detail, str) else str(detail)


# ---- Stop early with a clear message if the API isn't running ----------
try:
    api_get("/health")
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it first with: "
             "uv run uvicorn triage.api.main:app --port 8000")
    st.stop()


# ---- Sidebar: run the agent on a new complaint -------------------------
with st.sidebar:
    st.header("New complaint")
    complaint = st.text_area("Paste the consumer's complaint", height=220)
    if st.button("Run the agent", type="primary"):
        if len(complaint.strip()) < 20:
            st.error("Please paste a complaint of at least 20 characters.")
        else:
            with st.spinner("The agent is researching and drafting (about 30-60 seconds)..."):
                response = requests.post(f"{API_URL}/triage",
                                         json={"complaint": complaint}, timeout=300)
            if response.ok:
                result = response.json()
                if result.get("draft_id"):
                    st.success(f"Draft #{result['draft_id']} created and waiting for review.")
                else:
                    st.warning(f"No draft was saved (status: {result.get('status')}).")
                    for line in result.get("trace") or []:
                        st.text(line)
            else:
                st.error(error_text(response))


# ---- One draft, with everything a reviewer needs ------------------------
def show_draft(draft_id: int, reviewable: bool) -> None:
    details = api_get(f"/drafts/{draft_id}")
    draft = details["draft"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Complaint**")
        st.write(draft["complaint"])
        st.markdown(f"**Category:** `{draft['category']}`")
    with right:
        st.markdown("**Drafted response**")
        st.write(draft["draft_response"])
        st.markdown("**Citations:** " + ", ".join(draft["citations"]))

    with st.expander("Agent trace (what the agent did, step by step)"):
        if details["trace"]:
            for number, line in enumerate(details["trace"], start=1):
                st.text(f"{number}. {line}")
        else:
            st.caption("No trace was recorded for this draft.")

    with st.expander("Audit log (who did what, and when)"):
        st.dataframe(
            [{"time (UTC)": e["timestamp"], "who": e["actor"], "action": e["action"]}
             for e in details["audit_log"]],
            hide_index=True, use_container_width=True,
        )

    if not reviewable:
        return

    with st.form(f"review-{draft_id}"):
        reviewer = st.text_input("Your name")
        note = st.text_area("Note (required when rejecting)")
        approve_col, reject_col = st.columns(2)
        approve = approve_col.form_submit_button("Approve", type="primary")
        reject = reject_col.form_submit_button("Reject")

    if approve or reject:
        if not reviewer.strip():
            st.error("Please enter your name before reviewing.")
            return
        if reject and not note.strip():
            st.error("Please add a note explaining why you are rejecting this draft.")
            return
        decision = "approved" if approve else "rejected"
        response = requests.post(
            f"{API_URL}/drafts/{draft_id}/review",
            json={"decision": decision, "reviewer": reviewer.strip(), "note": note.strip()},
            timeout=30,
        )
        if response.ok:
            st.rerun()  # reload the page so the draft moves to the Reviewed tab
        else:
            st.error(error_text(response))


# ---- Main area: two tabs ------------------------------------------------
pending_tab, reviewed_tab = st.tabs(["Pending review", "Reviewed"])

with pending_tab:
    pending = api_get("/drafts", status="pending_approval")
    if not pending:
        st.info("No drafts are waiting for review. Run the agent from the sidebar.")
    for item in pending:
        with st.container(border=True):
            st.subheader(f"Draft #{item['id']}  ·  {item['category']}")
            st.caption(f"Created {item['created_at']} (UTC)")
            show_draft(item["id"], reviewable=True)

with reviewed_tab:
    reviewed = api_get("/drafts", status="approved") + api_get("/drafts", status="rejected")
    if not reviewed:
        st.info("No drafts have been reviewed yet.")
    for item in sorted(reviewed, key=lambda d: d["id"], reverse=True):
        with st.container(border=True):
            st.subheader(f"Draft #{item['id']}  ·  {item['status'].upper()} by {item['reviewer']}")
            if item.get("review_note"):
                st.caption(f"Note: {item['review_note']}")
            show_draft(item["id"], reviewable=False)