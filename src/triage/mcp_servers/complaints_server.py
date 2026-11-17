"""
complaints_server.py  (MCP server)

Gives the agent two READ-ONLY tools about past consumer complaints:
  1. search_similar_complaints - find past complaints similar in meaning
  2. get_category_stats        - how common each complaint category is

This server talks to the agent over "stdio" (standard input/output).
That means it must NEVER use plain print() to the screen: anything printed
there would corrupt the messages. We log to stderr AND to a log file
(logs/complaints_server.log), so we can see what happened even when the
server runs hidden in the background.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

# ---- Settings ----------------------------------------------------------
# This file lives in src/triage/mcp_servers/, so the project root is 3 folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complaints.parquet"
LOG_PATH = PROJECT_ROOT / "logs" / "complaints_server.log"
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION = "complaints"
DIMENSIONS = 768
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
SEARCH_TIMEOUT_SECONDS = 30


def log(message: str) -> None:
    """Write a timestamped message to stderr and to the log file."""
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    print(line, file=sys.stderr)
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


log("Server file loading...")

# Connections are created once, when the server starts, and reused for every call.
from triage.gcp import make_genai_client
genai_client = make_genai_client()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
log("Google and Qdrant clients created.")

# Create the MCP server and give it a name.
mcp = FastMCP("complaints")


def _run_search(query: str, limit: int) -> list[dict]:
    """The actual search work: Google embedding, then Qdrant lookup.
    These are BLOCKING network calls, so the tool below runs this function
    in a separate worker thread to keep the server's event loop free."""
    log("[search] asking Google for the embedding...")
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONS,
        ),
    )
    query_vector = result.embeddings[0].values
    log("[search] got the embedding from Google.")

    log("[search] asking Qdrant for similar complaints...")
    hits = qdrant.query_points(COLLECTION, query=query_vector, limit=limit).points
    log(f"[search] Qdrant returned {len(hits)} results.")

    return [
        {
            "complaint_id": hit.payload.get("complaint_id"),
            "similarity": round(hit.score, 3),
            "category": hit.payload.get("category"),
            "company": hit.payload.get("company"),
            "narrative_excerpt": str(hit.payload.get("narrative", ""))[:500],
        }
        for hit in hits
    ]


# ---- Tool 1 ------------------------------------------------------------
@mcp.tool()
async def search_similar_complaints(query: str, limit: int = 3) -> list[dict]:
    """Find past consumer complaints that are similar in MEANING to the query.

    Use this to see how similar credit-reporting problems were described and
    categorized before, which helps classify a new complaint and draft a
    response. Returns up to `limit` complaints (maximum 10), each with a
    similarity score (higher = closer), its category, the company, and an
    excerpt of the consumer's narrative.
    """
    limit = max(1, min(limit, 10))  # keep the limit between 1 and 10
    log(f"[search] called: '{query[:60]}' (limit={limit})")

    # Run the blocking work in a worker thread, and give up after 30 seconds
    # so a slow or stuck network call can never freeze the whole server.
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_search, query, limit),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log(f"[search] TIMED OUT after {SEARCH_TIMEOUT_SECONDS} seconds.")
        raise RuntimeError(
            f"Search took longer than {SEARCH_TIMEOUT_SECONDS} seconds. "
            "See logs/complaints_server.log."
        )

    log("[search] returning results to the client.")
    return results


# ---- Tool 2 ------------------------------------------------------------
@mcp.tool()
def get_category_stats() -> dict:
    """Return how many past complaints fall into each category, with percentages.

    Categories: incorrect_information, investigation_problem, improper_use,
    fraud_and_identity, cannot_access_report, other. Use this for context on
    how common a type of complaint is.
    """
    log("[stats] called")
    df = pd.read_parquet(DATA_PATH, columns=["category"])
    counts = df["category"].value_counts()
    total = int(counts.sum())
    log("[stats] returning results to the client.")
    return {
        "total_complaints": total,
        "categories": {
            category: {"count": int(count), "percent": round(count / total * 100, 1)}
            for category, count in counts.items()
        },
    }


# ---- Start the server --------------------------------------------------
if __name__ == "__main__":
    log("Complaints MCP server starting (stdio)...")
    mcp.run()
