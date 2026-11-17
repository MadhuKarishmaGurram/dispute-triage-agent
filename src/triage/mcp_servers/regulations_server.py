"""
regulations_server.py  (MCP server)

Gives the agent two READ-ONLY tools about the Fair Credit Reporting Act (FCRA):
  1. search_regulations       - find the law passages most relevant to a question
  2. list_regulation_sections - list which FCRA sections are available to cite

The law text comes from govinfo.gov (official U.S. Code) and was loaded into
Qdrant by scripts/build_regulations.py.

Like the complaints server, this talks over stdio, so it never prints to
the screen; it logs to stderr and to logs/regulations_server.log.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

# ---- Settings ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "regulations.jsonl"
LOG_PATH = PROJECT_ROOT / "logs" / "regulations_server.log"
load_dotenv(PROJECT_ROOT / ".env")

from triage.gcp import make_genai_client  # after load_dotenv so it sees .env

COLLECTION = "regulations"
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
genai_client = make_genai_client()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
log("Google and Qdrant clients created.")

mcp = FastMCP("regulations")


def _run_search(query: str, limit: int) -> list[dict]:
    """Blocking work: embed the question, then find the closest law passages."""
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

    log("[search] asking Qdrant for law passages...")
    hits = qdrant.query_points(COLLECTION, query=query_vector, limit=limit).points
    log(f"[search] Qdrant returned {len(hits)} passages.")

    return [
        {
            "citation": hit.payload.get("citation"),
            "title": hit.payload.get("title"),
            "similarity": round(hit.score, 3),
            "text": hit.payload.get("text"),
            "source_url": hit.payload.get("source_url"),
        }
        for hit in hits
    ]


# ---- Tool 1 ------------------------------------------------------------
@mcp.tool()
async def search_regulations(query: str, limit: int = 3) -> list[dict]:
    """Find the passages of the Fair Credit Reporting Act (FCRA) most relevant
    to a question, such as 'deadline to investigate a disputed item' or
    'who is allowed to pull a credit report'.

    Use this BEFORE stating any legal requirement, and cite only the
    citations returned here (for example '15 U.S.C. §1681i(a)'). Returns up
    to `limit` passages (maximum 8), each with its citation, section title,
    similarity score, full passage text, and official source URL.
    """
    limit = max(1, min(limit, 8))
    log(f"[search] called: '{query[:60]}' (limit={limit})")
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_search, query, limit),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log(f"[search] TIMED OUT after {SEARCH_TIMEOUT_SECONDS} seconds.")
        raise RuntimeError(
            f"Regulation search took longer than {SEARCH_TIMEOUT_SECONDS} seconds. "
            "See logs/regulations_server.log."
        )
    log("[search] returning passages to the client.")
    return results


# ---- Tool 2 ------------------------------------------------------------
@mcp.tool()
def list_regulation_sections() -> list[dict]:
    """List the FCRA sections available in this library, with their titles and
    the complaint category each one usually relates to. Use this to check
    what can be cited; do not cite sections that are not on this list.
    """
    log("[list] called")
    sections = {}
    with open(CHUNKS_PATH, encoding="utf-8") as chunks_file:
        for line in chunks_file:
            chunk = json.loads(line)
            sections.setdefault(chunk["section"], {
                "section": f"15 U.S.C. §{chunk['section']}",
                "title": chunk["title"],
                "related_category": chunk["related_category"],
            })
    log(f"[list] returning {len(sections)} sections.")
    return list(sections.values())


# ---- Start the server --------------------------------------------------
if __name__ == "__main__":
    log("Regulations MCP server starting (stdio)...")
    mcp.run()