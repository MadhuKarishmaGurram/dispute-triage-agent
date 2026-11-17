"""
build_regulations.py
Downloads key sections of the Fair Credit Reporting Act (FCRA) from
govinfo.gov (the official U.S. Code, published by the U.S. Government
Publishing Office), splits them into chunks, embeds each chunk, and stores
them in Qdrant so the agent can cite the real law.

Run: uv run python scripts/build_regulations.py
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()
from triage.gcp import make_genai_client  # imported after load_dotenv so it sees .env

# ---- Settings ----------------------------------------------------------
# FCRA sections we need, and the complaint category each one relates to.
SECTIONS = {
    "1681b": "improper_use",            # permissible purposes of consumer reports
    "1681c": "incorrect_information",   # what information may be reported, and for how long
    "1681c-1": "fraud_and_identity",    # fraud alerts
    "1681c-2": "fraud_and_identity",    # blocking information caused by identity theft
    "1681e": "incorrect_information",   # procedures to ensure accuracy
    "1681g": "cannot_access_report",    # consumers' right to see their file
    "1681i": "investigation_problem",   # disputes and the 30-day reinvestigation
    "1681j": "cannot_access_report",    # free credit reports
    "1681s-2": "incorrect_information", # duties of companies that report data
}
URL = "https://www.govinfo.gov/link/uscode/15/{section}?link-type=html"
HEADERS = {"User-Agent": "dispute-triage-agent (student portfolio project)"}

COLLECTION = "regulations"
DIMENSIONS = 768
MAX_CHUNK_CHARS = 1500
OUTPUT_PATH = Path("data/processed/regulations.jsonl")

# Lines that mark the END of the law text (history and notes follow them).
END_MARKERS = ("Editorial Notes", "Statutory Notes and Related Subsidiaries")

# A paragraph starts with a marker like (a), (1), (A) or (iv).
PARAGRAPH_START = re.compile(r"^\(([a-z]+|\d+|[A-Z]+)\)")
# Top-level subsections in these sections run (a) to (h). Note: "(i)" is
# treated as a roman numeral clause, not a subsection.
SUBSECTION_START = re.compile(r"^\(([a-h])\)")


def fetch_section(section: str) -> tuple[str, list[str]]:
    """Download one section and return (title, list of paragraphs)."""
    response = requests.get(URL.format(section=section), headers=HEADERS, timeout=30)
    response.raise_for_status()

    # Turn the web page into plain-text lines. The U.S. Code writes
    # '1681c-1' with a long dash, so we switch it to a normal hyphen.
    text = BeautifulSoup(response.text, "html.parser").get_text("\n")
    lines = [line.strip().replace("\u2013", "-") for line in text.splitlines()]
    lines = [line for line in lines if line and not line.isdigit()]  # drop footnote numbers

    # The law text starts at the heading line, e.g. "§1681i. Procedure in case ..."
    start = next(i for i, line in enumerate(lines) if line.startswith(f"§{section}."))
    title = lines[start].split(". ", 1)[-1]

    # ...and ends where the source credit or the notes begin.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("(Pub. L.") or lines[i] in END_MARKERS:
            end = i
            break

    # Web pages break text into lines in odd places, so rejoin them into
    # real paragraphs: a new paragraph starts with a marker like (a) or (1).
    paragraphs = []
    for line in lines[start + 1:end]:
        if PARAGRAPH_START.match(line) or not paragraphs:
            paragraphs.append(line)
        else:
            paragraphs[-1] += " " + line
    return title, paragraphs


def make_chunks(section: str, title: str, paragraphs: list[str], category: str) -> list[dict]:
    """Group paragraphs into chunks of about MAX_CHUNK_CHARS, never splitting
    a paragraph, and record which subsection each chunk starts in."""
    chunks, current, subsection = [], [], ""
    chunk_subsection = ""

    def close_chunk():
        if current:
            body = "\n".join(current)
            citation = f"15 U.S.C. §{section}{chunk_subsection}"
            chunks.append({
                "citation": citation,
                "section": section,
                "title": title,
                "related_category": category,
                "source_url": URL.format(section=section),
                # The title is included so the embedding knows the context.
                "text": f"{citation} — {title}\n{body}",
            })

    for paragraph in paragraphs:
        match = SUBSECTION_START.match(paragraph)
        if match:
            subsection = f"({match.group(1)})"
        if current and sum(len(p) for p in current) + len(paragraph) > MAX_CHUNK_CHARS:
            close_chunk()
            current = []
        if not current:
            chunk_subsection = subsection
        current.append(paragraph)
    close_chunk()
    return chunks


def embed(client, text: str) -> list[float]:
    """Embed one chunk, retrying with growing waits if Google has a hiccup."""
    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DIMENSIONS,
                ),
            )
            return result.embeddings[0].values
        except Exception as error:
            wait = 2 ** attempt
            print(f"  embedding failed ({str(error)[:60]}...) - retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Embedding failed 5 times in a row.")


def main() -> None:
    # STEP 1: Download, clean and chunk each section.
    all_chunks = []
    for section, category in SECTIONS.items():
        try:
            title, paragraphs = fetch_section(section)
        except Exception as error:
            print(f"§{section}: FAILED to download or read ({error}). Skipping.")
            continue
        chunks = make_chunks(section, title, paragraphs, category)
        all_chunks.extend(chunks)
        print(f"§{section} ({title}): {len(paragraphs)} paragraphs -> {len(chunks)} chunks")
        time.sleep(1)  # be polite to govinfo.gov

    if not all_chunks:
        print("No chunks were created. Stopping.")
        sys.exit(1)

    # STEP 2: Save the chunks to a readable file so we can inspect them.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk) + "\n")

    # STEP 3: Embed every chunk and store it in a fresh Qdrant collection.
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    genai_client = make_genai_client()
    qdrant = QdrantClient(url="http://localhost:6333")
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=number, vector=embed(genai_client, chunk["text"]), payload=chunk)
        for number, chunk in enumerate(all_chunks, start=1)
    ]
    qdrant.upsert(COLLECTION, points=points)

    # STEP 4: Summary.
    print("\n===== DONE =====")
    print(f"Law chunks stored in Qdrant: {qdrant.count(COLLECTION).count}")
    print(f"Readable copy saved to: {OUTPUT_PATH}")
    print("\nSample chunk (first 400 characters):")
    print(all_chunks[0]["text"][:400])


if __name__ == "__main__":
    main()