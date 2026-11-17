"""
build_index.py
Turns each complaint's narrative into an embedding (768 numbers that capture
its meaning) and stores it in Qdrant, so the agent can later find similar
complaints by meaning instead of by matching words.

Small test first:  uv run python scripts/build_index.py --limit 50
Full run:          uv run python scripts/build_index.py
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ---- Settings ----------------------------------------------------------
DATA_PATH = "data/processed/complaints.parquet"
COLLECTION = "complaints"   # Qdrant's name for a table of vectors
DIMENSIONS = 768            # numbers per embedding (same as our test)
MAX_CHARS = 6000            # trim very long narratives to stay within model limits
WORKERS = 4                 # how many embedding requests run at the same time
UPLOAD_BATCH = 100          # how many complaints we send to Qdrant at once


def embed_one(client, model, text):
    """Embed one complaint. If Google rejects the request (for example a
    temporary rate limit), wait and retry: 1s, 2s, 4s, 8s, 16s."""
    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model=model,
                contents=text[:MAX_CHARS],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",  # "this text is something to be found"
                    output_dimensionality=DIMENSIONS,
                ),
            )
            return result.embeddings[0].values
        except Exception as error:
            wait = 2 ** attempt
            print(f"  request failed ({str(error)[:80]}...) - retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Embedding failed 5 times in a row. Stopping.")


def main() -> None:
    # STEP 1: Read the --limit option and our settings from .env.
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only process this many complaints (for testing)")
    args = parser.parse_args()

    load_dotenv()
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION", "us-central1")
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if not project:
        print("GCP_PROJECT is missing from .env.")
        sys.exit(1)

    # STEP 2: Load the complaints (only the first N if --limit was given).
    df = pd.read_parquet(DATA_PATH)
    if args.limit:
        df = df.head(args.limit)
    records = df.to_dict(orient="records")
    print(f"Embedding {len(records)} complaints with {model}...")

    # STEP 3: Connect to Google (for embeddings) and Qdrant (for storage).
    genai_client = genai.Client(vertexai=True, project=project, location=region)
    qdrant = QdrantClient(url=qdrant_url)

    # STEP 4: Start with a fresh, empty collection every run, so running the
    # script twice never creates duplicates.
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
    )

    # STEP 5: Embed complaints 4 at a time, and upload them in groups of 100.
    start = time.time()
    points = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        vectors = pool.map(
            lambda record: embed_one(genai_client, model, str(record["narrative"])),
            records,
        )
        for number, (record, vector) in enumerate(zip(records, vectors), start=1):
            # The "payload" is the complaint's details, stored next to its vector.
            payload = {key: (None if pd.isna(value) else str(value))
                       for key, value in record.items()}
            points.append(PointStruct(id=number, vector=vector, payload=payload))

            if len(points) >= UPLOAD_BATCH:
                qdrant.upsert(COLLECTION, points=points)
                points = []
                print(f"  {number}/{len(records)} stored ({time.time() - start:.0f}s)")

    if points:  # upload whatever is left over
        qdrant.upsert(COLLECTION, points=points)

    # STEP 6: Confirm how many complaints are now in Qdrant.
    stored = qdrant.count(COLLECTION).count
    print("\n===== DONE =====")
    print(f"Complaints stored in Qdrant: {stored}")
    print(f"Time taken: {time.time() - start:.0f} seconds")


if __name__ == "__main__":
    main()