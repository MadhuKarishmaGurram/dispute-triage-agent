"""
test_search.py
Finds the complaints most similar in MEANING to a question.

Usage:
  uv run python scripts/test_search.py "your question here"
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

# STEP 1: Get the question (or use a default one).
question = " ".join(sys.argv[1:]) or "I paid off this debt but my credit report still shows it as unpaid."
print(f'Searching for: "{question}"\n')

# STEP 2: Read settings and connect to Google and Qdrant.
load_dotenv()
genai_client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_REGION", "us-central1"),
)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

# STEP 3: Turn the question into an embedding.
# RETRIEVAL_QUERY = "this text is a search question" (the complaints were
# embedded as RETRIEVAL_DOCUMENT = "things to be found").
result = genai_client.models.embed_content(
    model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
    contents=question,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
)
query_vector = result.embeddings[0].values

# STEP 4: Ask Qdrant for the 3 closest complaints.
hits = qdrant.query_points("complaints", query=query_vector, limit=3).points

# STEP 5: Show them. Similarity runs from 0 (unrelated) to 1 (identical meaning).
for rank, hit in enumerate(hits, start=1):
    details = hit.payload
    print(f"#{rank}  similarity: {hit.score:.3f}  |  category: {details['category']}")
    print(f"    company: {details['company']}")
    print("    " + str(details["narrative"])[:300].replace("\n", " ") + "...\n")