"""
test_embedding.py
Embeds ONE sentence with Gemini on Vertex AI, to check that our
Google Cloud connection works before we process all 5,000 complaints.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

# STEP 1: Read our settings from the .env file.
load_dotenv()
project = os.getenv("GCP_PROJECT")
region = os.getenv("GCP_REGION", "us-central1")
model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

if not project:
    print("GCP_PROJECT is missing from .env. Please add it and try again.")
    sys.exit(1)

print(f"Using model {model} in region {region}...")

# STEP 2: Connect to Vertex AI (Google Cloud), using the login you set up
# earlier with 'gcloud auth application-default login'.
client = genai.Client(vertexai=True, project=project, location=region)

# STEP 3: Turn one sentence into an embedding.
# output_dimensionality=768 asks for 768 numbers instead of the default 3,072:
# smaller and faster to search, with only a small loss in quality.
sentence = "I paid off this debt but my credit report still shows it as unpaid."

try:
    result = client.models.embed_content(
        model=model,
        contents=sentence,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
except Exception as error:
    print("The embedding call failed. Here is the error:")
    print(error)
    sys.exit(1)

# STEP 4: Show what came back.
vector = result.embeddings[0].values
print(f"Success! The sentence became a list of {len(vector)} numbers.")
print("The first 5 numbers are:", [round(v, 4) for v in vector[:5]])