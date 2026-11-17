"""
gcp.py
Creates our connection to Gemini on Vertex AI (Google Cloud).

Why this file exists: when our MCP server ran hidden in the background on
Windows, Google's default login lookup tried to run the 'gcloud' command to
find the project, and that hidden command got stuck. Here we load the saved
login file directly and pass the project ourselves, so 'gcloud' never runs.
Every part of the project that talks to Gemini uses this one function.
"""

import os
from pathlib import Path

import google.auth
from google import genai

# Permission to use Google Cloud services (including Vertex AI).
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _login_file_path() -> Path:
    """Find the login file created by 'gcloud auth application-default login'."""
    explicit = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return Path(explicit)
    if os.name == "nt":  # Windows
        return Path(os.environ["APPDATA"]) / "gcloud" / "application_default_credentials.json"
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def make_genai_client() -> genai.Client:
    """Create a Gemini client on Vertex AI without calling the gcloud command."""
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_REGION", "us-central1")
    if not project:
        raise ValueError("GCP_PROJECT is missing from .env.")

    path = _login_file_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No Google login file at {path}. "
            "Run: gcloud auth application-default login"
        )

    # Load the saved login directly from the file (no gcloud command involved).
    credentials, _ = google.auth.load_credentials_from_file(str(path), scopes=SCOPES)

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
    )