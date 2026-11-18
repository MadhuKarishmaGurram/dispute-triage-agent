"""
gcp.py
Creates our connections to Gemini on Vertex AI (Google Cloud):
  - make_genai_client(): Google's own client, used for embeddings
  - make_chat_model():   LangChain's Gemini chat model, used by the agent

Why this file exists: when our MCP server ran hidden in the background on
Windows, Google's default login lookup tried to run the 'gcloud' command to
find the project, and that hidden command got stuck. Here we load the saved
login file directly and pass the project ourselves, so 'gcloud' never runs.
Every part of the project that talks to Gemini uses this file.

Locations: embeddings use GCP_REGION (us-central1). The chat model uses
GEMINI_LOCATION (global), because newer Gemini models such as
gemini-3.5-flash are offered on the global endpoint, not in us-central1.
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


def _project() -> str:
    """Read the Google Cloud project ID from the environment (.env)."""
    project = os.getenv("GCP_PROJECT")
    if not project:
        raise ValueError("GCP_PROJECT is missing from .env.")
    return project


def _load_credentials():
    """Load the saved Google login directly from the file (no gcloud command)."""
    path = _login_file_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No Google login file at {path}. "
            "Run: gcloud auth application-default login"
        )
    credentials, _ = google.auth.load_credentials_from_file(str(path), scopes=SCOPES)
    return credentials


def make_genai_client() -> genai.Client:
    """Google's own Gemini client on Vertex AI (used for EMBEDDINGS).
    Location: GCP_REGION from .env (us-central1)."""
    return genai.Client(
        vertexai=True,
        project=_project(),
        location=os.getenv("GCP_REGION", "us-central1"),
        credentials=_load_credentials(),
    )


def make_chat_model():
    """LangChain's Gemini chat model on Vertex AI (used by the AGENT).
    Model: GEMINI_MODEL from .env. Location: GEMINI_LOCATION from .env (global)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        vertexai=True,
        project=_project(),
        location=os.getenv("GEMINI_LOCATION", "global"),
        credentials=_load_credentials(),
    )


def message_text(message) -> str:
    """Get the plain text out of a chat model's reply. Newer Gemini models can
    return a list of content blocks instead of one string, so handle both."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )