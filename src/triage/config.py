from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load runtime settings from the environment and a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gcp_project: str
    gcp_region: str
    gemini_model: str
    embedding_model: str
    qdrant_url: str
