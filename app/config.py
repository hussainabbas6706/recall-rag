"""Configuration loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Recall."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str | None = Field(default=None, repr=False)
    vault_path: Path = Path("tests/fixtures/vault")
    qdrant_url: HttpUrl = "http://localhost:6333"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=16, ge=1, le=16)
    local_embedding_collection_prefix: str = "recall_local"
    answer_model: str = "gemini-3.8-flash"
    top_k: int = Field(default=5, ge=1, le=20)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so every service uses one consistent configuration."""
    return Settings()
