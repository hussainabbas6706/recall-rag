"""Pydantic models defining Recall's public API contract."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "recall"


class IngestRequest(BaseModel):
    vault_path: str | None = Field(
        default=None,
        description="Optional local vault path. Defaults to VAULT_PATH.",
    )


class IngestResponse(BaseModel):
    status: str
    files_found: int
    message: str
