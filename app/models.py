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
    chunks_created: int = 0
    chunks_indexed: int = 0
    message: str


class Chunk(BaseModel):
    source_path: str = Field(
        description="Relative path from the vault root, e.g. 'projects/recall.md'"
    )
    heading_path: list[str] = Field(
        default_factory=list, description="Hierarchy of headings, e.g. ['Recall', 'MVP']"
    )
    chunk_index: int = Field(description="Deterministic 0-based sequence number within the file")
    content: str = Field(description="Clean text content for this chunk")
