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


class SearchResult(BaseModel):
    chunk: Chunk
    score: float = Field(description="Cosine similarity score between query and chunk vector")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of chunks to return")
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity score cutoff",
    )


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
