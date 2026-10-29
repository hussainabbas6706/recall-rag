"""Pydantic models defining Recall's public API contract."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "recall"


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vault_id: UUID


class IngestFailure(BaseModel):
    source_path: str
    reason: str


class IngestResponse(BaseModel):
    status: str
    files_found: int
    chunks_created: int = 0
    chunks_indexed: int = 0
    files_indexed: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    failures: list[IngestFailure] = Field(default_factory=list)
    message: str


class VaultEntry(BaseModel):
    path: str = Field(description="Path relative to the configured vault root")
    kind: Literal["folder", "file"]


class VaultResponse(BaseModel):
    entries: list[VaultEntry]


class VaultSelectionResponse(BaseModel):
    vault_id: UUID
    name: str
    entries: list[VaultEntry]


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
    vault_id: UUID = Field(description="Selected vault to search")
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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(SearchRequest):
    """Input for generating an answer from retrieved vault chunks."""

    history: list["ChatMessage"] = Field(default_factory=list, max_length=12)


class ChatCitation(BaseModel):
    """Source metadata corresponding to a numbered reference in an answer."""

    reference: int = Field(ge=1, description="1-based reference number used in the answer")
    source_path: str = Field(description="Relative path from the vault root")
    heading_path: list[str] = Field(
        default_factory=list, description="Heading hierarchy for the cited chunk"
    )


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation]
