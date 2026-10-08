"""FastAPI entrypoint for Recall."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.ingest import find_markdown_files
from app.models import HealthResponse, IngestRequest, IngestResponse

app = FastAPI(
    title="Recall",
    version="0.1.0",
    description="Search and chat with an Obsidian vault, with grounded citations.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm that the API process is running."""
    return HealthResponse()


@app.post("/ingest", response_model=IngestResponse)
def inspect_vault(request: IngestRequest) -> IngestResponse:
    """Discover Markdown files; vector indexing will be added in the next phase."""
    vault_path = Path(request.vault_path) if request.vault_path else get_settings().vault_path
    try:
        markdown_files = find_markdown_files(vault_path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return IngestResponse(
        status="ready_for_chunking",
        files_found=len(markdown_files),
        message="Markdown files discovered. Heading-aware chunking and vector indexing are next.",
    )
