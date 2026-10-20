"""FastAPI entrypoint for Recall."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.ingest import find_markdown_files, parse_vault_chunks
from app.models import HealthResponse, IngestRequest, IngestResponse
from app.vectorstore import get_qdrant_client, index_chunks

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
def ingest_vault(request: IngestRequest) -> IngestResponse:
    """Discover, chunk, and index Markdown notes into Qdrant."""
    settings = get_settings()
    vault_path = Path(request.vault_path) if request.vault_path else settings.vault_path
    try:
        files = find_markdown_files(vault_path)
        chunks = parse_vault_chunks(vault_path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        client = get_qdrant_client()
        chunks_indexed = index_chunks(
            client=client,
            collection_name=settings.qdrant_collection,
            chunks=chunks,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index chunks into Qdrant: {error}",
        ) from error

    return IngestResponse(
        status="indexed",
        files_found=len(files),
        chunks_created=len(chunks),
        chunks_indexed=chunks_indexed,
        message=(
            f"Discovered {len(files)} notes and indexed {chunks_indexed} "
            f"chunks into Qdrant collection '{settings.qdrant_collection}'."
        ),
    )
