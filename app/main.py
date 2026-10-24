"""FastAPI entrypoint for Recall."""

import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request

from app.config import get_settings
from app.generator import generate_answer
from app.ingest import find_markdown_files, parse_vault_chunks
from app.models import (
    ChatCitation,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.prompt import (
    INSUFFICIENT_CONTEXT_ANSWER,
    build_rag_prompt,
    sort_results_sequentially,
)
from app.vectorstore import get_qdrant_client, index_chunks, search_chunks

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recall",
    version="0.1.0",
    description="Search and chat with an Obsidian vault, with grounded citations.",
)


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    started = perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        logger.info(
            "timing stage=request.total method=%s path=%s status_code=%d duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code if response is not None else 500,
            (perf_counter() - started) * 1000,
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


@app.post("/search", response_model=SearchResponse)
def search_vault(request: SearchRequest) -> SearchResponse:
    """Search indexed Markdown chunks by semantic similarity."""
    settings = get_settings()
    try:
        client = get_qdrant_client()
        results = search_chunks(
            client=client,
            collection_name=settings.qdrant_collection,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query vectorstore: {error}",
        ) from error

    return SearchResponse(query=request.query, results=results)


@app.post("/chat", response_model=ChatResponse)
def chat_with_vault(request: ChatRequest) -> ChatResponse:
    """Retrieve relevant notes and generate an answer with source references."""
    settings = get_settings()
    try:
        client = get_qdrant_client()
        results = search_chunks(
            client=client,
            collection_name=settings.qdrant_collection,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query vectorstore: {error}",
        ) from error

    if not results:
        return ChatResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, citations=[])

    ordered_results = sort_results_sequentially(results)
    prompt = build_rag_prompt(request.query, ordered_results)

    try:
        answer = generate_answer(prompt)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {error}",
        ) from error

    citations = [
        ChatCitation(
            reference=reference,
            source_path=result.chunk.source_path,
            heading_path=result.chunk.heading_path,
        )
        for reference, result in enumerate(ordered_results, start=1)
    ]
    return ChatResponse(answer=answer, citations=citations)
