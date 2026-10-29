"""FastAPI entrypoint for Recall."""

import ipaddress
import logging
from time import perf_counter
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response

from app.config import get_settings
from app.generator import generate_answer
from app.indexing import index_vault_incrementally
from app.ingest import list_vault_entries
from app.models import (
    ChatCitation,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    VaultResponse,
    VaultSelectionResponse,
)
from app.prompt import (
    INSUFFICIENT_CONTEXT_ANSWER,
    build_rag_prompt,
    sort_results_sequentially,
)
from app.scoped_retrieval import (
    MAX_SCOPED_CONTEXT_CHARS,
    parse_explicit_reference,
    retrieve_heading_scoped_chunks,
)
from app.vault_picker import choose_vault_directory
from app.vault_session import collection_name_for_vault, vault_session
from app.vectorstore import get_qdrant_client, search_chunks

logger = logging.getLogger(__name__)
logging.getLogger("app").setLevel(logging.INFO)
EXPLICIT_TARGET_UNRESOLVED_ANSWER = (
    "I couldn't find that source in the indexed notes or it could not be resolved uniquely. "
    "Please check the filename and heading."
)

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


def require_local_application(request: Request) -> None:
    """Restrict folder selection and selected-vault operations to loopback clients."""
    client = request.client
    if client is None:
        raise HTTPException(
            status_code=403,
            detail="This operation is available only from the local Recall app.",
        )
    try:
        if not ipaddress.ip_address(client.host).is_loopback:
            raise HTTPException(
                status_code=403,
                detail="This operation is available only from the local Recall app.",
            )
    except ValueError as error:
        raise HTTPException(
            status_code=403,
            detail="This operation is available only from the local Recall app.",
        ) from error

    origin = request.headers.get("origin")
    if origin:
        try:
            parsed_origin = urlsplit(origin)
        except ValueError as error:
            raise HTTPException(
                status_code=403,
                detail="This operation is available only from the local Recall app.",
            ) from error
        hostname = parsed_origin.hostname
        is_local_origin = hostname == "localhost"
        if hostname:
            try:
                is_local_origin = is_local_origin or ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                pass
        if parsed_origin.scheme not in {"http", "https"} or not is_local_origin:
            raise HTTPException(
                status_code=403,
                detail="This operation is available only from the local Recall app.",
            )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm that the API process is running."""
    return HealthResponse()


@app.get("/vault", response_model=VaultResponse)
def get_vault(request: Request) -> VaultResponse:
    """List folders and Markdown files in the configured vault."""
    require_local_application(request)
    try:
        entries = list_vault_entries(get_settings().vault_path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise HTTPException(
            status_code=400,
            detail="The configured vault path is missing or is not a directory.",
        ) from error

    return VaultResponse(entries=entries)


@app.get("/vault/selection", response_model=VaultSelectionResponse)
def get_vault_selection(request: Request) -> VaultSelectionResponse | Response:
    """Return the active in-memory selection without exposing its absolute path."""
    require_local_application(request)
    selection = vault_session.current()
    if selection is None:
        return Response(status_code=204)
    if vault_session.resolve(selection.vault_id) is None:
        vault_session.clear()
        return Response(status_code=204)
    try:
        entries = list_vault_entries(selection.path)
    except (OSError, ValueError) as error:
        vault_session.clear()
        raise HTTPException(
            status_code=409,
            detail="The selected vault is no longer available. Select it again.",
        ) from error
    return VaultSelectionResponse(vault_id=selection.vault_id, name=selection.name, entries=entries)


@app.post("/vault/select", response_model=VaultSelectionResponse)
def select_vault(request: Request) -> VaultSelectionResponse | Response:
    """Open the local OS folder chooser and select a vault for this backend session."""
    require_local_application(request)
    try:
        selected_path = choose_vault_directory()
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The native folder chooser is unavailable. Run Recall on the desktop "
                "where the vault folder is stored."
            ),
        ) from error

    if selected_path is None:
        return Response(status_code=204)

    try:
        selection, entries = vault_session.select(selected_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=400,
            detail="The selected folder is unavailable or is not a readable directory.",
        ) from error

    return VaultSelectionResponse(
        vault_id=selection.vault_id,
        name=selection.name,
        entries=entries,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_vault(payload: IngestRequest, request: Request) -> IngestResponse:
    """Discover, chunk, and index Markdown notes into Qdrant."""
    require_local_application(request)
    selection = vault_session.resolve(payload.vault_id)
    if selection is None:
        raise HTTPException(
            status_code=409,
            detail="This vault selection is no longer active. Select the vault again.",
        )

    try:
        client = get_qdrant_client()
        indexing = index_vault_incrementally(
            client=client,
            collection_name=collection_name_for_vault(selection.path),
            vault_path=selection.path,
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        raise HTTPException(
            status_code=400,
            detail="The selected vault path is missing or is not a directory.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index chunks into Qdrant: {error}",
        ) from error

    return IngestResponse(
        status="partial" if indexing.failures else "indexed",
        files_found=indexing.files_found,
        chunks_created=indexing.chunks_indexed,
        chunks_indexed=indexing.chunks_indexed,
        files_indexed=indexing.files_indexed,
        files_updated=indexing.files_updated,
        files_unchanged=indexing.files_unchanged,
        files_deleted=indexing.files_deleted,
        failures=[
            {"source_path": failure.source_path, "reason": failure.reason}
            for failure in indexing.failures
        ],
        message=(
            f"Discovered {indexing.files_found} notes: {indexing.files_indexed} new, "
            f"{indexing.files_updated} updated, {indexing.files_unchanged} unchanged, "
            f"{indexing.files_deleted} deleted; indexed {indexing.chunks_indexed} chunks, "
            f"{len(indexing.failures)} failed for vault '{selection.name}'."
        ),
    )


@app.post("/search", response_model=SearchResponse)
def search_vault(payload: SearchRequest, request: Request) -> SearchResponse:
    """Search indexed Markdown chunks by semantic similarity."""
    require_local_application(request)
    selection = vault_session.resolve(payload.vault_id)
    if selection is None:
        raise HTTPException(
            status_code=409,
            detail="This vault selection is no longer active. Select the vault again.",
        )
    collection_name = collection_name_for_vault(selection.path)
    try:
        client = get_qdrant_client()
        results = search_chunks(
            client=client,
            collection_name=collection_name,
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query vectorstore: {error}",
        ) from error

    return SearchResponse(query=payload.query, results=results)


@app.post("/chat", response_model=ChatResponse)
def chat_with_vault(payload: ChatRequest, request: Request) -> ChatResponse:
    """Retrieve relevant notes and generate an answer with source references."""
    require_local_application(request)
    selection = vault_session.resolve(payload.vault_id)
    if selection is None:
        raise HTTPException(
            status_code=409,
            detail="This vault selection is no longer active. Select the vault again.",
        )
    collection_name = collection_name_for_vault(selection.path)
    history = payload.history[-8:]
    retrieval_query = "\n".join(
        [*(f"{message.role}: {message.content}" for message in history), payload.query]
    )
    context_note = None
    try:
        client = get_qdrant_client()
        semantic_results = search_chunks(
            client=client,
            collection_name=collection_name,
            query=retrieval_query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
        explicit_reference = parse_explicit_reference(payload.query)
        if explicit_reference is None:
            results = semantic_results
        else:
            file_reference, heading_reference = explicit_reference
            scoped_results = retrieve_heading_scoped_chunks(
                client=client,
                collection_name=collection_name,
                file_reference=file_reference,
                heading_reference=heading_reference,
            )
            if scoped_results is None:
                context_note = (
                    "The explicitly named file and heading could not be resolved uniquely "
                    "in the index. Do not present other sources as that target; explain "
                    "that the requested target could not be identified unless the context "
                    "itself clearly establishes it."
                )
                return ChatResponse(
                    answer=EXPLICIT_TARGET_UNRESOLVED_ANSWER,
                    citations=[],
                )
            if sum(len(result.chunk.content) for result in scoped_results) > MAX_SCOPED_CONTEXT_CHARS:
                context_note = (
                    "Only part of the explicitly requested section fit in the context. "
                    "Do not claim the section is complete or infer omitted content."
                )
            results = scoped_results
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
    prompt = build_rag_prompt(
        payload.query,
        ordered_results,
        history=history,
        context_note=context_note,
    )

    try:
        answer = generate_answer(prompt)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {error}",
        ) from error

    if INSUFFICIENT_CONTEXT_ANSWER in answer:
        return ChatResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, citations=[])

    citations = [
        ChatCitation(
            reference=reference,
            source_path=result.chunk.source_path,
            heading_path=result.chunk.heading_path,
        )
        for reference, result in enumerate(ordered_results, start=1)
    ]
    return ChatResponse(answer=answer, citations=citations)
