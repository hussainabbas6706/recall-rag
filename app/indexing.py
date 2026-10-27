"""Incremental, per-note vault indexing with persisted content fingerprints."""

import logging
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from app.chunking import chunk_markdown
from app.config import get_settings
from app.embeddings import EMBEDDING_DIMENSIONS, embed_texts
from app.ingest import find_markdown_files
from app.vectorstore import ensure_collection_exists

logger = logging.getLogger(__name__)

_NOTE_PROCESSING_ERRORS = (
    ApiException,
    httpx.HTTPError,
    OSError,
    RuntimeError,
    UnicodeError,
    ValueError,
)


@dataclass
class IndexingFailure:
    source_path: str
    reason: str


@dataclass
class IndexingResult:
    files_found: int
    files_indexed: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    chunks_indexed: int = 0
    failures: list[IndexingFailure] = field(default_factory=list)


def _scroll_all_points(client: QdrantClient, collection_name: str) -> list[models.Record]:
    records: list[models.Record] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=["source_path", "content_hash", "embedding_model", "chunk_index"],
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            return records


def _point_id(source_path: str, content_hash: str, chunk_index: int, model_name: str) -> str:
    identity = f"{model_name}:{source_path}:{content_hash}:{chunk_index}"
    return str(uuid5(NAMESPACE_URL, identity))


def _delete_points(client: QdrantClient, collection_name: str, point_ids: list[str]) -> None:
    if point_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )


def _safe_failure_reason(error: Exception, vault_root: Path) -> str:
    reason = str(error).replace(str(vault_root), "<vault>")
    return f"{type(error).__name__}: {reason[:240]}"


def index_vault_incrementally(
    client: QdrantClient,
    collection_name: str,
    vault_path: Path,
) -> IndexingResult:
    """Index changed notes, skip unchanged notes, and remove deleted-note vectors."""
    settings = get_settings()
    vault_root = vault_path.resolve(strict=True)
    discovery_started = perf_counter()
    files = find_markdown_files(vault_root)
    discovery_seconds = perf_counter() - discovery_started
    result = IndexingResult(files_found=len(files))
    collection_setup_started = perf_counter()
    ensure_collection_exists(
        client,
        collection_name=collection_name,
        vector_size=EMBEDDING_DIMENSIONS,
    )
    points_by_source: dict[str, list[models.Record]] = {}
    for point in _scroll_all_points(client, collection_name):
        if point.payload is None:
            continue
        source_path = point.payload.get("source_path")
        if isinstance(source_path, str):
            points_by_source.setdefault(source_path, []).append(point)
    collection_setup_seconds = perf_counter() - collection_setup_started
    file_read_seconds = 0.0
    chunking_seconds = 0.0
    embedding_seconds = 0.0
    qdrant_write_seconds = 0.0

    current_sources: set[str] = set()
    for file_path in files:
        relative_path = file_path.relative_to(vault_root).as_posix()
        current_sources.add(relative_path)
        previous_points = points_by_source.get(relative_path, [])

        try:
            if file_path.is_symlink():
                raise ValueError("Symbolic links are not indexed.")
            resolved_file = file_path.resolve(strict=True)
            if not resolved_file.is_relative_to(vault_root) or not resolved_file.is_file():
                raise ValueError("The note no longer resolves inside the selected vault.")

            read_started = perf_counter()
            raw_text = resolved_file.read_text(encoding="utf-8")
            file_read_seconds += perf_counter() - read_started
            content_hash = sha256(raw_text.encode("utf-8")).hexdigest()
            chunking_started = perf_counter()
            chunks = chunk_markdown(source_path=relative_path, raw_text=raw_text)
            chunking_seconds += perf_counter() - chunking_started
            expected_indices = set(range(len(chunks)))
            old_indices = {
                point.payload.get("chunk_index")
                for point in previous_points
                if point.payload is not None
            }
            if (
                previous_points
                and old_indices == expected_indices
                and all(
                    point.payload is not None
                    and point.payload.get("content_hash") == content_hash
                    and point.payload.get("embedding_model") == settings.local_embedding_model
                    for point in previous_points
                )
            ):
                result.files_unchanged += 1
                continue
            if not chunks and not previous_points:
                result.files_unchanged += 1
                continue

            new_ids: list[str] = []
            if chunks:
                try:
                    for start in range(0, len(chunks), settings.embedding_batch_size):
                        batch = chunks[start : start + settings.embedding_batch_size]
                        embedding_started = perf_counter()
                        try:
                            vectors = embed_texts([chunk.content for chunk in batch])
                        finally:
                            embedding_seconds += perf_counter() - embedding_started
                        points = []
                        for chunk, vector in zip(batch, vectors, strict=True):
                            point_id = _point_id(
                                relative_path,
                                content_hash,
                                chunk.chunk_index,
                                settings.local_embedding_model,
                            )
                            new_ids.append(point_id)
                            payload = chunk.model_dump()
                            payload["content_hash"] = content_hash
                            payload["embedding_model"] = settings.local_embedding_model
                            points.append(
                                models.PointStruct(id=point_id, vector=vector, payload=payload)
                            )
                        write_started = perf_counter()
                        try:
                            client.upsert(
                                collection_name=collection_name,
                                points=points,
                                wait=True,
                            )
                        finally:
                            qdrant_write_seconds += perf_counter() - write_started
                except _NOTE_PROCESSING_ERRORS:
                    old_ids = {str(point.id) for point in previous_points}
                    cleanup_started = perf_counter()
                    _delete_points(
                        client,
                        collection_name,
                        [point_id for point_id in new_ids if point_id not in old_ids],
                    )
                    qdrant_write_seconds += perf_counter() - cleanup_started
                    raise

            new_id_set = set(new_ids)
            old_ids_to_remove = [
                str(point.id) for point in previous_points if str(point.id) not in new_id_set
            ]
            write_started = perf_counter()
            _delete_points(client, collection_name, old_ids_to_remove)
            qdrant_write_seconds += perf_counter() - write_started
            if previous_points:
                result.files_updated += 1
            else:
                result.files_indexed += 1
            result.chunks_indexed += len(chunks)
        except _NOTE_PROCESSING_ERRORS as error:
            failure = IndexingFailure(
                source_path=relative_path,
                reason=_safe_failure_reason(error, vault_root),
            )
            result.failures.append(failure)
            logger.warning(
                "indexing note failed source_path=%s error_type=%s reason=%s",
                relative_path,
                type(error).__name__,
                failure.reason,
            )

    for source_path, points in points_by_source.items():
        if source_path in current_sources:
            continue
        try:
            write_started = perf_counter()
            _delete_points(client, collection_name, [str(point.id) for point in points])
            qdrant_write_seconds += perf_counter() - write_started
            result.files_deleted += 1
        except _NOTE_PROCESSING_ERRORS as error:
            failure = IndexingFailure(
                source_path=source_path,
                reason=_safe_failure_reason(error, vault_root),
            )
            result.failures.append(failure)
            logger.warning(
                "indexing stale note removal failed source_path=%s error_type=%s reason=%s",
                source_path,
                type(error).__name__,
                failure.reason,
            )

    logger.info(
        "timing stage=ingestion.incremental files=%d indexed=%d updated=%d unchanged=%d "
        "deleted=%d chunks_indexed=%d failures=%d discovery_ms=%.2f file_read_ms=%.2f "
        "chunking_ms=%.2f collection_setup_ms=%.2f embedding_ms=%.2f "
        "qdrant_write_ms=%.2f",
        result.files_found,
        result.files_indexed,
        result.files_updated,
        result.files_unchanged,
        result.files_deleted,
        result.chunks_indexed,
        len(result.failures),
        discovery_seconds * 1000,
        file_read_seconds * 1000,
        chunking_seconds * 1000,
        collection_setup_seconds * 1000,
        embedding_seconds * 1000,
        qdrant_write_seconds * 1000,
    )
    return result
