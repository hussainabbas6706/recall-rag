"""Metadata-scoped retrieval for questions that name a file and heading."""

import re
from pathlib import PurePosixPath

from qdrant_client import QdrantClient, models

from app.models import Chunk, SearchResult

MAX_SCOPED_CONTEXT_CHARS = 12_000
_QUOTED_VALUE = re.compile(r"""(?P<quote>`|"|')(?P<value>.+?)(?P=quote)""")


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _heading_matches(heading: str, requested: str) -> bool:
    normalized_heading = _normalize_label(heading)
    normalized_requested = _normalize_label(requested)
    if normalized_heading == normalized_requested:
        return True

    for separator in ("—", "–", " - ", ":"):
        if separator in heading:
            suffix = heading.rsplit(separator, maxsplit=1)[-1]
            if _normalize_label(suffix) == normalized_requested:
                return True
    return False


def _explicit_file_references(question: str) -> list[str]:
    references = [
        match.group("value").strip()
        for match in _QUOTED_VALUE.finditer(question)
        if match.group("value").strip().casefold().endswith(".md")
    ]

    pattern = re.compile(
        r"\b(?:file|note|in|from)\s+(?:(?:the\s+)?(?:file|note)\s+)?"
        r"(?P<value>[^`\"'<>?!;\n]+?\.md)(?![\w])",
        re.IGNORECASE,
    )
    references.extend(
        match.group("value").strip().rstrip(".,)")
        for match in pattern.finditer(question)
    )
    return list(dict.fromkeys(references))


def _explicit_heading_references(question: str) -> list[str]:
    references: list[str] = []
    for match in _QUOTED_VALUE.finditer(question):
        value = match.group("value").strip()
        if not value or value.casefold().endswith(".md"):
            continue
        before = question[max(0, match.start() - 48) : match.start()].casefold()
        after = question[match.end() : match.end() + 36].casefold()
        if (
            re.search(r"\b(?:heading|section|under|titled|named)\b", before)
            or re.match(r"\s*(?:section|heading)\b", after)
        ):
            references.append(value)

    if not references:
        patterns = (
            r"\bunder\s+(?:the\s+)?(.+?)\s+(?:section|heading)\b",
            (
                r"\b(?:section|heading)\s+(?:titled|named|called)\s+(.+?)"
                r"(?=\s+(?:in|from)\b|[?.!,;]|$)"
            ),
        )
        for pattern in patterns:
            references.extend(
                match.group(1).strip().strip("`\"'")
                for match in re.finditer(pattern, question, re.IGNORECASE)
            )

    return list(dict.fromkeys(references))


def parse_explicit_reference(question: str) -> tuple[str, str] | None:
    """Return a file and heading only when both are explicitly identified."""
    normalized_question = question.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    files = _explicit_file_references(normalized_question)
    headings = _explicit_heading_references(normalized_question)
    if len(files) != 1 or len(headings) != 1:
        return None
    return files[0], headings[0]


def _scroll_records(
    client: QdrantClient,
    collection_name: str,
    *,
    scroll_filter: models.Filter | None = None,
    with_payload: list[str],
) -> list[models.Record]:
    records: list[models.Record] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            return records


def _strip_leading_decoration(value: str) -> str:
    """Remove only leading non-word decoration from a filename.

    This allows a note like `🗓️ WEEKLY GOALS.md` to match a user query for
    `WEEKLY GOALS.md` without stripping meaningful characters from the actual
    name itself.
    """
    cleaned = value.strip()
    return re.sub(r"^[^\w]+", "", cleaned)


def _resolve_source_path(
    client: QdrantClient,
    collection_name: str,
    file_reference: str,
) -> str | None:
    records = _scroll_records(
        client,
        collection_name,
        with_payload=["source_path"],
    )
    source_paths = {
        record.payload["source_path"]
        for record in records
        if record.payload and isinstance(record.payload.get("source_path"), str)
    }

    normalized_reference = file_reference.replace("\\", "/").strip("/")
    if "/" in normalized_reference:
        matches = [
            path
            for path in source_paths
            if path.casefold() == normalized_reference.casefold()
        ]
    else:
        requested_name = PurePosixPath(normalized_reference).name.casefold()
        requested_decoration_stripped = _strip_leading_decoration(requested_name)
        matches = [
            path
            for path in source_paths
            if PurePosixPath(path).name.casefold() == requested_name
            or (
                requested_decoration_stripped
                and _strip_leading_decoration(PurePosixPath(path).name.casefold())
                == requested_decoration_stripped
            )
        ]
    return matches[0] if len(matches) == 1 else None


def retrieve_heading_scoped_chunks(
    client: QdrantClient,
    collection_name: str,
    file_reference: str,
    heading_reference: str,
) -> list[SearchResult] | None:
    """Resolve one indexed file/heading and return its chunks and descendants.

    `None` means the target was missing or ambiguous; callers should retain
    ordinary semantic results rather than guessing at a different target.
    """
    if not client.collection_exists(collection_name=collection_name):
        return None

    source_path = _resolve_source_path(client, collection_name, file_reference)
    if source_path is None:
        return None

    source_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_path",
                match=models.MatchValue(value=source_path),
            )
        ]
    )
    records = _scroll_records(
        client,
        collection_name,
        scroll_filter=source_filter,
        with_payload=["source_path", "heading_path", "chunk_index", "content"],
    )
    heading_label = _normalize_label(heading_reference)
    matching_prefixes: set[tuple[str, ...]] = set()
    for record in records:
        payload = record.payload or {}
        heading_path = payload.get("heading_path")
        if not isinstance(heading_path, list) or not all(
            isinstance(heading, str) for heading in heading_path
        ):
            continue
        for index, heading in enumerate(heading_path):
            if _heading_matches(heading, heading_label):
                matching_prefixes.add(tuple(heading_path[: index + 1]))

    if len(matching_prefixes) != 1:
        return None
    requested_prefix = next(iter(matching_prefixes))
    results = []
    for record in records:
        payload = record.payload or {}
        heading_path = payload.get("heading_path")
        content = payload.get("content")
        chunk_index = payload.get("chunk_index")
        if (
            not isinstance(heading_path, list)
            or tuple(heading_path[: len(requested_prefix)]) != requested_prefix
            or not isinstance(content, str)
            or not isinstance(chunk_index, int)
        ):
            continue
        results.append(
            SearchResult(
                chunk=Chunk(
                    source_path=source_path,
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                    content=content,
                ),
                score=0.0,
            )
        )
    return sorted(results, key=lambda result: result.chunk.chunk_index)


def merge_scoped_and_semantic_results(
    semantic_results: list[SearchResult],
    scoped_results: list[SearchResult],
    max_context_chars: int = MAX_SCOPED_CONTEXT_CHARS,
) -> list[SearchResult]:
    """Prefer scoped evidence, deduplicate stable chunks, and cap body text."""
    merged_by_identity: dict[tuple[str, int], SearchResult] = {}
    ordered_candidates = [
        *sorted(scoped_results, key=lambda result: result.chunk.chunk_index),
        *semantic_results,
    ]
    for result in ordered_candidates:
        identity = (result.chunk.source_path.casefold(), result.chunk.chunk_index)
        existing = merged_by_identity.get(identity)
        if existing is None or result.score > existing.score:
            merged_by_identity[identity] = result

    selected: list[SearchResult] = []
    used_chars = 0
    for result in merged_by_identity.values():
        content_size = len(result.chunk.content)
        if used_chars + content_size > max_context_chars:
            continue
        selected.append(result)
        used_chars += content_size
    return selected
