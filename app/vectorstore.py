"""Qdrant client management and collection lifecycle."""

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.embeddings import embed_query, embed_texts
from app.models import Chunk, SearchResult


def get_qdrant_client(url: str | None = None) -> QdrantClient:
    """Return a QdrantClient connected to the configured Qdrant instance."""
    target_url = url or str(get_settings().qdrant_url)
    return QdrantClient(url=target_url, check_compatibility=False)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 768,
) -> bool:
    """Ensure a Qdrant collection exists with Cosine distance and given vector size.

    Returns:
        bool: True if a new collection was created, False if it already existed.
    """
    if client.collection_exists(collection_name=collection_name):
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    return True


def generate_point_id(source_path: str, chunk_index: int) -> str:
    """Generate a deterministic UUID string from source path and chunk index."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_path}:{chunk_index}"))


def index_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: Sequence[Chunk],
    batch_size: int = 64,
    embedding_model=None,
) -> int:
    """Embed chunks and index them into Qdrant in batches.

    Returns:
        int: Number of chunks successfully upserted into Qdrant.
    """
    if not chunks:
        return 0

    settings = get_settings()
    ensure_collection_exists(
        client=client,
        collection_name=collection_name,
        vector_size=settings.embedding_dimensions,
    )

    total_indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]
        vectors = embed_texts(texts, model=embedding_model)

        points = [
            models.PointStruct(
                id=generate_point_id(chunk.source_path, chunk.chunk_index),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        total_indexed += len(points)

    return total_indexed


def search_chunks(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 5,
    score_threshold: float | None = None,
    embedding_model=None,
) -> list[SearchResult]:
    """Search for relevant chunks in Qdrant matching the query text.

    Args:
        client: Connected QdrantClient instance.
        collection_name: Target collection name.
        query: User's natural language search text.
        top_k: Maximum number of closest points to return.
        score_threshold: Optional minimum cosine similarity cutoff.
        embedding_model: Optional GoogleGenerativeAIEmbeddings instance (for testing).

    Returns:
        list[SearchResult]: Ranked matching chunks with similarity scores.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty.")

    if not client.collection_exists(collection_name=collection_name):
        return []

    query_vector = embed_query(clean_query, model=embedding_model)

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    results: list[SearchResult] = []
    for point in response.points:
        if point.payload is None:
            continue
        chunk = Chunk(
            source_path=point.payload["source_path"],
            heading_path=point.payload.get("heading_path", []),
            chunk_index=point.payload["chunk_index"],
            content=point.payload["content"],
        )
        results.append(SearchResult(chunk=chunk, score=point.score))

    return results
