"""Unit tests for Qdrant client connection and collection management."""

import pytest
from qdrant_client import QdrantClient, models

from app.vectorstore import ensure_collection_exists, get_qdrant_client


def test_ensure_collection_exists_creates_collection() -> None:
    client = QdrantClient(":memory:")
    collection_name = "test_chunks"

    assert not client.collection_exists(collection_name)

    created = ensure_collection_exists(client, collection_name=collection_name, vector_size=768)
    assert created is True
    assert client.collection_exists(collection_name)

    # Inspect the created collection configuration
    collection_info = client.get_collection(collection_name)
    vectors_config = collection_info.config.params.vectors
    assert isinstance(vectors_config, models.VectorParams)
    assert vectors_config.size == 768
    assert vectors_config.distance == models.Distance.COSINE


def test_ensure_collection_exists_is_idempotent() -> None:
    client = QdrantClient(":memory:")
    collection_name = "test_chunks"

    # First call creates the collection
    first_call = ensure_collection_exists(client, collection_name=collection_name, vector_size=768)
    assert first_call is True

    # Second call detects it already exists and returns False safely
    second_call = ensure_collection_exists(client, collection_name=collection_name, vector_size=768)
    assert second_call is False


def test_get_qdrant_client_initialization() -> None:
    client = get_qdrant_client(url="http://localhost:6333")
    assert isinstance(client, QdrantClient)


def test_generate_point_id_deterministic() -> None:
    from app.vectorstore import generate_point_id

    id1 = generate_point_id("notes/ai.md", 0)
    id2 = generate_point_id("notes/ai.md", 0)
    id3 = generate_point_id("notes/ai.md", 1)
    id4 = generate_point_id("notes/other.md", 0)

    assert id1 == id2
    assert id1 != id3
    assert id1 != id4


def test_index_chunks_empty_returns_zero() -> None:
    from app.vectorstore import index_chunks

    client = QdrantClient(":memory:")
    assert index_chunks(client, "test_collection", []) == 0


def test_index_chunks_indexes_payload_and_vectors() -> None:
    from unittest.mock import MagicMock

    from app.models import Chunk
    from app.vectorstore import generate_point_id, index_chunks

    client = QdrantClient(":memory:")
    collection_name = "test_indexing"

    chunks = [
        Chunk(
            source_path="projects/recall.md",
            heading_path=["Recall", "Architecture"],
            chunk_index=0,
            content="Recall uses Qdrant for vector retrieval.",
        ),
        Chunk(
            source_path="projects/recall.md",
            heading_path=["Recall", "Chunking"],
            chunk_index=1,
            content="Heading-aware chunking preserves document hierarchy.",
        ),
    ]

    mock_embeddings = MagicMock()
    # Return dummy vectors of 768 dimensions
    dummy_vec1 = [0.1] * 768
    dummy_vec2 = [0.2] * 768
    mock_embeddings.embed_documents.return_value = [dummy_vec1, dummy_vec2]

    indexed_count = index_chunks(
        client=client,
        collection_name=collection_name,
        chunks=chunks,
        batch_size=2,
        embedding_model=mock_embeddings,
    )

    assert indexed_count == 2

    # Verify points in Qdrant
    point_id_1 = generate_point_id(chunks[0].source_path, chunks[0].chunk_index)
    point_id_2 = generate_point_id(chunks[1].source_path, chunks[1].chunk_index)

    retrieved = client.retrieve(
        collection_name=collection_name,
        ids=[point_id_1, point_id_2],
        with_payload=True,
        with_vectors=True,
    )

    assert len(retrieved) == 2
    by_id = {str(point.id): point for point in retrieved}

    assert by_id[point_id_1].payload["source_path"] == "projects/recall.md"
    assert by_id[point_id_1].payload["heading_path"] == ["Recall", "Architecture"]
    assert by_id[point_id_1].payload["chunk_index"] == 0
    assert by_id[point_id_1].payload["content"] == "Recall uses Qdrant for vector retrieval."
    # Qdrant normalizes vectors to unit length under Cosine distance
    assert len(by_id[point_id_1].vector) == 768
    norm = sum(x * x for x in by_id[point_id_1].vector) ** 0.5
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Test idempotency: re-indexing does not duplicate points
    reindexed_count = index_chunks(
        client=client,
        collection_name=collection_name,
        chunks=chunks,
        batch_size=2,
        embedding_model=mock_embeddings,
    )
    assert reindexed_count == 2
    count_info = client.count(collection_name=collection_name)
    assert count_info.count == 2
