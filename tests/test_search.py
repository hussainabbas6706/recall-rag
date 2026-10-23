"""Tests for the /search API endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import Chunk, SearchResult


def test_search_endpoint_returns_ranked_results() -> None:
    client = TestClient(app)

    dummy_chunk = Chunk(
        source_path="projects/recall.md",
        heading_path=["Recall", "MVP"],
        chunk_index=0,
        content="Recall is a naive RAG assistant.",
    )
    dummy_results = [SearchResult(chunk=dummy_chunk, score=0.88)]

    with patch("app.main.search_chunks", return_value=dummy_results) as mock_search:
        response = client.post("/search", json={"query": "What is Recall?", "top_k": 3})

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is Recall?"
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.88
        assert data["results"][0]["chunk"]["source_path"] == "projects/recall.md"
        assert data["results"][0]["chunk"]["heading_path"] == ["Recall", "MVP"]
        mock_search.assert_called_once()


def test_search_endpoint_empty_query_rejected() -> None:
    client = TestClient(app)
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422


def test_search_endpoint_invalid_top_k_rejected() -> None:
    client = TestClient(app)
    # top_k must be between 1 and 20
    response = client.post("/search", json={"query": "valid query", "top_k": 0})
    assert response.status_code == 422

    response = client.post("/search", json={"query": "valid query", "top_k": 50})
    assert response.status_code == 422


def test_search_endpoint_vectorstore_failure_returns_500() -> None:
    client = TestClient(app)
    with patch("app.main.search_chunks", side_effect=RuntimeError("Qdrant unreachable")):
        response = client.post("/search", json={"query": "test query"})
        assert response.status_code == 500
        assert "Failed to query vectorstore" in response.json()["detail"]
