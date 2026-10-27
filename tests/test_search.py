"""Tests for the /search API endpoint."""

from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.models import Chunk, SearchResult


def test_search_endpoint_returns_ranked_results(selected_fixture_vault) -> None:
    client = TestClient(app, client=("127.0.0.1", 50000))

    dummy_chunk = Chunk(
        source_path="projects/recall.md",
        heading_path=["Recall", "MVP"],
        chunk_index=0,
        content="Recall is a naive RAG assistant.",
    )
    dummy_results = [SearchResult(chunk=dummy_chunk, score=0.88)]

    with patch("app.main.search_chunks", return_value=dummy_results) as mock_search:
        response = client.post(
            "/search",
            json={
                "query": "What is Recall?",
                "top_k": 3,
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is Recall?"
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.88
        assert data["results"][0]["chunk"]["source_path"] == "projects/recall.md"
        assert data["results"][0]["chunk"]["heading_path"] == ["Recall", "MVP"]
        mock_search.assert_called_once()


def test_search_endpoint_empty_query_rejected(selected_fixture_vault) -> None:
    client = TestClient(app, client=("127.0.0.1", 50000))
    response = client.post(
        "/search",
        json={"query": "", "vault_id": str(selected_fixture_vault.vault_id)},
    )
    assert response.status_code == 422


def test_search_endpoint_invalid_top_k_rejected(selected_fixture_vault) -> None:
    client = TestClient(app, client=("127.0.0.1", 50000))
    # top_k must be between 1 and 20
    payload = {"query": "valid query", "vault_id": str(selected_fixture_vault.vault_id)}
    response = client.post("/search", json={**payload, "top_k": 0})
    assert response.status_code == 422

    response = client.post("/search", json={**payload, "top_k": 50})
    assert response.status_code == 422


def test_search_endpoint_vectorstore_failure_returns_500(selected_fixture_vault) -> None:
    client = TestClient(app, client=("127.0.0.1", 50000))
    with patch("app.main.search_chunks", side_effect=RuntimeError("Qdrant unreachable")):
        response = client.post(
            "/search",
            json={"query": "test query", "vault_id": str(selected_fixture_vault.vault_id)},
        )
        assert response.status_code == 500
        assert "Failed to query vectorstore" in response.json()["detail"]


def test_search_rejects_stale_vault_selection(selected_fixture_vault) -> None:
    client = TestClient(app, client=("127.0.0.1", 50000))
    with patch("app.main.search_chunks") as search:
        response = client.post(
            "/search",
            json={"query": "test query", "vault_id": str(uuid4())},
        )

    assert response.status_code == 409
    search.assert_not_called()
