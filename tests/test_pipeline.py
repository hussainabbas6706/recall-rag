"""Fixture-backed API flow using in-memory Qdrant and deterministic embeddings."""

from unittest.mock import patch

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app


def test_fixture_ingest_search_and_chat_flow(
    deterministic_embeddings, selected_fixture_vault
) -> None:
    qdrant = QdrantClient(":memory:")
    client = TestClient(app, client=("127.0.0.1", 50000))

    with (
        patch("app.main.get_qdrant_client", return_value=qdrant),
        patch(
            "app.indexing.embed_texts",
            side_effect=lambda texts, model=None: deterministic_embeddings.embed_documents(
                texts
            ),
        ),
        patch(
            "app.vectorstore.embed_query",
            side_effect=lambda query, model=None: deterministic_embeddings.embed_query(query),
        ),
        patch(
            "app.main.generate_answer",
            return_value="Recall retrieves Markdown chunks from Qdrant [1].",
        ) as generate,
    ):
        ingest_response = client.post(
            "/ingest",
            json={"vault_id": str(selected_fixture_vault.vault_id)},
        )
        assert ingest_response.status_code == 200, ingest_response.text
        assert ingest_response.json()["chunks_indexed"] == 4

        search_response = client.post(
            "/search",
            json={
                "query": "What does Recall retrieve and answer with?",
                "top_k": 3,
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )
        assert search_response.status_code == 200
        search_results = search_response.json()["results"]
        assert search_results
        assert search_results[0]["chunk"]["source_path"] == "projects/recall.md"

        chat_response = client.post(
            "/chat",
            json={
                "query": "What does Recall retrieve and answer with?",
                "top_k": 3,
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )
        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        assert chat_data["answer"] == "Recall retrieves Markdown chunks from Qdrant [1]."
        assert chat_data["citations"][0]["reference"] == 1
        assert chat_data["citations"][0]["source_path"] == "projects/recall.md"
        assert chat_data["citations"][0]["heading_path"] == ["Recall", "MVP"]

        prompt = generate.call_args.args[0]
        assert "[1] projects/recall.md | Recall > MVP" in prompt
        assert "Markdown chunks from Qdrant" in prompt

        unsupported_response = client.post(
            "/chat",
            json={
                "query": "What is the weather on Mars?",
                "top_k": 3,
                "score_threshold": 0.99,
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )
        assert unsupported_response.status_code == 200
        assert unsupported_response.json() == {
            "answer": "I don't have enough information to answer that.",
            "citations": [],
        }

    assert generate.call_count == 1
