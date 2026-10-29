"""Tests for the chat API data contract."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import ChatCitation, ChatRequest, ChatResponse, Chunk, SearchResult
from app.vault_session import collection_name_for_vault

client = TestClient(app, client=("127.0.0.1", 50000))


def test_chat_request_reuses_search_validation_and_defaults() -> None:
    request = ChatRequest(query="What is Recall?", vault_id=uuid4())

    assert request.top_k == 5
    assert request.score_threshold is None

    with pytest.raises(ValidationError):
        ChatRequest(query="", top_k=0, vault_id=uuid4())


def test_chat_response_serializes_answer_and_citations() -> None:
    response = ChatResponse(
        answer="Recall is a local assistant. [1]",
        citations=[
            ChatCitation(
                reference=1,
                source_path="projects/recall.md",
                heading_path=["Recall", "MVP"],
            )
        ],
    )

    assert response.model_dump() == {
        "answer": "Recall is a local assistant. [1]",
        "citations": [
            {
                "reference": 1,
                "source_path": "projects/recall.md",
                "heading_path": ["Recall", "MVP"],
            }
        ],
    }


def test_chat_citation_reference_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ChatCitation(reference=0, source_path="notes/example.md")


def _search_result(
    source_path: str,
    heading_path: list[str],
    content: str,
    score: float,
    chunk_index: int,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            source_path=source_path,
            heading_path=heading_path,
            chunk_index=chunk_index,
            content=content,
        ),
        score=score,
    )


def test_chat_endpoint_returns_answer_with_matching_citation_order(selected_fixture_vault) -> None:
    results = [
        _search_result("notes/a.md", ["A", "Second"], "A second section.", 0.95, 2),
        _search_result("notes/b.md", ["B"], "B content.", 0.70, 0),
        _search_result("notes/a.md", ["A", "First"], "A first section.", 0.80, 0),
    ]
    with (
        patch("app.main.search_chunks", return_value=results),
        patch("app.main.generate_answer", return_value="Answer grounded in [1].") as generate,
    ):
        response = client.post(
            "/chat",
            json={
                "query": "What do the notes say?",
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Answer grounded in [1]."
    assert data["citations"] == [
        {
            "reference": 1,
            "source_path": "notes/a.md",
            "heading_path": ["A", "First"],
        },
        {
            "reference": 2,
            "source_path": "notes/a.md",
            "heading_path": ["A", "Second"],
        },
        {
            "reference": 3,
            "source_path": "notes/b.md",
            "heading_path": ["B"],
        },
    ]
    prompt = generate.call_args.args[0]
    assert prompt.index("[1] notes/a.md | A > First") < prompt.index("[2] notes/a.md | A > Second")
    assert prompt.index("[2] notes/a.md | A > Second") < prompt.index("[3] notes/b.md | B")


def test_chat_endpoint_returns_fallback_without_generating_answer(selected_fixture_vault) -> None:
    with (
        patch("app.main.search_chunks", return_value=[]),
        patch("app.main.generate_answer") as generate,
    ):
        response = client.post(
            "/chat",
            json={
                "query": "A question with no matches",
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't have enough information to answer that.",
        "citations": [],
    }
    generate.assert_not_called()


def test_chat_endpoint_rejects_invalid_request(selected_fixture_vault) -> None:
    response = client.post(
        "/chat",
        json={
            "query": "valid question",
            "top_k": 0,
            "vault_id": str(selected_fixture_vault.vault_id),
        },
    )

    assert response.status_code == 422


def test_chat_endpoint_returns_error_when_retrieval_fails(selected_fixture_vault) -> None:
    with patch("app.main.search_chunks", side_effect=RuntimeError("Qdrant unreachable")):
        response = client.post(
            "/chat",
            json={"query": "A valid question", "vault_id": str(selected_fixture_vault.vault_id)},
        )

    assert response.status_code == 500
    assert "Failed to query vectorstore" in response.json()["detail"]


def test_chat_endpoint_returns_bad_request_for_invalid_query(selected_fixture_vault) -> None:
    with patch("app.main.search_chunks", side_effect=ValueError("Query string cannot be empty.")):
        response = client.post(
            "/chat",
            json={"query": " ", "vault_id": str(selected_fixture_vault.vault_id)},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query string cannot be empty."


def test_chat_endpoint_returns_error_when_generation_fails(selected_fixture_vault) -> None:
    results = [_search_result("notes/a.md", ["A"], "Relevant text.", 0.9, 0)]
    with (
        patch("app.main.search_chunks", return_value=results),
        patch("app.main.generate_answer", side_effect=RuntimeError("Gemini unavailable")),
    ):
        response = client.post(
            "/chat",
            json={"query": "A valid question", "vault_id": str(selected_fixture_vault.vault_id)},
        )

    assert response.status_code == 500
    assert "Failed to generate answer" in response.json()["detail"]


def test_chat_endpoint_uses_selected_vault_collection_and_history(selected_fixture_vault) -> None:
    results = [_search_result("notes/a.md", ["A"], "A local source.", 0.9, 0)]
    history = [
        {"role": "user", "content": "What does the note say about Recall?"},
        {"role": "assistant", "content": "It describes a local RAG assistant."},
    ]
    with (
        patch("app.main.search_chunks", return_value=results) as search,
        patch("app.main.generate_answer", return_value="It uses Qdrant [1].") as generate,
    ):
        response = client.post(
            "/chat",
            json={
                "query": "Which vector store does it use?",
                "vault_id": str(selected_fixture_vault.vault_id),
                "history": history,
            },
        )

    assert response.status_code == 200
    assert search.call_args.kwargs["collection_name"] == (
        collection_name_for_vault(selected_fixture_vault.path)
    )
    assert "What does the note say about Recall?" in search.call_args.kwargs["query"]
    assert "--- CONVERSATION HISTORY ---" in generate.call_args.args[0]
    assert response.json()["citations"][0]["source_path"] == "notes/a.md"


def test_chat_rejects_a_vault_id_from_an_old_selection(selected_fixture_vault) -> None:
    old_vault_id = selected_fixture_vault.vault_id
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        from app.vault_session import vault_session

        vault_session.select(Path(directory))
        with patch("app.main.search_chunks") as search:
            response = client.post(
                "/chat",
                json={"query": "Old vault question", "vault_id": str(old_vault_id)},
            )

    assert response.status_code == 409
    search.assert_not_called()


def test_chat_endpoint_returns_no_citations_when_llm_has_insufficient_context(selected_fixture_vault) -> None:
    results = [_search_result('notes/a.md', ['A'], 'Irrelevant content.', 0.2, 0)]
    with (
        patch('app.main.search_chunks', return_value=results),
        patch('app.main.generate_answer', return_value='I don\'t have enough information to answer that.') as generate,
    ):
        response = client.post(
            '/chat',
            json={
                'query': 'A question that context cannot answer',
                'vault_id': str(selected_fixture_vault.vault_id),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        'answer': 'I don\'t have enough information to answer that.',
        'citations': [],
    }
    generate.assert_called_once()
