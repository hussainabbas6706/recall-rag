"""Tests for the chat API data contract."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import ChatCitation, ChatRequest, ChatResponse, Chunk, SearchResult

client = TestClient(app)


def test_chat_request_reuses_search_validation_and_defaults() -> None:
    request = ChatRequest(query="What is Recall?")

    assert request.top_k == 5
    assert request.score_threshold is None

    with pytest.raises(ValidationError):
        ChatRequest(query="", top_k=0)


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


def test_chat_endpoint_returns_answer_with_matching_citation_order() -> None:
    results = [
        _search_result("notes/a.md", ["A", "Second"], "A second section.", 0.95, 2),
        _search_result("notes/b.md", ["B"], "B content.", 0.70, 0),
        _search_result("notes/a.md", ["A", "First"], "A first section.", 0.80, 0),
    ]
    with (
        patch("app.main.search_chunks", return_value=results),
        patch("app.main.generate_answer", return_value="Answer grounded in [1].") as generate,
    ):
        response = client.post("/chat", json={"query": "What do the notes say?"})

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


def test_chat_endpoint_returns_fallback_without_generating_answer() -> None:
    with (
        patch("app.main.search_chunks", return_value=[]),
        patch("app.main.generate_answer") as generate,
    ):
        response = client.post("/chat", json={"query": "A question with no matches"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't have enough information to answer that.",
        "citations": [],
    }
    generate.assert_not_called()


def test_chat_endpoint_rejects_invalid_request() -> None:
    response = client.post("/chat", json={"query": "valid question", "top_k": 0})

    assert response.status_code == 422


def test_chat_endpoint_returns_error_when_retrieval_fails() -> None:
    with patch("app.main.search_chunks", side_effect=RuntimeError("Qdrant unreachable")):
        response = client.post("/chat", json={"query": "A valid question"})

    assert response.status_code == 500
    assert "Failed to query vectorstore" in response.json()["detail"]


def test_chat_endpoint_returns_error_when_generation_fails() -> None:
    results = [_search_result("notes/a.md", ["A"], "Relevant text.", 0.9, 0)]
    with (
        patch("app.main.search_chunks", return_value=results),
        patch("app.main.generate_answer", side_effect=RuntimeError("Gemini unavailable")),
    ):
        response = client.post("/chat", json={"query": "A valid question"})

    assert response.status_code == 500
    assert "Failed to generate answer" in response.json()["detail"]
