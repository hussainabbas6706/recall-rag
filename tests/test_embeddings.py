"""Unit tests for embedding generation."""

from unittest.mock import MagicMock

import pytest

from app.embeddings import embed_query, embed_texts, get_embedding_model


def test_get_embedding_model_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        get_embedding_model(api_key="")


def test_embed_texts_empty_returns_empty_list() -> None:
    assert embed_texts([]) == []


def test_embed_texts_calls_model_with_batch() -> None:
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

    texts = ["First chunk text", "Second chunk text"]
    result = embed_texts(texts, model=mock_model)

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.embed_documents.assert_called_once_with(texts, output_dimensionality=768)


def test_embed_query_calls_model() -> None:
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

    result = embed_query("What is Recall?", model=mock_model)

    assert result == [0.1, 0.2, 0.3]
    mock_model.embed_query.assert_called_once_with("What is Recall?", output_dimensionality=768)


def test_embed_query_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        embed_query("   ")
