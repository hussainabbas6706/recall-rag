"""Unit tests for embedding generation."""

from unittest.mock import MagicMock

import pytest

from app.embeddings import LocalSentenceTransformer, embed_query, embed_texts, get_embedding_model


def test_get_embedding_model_loads_cpu_model_once(monkeypatch) -> None:
    from app import embeddings

    loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", loader)
    monkeypatch.setattr(embeddings, "_embedding_model", None)
    first = get_embedding_model()
    second = get_embedding_model()

    assert first is second
    assert isinstance(first, LocalSentenceTransformer)
    loader.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")


def test_embed_texts_empty_returns_empty_list() -> None:
    assert embed_texts([]) == []


def test_embed_texts_calls_model_with_batch() -> None:
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1] * 384, [0.3] * 384]

    texts = ["First chunk text", "Second chunk text"]
    result = embed_texts(texts, model=mock_model)

    assert result == [[0.1] * 384, [0.3] * 384]
    mock_model.embed_documents.assert_called_once_with(texts, batch_size=16)


def test_embed_query_calls_model() -> None:
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1] * 384

    result = embed_query("What is Recall?", model=mock_model)

    assert result == [0.1] * 384
    mock_model.embed_query.assert_called_once_with("What is Recall?")


def test_embed_query_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        embed_query("   ")


def test_embed_texts_rejects_empty_document() -> None:
    with pytest.raises(ValueError, match="Document text cannot be empty"):
        embed_texts(["  "], model=MagicMock())


def test_embed_texts_rejects_wrong_vector_dimensions() -> None:
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1, 0.2]]
    with pytest.raises(ValueError, match="exactly 384 dimensions"):
        embed_texts(["text"], model=mock_model)


def test_local_provider_uses_same_model_for_documents_and_queries() -> None:
    class Encoded:
        def __init__(self, vector: list[float]) -> None:
            self.vector = vector

        def tolist(self) -> list[list[float]]:
            return [self.vector]

    model = MagicMock()
    model.encode.side_effect = [Encoded([0.5] * 384), Encoded([0.25] * 384)]
    provider = LocalSentenceTransformer(model, batch_size=8)

    assert provider.embed_documents(["note text"]) == [[0.5] * 384]
    assert provider.embed_query("question") == [0.25] * 384
    assert model.encode.call_args_list[0].args[0] == ["note text"]
    assert model.encode.call_args_list[0].kwargs["batch_size"] == 8
    assert model.encode.call_args_list[1].args[0] == ["question"]
