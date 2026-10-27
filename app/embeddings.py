"""Local sentence-transformer embeddings for document and query vectors."""

from threading import Lock
from typing import Protocol

from app.config import get_settings

EMBEDDING_DIMENSIONS = 384
_model_lock = Lock()


class EmbeddingModel(Protocol):
    """Minimal interface required by the vector-store pipeline."""

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class _EncodedVectors(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> _EncodedVectors: ...


class LocalSentenceTransformer:
    """Adapt a cached Sentence Transformers model to Recall's embedding interface."""

    def __init__(self, model: _SentenceTransformerModel, batch_size: int) -> None:
        self._model = model
        self._batch_size = batch_size
        self._lock = Lock()

    def embed_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        with self._lock:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size or self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [[float(value) for value in vector] for vector in vectors.tolist()]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text], batch_size=1)[0]


_embedding_model: LocalSentenceTransformer | None = None


def _load_sentence_transformer(model_name: str) -> _SentenceTransformerModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def get_embedding_model() -> LocalSentenceTransformer:
    """Load the configured local embedding model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                settings = get_settings()
                model = _load_sentence_transformer(settings.local_embedding_model)
                _embedding_model = LocalSentenceTransformer(model, settings.embedding_batch_size)
    return _embedding_model


def _validate_vectors(vectors: list[list[float]], expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding model returned {len(vectors)} vectors; expected {expected_count}."
        )
    if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
        raise ValueError(
            f"Embedding vectors must have exactly {EMBEDDING_DIMENSIONS} dimensions."
        )
    return vectors


def embed_texts(
    texts: list[str],
    model: EmbeddingModel | None = None,
) -> list[list[float]]:
    """Generate validated local vectors for a batch of document chunks."""
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Document text cannot be empty.")

    settings = get_settings()
    embedding_model = model or get_embedding_model()
    vectors = embedding_model.embed_documents(texts, batch_size=settings.embedding_batch_size)
    return _validate_vectors(vectors, len(texts))


def embed_query(
    query: str,
    model: EmbeddingModel | None = None,
) -> list[float]:
    """Generate a query vector with the same local model used for documents."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty.")

    embedding_model = model or get_embedding_model()
    return _validate_vectors([embedding_model.embed_query(clean_query)], 1)[0]
