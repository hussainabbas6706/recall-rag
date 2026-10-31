"""Cross-encoder reranking module for two-stage retrieval."""

import logging
from threading import Lock
from typing import Protocol

from app.config import get_settings
from app.models import SearchResult

logger = logging.getLogger(__name__)

_reranker_lock = Lock()


class RerankerModel(Protocol):
    """Protocol for reranker models to allow deterministic mocking in tests."""

    def predict(self, sentences: list[list[str]]) -> list[float]: ...


class LocalCrossEncoder:
    """Wraps a cached sentence-transformers CrossEncoder for CPU inference."""

    def __init__(self, model: RerankerModel) -> None:
        self._model = model
        self._lock = Lock()

    def predict(self, pairs: list[list[str]]) -> list[float]:
        with self._lock:
            scores = self._model.predict(pairs)
        # Handle numpy ndarray or raw list
        if hasattr(scores, "tolist"):
            return [float(s) for s in scores.tolist()]
        return [float(s) for s in scores]


_reranker_instance: LocalCrossEncoder | None = None


def _load_cross_encoder(model_name: str) -> RerankerModel:
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder model: %s on CPU", model_name)
    return CrossEncoder(model_name, device="cpu")


def get_reranker_model() -> LocalCrossEncoder:
    """Load the configured local cross-encoder model once and reuse it."""
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                settings = get_settings()
                model = _load_cross_encoder(settings.rerank_model)
                _reranker_instance = LocalCrossEncoder(model)
    return _reranker_instance


def rerank_search_results(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
    reranker: RerankerModel | None = None,
) -> list[SearchResult]:
    """Rerank a list of SearchResults using cross-attention between query and chunk text.

    Args:
        query: User natural language search query.
        results: Candidate SearchResult items from first-stage vector search.
        top_k: Number of highest-ranked results to return.
        reranker: Optional custom or mock reranker instance (for testing).

    Returns:
        List of SearchResult items sorted by cross-encoder relevance score, sliced to top_k.
    """
    if not results:
        return []

    if len(results) == 1:
        return results[:top_k]

    clean_query = query.strip()
    if not clean_query:
        return results[:top_k]

    active_reranker = reranker or get_reranker_model()

    # Build pairs of (query, chunk_content)
    pairs = [[clean_query, r.chunk.content] for r in results]

    try:
        raw_scores = active_reranker.predict(pairs)
        scores = (
            raw_scores.tolist() if hasattr(raw_scores, "tolist") else [float(s) for s in raw_scores]
        )
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Reranking failed (%s); falling back to original vector ranking", e)
        return results[:top_k]

    # Combine original results with new cross-encoder scores
    scored_results: list[SearchResult] = []
    for result, score in zip(results, scores, strict=True):
        scored_results.append(
            SearchResult(
                chunk=result.chunk,
                score=round(float(score), 4),
            )
        )

    # Sort descending by cross-encoder score
    scored_results.sort(key=lambda r: r.score, reverse=True)
    return scored_results[:top_k]
