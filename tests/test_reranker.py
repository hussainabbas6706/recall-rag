"""Unit tests for the two-stage Cross-Encoder reranker."""

from app.models import Chunk, SearchResult
from app.reranker import rerank_search_results


class MockReranker:
    """Mock reranker that scores chunks based on keyword matching."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores

    def predict(self, pairs: list[list[str]]) -> list[float]:
        if self.scores is not None:
            return self.scores
        # Default mock heuristic: score higher if query tokens appear in chunk
        results = []
        for query, doc in pairs:
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            results.append(float(overlap))
        return results


def _make_result(source: str, content: str, initial_score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            source_path=source,
            heading_path=["Topic"],
            chunk_index=0,
            content=content,
        ),
        score=initial_score,
    )


def test_rerank_promotes_relevant_chunk_to_rank_one() -> None:
    # Chunk A was ranked #1 by vector search (score 0.9), but Chunk B is actually about risk register
    chunk_a = _make_result("notes/general.md", "General notes on daily team updates.", initial_score=0.9)
    chunk_b = _make_result("notes/project-plan.md", "The project risk register tracks delays.", initial_score=0.6)

    # Initial order: [chunk_a, chunk_b]
    candidates = [chunk_a, chunk_b]
    mock = MockReranker(scores=[0.1, 0.95])  # Reranker scores Chunk B much higher

    reranked = rerank_search_results("risk register", candidates, top_k=2, reranker=mock)

    assert len(reranked) == 2
    # Chunk B should now be promoted to rank 1!
    assert reranked[0].chunk.source_path == "notes/project-plan.md"
    assert reranked[0].score == 0.95
    assert reranked[1].chunk.source_path == "notes/general.md"


def test_rerank_respects_top_k_limit() -> None:
    candidates = [
        _make_result(f"notes/{i}.md", f"Content {i}")
        for i in range(10)
    ]
    mock = MockReranker(scores=[float(i) for i in range(10)])

    reranked = rerank_search_results("query", candidates, top_k=3, reranker=mock)
    assert len(reranked) == 3
    # Highest score (9.0) should be first
    assert reranked[0].chunk.source_path == "notes/9.md"


def test_rerank_empty_or_single_result() -> None:
    assert rerank_search_results("query", []) == []

    single = [_make_result("notes/one.md", "Content")]
    assert rerank_search_results("query", single, top_k=5) == single


def test_rerank_fallback_on_error() -> None:
    class FailingReranker:
        def predict(self, pairs: list[list[str]]) -> list[float]:
            raise RuntimeError("Inference boom")

    candidates = [
        _make_result("notes/a.md", "A", initial_score=0.8),
        _make_result("notes/b.md", "B", initial_score=0.7),
    ]

    # Should gracefully return original vector candidates sliced to top_k without crashing
    fallback = rerank_search_results("query", candidates, top_k=2, reranker=FailingReranker())
    assert len(fallback) == 2
    assert fallback[0].chunk.source_path == "notes/a.md"

