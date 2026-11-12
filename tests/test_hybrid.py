"""Unit tests for BM25 Okapi sparse indexing, RRF, and hybrid search."""

from unittest.mock import MagicMock

from app.hybrid import (
    BM25Index,
    get_bm25_index_for_collection,
    hybrid_search,
    invalidate_bm25_cache,
    reciprocal_rank_fusion,
    tokenize,
)
from app.models import Chunk, SearchResult


def _make_chunk(source: str, index: int, content: str) -> Chunk:
    return Chunk(
        source_path=source,
        heading_path=["Header"],
        chunk_index=index,
        content=content,
    )


def _make_result(source: str, index: int, content: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk=_make_chunk(source, index, content),
        score=score,
    )


def test_tokenize_preserves_identifiers_and_replaces_punctuation() -> None:
    text = "Set HF_TOKEN=123 for fastapi-app v1.5! Check (NumPy, uv, Jupyter)."
    tokens = tokenize(text)
    assert "hf_token" in tokens
    assert "fastapi" in tokens
    assert "numpy" in tokens
    assert "uv" in tokens
    assert "jupyter" in tokens
    assert "v1.5" in tokens


def test_bm25_promotes_exact_rare_keyword_needle() -> None:
    chunks = [
        _make_chunk("notes/a.md", 0, "General roadmap planning and weekly routines."),
        _make_chunk("notes/b.md", 0, "Python setup instructions with uv and Jupyter."),
        _make_chunk("notes/c.md", 0, "Make sure HF_TOKEN is configured for model downloads."),
    ]
    index = BM25Index(chunks)

    results = index.search("HF_TOKEN", top_k=2)
    assert len(results) == 1
    assert results[0].chunk.source_path == "notes/c.md"
    assert results[0].score > 0.0


def test_bm25_empty_query_or_no_match() -> None:
    chunks = [_make_chunk("notes/a.md", 0, "Just some normal text.")]
    index = BM25Index(chunks)

    assert index.search("") == []
    assert index.search("nonexistenttermxyz123") == []


def test_rrf_fuses_dense_and_sparse_with_agreement_boost() -> None:
    chunk_a = _make_result("notes/a.md", 0, "Content A")
    chunk_b = _make_result("notes/b.md", 0, "Content B")
    chunk_c = _make_result("notes/c.md", 0, "Content C")

    dense_list = [chunk_a, chunk_b]
    sparse_list = [chunk_a, chunk_c]

    fused = reciprocal_rank_fusion([dense_list, sparse_list], k=60, top_k=5)

    assert len(fused) == 3
    assert fused[0].chunk.source_path == "notes/a.md"
    assert fused[0].score == round(2.0 / 61.0, 6)
    assert {fused[1].chunk.source_path, fused[2].chunk.source_path} == {"notes/b.md", "notes/c.md"}


def test_rrf_respects_top_k_and_handles_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

    items = [_make_result(f"notes/{i}.md", 0, f"Content {i}") for i in range(10)]
    fused = reciprocal_rank_fusion([items], k=60, top_k=4)
    assert len(fused) == 4
    assert fused[0].chunk.source_path == "notes/0.md"


def test_hybrid_search_combines_vector_and_bm25() -> None:
    client = MagicMock()
    # Mock vector search returning chunk_b
    chunk_b = _make_result("notes/b.md", 0, "Overview of Python frameworks")

    # Mock BM25 index returning chunk_a with exact keyword match
    chunk_a = _make_chunk("notes/a.md", 0, "Setup HF_TOKEN in environment")
    bm25 = BM25Index([chunk_a])

    from unittest.mock import patch
    with patch("app.hybrid.search_chunks", return_value=[chunk_b]):
        results = hybrid_search(
            client=client,
            collection_name="test_col",
            query="HF_TOKEN",
            top_k=2,
            bm25_index=bm25,
            enable_sparse=True,
        )

    assert len(results) == 2
    sources = [r.chunk.source_path for r in results]
    assert "notes/a.md" in sources
    assert "notes/b.md" in sources


def test_bm25_cache_invalidation() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    idx1 = get_bm25_index_for_collection(client, "test_col")
    idx2 = get_bm25_index_for_collection(client, "test_col")
    assert idx1 is idx2

    invalidate_bm25_cache("test_col")
    idx3 = get_bm25_index_for_collection(client, "test_col")
    assert idx1 is not idx3
