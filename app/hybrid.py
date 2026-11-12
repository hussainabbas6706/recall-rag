"""Hybrid search module: BM25 Okapi sparse retrieval and Reciprocal Rank Fusion (RRF)."""

import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from threading import Lock
from typing import Any

from qdrant_client import QdrantClient

from app.models import Chunk, SearchResult
from app.vectorstore import search_chunks

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric and identifier tokens."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Index:
    """In-memory BM25 Okapi index over a collection of vault Chunks."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = list(chunks)
        self.doc_count = len(self.chunks)

        self.doc_tokens: list[list[str]] = [tokenize(c.content) for c in self.chunks]
        self.doc_lens: list[int] = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(self.doc_count, 1)

        # Term document frequency: how many documents contain term t
        self.doc_freq: dict[str, int] = defaultdict(int)
        # Term frequency per document: list of Counter dictionaries
        self.term_freqs: list[Counter[str]] = []

        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freq[term] += 1

        # Precompute IDF for all observed terms
        self.idf: dict[str, float] = {}
        for term, freq in self.doc_freq.items():
            # Robertson-Spärck Jones IDF with smoothing + 1.0 to ensure positive weights
            self.idf[term] = math.log(((self.doc_count - freq + 0.5) / (freq + 0.5)) + 1.0)

    def search(self, query: str, top_k: int = 15) -> list[SearchResult]:
        """Search chunks using BM25 Okapi scoring.

        Args:
            query: Natural language or keyword query string.
            top_k: Maximum number of matching chunks to return.

        Returns:
            List of SearchResult objects sorted descending by BM25 score.
        """
        query_tokens = tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []

        scores: list[float] = [0.0] * self.doc_count

        for token in query_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]

            for doc_idx, tf_map in enumerate(self.term_freqs):
                freq = tf_map.get(token, 0)
                if freq == 0:
                    continue

                doc_len = self.doc_lens[doc_idx]
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-6))
                )
                scores[doc_idx] += token_idf * (freq * (self.k1 + 1.0) / denominator)

        # Pair chunks with non-zero scores
        scored_pairs = [
            (self.chunks[idx], score)
            for idx, score in enumerate(scores)
            if score > 0.0
        ]
        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(chunk=chunk, score=round(score, 4))
            for chunk, score in scored_pairs[:top_k]
        ]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchResult]],
    k: int = 60,
    top_k: int = 15,
) -> list[SearchResult]:
    """Combine multiple ranked SearchResult lists using Reciprocal Rank Fusion (RRF).

    Score formula for document d across rankings r in R:
        RRF(d) = sum_{r in R} 1 / (k + rank_r(d))

    Args:
        ranked_lists: Sequence of ranked SearchResult lists (e.g. [dense_results, sparse_results]).
        k: Smoothing constant preventing early ranks from dominating (default: 60).
        top_k: Maximum number of fused candidate results to return.

    Returns:
        Deduplicated SearchResults sorted descending by combined RRF score.
    """
    if not ranked_lists:
        return []

    # Map unique chunk key (source_path, chunk_index) -> (Chunk, accumulated_rrf_score)
    fused_scores: dict[tuple[str, int], float] = defaultdict(float)
    chunk_registry: dict[tuple[str, int], Chunk] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            key = (result.chunk.source_path, result.chunk.chunk_index)
            chunk_registry[key] = result.chunk
            fused_scores[key] += 1.0 / (k + rank)

    sorted_keys = sorted(fused_scores.keys(), key=lambda key: fused_scores[key], reverse=True)

    return [
        SearchResult(
            chunk=chunk_registry[key],
            score=round(fused_scores[key], 6),
        )
        for key in sorted_keys[:top_k]
    ]


_bm25_indices: dict[str, BM25Index] = {}
_indices_lock = Lock()


def get_bm25_index_for_collection(client: QdrantClient, collection_name: str) -> BM25Index:
    """Fetch or build a cached BM25 index over all chunks in a Qdrant collection."""
    with _indices_lock:
        if collection_name in _bm25_indices:
            return _bm25_indices[collection_name]

    try:
        if not client.collection_exists(collection_name=collection_name):
            empty_index = BM25Index([])
            with _indices_lock:
                _bm25_indices[collection_name] = empty_index
            return empty_index

        records = []
        offset = None
        while True:
            page, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["source_path", "heading_path", "chunk_index", "content"],
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                break
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "BM25 index initialization failed for %s (%s); falling back to empty index",
            collection_name,
            error,
        )
        return BM25Index([])

    chunks: list[Chunk] = []
    for r in records:
        if not r.payload:
            continue
        try:
            chunk = Chunk(
                source_path=r.payload["source_path"],
                heading_path=r.payload.get("heading_path", []),
                chunk_index=r.payload["chunk_index"],
                content=r.payload["content"],
            )
            chunks.append(chunk)
        except (KeyError, TypeError, ValueError):
            continue

    index = BM25Index(chunks)
    with _indices_lock:
        _bm25_indices[collection_name] = index
    return index


def invalidate_bm25_cache(collection_name: str | None = None) -> None:
    """Clear BM25 cache for a specific collection or all collections."""
    with _indices_lock:
        if collection_name:
            _bm25_indices.pop(collection_name, None)
        else:
            _bm25_indices.clear()


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 15,
    dense_top_k: int = 15,
    sparse_top_k: int = 15,
    score_threshold: float | None = None,
    embedding_model: Any = None,
    bm25_index: BM25Index | None = None,
    rrf_k: int = 60,
    enable_sparse: bool = True,
) -> list[SearchResult]:
    """Execute hybrid retrieval combining dense vector search and BM25 sparse search via RRF.

    Args:
        client: Qdrant client connection.
        collection_name: Target Qdrant collection.
        query: User search query.
        top_k: Total fused candidate results to return.
        dense_top_k: Number of candidates fetched via dense vector search.
        sparse_top_k: Number of candidates fetched via BM25 sparse search.
        score_threshold: Optional cosine score threshold for dense search.
        embedding_model: Optional embedding model provider.
        bm25_index: Optional prebuilt BM25 index (for tests or custom contexts).
        rrf_k: Reciprocal Rank Fusion constant (default: 60).
        enable_sparse: If False, falls back to dense-only retrieval.

    Returns:
        List of SearchResult objects fused via RRF.
    """
    dense_results = search_chunks(
        client=client,
        collection_name=collection_name,
        query=query,
        top_k=dense_top_k,
        score_threshold=score_threshold,
        embedding_model=embedding_model,
    )

    if not enable_sparse:
        return dense_results[:top_k]

    active_bm25 = bm25_index or get_bm25_index_for_collection(client, collection_name)
    sparse_results = active_bm25.search(query=query, top_k=sparse_top_k)

    if not sparse_results:
        return dense_results[:top_k]
    if not dense_results:
        return sparse_results[:top_k]

    return reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, top_k=top_k)
