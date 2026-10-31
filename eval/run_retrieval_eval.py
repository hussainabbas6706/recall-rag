"""Standalone retrieval benchmark runner for Recall.

Measures Recall@k, Precision@k, and MRR across categorized test queries
using either local neural embeddings (all-MiniLM-L6-v2) or deterministic mock vectors.
Saves results to eval/results/ for milestone comparison.
"""

import argparse
import json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.embeddings import get_embedding_model
from app.ingest import parse_vault_chunks
from app.models import SearchResult
from app.reranker import get_reranker_model, rerank_search_results
from app.scoped_retrieval import parse_explicit_reference, retrieve_heading_scoped_chunks
from app.vectorstore import index_chunks, search_chunks
from eval.fixtures.synthetic_vault import build_synthetic_vault

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval")


class MockDeterministicEmbeddings:
    """Fast hashing embedder for quick verification without loading neural models."""

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        del batch_size
        return [self._embed(text, 384) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, 384)

    @staticmethod
    def _embed(text: str, dimensions: int) -> list[float]:
        import hashlib
        import re

        vector = [0.0] * dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % dimensions
            vector[index] += 1.0
        return vector


def evaluate_query(
    case: dict[str, Any],
    client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
    embedding_model: Any = None,
    use_reranker: bool = False,
    reranker_model: Any = None,
    candidate_k: int = 15,
) -> dict[str, Any]:
    """Execute end-to-end retrieval for a single evaluation test case."""
    query = case["query"]
    expected_sources = case.get("expected_sources", [])
    query_type = case.get("query_type", "general")

    retrieved_results: list[SearchResult] = []
    route_used = "semantic_search"

    explicit_ref = parse_explicit_reference(query)
    if explicit_ref is not None:
        file_ref, heading_ref = explicit_ref
        scoped_results = retrieve_heading_scoped_chunks(
            client=client,
            collection_name=collection_name,
            file_reference=file_ref,
            heading_reference=heading_ref,
        )
        if scoped_results is not None:
            retrieved_results = scoped_results[:top_k]
            route_used = "scoped_retrieval"
        else:
            route_used = "scoped_unresolved"
            retrieved_results = []
    else:
        fetch_k = candidate_k if use_reranker else top_k
        candidates = search_chunks(
            client=client,
            collection_name=collection_name,
            query=query,
            top_k=fetch_k,
            embedding_model=embedding_model,
        )
        if use_reranker and candidates:
            retrieved_results = rerank_search_results(
                query=query,
                results=candidates,
                top_k=top_k,
                reranker=reranker_model,
            )
            route_used = "two_stage_reranked"
        else:
            retrieved_results = candidates[:top_k]

    retrieved_sources = [r.chunk.source_path for r in retrieved_results]
    retrieved_top_k = retrieved_sources[:top_k]

    # Metrics calculation
    if not expected_sources:
        # For unanswerable/missing queries: success means 0 retrieved sources or no false hits
        is_unanswerable = query_type in ("unanswerable", "missing_target")
        recall = 1.0 if (not retrieved_top_k or is_unanswerable) else 0.0
        precision = 1.0 if not retrieved_top_k else 0.0
        mrr = 1.0 if (not retrieved_top_k or is_unanswerable) else 0.0
        hit = 1 if (not retrieved_top_k or is_unanswerable) else 0
    else:
        hits = sum(1 for source in expected_sources if source in retrieved_top_k)
        recall = hits / len(expected_sources)
        precision = hits / max(len(retrieved_top_k), 1)
        hit = 1 if hits > 0 else 0

        first_rank = None
        for rank, source in enumerate(retrieved_sources, start=1):
            if source in expected_sources:
                first_rank = rank
                break
        mrr = (1.0 / first_rank) if first_rank is not None else 0.0

    return {
        "id": case.get("id"),
        "query": query,
        "query_type": query_type,
        "route_used": route_used,
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_top_k,
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


class MockReranker:
    """Mock reranker that scores candidate pairs by token overlap for fast testing."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        results = []
        for query, doc in pairs:
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            results.append(float(overlap))
        return results


def run_benchmark(
    queries_file: Path,
    use_mock: bool = False,
    use_reranker: bool = False,
    candidate_k: int = 15,
    top_k: int = 5,
    save_output: Path | None = None,
) -> dict[str, Any]:
    """Run full benchmark evaluation on synthetic vault."""
    queries: list[dict[str, Any]] = json.loads(queries_file.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        vault_path = build_synthetic_vault(Path(tmp_dir) / "synthetic_vault")
        chunks = parse_vault_chunks(vault_path)

        embedding_model = MockDeterministicEmbeddings() if use_mock else get_embedding_model()
        reranker_model = MockReranker() if use_mock else (get_reranker_model() if use_reranker else None)
        qdrant = QdrantClient(":memory:")
        collection = "benchmark_eval"

        indexed = index_chunks(
            client=qdrant,
            collection_name=collection,
            chunks=chunks,
            embedding_model=embedding_model,
        )

        results = [
            evaluate_query(
                case=case,
                client=qdrant,
                collection_name=collection,
                top_k=top_k,
                embedding_model=embedding_model,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
                candidate_k=candidate_k,
            )
            for case in queries
        ]

    # Aggregate metrics
    overall_recall = sum(r["recall"] for r in results) / len(results)
    overall_precision = sum(r["precision"] for r in results) / len(results)
    overall_mrr = sum(r["mrr"] for r in results) / len(results)
    overall_hit_rate = sum(r["hit"] for r in results) / len(results)

    # Category breakdown
    category_metrics: dict[str, dict[str, float]] = defaultdict(lambda: {"recall": 0.0, "mrr": 0.0, "count": 0.0})
    for r in results:
        q_type = r["query_type"]
        category_metrics[q_type]["recall"] += r["recall"]
        category_metrics[q_type]["mrr"] += r["mrr"]
        category_metrics[q_type]["count"] += 1.0

    category_summary = {}
    for q_type, data in sorted(category_metrics.items()):
        count = data["count"]
        category_summary[q_type] = {
            "count": int(count),
            "recall": round(data["recall"] / count, 3),
            "mrr": round(data["mrr"] / count, 3),
        }

    pipeline_name = "Two-Stage Retrieve + Rerank" if use_reranker else "Single-Stage Dense"
    model_name = "deterministic_mock" if use_mock else "sentence-transformers/all-MiniLM-L6-v2"
    if use_reranker and not use_mock:
        model_name += " + cross-encoder/ms-marco-MiniLM-L-6-v2"

    report = {
        "pipeline": pipeline_name,
        "model": model_name,
        "use_reranker": use_reranker,
        "top_k": top_k,
        "candidate_k": candidate_k if use_reranker else top_k,
        "total_queries": len(results),
        "total_chunks_indexed": indexed,
        "overall_recall": round(overall_recall, 3),
        "overall_precision": round(overall_precision, 3),
        "overall_mrr": round(overall_mrr, 3),
        "overall_hit_rate": round(overall_hit_rate, 3),
        "by_category": category_summary,
        "details": results,
    }

    if save_output:
        save_output.parent.mkdir(parents=True, exist_ok=True)
        save_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print a clean CLI table of benchmark results."""
    if sys.stdout.encoding.lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    print("\n" + "=" * 70)
    print(f"RECALL RETRIEVAL BENCHMARK — {report.get('pipeline', 'BASELINE')} ({report['model']})")
    print("=" * 70)
    print(f"Total Queries:         {report['total_queries']}")
    print(f"Top-K Limit:           {report['top_k']}")
    if report.get("use_reranker"):
        print(f"Candidate K (Stage 1): {report.get('candidate_k')}")
    print(f"Overall Recall@{report['top_k']}:     {report['overall_recall'] * 100:.1f}%")
    print(f"Overall MRR:           {report['overall_mrr']:.3f}")
    print(f"Overall Hit Rate:      {report['overall_hit_rate'] * 100:.1f}%")
    print("-" * 70)
    print(f"{'Category':<24} {'Count':<8} {'Recall@k':<12} {'MRR':<10}")
    print("-" * 70)
    for category, stats in report["by_category"].items():
        print(
            f"{category:<24} {stats['count']:<8} "
            f"{stats['recall'] * 100:>6.1f}%     {stats['mrr']:>6.3f}"
        )
    print("=" * 70)

    print("\nDetailed Query Results:")
    for d in report["details"]:
        status = "PASS" if d["hit"] else "FAIL"
        safe_query = d["query"].encode("ascii", errors="replace").decode("ascii")
        safe_expected = [s.encode("ascii", errors="replace").decode("ascii") for s in d["expected_sources"]]
        safe_retrieved = [s.encode("ascii", errors="replace").decode("ascii") for s in d["retrieved_sources"]]
        print(f"[{status}] ({d['query_type']}) {safe_query}")
        print(f"       Expected:  {safe_expected}")
        print(f"       Retrieved: {safe_retrieved}")
        print(f"       Recall: {d['recall']:.2f} | MRR: {d['mrr']:.2f} | Route: {d['route_used']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall Retrieval Evaluation Benchmark")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("eval/fixtures/retrieval_eval_queries.json"),
        help="Path to evaluation queries JSON",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock embeddings instead of all-MiniLM-L6-v2 (fast mode)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable two-stage Cross-Encoder reranking",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=15,
        help="Stage 1 over-fetch candidate limit before reranking (default: 15)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top k retrieval count (default: 5)")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Path to save output JSON metrics",
    )
    args = parser.parse_args()

    save_path = args.save
    if save_path is None:
        save_path = Path("eval/results/two_stage_reranked.json") if args.rerank else Path("eval/results/baseline_dense.json")

    report = run_benchmark(
        queries_file=args.queries,
        use_mock=args.mock,
        use_reranker=args.rerank,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        save_output=save_path,
    )
    print_report(report)


if __name__ == "__main__":
    main()
