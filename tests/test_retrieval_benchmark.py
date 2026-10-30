"""Test suite integration for retrieval benchmark."""

from pathlib import Path

from eval.run_retrieval_eval import run_benchmark


def test_retrieval_benchmark_mock(tmp_path: Path) -> None:
    """Ensure retrieval evaluation benchmark runs cleanly and meets baseline thresholds."""
    queries_file = Path("eval/fixtures/retrieval_eval_queries.json")
    assert queries_file.exists(), "Benchmark queries fixture missing."

    output_file = tmp_path / "baseline.json"
    report = run_benchmark(
        queries_file=queries_file,
        use_mock=True,
        top_k=5,
        save_output=output_file,
    )

    assert report["total_queries"] >= 15
    assert report["overall_recall"] >= 0.70
    assert report["overall_mrr"] >= 0.70
    assert output_file.exists()

