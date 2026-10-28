"""Deterministic retrieval evaluation for the current dense-search baseline."""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from app.ingest import parse_vault_chunks
from app.scoped_retrieval import parse_explicit_reference, retrieve_heading_scoped_chunks
from app.vectorstore import index_chunks, search_chunks

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
EVALUATION_CASES = Path(__file__).parent / "fixtures" / "retrieval_eval.json"


def _build_synthetic_vault(root: Path) -> Path:
    """Create a small benchmark vault with decorated and undecorated filenames."""
    vault = root / "synthetic_eval_vault"
    (vault / "notes").mkdir(parents=True, exist_ok=True)
    (vault / "notes" / "archive").mkdir(exist_ok=True)

    files = {
        "notes/weekly-goals.md": """# Weekly goals

## Launch checklist

- confirm the launch checklist is complete
- review the final sign-off with stakeholders
- publish the release notes for the team

## Daily focus

- finish the key tasks before the review
- update the recap in the next standup
""",
        "notes/🗓️ WEEKLY GOALS.md": """# Weekly goals

## Launch checklist

- confirm the launch checklist is complete
- review the final sign-off with stakeholders
- publish the release notes for the team

## Weekly rhythm

- track the sprint work in the delivery log
- summarize progress before Friday
""",
        "notes/daily-routine.md": """# Daily routine

## Morning review

- check the next delivery window
- note blockers before the standup

## Evening recap

- update the task list for the next day
- log follow-ups from the review
""",
        "notes/master-timeline.md": """# Master timeline

## Milestones

- kickoff starts after the planning review
- release goes live after final QA
- follow-up review closes the project
""",
        "notes/project-plan.md": """# Project plan

## Decision log

- keep the launch plan local to the repository notes
- confirm the final review before publishing

## Risk register

- the status depends on the launch checklist and review timing
""",
        "notes/archive/old-plan.md": """# Old plan

## Archive

- this note is historical and not part of the current launch plan
""",
    }

    for relative_path, content in files.items():
        file_path = vault / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    return vault


def _query_case_summary(results, case: dict) -> dict:
    expected_sources = case.get("expected_sources", [])
    retrieved_sources = [result.chunk.source_path for result in results]
    retrieved_top_3 = retrieved_sources[:3]

    hits = sum(1 for source in expected_sources if source in retrieved_top_3)
    recall_at_3 = hits / len(expected_sources) if expected_sources else 1.0 if not retrieved_top_3 else 0.0
    context_precision = hits / max(len(retrieved_top_3), 1)

    first_relevant_rank = None
    for index, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            first_relevant_rank = index
            break
    reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank is not None else 0.0

    return {
        "query": case["query"],
        "query_type": case.get("query_type", "general"),
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "recall_at_3": recall_at_3,
        "context_precision": context_precision,
        "mrr": reciprocal_rank,
    }


def test_retrieval_evaluation_baseline(tmp_path, deterministic_embeddings) -> None:
    cases = json.loads(EVALUATION_CASES.read_text(encoding="utf-8"))
    vault_path = _build_synthetic_vault(tmp_path)
    chunks = parse_vault_chunks(vault_path)
    qdrant = QdrantClient(":memory:")
    collection_name = "synthetic_retrieval_evaluation"

    indexed = index_chunks(
        client=qdrant,
        collection_name=collection_name,
        chunks=chunks,
        embedding_model=deterministic_embeddings,
    )
    assert indexed == len(chunks)

    summary: list[dict] = []
    scope_passes: list[bool] = []
    explicit_queries = 0

    for case in cases:
        results = search_chunks(
            client=qdrant,
            collection_name=collection_name,
            query=case["query"],
            top_k=5,
            embedding_model=deterministic_embeddings,
        )
        summary.append(_query_case_summary(results, case))

        if case.get("expected_target"):
            explicit_queries += 1
            ref = parse_explicit_reference(case["query"])
            resolved = retrieve_heading_scoped_chunks(
                qdrant,
                collection_name,
                case["expected_target"]["file"],
                case["expected_target"]["heading"],
            )
            expected_target = (
                case["expected_target"]["file"],
                case["expected_target"]["heading"],
            )
            scope_passes.append(ref == expected_target and resolved is not None)

    recall_values = [item["recall_at_3"] for item in summary]
    precision_values = [item["context_precision"] for item in summary]
    mrr_values = [item["mrr"] for item in summary]
    overall_recall_at_3 = sum(recall_values) / len(recall_values)
    mean_context_precision = sum(precision_values) / len(precision_values)
    mean_reciprocal_rank = sum(mrr_values) / len(mrr_values)
    scope_resolution_rate = (sum(scope_passes) / len(scope_passes)) if scope_passes else 0.0

    print(
        "Synthetic retrieval baseline: "
        f"cases={len(cases)} recall@3={overall_recall_at_3:.2f} "
        f"context_precision={mean_context_precision:.2f} "
        f"mrr={mean_reciprocal_rank:.2f} "
        f"scope_resolution={scope_resolution_rate:.2f}"
    )
    for item in summary:
        safe_query = item["query"].encode("ascii", "ignore").decode("ascii")
        safe_sources = [
            source.encode("ascii", "ignore").decode("ascii") for source in item["retrieved_sources"][:3]
        ]
        print(
            f"{item['query_type']}: {safe_query} -> "
            f"recall@3={item['recall_at_3']:.2f}, precision={item['context_precision']:.2f}, "
            f"mrr={item['mrr']:.2f}, sources={safe_sources}"
        )

    assert len(cases) >= 15
    assert explicit_queries >= 1
    assert 0.0 <= overall_recall_at_3 <= 1.0
    assert 0.0 <= mean_context_precision <= 1.0
    assert 0.0 <= mean_reciprocal_rank <= 1.0
    assert 0.0 <= scope_resolution_rate <= 1.0
