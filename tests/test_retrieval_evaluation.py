"""Small fixture benchmark for deterministic retrieval regression checks."""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from app.ingest import parse_vault_chunks
from app.vectorstore import index_chunks, search_chunks

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
EVALUATION_CASES = Path(__file__).parent / "fixtures" / "retrieval_eval.json"


def test_fixture_retrieval_hit_at_3_and_mean_reciprocal_rank(
    deterministic_embeddings,
) -> None:
    cases = json.loads(EVALUATION_CASES.read_text(encoding="utf-8"))
    chunks = parse_vault_chunks(FIXTURE_VAULT)
    qdrant = QdrantClient(":memory:")
    collection_name = "fixture_retrieval_evaluation"

    indexed = index_chunks(
        client=qdrant,
        collection_name=collection_name,
        chunks=chunks,
        embedding_model=deterministic_embeddings,
    )
    assert indexed == len(chunks)

    reciprocal_ranks: list[float] = []
    hits_at_3 = 0
    per_query_ranks: dict[str, int | None] = {}
    for case in cases:
        results = search_chunks(
            client=qdrant,
            collection_name=collection_name,
            query=case["query"],
            top_k=3,
            embedding_model=deterministic_embeddings,
        )
        ranked_sources = [result.chunk.source_path for result in results]
        try:
            rank = ranked_sources.index(case["expected_source"]) + 1
        except ValueError:
            rank = None

        per_query_ranks[case["query"]] = rank
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)

    hit_at_3 = hits_at_3 / len(cases)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(cases)
    print(
        f"Fixture retrieval: Hit@3={hit_at_3:.2f}, "
        f"MRR={mean_reciprocal_rank:.2f}, ranks={per_query_ranks}"
    )

    assert hit_at_3 == 1.0, per_query_ranks
