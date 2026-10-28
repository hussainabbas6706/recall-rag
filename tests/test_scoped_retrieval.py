"""Tests for explicit file-and-heading retrieval."""

from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from app.main import app
from app.models import Chunk, SearchResult
from app.scoped_retrieval import (
    MAX_SCOPED_CONTEXT_CHARS,
    merge_scoped_and_semantic_results,
    parse_explicit_reference,
    retrieve_heading_scoped_chunks,
)
from app.vectorstore import ensure_collection_exists

client = TestClient(app, client=("127.0.0.1", 50000))
COLLECTION = "test_heading_scoped"
PARENT = "WEEKLY GOALS — Check off as you go"


def _result(
    source_path: str,
    heading_path: list[str],
    chunk_index: int,
    content: str,
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            source_path=source_path,
            heading_path=heading_path,
            chunk_index=chunk_index,
            content=content,
        ),
        score=score,
    )


def _indexed_collection() -> QdrantClient:
    qdrant = QdrantClient(":memory:")
    ensure_collection_exists(qdrant, COLLECTION)
    points = [
        ("notes/WEEKLY GOALS.md", [PARENT], 0, "Update this every Sunday."),
        ("notes/WEEKLY GOALS.md", [PARENT, "Week 1"], 1, "Week 1 goal: finish the first lesson."),
        ("notes/WEEKLY GOALS.md", [PARENT, "Week 2"], 2, "Week 2 goal: complete the project."),
        ("notes/WEEKLY GOALS.md", ["Other section"], 3, "Sibling content must not be included."),
        (
            "notes/DAILY ROUTINE.md",
            [PARENT, "Week 3"],
            0,
            "Similar but unrelated daily routine content.",
        ),
    ]
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": source_path,
                    "heading_path": heading_path,
                    "chunk_index": chunk_index,
                    "content": content,
                },
            )
            for source_path, heading_path, chunk_index, content in points
        ],
        wait=True,
    )
    return qdrant


def test_parse_explicit_file_and_heading_reference() -> None:
    question = (
        "List instructions under the 'Check off as you go' section "
        "in `WEEKLY GOALS.md`."
    )

    assert parse_explicit_reference(question) == ("WEEKLY GOALS.md", "Check off as you go")


def test_parse_explicit_reference_accepts_smart_quotes_and_unquoted_filename() -> None:
    question = (
        "Based only on my indexed Obsidian notes, find my weekly goals. "
        "List the specific instructions under the “Check off as you go” "
        "section in WEEKLY GOALS.md. If you cannot find the complete section "
        "in the retrieved context, say so instead of guessing. Include the "
        "source filename and heading for each instruction."
    )

    assert parse_explicit_reference(question) == ("WEEKLY GOALS.md", "Check off as you go")


def test_parse_explicit_reference_rejects_missing_or_ambiguous_identifiers() -> None:
    assert parse_explicit_reference("What does WEEKLY GOALS.md say?") is None
    assert (
        parse_explicit_reference(
            "Compare the 'Check off as you go' section in `WEEKLY GOALS.md` "
            "with the 'Sunday plan' section in `DAILY ROUTINE.md`."
        )
        is None
    )


def test_scoped_retrieval_includes_nested_chunks_not_siblings_or_other_notes() -> None:
    qdrant = _indexed_collection()

    results = retrieve_heading_scoped_chunks(
        qdrant,
        COLLECTION,
        "WEEKLY GOALS.md",
        "Check off as you go",
    )

    assert results is not None
    assert [(result.chunk.source_path, result.chunk.chunk_index) for result in results] == [
        ("notes/WEEKLY GOALS.md", 0),
        ("notes/WEEKLY GOALS.md", 1),
        ("notes/WEEKLY GOALS.md", 2),
    ]


def test_scoped_retrieval_resolves_unique_decorated_filename() -> None:
    qdrant = QdrantClient(":memory:")
    ensure_collection_exists(qdrant, COLLECTION)
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": "notes/🗓️ WEEKLY GOALS.md",
                    "heading_path": [PARENT],
                    "chunk_index": 0,
                    "content": "Update this every Sunday.",
                },
            ),
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": "notes/🗓️ WEEKLY GOALS.md",
                    "heading_path": [PARENT, "Week 1"],
                    "chunk_index": 1,
                    "content": "Week 1 goal: finish the first lesson.",
                },
            ),
        ],
        wait=True,
    )

    results = retrieve_heading_scoped_chunks(
        qdrant,
        COLLECTION,
        "WEEKLY GOALS.md",
        "Check off as you go",
    )

    assert results is not None
    assert [(result.chunk.source_path, result.chunk.chunk_index) for result in results] == [
        ("notes/🗓️ WEEKLY GOALS.md", 0),
        ("notes/🗓️ WEEKLY GOALS.md", 1),
    ]


def test_scoped_retrieval_does_not_guess_missing_or_ambiguous_targets() -> None:
    qdrant = _indexed_collection()
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": "archive/WEEKLY GOALS.md",
                    "heading_path": [PARENT],
                    "chunk_index": 0,
                    "content": "A second file with the same basename.",
                },
            )
        ],
        wait=True,
    )

    assert (
        retrieve_heading_scoped_chunks(
            qdrant, COLLECTION, "WEEKLY GOALS.md", "Check off as you go"
        )
        is None
    )
    assert (
        retrieve_heading_scoped_chunks(
            qdrant, COLLECTION, "notes/WEEKLY GOALS.md", "No such heading"
        )
        is None
    )

    qdrant_with_decorated_duplicate = QdrantClient(":memory:")
    ensure_collection_exists(qdrant_with_decorated_duplicate, COLLECTION)
    qdrant_with_decorated_duplicate.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": "notes/WEEKLY GOALS.md",
                    "heading_path": [PARENT],
                    "chunk_index": 0,
                    "content": "Undecorated copy.",
                },
            ),
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": "notes/🗓️ WEEKLY GOALS.md",
                    "heading_path": [PARENT],
                    "chunk_index": 0,
                    "content": "Decorated copy.",
                },
            ),
        ],
        wait=True,
    )
    assert (
        retrieve_heading_scoped_chunks(
            qdrant_with_decorated_duplicate,
            COLLECTION,
            "WEEKLY GOALS.md",
            "Check off as you go",
        )
        is None
    )


def test_scoped_retrieval_rejects_ambiguous_heading_paths() -> None:
    qdrant = _indexed_collection()
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 383,
                payload={
                    "source_path": "notes/AMBIGUOUS.md",
                    "heading_path": [parent, "Target"],
                    "chunk_index": index,
                    "content": f"Target under {parent}.",
                },
            )
            for index, parent in enumerate(["Parent A", "Parent B"])
        ],
        wait=True,
    )

    assert (
        retrieve_heading_scoped_chunks(
            qdrant, COLLECTION, "AMBIGUOUS.md", "Target"
        )
        is None
    )


def test_merge_prefers_scoped_chunks_deduplicates_and_bounds_context() -> None:
    scoped = [
        _result("notes/WEEKLY GOALS.md", [PARENT], 0, "parent"),
        _result("notes/WEEKLY GOALS.md", [PARENT, "Week 1"], 1, "x" * 8),
        _result("notes/WEEKLY GOALS.md", [PARENT, "Week 2"], 2, "y" * 8),
    ]
    semantic = [
        _result("notes/WEEKLY GOALS.md", [PARENT], 0, "parent", score=0.8),
        _result("notes/DAILY ROUTINE.md", ["Sunday"], 0, "unrelated", score=0.7),
    ]

    results = merge_scoped_and_semantic_results(semantic, scoped, max_context_chars=16)

    assert [(result.chunk.source_path, result.chunk.chunk_index) for result in results] == [
        ("notes/WEEKLY GOALS.md", 0),
        ("notes/WEEKLY GOALS.md", 1),
    ]
    assert sum(len(result.chunk.content) for result in results) <= 16
    assert len({(result.chunk.source_path, result.chunk.chunk_index) for result in results}) == len(
        results
    )


def test_large_scoped_section_obeys_configured_context_bound() -> None:
    scoped = [
        _result("notes/WEEKLY GOALS.md", [PARENT, f"Week {index}"], index, "x" * 1000)
        for index in range(20)
    ]

    results = merge_scoped_and_semantic_results([], scoped)

    assert sum(len(result.chunk.content) for result in results) <= MAX_SCOPED_CONTEXT_CHARS
    assert len(results) == MAX_SCOPED_CONTEXT_CHARS // 1000


def test_chat_uses_scoped_evidence_and_cites_exact_context(selected_fixture_vault) -> None:
    qdrant = _indexed_collection()
    unrelated_content = "DAILY NOISE " + ("x" * MAX_SCOPED_CONTEXT_CHARS)
    semantic = [
        _result("notes/DAILY ROUTINE.md", ["Sunday"], 0, unrelated_content, 0.9),
        _result(
            "notes/WEEKLY GOALS.md",
            [PARENT],
            0,
            "Update this every Sunday.",
            0.8,
        ),
    ]
    with (
        patch("app.main.get_qdrant_client", return_value=qdrant),
        patch("app.main.collection_name_for_vault", return_value=COLLECTION),
        patch("app.main.search_chunks", return_value=semantic),
        patch("app.main.generate_answer", return_value="Week one and two [1][2][3].") as generate,
    ):
        response = client.post(
            "/chat",
            json={
                "query": (
                    "List the specific instructions under the “Check off as you go” "
                    "section in WEEKLY GOALS.md. If you cannot find the complete "
                    "section in the retrieved context, say so instead of guessing."
                ),
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )

    assert response.status_code == 200
    prompt = generate.call_args.args[0]
    assert "DAILY NOISE" not in prompt
    assert "Update this every Sunday." in prompt
    assert "Week 1 goal: finish the first lesson." in prompt
    assert "Week 2 goal: complete the project." in prompt
    assert "Sibling content must not be included." not in prompt
    assert "Similar but unrelated daily routine content." not in prompt
    citations = response.json()["citations"]
    assert [(item["source_path"], item["heading_path"]) for item in citations] == [
        ("notes/WEEKLY GOALS.md", [PARENT]),
        ("notes/WEEKLY GOALS.md", [PARENT, "Week 1"]),
        ("notes/WEEKLY GOALS.md", [PARENT, "Week 2"]),
    ]
    for reference, citation in enumerate(citations, start=1):
        context_line = f"[{reference}] {citation['source_path']} | "
        assert context_line in prompt


def test_chat_discloses_when_explicit_target_cannot_be_resolved(selected_fixture_vault) -> None:
    qdrant = _indexed_collection()
    semantic = [_result("notes/DAILY ROUTINE.md", ["Sunday"], 0, "Sunday routine.")]
    with (
        patch("app.main.get_qdrant_client", return_value=qdrant),
        patch("app.main.collection_name_for_vault", return_value=COLLECTION),
        patch("app.main.search_chunks", return_value=semantic),
        patch("app.main.generate_answer", return_value="The target was not identified.") as generate,
    ):
        response = client.post(
            "/chat",
            json={
                "query": "What is in the 'Missing section' section of `WEEKLY GOALS.md`?",
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )

    assert response.status_code == 200
    prompt = generate.call_args.args[0]
    assert "could not be resolved uniquely" in prompt
    assert "Do not present other sources as that target" in prompt
    assert "Sunday routine." in prompt
    assert response.json()["citations"] == [
        {
            "reference": 1,
            "source_path": "notes/DAILY ROUTINE.md",
            "heading_path": ["Sunday"],
        }
    ]


def test_ordinary_chat_keeps_semantic_results(selected_fixture_vault) -> None:
    semantic = [_result("notes/a.md", ["A"], 0, "Semantic result.")]
    with (
        patch("app.main.search_chunks", return_value=semantic) as search,
        patch("app.main.retrieve_heading_scoped_chunks") as scoped,
        patch("app.main.generate_answer", return_value="Answer [1]."),
    ):
        response = client.post(
            "/chat",
            json={
                "query": "What does my note say about weekly planning?",
                "vault_id": str(selected_fixture_vault.vault_id),
            },
        )

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_path"] == "notes/a.md"
    search.assert_called_once()
    scoped.assert_not_called()
