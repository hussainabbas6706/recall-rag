from pathlib import Path

from fastapi.testclient import TestClient

from app.ingest import find_markdown_files, parse_vault_chunks
from app.main import app

FIXTURE_VAULT = Path("tests/fixtures/vault")


def test_find_markdown_files_returns_fixture_notes() -> None:
    files = find_markdown_files(FIXTURE_VAULT)

    assert [file.relative_to(FIXTURE_VAULT).as_posix() for file in files] == [
        "career/job-applications.md",
        "projects/atlas.md",
        "projects/recall.md",
    ]


def test_parse_vault_chunks_fixture_vault() -> None:
    chunks = parse_vault_chunks(FIXTURE_VAULT)

    # 3 notes in fixture vault produce 4 total chunks
    assert len(chunks) == 4

    # Career note
    career_chunk = next(c for c in chunks if c.source_path == "career/job-applications.md")
    assert career_chunk.heading_path == ["Job applications", "Targets"]
    assert career_chunk.chunk_index == 0
    assert "Acme AI" in career_chunk.content

    # Atlas note
    atlas_chunk = next(c for c in chunks if c.source_path == "projects/atlas.md")
    assert atlas_chunk.heading_path == ["Atlas", "Architecture"]
    assert atlas_chunk.chunk_index == 0
    assert "LangGraph" in atlas_chunk.content

    # Recall note (has two sections: MVP and Deadline)
    recall_chunks = [c for c in chunks if c.source_path == "projects/recall.md"]
    assert len(recall_chunks) == 2
    assert recall_chunks[0].heading_path == ["Recall", "MVP"]
    assert recall_chunks[0].chunk_index == 0
    assert recall_chunks[1].heading_path == ["Recall", "Deadline"]
    assert recall_chunks[1].chunk_index == 1


def test_ingest_endpoint_chunks_fixture_vault() -> None:
    client = TestClient(app)
    response = client.post("/ingest", json={"vault_path": str(FIXTURE_VAULT)})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "chunked"
    assert data["files_found"] == 3
    assert data["chunks_created"] == 4


def test_ingest_endpoint_nonexistent_vault() -> None:
    client = TestClient(app)
    response = client.post("/ingest", json={"vault_path": "non/existent/path/xyz"})

    assert response.status_code == 400

