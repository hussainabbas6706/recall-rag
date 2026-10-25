from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ingest import find_markdown_files, list_vault_entries, parse_vault_chunks
from app.main import app
from app.models import IngestRequest

FIXTURE_VAULT = Path("tests/fixtures/vault")
client = TestClient(app)


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


def test_ingest_endpoint_indexes_fixture_vault() -> None:
    settings = SimpleNamespace(
        vault_path=FIXTURE_VAULT,
        qdrant_collection="test_collection",
    )
    with (
        patch("app.main.get_settings", return_value=settings),
        patch("app.main.index_chunks", return_value=4) as mock_index,
    ):
        response = client.post("/ingest", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["files_found"] == 3
        assert data["chunks_created"] == 4
        assert data["chunks_indexed"] == 4
        mock_index.assert_called_once()


def test_ingest_endpoint_indexing_failure_returns_500() -> None:
    settings = SimpleNamespace(
        vault_path=FIXTURE_VAULT,
        qdrant_collection="test_collection",
    )
    with (
        patch("app.main.get_settings", return_value=settings),
        patch("app.main.index_chunks", side_effect=RuntimeError("Connection refused")),
    ):
        response = client.post("/ingest", json={})

        assert response.status_code == 500
        assert "Failed to index chunks into Qdrant" in response.json()["detail"]


def test_ingest_request_rejects_extra_path_fields() -> None:
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"vault_path": "C:/outside-vault"})


def test_ingest_endpoint_rejects_external_path_without_reading_or_indexing(
    tmp_path: Path,
) -> None:
    external_note = tmp_path / "outside.md"
    external_note.write_text("must not be ingested", encoding="utf-8")
    with (
        patch("app.main.find_markdown_files") as find_files,
        patch("app.main.index_chunks") as index,
    ):
        response = client.post("/ingest", json={"vault_path": str(tmp_path)})

    assert response.status_code == 422
    find_files.assert_not_called()
    index.assert_not_called()


def test_ingest_endpoint_uses_configured_vault_and_hides_invalid_path(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        vault_path=tmp_path / "missing-vault",
        qdrant_collection="test_collection",
    )
    with (
        patch("app.main.get_settings", return_value=settings),
        patch("app.main.index_chunks") as index,
    ):
        response = client.post("/ingest", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The configured vault path is missing or is not a directory."
    )
    assert str(tmp_path) not in response.text
    index.assert_not_called()


def test_list_vault_entries_returns_nested_fixture_tree() -> None:
    entries = list_vault_entries(FIXTURE_VAULT)

    assert [(entry.path, entry.kind) for entry in entries] == [
        ("career", "folder"),
        ("career/job-applications.md", "file"),
        ("projects", "folder"),
        ("projects/atlas.md", "file"),
        ("projects/recall.md", "file"),
    ]


def test_list_vault_entries_returns_empty_list_for_empty_vault(tmp_path: Path) -> None:
    assert list_vault_entries(tmp_path) == []


def test_vault_endpoint_lists_configured_vault_without_note_contents(tmp_path: Path) -> None:
    note = tmp_path / "notes" / "example.md"
    note.parent.mkdir()
    note.write_text("private note contents", encoding="utf-8")
    with patch("app.main.get_settings", return_value=SimpleNamespace(vault_path=tmp_path)):
        response = client.get("/vault")

    assert response.status_code == 200
    assert response.json() == {
        "entries": [
            {"path": "notes", "kind": "folder"},
            {"path": "notes/example.md", "kind": "file"},
        ]
    }
    assert "private note contents" not in response.text
    assert str(tmp_path) not in response.text


def test_list_vault_entries_excludes_hidden_internal_and_non_markdown_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "visible").mkdir()
    (tmp_path / "visible" / "keep.MD").write_text("# visible", encoding="utf-8")
    (tmp_path / "visible" / "image.png").write_bytes(b"not markdown")
    for excluded_name in (".obsidian", ".git", ".trash", ".private"):
        excluded = tmp_path / excluded_name
        excluded.mkdir()
        (excluded / "secret.md").write_text("# excluded", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# excluded", encoding="utf-8")

    entries = list_vault_entries(tmp_path)

    assert [(entry.path, entry.kind) for entry in entries] == [
        ("visible", "folder"),
        ("visible/keep.MD", "file"),
    ]


def test_list_vault_entries_skips_symlinks_that_point_outside_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    (outside / "secret.md").write_text("# outside", encoding="utf-8")

    try:
        (vault / "linked-folder").symlink_to(outside, target_is_directory=True)
        (vault / "linked-note.md").symlink_to(outside / "secret.md")
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Symlink creation is not supported: {error}")

    assert list_vault_entries(vault) == []
    assert find_markdown_files(vault) == []


def test_vault_endpoint_reports_missing_configured_vault(
    tmp_path: Path,
) -> None:
    configured_path = tmp_path / "missing-vault"
    with patch(
        "app.main.get_settings",
        return_value=SimpleNamespace(vault_path=configured_path),
    ):
        response = client.get("/vault")

    assert response.status_code == 400
    assert "configured vault path is missing" in response.json()["detail"]
    assert str(configured_path) not in response.text


def test_vault_endpoint_rejects_configured_path_that_is_a_file(tmp_path: Path) -> None:
    configured_path = tmp_path / "not-a-vault"
    configured_path.write_text("not a directory", encoding="utf-8")
    with patch(
        "app.main.get_settings",
        return_value=SimpleNamespace(vault_path=configured_path),
    ):
        response = client.get("/vault")

    assert response.status_code == 400
    assert str(configured_path) not in response.text
