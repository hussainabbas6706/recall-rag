from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.indexing import IndexingFailure, IndexingResult
from app.ingest import find_markdown_files, list_vault_entries, parse_vault_chunks
from app.main import app
from app.models import IngestRequest
from app.vault_session import collection_name_for_vault

FIXTURE_VAULT = Path("tests/fixtures/vault")
client = TestClient(app, client=("127.0.0.1", 50000))


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


def test_ingest_endpoint_indexes_fixture_vault(selected_fixture_vault) -> None:
    with (
        patch(
            "app.main.index_vault_incrementally",
            return_value=IndexingResult(files_found=3, files_indexed=3, chunks_indexed=4),
        ) as mock_index,
    ):
        response = client.post("/ingest", json={"vault_id": str(selected_fixture_vault.vault_id)})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["files_found"] == 3
        assert data["chunks_created"] == 4
        assert data["chunks_indexed"] == 4
        assert data["files_indexed"] == 3
        mock_index.assert_called_once()
        assert mock_index.call_args.kwargs["collection_name"] == (
            collection_name_for_vault(selected_fixture_vault.path)
        )


def test_ingest_endpoint_indexing_failure_returns_500(selected_fixture_vault) -> None:
    with (
        patch(
            "app.main.index_vault_incrementally",
            side_effect=RuntimeError("Connection refused"),
        ),
    ):
        response = client.post("/ingest", json={"vault_id": str(selected_fixture_vault.vault_id)})

        assert response.status_code == 500
        assert "Failed to index chunks into Qdrant" in response.json()["detail"]


def test_ingest_endpoint_reports_partial_note_failures(selected_fixture_vault) -> None:
    result = IndexingResult(
        files_found=2,
        files_indexed=1,
        chunks_indexed=3,
        failures=[IndexingFailure("broken.md", "UnicodeError: invalid text encoding")],
    )
    with patch("app.main.index_vault_incrementally", return_value=result):
        response = client.post(
            "/ingest",
            json={"vault_id": str(selected_fixture_vault.vault_id)},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "partial"
    assert data["files_indexed"] == 1
    assert data["failures"] == [
        {"source_path": "broken.md", "reason": "UnicodeError: invalid text encoding"}
    ]


def test_ingest_request_rejects_extra_path_fields() -> None:
    with pytest.raises(ValidationError):
        IngestRequest.model_validate(
            {
                "vault_id": "f9cb6e98-4955-4a62-a735-c0cb72baaf06",
                "vault_path": "C:/outside-vault",
            }
        )


def test_ingest_endpoint_rejects_external_path_without_reading_or_indexing(
    tmp_path: Path, selected_fixture_vault
) -> None:
    external_note = tmp_path / "outside.md"
    external_note.write_text("must not be ingested", encoding="utf-8")
    with (
        patch("app.indexing.find_markdown_files") as find_files,
        patch("app.main.index_vault_incrementally") as index,
    ):
        response = client.post(
            "/ingest",
            json={"vault_id": str(selected_fixture_vault.vault_id), "vault_path": str(tmp_path)},
        )

    assert response.status_code == 422
    find_files.assert_not_called()
    index.assert_not_called()


def test_ingest_endpoint_requires_current_vault_selection() -> None:
    from uuid import uuid4

    with (
        patch("app.main.index_vault_incrementally") as index,
    ):
        response = client.post("/ingest", json={"vault_id": str(uuid4())})

    assert response.status_code == 409
    assert response.json()["detail"] == "This vault selection is no longer active. Select the vault again."
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


def test_vault_select_endpoint_returns_only_relative_tree_data(tmp_path: Path) -> None:
    note = tmp_path / "projects" / "fixture.md"
    note.parent.mkdir()
    note.write_text("private fixture note content", encoding="utf-8")
    with patch("app.main.choose_vault_directory", return_value=tmp_path):
        response = client.post("/vault/select")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == tmp_path.name
    assert data["entries"] == [
        {"path": "projects", "kind": "folder"},
        {"path": "projects/fixture.md", "kind": "file"},
    ]
    assert "private fixture note content" not in response.text
    assert str(tmp_path) not in response.text


def test_vault_selection_endpoint_restores_active_selection(
    selected_fixture_vault,
) -> None:
    response = client.get("/vault/selection")

    assert response.status_code == 200
    assert response.json()["vault_id"] == str(selected_fixture_vault.vault_id)
    assert response.json()["name"] == selected_fixture_vault.path.name
    assert all(
        not Path(entry["path"]).is_absolute()
        for entry in response.json()["entries"]
    )


def test_vault_selection_endpoint_returns_no_content_when_unselected() -> None:
    assert client.get("/vault/selection").status_code == 204


def test_vault_select_cancellation_preserves_active_selection(
    selected_fixture_vault,
) -> None:
    with patch("app.main.choose_vault_directory", return_value=None):
        response = client.post("/vault/select")

    assert response.status_code == 204
    restored = client.get("/vault/selection")
    assert restored.status_code == 200
    assert restored.json()["vault_id"] == str(selected_fixture_vault.vault_id)


def test_vault_select_rejects_non_local_origin() -> None:
    non_local_client = TestClient(app, client=("127.0.0.1", 50000))
    with patch("app.main.choose_vault_directory") as choose:
        response = non_local_client.post(
            "/vault/select",
            headers={"Origin": "https://example.com"},
        )

    assert response.status_code == 403
    choose.assert_not_called()


def test_vault_select_rejects_non_loopback_client() -> None:
    remote_client = TestClient(app, client=("192.0.2.10", 50000))
    with patch("app.main.choose_vault_directory") as choose:
        response = remote_client.post("/vault/select")

    assert response.status_code == 403
    choose.assert_not_called()


def test_vault_select_rejects_invalid_folder_without_replacing_selection(
    tmp_path: Path, selected_fixture_vault
) -> None:
    not_a_directory = tmp_path / "note.md"
    not_a_directory.write_text("fixture", encoding="utf-8")
    with patch("app.main.choose_vault_directory", return_value=not_a_directory):
        response = client.post("/vault/select")

    assert response.status_code == 400
    assert str(tmp_path) not in response.text
    restored = client.get("/vault/selection")
    assert restored.status_code == 200
    assert restored.json()["vault_id"] == str(selected_fixture_vault.vault_id)


def test_vault_switch_issues_new_id_and_invalidates_previous_selection(
    tmp_path: Path, selected_fixture_vault
) -> None:
    old_vault_id = selected_fixture_vault.vault_id
    next_vault = tmp_path / "next-vault"
    next_vault.mkdir()
    with patch("app.main.choose_vault_directory", return_value=next_vault):
        response = client.post("/vault/select")

    assert response.status_code == 200
    new_vault_id = response.json()["vault_id"]
    assert new_vault_id != str(old_vault_id)
    with patch("app.main.search_chunks") as search:
        stale_search = client.post(
            "/search",
            json={"query": "old notes", "vault_id": str(old_vault_id)},
        )
    assert stale_search.status_code == 409
    search.assert_not_called()


def test_vault_select_reports_unavailable_native_picker() -> None:
    with patch(
        "app.main.choose_vault_directory",
        side_effect=RuntimeError("native details must not leak"),
    ):
        response = client.post("/vault/select")

    assert response.status_code == 503
    assert "native details must not leak" not in response.text


def test_vault_select_rejects_symlink_root_when_supported(tmp_path: Path) -> None:
    linked_vault = tmp_path / "linked-vault"
    try:
        linked_vault.symlink_to(FIXTURE_VAULT.resolve(), target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Symlink creation is not supported: {error}")

    with patch("app.main.choose_vault_directory", return_value=linked_vault):
        response = client.post("/vault/select")

    assert response.status_code == 400
    assert "symbolic link" in response.json()["detail"]
    assert str(tmp_path) not in response.text
