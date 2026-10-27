"""Fixture-backed incremental indexing tests without model downloads or API calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient, models

from app.indexing import index_vault_incrementally
from app.vault_session import collection_name_for_vault


def _embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    del batch_size
    vectors = []
    for text in texts:
        vector = [0.0] * 384
        vector[sum(text.encode("utf-8")) % 384] = 1.0
        vectors.append(vector)
    return vectors


def test_new_unchanged_modified_and_deleted_notes(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    collection_name = "incremental_test"
    note = tmp_path / "topic.md"
    note.write_text("# Topic\n\nInitial content.", encoding="utf-8")

    with patch("app.indexing.embed_texts", side_effect=_embed) as embed:
        first = index_vault_incrementally(client, collection_name, tmp_path)
        assert first.files_indexed == 1
        assert first.chunks_indexed == 1
        assert embed.call_count == 1

        embed.reset_mock()
        unchanged = index_vault_incrementally(client, collection_name, tmp_path)
        assert unchanged.files_unchanged == 1
        assert unchanged.chunks_indexed == 0
        embed.assert_not_called()

        note.write_text(
            "# Topic\n\nUpdated content.\n\n## Details\n\nMore information.",
            encoding="utf-8",
        )
        updated = index_vault_incrementally(client, collection_name, tmp_path)
        assert updated.files_updated == 1
        assert updated.chunks_indexed == 2
        assert client.count(collection_name).count == 2
        indexed_content = {
            point.payload["content"]
            for point in client.scroll(collection_name, with_payload=True)[0]
        }
        assert "Initial content." not in indexed_content

        note.unlink()
        deleted = index_vault_incrementally(client, collection_name, tmp_path)

    assert deleted.files_deleted == 1
    assert client.count(collection_name).count == 0


def test_failed_replacement_preserves_previously_indexed_note(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    collection_name = "failed_replacement_test"
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nOriginal source text.", encoding="utf-8")
    with patch("app.indexing.embed_texts", side_effect=_embed):
        index_vault_incrementally(client, collection_name, tmp_path)
        note.write_text("# Note\n\nChanged source text.", encoding="utf-8")
        with patch("app.indexing.embed_texts", side_effect=RuntimeError("local model failed")):
            result = index_vault_incrementally(client, collection_name, tmp_path)

    assert result.files_updated == 0
    assert len(result.failures) == 1
    assert result.failures[0].source_path == "note.md"
    assert "local model failed" in result.failures[0].reason
    points, _ = client.scroll(collection_name, with_payload=True)
    assert len(points) == 1
    assert points[0].payload["content"] == "Original source text."


def test_partial_vector_write_is_rolled_back_before_preserving_old_note(
    tmp_path: Path,
) -> None:
    client = QdrantClient(":memory:")
    collection_name = "partial_write_test"
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nOriginal source text.", encoding="utf-8")
    with patch("app.indexing.embed_texts", side_effect=_embed):
        index_vault_incrementally(client, collection_name, tmp_path)
        note.write_text(
            "\n\n".join(f"## Section {index}\n\nUpdated chunk {index}." for index in range(17)),
            encoding="utf-8",
        )
        original_upsert = client.upsert
        upsert_calls = 0

        def fail_second_upsert(**kwargs):
            nonlocal upsert_calls
            upsert_calls += 1
            if upsert_calls == 2:
                raise RuntimeError("simulated Qdrant write failure")
            return original_upsert(**kwargs)

        with (
            patch("app.indexing.embed_texts", side_effect=_embed),
            patch.object(client, "upsert", side_effect=fail_second_upsert),
        ):
            result = index_vault_incrementally(client, collection_name, tmp_path)

    assert len(result.failures) == 1
    points, _ = client.scroll(collection_name, with_payload=True)
    assert len(points) == 1
    assert points[0].payload["content"] == "Original source text."


def test_model_version_uses_separate_stable_vault_collection(tmp_path: Path) -> None:
    first_settings = SimpleNamespace(
        local_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        local_embedding_collection_prefix="recall_local",
    )
    second_settings = SimpleNamespace(
        local_embedding_model="another/model-version",
        local_embedding_collection_prefix="recall_local",
    )
    with patch("app.vault_session.get_settings", return_value=first_settings):
        first_collection = collection_name_for_vault(tmp_path)
    with patch("app.vault_session.get_settings", return_value=second_settings):
        next_collection = collection_name_for_vault(tmp_path)

    assert first_collection != next_collection
    assert first_collection != "recall_chunks"
    with patch("app.vault_session.get_settings", return_value=first_settings):
        assert first_collection == collection_name_for_vault(tmp_path)


def test_incompatible_existing_collection_fails_without_repurposing() -> None:
    client = QdrantClient(":memory:")
    collection_name = "legacy_gemini_collection"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id="2a4d5e0d-56e3-4368-8ba5-aecb05806c37",
                vector=[0.1] * 768,
                payload={"source_path": "legacy.md"},
            )
        ],
    )
    with (
        patch("app.indexing.get_settings") as indexing_settings,
    ):
        indexing_settings.return_value.local_embedding_model = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        indexing_settings.return_value.embedding_batch_size = 16
        with pytest.raises(ValueError, match="Choose a new collection and re-index"):
            index_vault_incrementally(client, collection_name, Path("tests/fixtures/vault"))

    assert client.count(collection_name).count == 1
    assert client.get_collection(collection_name).config.params.vectors.size == 768
