"""Shared deterministic test helpers."""

import hashlib
import re

import pytest

from app.vault_session import vault_session


class DeterministicEmbeddings:
    """Small token-vector model for repeatable tests without Gemini calls."""

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[list[float]]:
        del batch_size
        return [self._embed(text, 384) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, 384)

    @staticmethod
    def _embed(text: str, dimensions: int) -> list[float]:
        vector = [0.0] * dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % dimensions
            vector[index] += 1.0
        return vector


@pytest.fixture
def deterministic_embeddings() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()


@pytest.fixture(autouse=True)
def clear_selected_vault():
    vault_session.clear()
    yield
    vault_session.clear()


@pytest.fixture
def selected_fixture_vault():
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "vault"
    selection, _ = vault_session.select(fixture_path)
    return selection
