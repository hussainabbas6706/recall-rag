"""Shared deterministic test helpers."""

import hashlib
import re

import pytest


class DeterministicEmbeddings:
    """Small token-vector model for repeatable tests without Gemini calls."""

    def embed_documents(
        self,
        texts: list[str],
        output_dimensionality: int = 768,
    ) -> list[list[float]]:
        return [self._embed(text, output_dimensionality) for text in texts]

    def embed_query(self, text: str, output_dimensionality: int = 768) -> list[float]:
        return self._embed(text, output_dimensionality)

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
