"""Vault discovery and Markdown parsing into structured chunks."""

import logging
from pathlib import Path
from time import perf_counter

from app.chunking import chunk_markdown
from app.models import Chunk

logger = logging.getLogger(__name__)

IGNORED_DIRECTORY_NAMES = {".git", ".obsidian", ".trash"}


def find_markdown_files(vault_path: Path) -> list[Path]:
    """Return Markdown files, excluding Obsidian and VCS implementation directories."""
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")
    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")

    started = perf_counter()
    files = sorted(
        path
        for path in vault_path.rglob("*.md")
        if not any(parent.name in IGNORED_DIRECTORY_NAMES for parent in path.parents)
    )
    logger.info(
        "timing stage=ingestion.discovery files=%d duration_ms=%.2f",
        len(files),
        (perf_counter() - started) * 1000,
    )
    return files


def parse_vault_chunks(vault_path: Path) -> list[Chunk]:
    """Scan vault and parse all Markdown notes into structured chunks."""
    files = find_markdown_files(vault_path)
    all_chunks: list[Chunk] = []
    file_read_seconds = 0.0
    chunking_seconds = 0.0

    for file_path in files:
        relative_path = file_path.relative_to(vault_path).as_posix()
        read_started = perf_counter()
        raw_text = file_path.read_text(encoding="utf-8")
        file_read_seconds += perf_counter() - read_started

        chunking_started = perf_counter()
        file_chunks = chunk_markdown(source_path=relative_path, raw_text=raw_text)
        chunking_seconds += perf_counter() - chunking_started
        all_chunks.extend(file_chunks)

    logger.info(
        "timing stage=ingestion.parse files=%d chunks=%d file_read_ms=%.2f chunking_ms=%.2f",
        len(files),
        len(all_chunks),
        file_read_seconds * 1000,
        chunking_seconds * 1000,
    )
    return all_chunks
